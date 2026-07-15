#include "voxel_mesher_blocky.h"
#include "../../constants/cube_tables.h"
#include "../../storage/voxel_buffer.h"
#include "../../util/funcs.h"
#include "../../util/span.h"
#include <core/os/os.h>

// Utility functions
namespace {

template <typename T>
void raw_copy_to(PoolVector<T> &to, const Vector<T> &from) {
	to.resize(from.size());
	typename PoolVector<T>::Write w = to.write();
	memcpy(w.ptr(), from.ptr(), from.size() * sizeof(T));
}

const int g_opposite_side[6] = {
	Cube::SIDE_NEGATIVE_X,
	Cube::SIDE_POSITIVE_X,
	Cube::SIDE_POSITIVE_Y,
	Cube::SIDE_NEGATIVE_Y,
	Cube::SIDE_POSITIVE_Z,
	Cube::SIDE_NEGATIVE_Z
};

inline bool is_face_visible(const VoxelLibrary::BakedData &lib, const Voxel::BakedData &vt, uint32_t other_voxel_id, int side) {
	if (other_voxel_id < lib.models.size()) {
		const Voxel::BakedData &other_vt = lib.models[other_voxel_id];
		// A non-occluding neighbour (invisible-but-collidable voxel) is treated as
		// air, but only for an *occluding* vt: a solid block still meshes its face
		// toward it (no hole at the seam). Between two non-occluding voxels the
		// shared face falls through to the transparency rule and is culled like
		// any interior face — otherwise a stack of them would mesh interior
		// horizontal faces and act like climbable ledges.
		if (other_vt.empty || (!other_vt.occludes_neighbors && vt.occludes_neighbors) ||
				(other_vt.transparency_index > vt.transparency_index)) {
			return true;
		} else {
			const unsigned int ai = vt.model.side_pattern_indices[side];
			const unsigned int bi = other_vt.model.side_pattern_indices[g_opposite_side[side]];
			// Patterns are not the same, and B does not occlude A
			return (ai != bi) && !lib.get_side_pattern_occlusion(bi, ai);
		}
	}
	return true;
}

inline bool contributes_to_ao(const VoxelLibrary::BakedData &lib, uint32_t voxel_id) {
	if (voxel_id < lib.models.size()) {
		const Voxel::BakedData &t = lib.models[voxel_id];
		return t.contributes_to_ao;
	}
	return true;
}

} // namespace

// ---- Greedy meshing --------------------------------------------------------
// Merges coplanar, same-voxel, UNIFORMLY-SHADED cube faces into large
// rectangles. Faces with per-corner AO gradients and non-cube models are left
// to generate_blocky_mesh (which skips the faces handled here). A uniformly
// shaded face bakes to one flat colour, so the merged quad is exact — no AO
// detail is lost.

static inline float mb_axis_get(const Vector3 &v, int a) {
	return a == 0 ? v.x : (a == 1 ? v.y : v.z);
}
static inline void mb_axis_set(Vector3 &v, int a, float val) {
	if (a == 0) {
		v.x = val;
	} else if (a == 1) {
		v.y = val;
	} else {
		v.z = val;
	}
}

// Per-corner AO shading for one face (mirrors generate_blocky_mesh). Enu bakes
// AO into vertex colours and its "air" model contributes to AO, so exposed
// faces are never fully lit — instead the greedy pass merges faces whose four
// corners share ONE shade value, which a single flat colour reproduces exactly.
// Returns that common shade (0..3) if the face is uniformly shaded, else -1.
template <typename Type_T>
static int mb_face_uniform_shade(const Span<Type_T> type_buffer, int voxel_index, int side,
		const VoxelLibrary::BakedData &library,
		const VoxelFixedArray<int, Cube::EDGE_COUNT> &edge_neighbor_lut,
		const VoxelFixedArray<int, Cube::CORNER_COUNT> &corner_neighbor_lut) {
	int shaded_corner[8] = { 0 };
	for (unsigned int j = 0; j < 4; ++j) {
		const unsigned int edge = Cube::g_side_edges[side][j];
		const int edge_neighbor_id = type_buffer[voxel_index + edge_neighbor_lut[edge]];
		if (contributes_to_ao(library, edge_neighbor_id)) {
			++shaded_corner[Cube::g_edge_corners[edge][0]];
			++shaded_corner[Cube::g_edge_corners[edge][1]];
		}
	}
	for (unsigned int j = 0; j < 4; ++j) {
		const unsigned int corner = Cube::g_side_corners[side][j];
		if (shaded_corner[corner] == 2) {
			shaded_corner[corner] = 3;
		} else {
			const int corner_neigbor_id = type_buffer[voxel_index + corner_neighbor_lut[corner]];
			if (contributes_to_ao(library, corner_neigbor_id)) {
				++shaded_corner[corner];
			}
		}
	}
	const int s0 = shaded_corner[Cube::g_side_corners[side][0]];
	for (unsigned int j = 1; j < 4; ++j) {
		if (shaded_corner[Cube::g_side_corners[side][j]] != s0) {
			return -1;
		}
	}
	return s0;
}

// If this voxel's face on `side` is one the greedy pass handles — a plain cube
// face (single unit quad, no inner mesh), visible, and uniformly shaded — return
// its shade (0..3); else -1. Both the greedy pass and generate_blocky_mesh test
// this so exactly one emits each face. With bake_occlusion off, shade is 0.
template <typename Type_T>
static inline int mb_greedy_face(const Span<Type_T> type_buffer, int voxel_index, int side,
		const VoxelLibrary::BakedData &library, bool bake_occlusion,
		const VoxelFixedArray<int, Cube::SIDE_COUNT> &side_neighbor_lut,
		const VoxelFixedArray<int, Cube::EDGE_COUNT> &edge_neighbor_lut,
		const VoxelFixedArray<int, Cube::CORNER_COUNT> &corner_neighbor_lut) {
	const int voxel_id = type_buffer[voxel_index];
	if (voxel_id == 0 || !library.has_model(voxel_id)) {
		return -1;
	}
	const Voxel::BakedData &voxel = library.models[voxel_id];
	if (voxel.model.positions.size() != 0 || voxel.model.side_positions[side].size() != 4) {
		return -1; // not a plain cube face
	}
	const uint32_t neighbor_voxel_id = type_buffer[voxel_index + side_neighbor_lut[side]];
	if (!is_face_visible(library, voxel, neighbor_voxel_id, side)) {
		return -1;
	}
	if (!bake_occlusion) {
		return 0;
	}
	return mb_face_uniform_shade(type_buffer, voxel_index, side, library, edge_neighbor_lut, corner_neighbor_lut);
}

// Emit one merged rectangle: `w`x`h` voxels on `side`, base voxel at padded
// (nc along normal axis na, u0 along ua, v0 along va).
static void mb_emit_rect(
		VoxelFixedArray<VoxelMesherBlocky::Arrays, VoxelMesherBlocky::MAX_MATERIALS> &out_arrays_per_material,
		const VoxelLibrary::BakedData &library, int voxel_id, int side,
		int na, int ua, int va, int nc, int u0, int v0, int w, int h,
		int shade, float baked_occlusion_darkness) {
	const Voxel::BakedData &voxel = library.models[voxel_id];
	VoxelMesherBlocky::Arrays &arrays = out_arrays_per_material[voxel.material_id];
	const std::vector<Vector3> &sp = voxel.model.side_positions[side];
	const std::vector<Vector2> &su = voxel.model.side_uvs[side];
	const std::vector<int> &si = voxel.model.side_indices[side];
	const std::vector<float> &st = voxel.model.side_tangents[side];
	const int base = arrays.positions.size();
	const int PAD = VoxelMesherBlocky::PADDING;
	const Vector3 normal = Cube::g_side_normals[side].to_vec3();
	// Uniform shade over the whole face: one flat colour matches what the
	// per-face path bakes (each vertex takes its nearest corner's shade, and
	// all four corners share `shade` here).
	float gs = 1.0f - CLAMP(baked_occlusion_darkness * (float)shade, 0.0f, 1.0f);
	const Color rect_color = Color(gs, gs, gs) * voxel.color;

	for (unsigned int k = 0; k < sp.size(); ++k) {
		const Vector3 &uv_vert = sp[k]; // unit-cube face vertex, components 0/1
		Vector3 out;
		mb_axis_set(out, na, (nc - PAD) + mb_axis_get(uv_vert, na));
		mb_axis_set(out, ua, (u0 - PAD) + (mb_axis_get(uv_vert, ua) == 0.f ? 0.f : (float)w));
		mb_axis_set(out, va, (v0 - PAD) + (mb_axis_get(uv_vert, va) == 0.f ? 0.f : (float)h));
		arrays.positions.push_back(out);
		arrays.normals.push_back(normal);
		arrays.colors.push_back(rect_color);
		// Tile the texture across the merged quad (0..w, 0..h) instead of
		// stretching one 0..1 copy. Needs REPEAT wrap on the material.
		const Vector2 uv = (k < su.size()) ? su[k] : Vector2();
		arrays.uvs.push_back(Vector2(uv.x * w, uv.y * h));
		if (st.size() >= (k + 1) * 4) {
			for (int t = 0; t < 4; ++t) {
				arrays.tangents.push_back(st[k * 4 + t]);
			}
		}
	}
	for (unsigned int j = 0; j < si.size(); ++j) {
		arrays.indices.push_back(base + si[j]);
	}
}

template <typename Type_T>
static void generate_blocky_greedy_pass(
		VoxelFixedArray<VoxelMesherBlocky::Arrays, VoxelMesherBlocky::MAX_MATERIALS> &out_arrays_per_material,
		const Span<Type_T> type_buffer, const Vector3i block_size,
		const VoxelLibrary::BakedData &library, bool bake_occlusion,
		float baked_occlusion_darkness, bool cull_down_faces) {
	const int row_size = block_size.y;
	const int deck_size = block_size.x * row_size;
	const int stride[3] = { row_size, 1, deck_size }; // X, Y, Z index strides

	// Neighbor LUTs (same construction as generate_blocky_mesh) for visibility
	// and AO. Duplicated so this pass stands alone.
	VoxelFixedArray<int, Cube::SIDE_COUNT> side_neighbor_lut;
	side_neighbor_lut[Cube::SIDE_LEFT] = row_size;
	side_neighbor_lut[Cube::SIDE_RIGHT] = -row_size;
	side_neighbor_lut[Cube::SIDE_BACK] = -deck_size;
	side_neighbor_lut[Cube::SIDE_FRONT] = deck_size;
	side_neighbor_lut[Cube::SIDE_BOTTOM] = -1;
	side_neighbor_lut[Cube::SIDE_TOP] = 1;
	VoxelFixedArray<int, Cube::EDGE_COUNT> edge_neighbor_lut;
	edge_neighbor_lut[Cube::EDGE_BOTTOM_BACK] = side_neighbor_lut[Cube::SIDE_BOTTOM] + side_neighbor_lut[Cube::SIDE_BACK];
	edge_neighbor_lut[Cube::EDGE_BOTTOM_FRONT] = side_neighbor_lut[Cube::SIDE_BOTTOM] + side_neighbor_lut[Cube::SIDE_FRONT];
	edge_neighbor_lut[Cube::EDGE_BOTTOM_LEFT] = side_neighbor_lut[Cube::SIDE_BOTTOM] + side_neighbor_lut[Cube::SIDE_LEFT];
	edge_neighbor_lut[Cube::EDGE_BOTTOM_RIGHT] = side_neighbor_lut[Cube::SIDE_BOTTOM] + side_neighbor_lut[Cube::SIDE_RIGHT];
	edge_neighbor_lut[Cube::EDGE_BACK_LEFT] = side_neighbor_lut[Cube::SIDE_BACK] + side_neighbor_lut[Cube::SIDE_LEFT];
	edge_neighbor_lut[Cube::EDGE_BACK_RIGHT] = side_neighbor_lut[Cube::SIDE_BACK] + side_neighbor_lut[Cube::SIDE_RIGHT];
	edge_neighbor_lut[Cube::EDGE_FRONT_LEFT] = side_neighbor_lut[Cube::SIDE_FRONT] + side_neighbor_lut[Cube::SIDE_LEFT];
	edge_neighbor_lut[Cube::EDGE_FRONT_RIGHT] = side_neighbor_lut[Cube::SIDE_FRONT] + side_neighbor_lut[Cube::SIDE_RIGHT];
	edge_neighbor_lut[Cube::EDGE_TOP_BACK] = side_neighbor_lut[Cube::SIDE_TOP] + side_neighbor_lut[Cube::SIDE_BACK];
	edge_neighbor_lut[Cube::EDGE_TOP_FRONT] = side_neighbor_lut[Cube::SIDE_TOP] + side_neighbor_lut[Cube::SIDE_FRONT];
	edge_neighbor_lut[Cube::EDGE_TOP_LEFT] = side_neighbor_lut[Cube::SIDE_TOP] + side_neighbor_lut[Cube::SIDE_LEFT];
	edge_neighbor_lut[Cube::EDGE_TOP_RIGHT] = side_neighbor_lut[Cube::SIDE_TOP] + side_neighbor_lut[Cube::SIDE_RIGHT];
	VoxelFixedArray<int, Cube::CORNER_COUNT> corner_neighbor_lut;
	corner_neighbor_lut[Cube::CORNER_BOTTOM_BACK_LEFT] = side_neighbor_lut[Cube::SIDE_BOTTOM] + side_neighbor_lut[Cube::SIDE_BACK] + side_neighbor_lut[Cube::SIDE_LEFT];
	corner_neighbor_lut[Cube::CORNER_BOTTOM_BACK_RIGHT] = side_neighbor_lut[Cube::SIDE_BOTTOM] + side_neighbor_lut[Cube::SIDE_BACK] + side_neighbor_lut[Cube::SIDE_RIGHT];
	corner_neighbor_lut[Cube::CORNER_BOTTOM_FRONT_RIGHT] = side_neighbor_lut[Cube::SIDE_BOTTOM] + side_neighbor_lut[Cube::SIDE_FRONT] + side_neighbor_lut[Cube::SIDE_RIGHT];
	corner_neighbor_lut[Cube::CORNER_BOTTOM_FRONT_LEFT] = side_neighbor_lut[Cube::SIDE_BOTTOM] + side_neighbor_lut[Cube::SIDE_FRONT] + side_neighbor_lut[Cube::SIDE_LEFT];
	corner_neighbor_lut[Cube::CORNER_TOP_BACK_LEFT] = side_neighbor_lut[Cube::SIDE_TOP] + side_neighbor_lut[Cube::SIDE_BACK] + side_neighbor_lut[Cube::SIDE_LEFT];
	corner_neighbor_lut[Cube::CORNER_TOP_BACK_RIGHT] = side_neighbor_lut[Cube::SIDE_TOP] + side_neighbor_lut[Cube::SIDE_BACK] + side_neighbor_lut[Cube::SIDE_RIGHT];
	corner_neighbor_lut[Cube::CORNER_TOP_FRONT_RIGHT] = side_neighbor_lut[Cube::SIDE_TOP] + side_neighbor_lut[Cube::SIDE_FRONT] + side_neighbor_lut[Cube::SIDE_RIGHT];
	corner_neighbor_lut[Cube::CORNER_TOP_FRONT_LEFT] = side_neighbor_lut[Cube::SIDE_TOP] + side_neighbor_lut[Cube::SIDE_FRONT] + side_neighbor_lut[Cube::SIDE_LEFT];

	const int mn[3] = { (int)VoxelMesherBlocky::PADDING, (int)VoxelMesherBlocky::PADDING, (int)VoxelMesherBlocky::PADDING };
	const int mx[3] = { block_size.x - (int)VoxelMesherBlocky::PADDING,
		block_size.y - (int)VoxelMesherBlocky::PADDING, block_size.z - (int)VoxelMesherBlocky::PADDING };

	// side -> (normal axis, in-plane axes u, v)
	struct SideAxes { int side; int na; int ua; int va; };
	static const SideAxes SIDES[6] = {
		{ Cube::SIDE_LEFT, 0, 1, 2 }, { Cube::SIDE_RIGHT, 0, 1, 2 },
		{ Cube::SIDE_BOTTOM, 1, 0, 2 }, { Cube::SIDE_TOP, 1, 0, 2 },
		{ Cube::SIDE_BACK, 2, 0, 1 }, { Cube::SIDE_FRONT, 2, 0, 1 }
	};

	std::vector<int> mask;
	for (int s = 0; s < 6; ++s) {
		const int side = SIDES[s].side;
		if (cull_down_faces && side == Cube::SIDE_NEGATIVE_Y) {
			continue;
		}
		const int na = SIDES[s].na, ua = SIDES[s].ua, va = SIDES[s].va;
		const int W = mx[ua] - mn[ua];
		const int H = mx[va] - mn[va];
		if (W <= 0 || H <= 0) {
			continue;
		}
		mask.assign((size_t)W * H, 0);
		for (int nc = mn[na]; nc < mx[na]; ++nc) {
			for (int vv = 0; vv < H; ++vv) {
				for (int uu = 0; uu < W; ++uu) {
					const int idx = nc * stride[na] + (mn[ua] + uu) * stride[ua] + (mn[va] + vv) * stride[va];
					int key = 0;
					const int shade = mb_greedy_face(type_buffer, idx, side, library, bake_occlusion,
							side_neighbor_lut, edge_neighbor_lut, corner_neighbor_lut);
					if (shade >= 0) {
						// key packs voxel id (>=1) and shade (0..3) so only faces
						// with identical colour AND shade merge. Never 0 for a real
						// voxel, so 0 cleanly marks an ineligible cell.
						key = (type_buffer[idx] << 2) | shade;
					}
					mask[uu + vv * W] = key;
				}
			}
			for (int j = 0; j < H; ++j) {
				for (int i = 0; i < W;) {
					const int key = mask[i + j * W];
					if (key == 0) {
						++i;
						continue;
					}
					int w = 1;
					while (i + w < W && mask[(i + w) + j * W] == key) {
						++w;
					}
					int hh = 1;
					bool stop = false;
					while (j + hh < H && !stop) {
						for (int k = 0; k < w; ++k) {
							if (mask[(i + k) + (j + hh) * W] != key) {
								stop = true;
								break;
							}
						}
						if (!stop) {
							++hh;
						}
					}
					mb_emit_rect(out_arrays_per_material, library, key >> 2, side, na, ua, va,
							nc, mn[ua] + i, mn[va] + j, w, hh, key & 3, baked_occlusion_darkness);
					for (int b = 0; b < hh; ++b) {
						for (int a = 0; a < w; ++a) {
							mask[(i + a) + (j + b) * W] = 0;
						}
					}
					i += w;
				}
			}
		}
	}
}


template <typename Type_T>
static void generate_blocky_mesh(
		VoxelFixedArray<VoxelMesherBlocky::Arrays, VoxelMesherBlocky::MAX_MATERIALS> &out_arrays_per_material,
		const Span<Type_T> type_buffer,
		const Vector3i block_size,
		const VoxelLibrary::BakedData &library,
		bool bake_occlusion, float baked_occlusion_darkness, bool cull_down_faces, bool greedy) {
	ERR_FAIL_COND(block_size.x < static_cast<int>(2 * VoxelMesherBlocky::PADDING) ||
				  block_size.y < static_cast<int>(2 * VoxelMesherBlocky::PADDING) ||
				  block_size.z < static_cast<int>(2 * VoxelMesherBlocky::PADDING));

	// Build lookup tables so to speed up voxel access.
	// These are values to add to an address in order to get given neighbor.

	const int row_size = block_size.y;
	const int deck_size = block_size.x * row_size;

	// Data must be padded, hence the off-by-one
	const Vector3i min = Vector3i(VoxelMesherBlocky::PADDING);
	const Vector3i max = block_size - Vector3i(VoxelMesherBlocky::PADDING);

	int index_offsets[VoxelMesherBlocky::MAX_MATERIALS] = { 0 };

	VoxelFixedArray<int, Cube::SIDE_COUNT> side_neighbor_lut;
	side_neighbor_lut[Cube::SIDE_LEFT] = row_size;
	side_neighbor_lut[Cube::SIDE_RIGHT] = -row_size;
	side_neighbor_lut[Cube::SIDE_BACK] = -deck_size;
	side_neighbor_lut[Cube::SIDE_FRONT] = deck_size;
	side_neighbor_lut[Cube::SIDE_BOTTOM] = -1;
	side_neighbor_lut[Cube::SIDE_TOP] = 1;

	VoxelFixedArray<int, Cube::EDGE_COUNT> edge_neighbor_lut;
	edge_neighbor_lut[Cube::EDGE_BOTTOM_BACK] = side_neighbor_lut[Cube::SIDE_BOTTOM] + side_neighbor_lut[Cube::SIDE_BACK];
	edge_neighbor_lut[Cube::EDGE_BOTTOM_FRONT] = side_neighbor_lut[Cube::SIDE_BOTTOM] + side_neighbor_lut[Cube::SIDE_FRONT];
	edge_neighbor_lut[Cube::EDGE_BOTTOM_LEFT] = side_neighbor_lut[Cube::SIDE_BOTTOM] + side_neighbor_lut[Cube::SIDE_LEFT];
	edge_neighbor_lut[Cube::EDGE_BOTTOM_RIGHT] = side_neighbor_lut[Cube::SIDE_BOTTOM] + side_neighbor_lut[Cube::SIDE_RIGHT];
	edge_neighbor_lut[Cube::EDGE_BACK_LEFT] = side_neighbor_lut[Cube::SIDE_BACK] + side_neighbor_lut[Cube::SIDE_LEFT];
	edge_neighbor_lut[Cube::EDGE_BACK_RIGHT] = side_neighbor_lut[Cube::SIDE_BACK] + side_neighbor_lut[Cube::SIDE_RIGHT];
	edge_neighbor_lut[Cube::EDGE_FRONT_LEFT] = side_neighbor_lut[Cube::SIDE_FRONT] + side_neighbor_lut[Cube::SIDE_LEFT];
	edge_neighbor_lut[Cube::EDGE_FRONT_RIGHT] = side_neighbor_lut[Cube::SIDE_FRONT] + side_neighbor_lut[Cube::SIDE_RIGHT];
	edge_neighbor_lut[Cube::EDGE_TOP_BACK] = side_neighbor_lut[Cube::SIDE_TOP] + side_neighbor_lut[Cube::SIDE_BACK];
	edge_neighbor_lut[Cube::EDGE_TOP_FRONT] = side_neighbor_lut[Cube::SIDE_TOP] + side_neighbor_lut[Cube::SIDE_FRONT];
	edge_neighbor_lut[Cube::EDGE_TOP_LEFT] = side_neighbor_lut[Cube::SIDE_TOP] + side_neighbor_lut[Cube::SIDE_LEFT];
	edge_neighbor_lut[Cube::EDGE_TOP_RIGHT] = side_neighbor_lut[Cube::SIDE_TOP] + side_neighbor_lut[Cube::SIDE_RIGHT];

	VoxelFixedArray<int, Cube::CORNER_COUNT> corner_neighbor_lut;

	corner_neighbor_lut[Cube::CORNER_BOTTOM_BACK_LEFT] =
			side_neighbor_lut[Cube::SIDE_BOTTOM] +
			side_neighbor_lut[Cube::SIDE_BACK] +
			side_neighbor_lut[Cube::SIDE_LEFT];

	corner_neighbor_lut[Cube::CORNER_BOTTOM_BACK_RIGHT] =
			side_neighbor_lut[Cube::SIDE_BOTTOM] +
			side_neighbor_lut[Cube::SIDE_BACK] +
			side_neighbor_lut[Cube::SIDE_RIGHT];

	corner_neighbor_lut[Cube::CORNER_BOTTOM_FRONT_RIGHT] =
			side_neighbor_lut[Cube::SIDE_BOTTOM] +
			side_neighbor_lut[Cube::SIDE_FRONT] +
			side_neighbor_lut[Cube::SIDE_RIGHT];

	corner_neighbor_lut[Cube::CORNER_BOTTOM_FRONT_LEFT] =
			side_neighbor_lut[Cube::SIDE_BOTTOM] +
			side_neighbor_lut[Cube::SIDE_FRONT] +
			side_neighbor_lut[Cube::SIDE_LEFT];

	corner_neighbor_lut[Cube::CORNER_TOP_BACK_LEFT] =
			side_neighbor_lut[Cube::SIDE_TOP] +
			side_neighbor_lut[Cube::SIDE_BACK] +
			side_neighbor_lut[Cube::SIDE_LEFT];

	corner_neighbor_lut[Cube::CORNER_TOP_BACK_RIGHT] =
			side_neighbor_lut[Cube::SIDE_TOP] +
			side_neighbor_lut[Cube::SIDE_BACK] +
			side_neighbor_lut[Cube::SIDE_RIGHT];

	corner_neighbor_lut[Cube::CORNER_TOP_FRONT_RIGHT] =
			side_neighbor_lut[Cube::SIDE_TOP] +
			side_neighbor_lut[Cube::SIDE_FRONT] +
			side_neighbor_lut[Cube::SIDE_RIGHT];

	corner_neighbor_lut[Cube::CORNER_TOP_FRONT_LEFT] =
			side_neighbor_lut[Cube::SIDE_TOP] +
			side_neighbor_lut[Cube::SIDE_FRONT] +
			side_neighbor_lut[Cube::SIDE_LEFT];

	//uint64_t time_prep = OS::get_singleton()->get_ticks_usec() - time_before;
	//time_before = OS::get_singleton()->get_ticks_usec();

	for (unsigned int z = min.z; z < (unsigned int)max.z; ++z) {
		for (unsigned int x = min.x; x < (unsigned int)max.x; ++x) {
			for (unsigned int y = min.y; y < (unsigned int)max.y; ++y) {
				// min and max are chosen such that you can visit 1 neighbor away from the current voxel without size check

				const int voxel_index = y + x * row_size + z * deck_size;
				const int voxel_id = type_buffer[voxel_index];

				if (voxel_id != 0 && library.has_model(voxel_id)) {
					const Voxel::BakedData &voxel = library.models[voxel_id];

					VoxelMesherBlocky::Arrays &arrays = out_arrays_per_material[voxel.material_id];
					int &index_offset = index_offsets[voxel.material_id];

					// Hybrid approach: extract cube faces and decimate those that aren't visible,
					// and still allow voxels to have geometry that is not a cube

					// Sides
					for (unsigned int side = 0; side < Cube::SIDE_COUNT; ++side) {
						if (cull_down_faces && side == Cube::SIDE_NEGATIVE_Y) {
							continue;
						}
						const std::vector<Vector3> &side_positions = voxel.model.side_positions[side];
						const unsigned int vertex_count = side_positions.size();

						if (vertex_count == 0) {
							continue;
						}

						const uint32_t neighbor_voxel_id = type_buffer[voxel_index + side_neighbor_lut[side]];

						if (!is_face_visible(library, voxel, neighbor_voxel_id, side)) {
							continue;
						}

						// The face is visible

						int shaded_corner[8] = { 0 };

						if (bake_occlusion) {
							// Combinatory solution for https://0fps.net/2013/07/03/ambient-occlusion-for-minecraft-like-worlds/
							// (inverted)
							//	function vertexAO(side1, side2, corner) {
							//	  if(side1 && side2) {
							//		return 0
							//	  }
							//	  return 3 - (side1 + side2 + corner)
							//	}

							for (unsigned int j = 0; j < 4; ++j) {
								const unsigned int edge = Cube::g_side_edges[side][j];
								const int edge_neighbor_id = type_buffer[voxel_index + edge_neighbor_lut[edge]];
								if (contributes_to_ao(library, edge_neighbor_id)) {
									++shaded_corner[Cube::g_edge_corners[edge][0]];
									++shaded_corner[Cube::g_edge_corners[edge][1]];
								}
							}
							for (unsigned int j = 0; j < 4; ++j) {
								const unsigned int corner = Cube::g_side_corners[side][j];
								if (shaded_corner[corner] == 2) {
									shaded_corner[corner] = 3;
								} else {
									const int corner_neigbor_id = type_buffer[voxel_index + corner_neighbor_lut[corner]];
									if (contributes_to_ao(library, corner_neigbor_id)) {
										++shaded_corner[corner];
									}
								}
							}
						}

						// The greedy pass emits uniformly-shaded cube faces as merged
						// rectangles; skip them here so they aren't drawn twice. Must
						// match mb_greedy_face exactly (same uniform-shade test).
						if (greedy && voxel.model.positions.size() == 0 && side_positions.size() == 4) {
							bool uniform = true;
							const int s0 = shaded_corner[Cube::g_side_corners[side][0]];
							for (unsigned int j = 1; j < 4; ++j) {
								if (shaded_corner[Cube::g_side_corners[side][j]] != s0) {
									uniform = false;
									break;
								}
							}
							if (uniform) {
								continue;
							}
						}

						const std::vector<Vector2> &side_uvs = voxel.model.side_uvs[side];
						const std::vector<float> &side_tangents = voxel.model.side_tangents[side];

						// Subtracting 1 because the data is padded
						Vector3 pos(x - 1, y - 1, z - 1);

						// Append vertices of the faces in one go, don't use push_back

						{
							const int append_index = arrays.positions.size();
							arrays.positions.resize(arrays.positions.size() + vertex_count);
							Vector3 *w = arrays.positions.data() + append_index;
							for (unsigned int i = 0; i < vertex_count; ++i) {
								w[i] = side_positions[i] + pos;
							}
						}

						{
							const int append_index = arrays.uvs.size();
							arrays.uvs.resize(arrays.uvs.size() + vertex_count);
							memcpy(arrays.uvs.data() + append_index, side_uvs.data(), vertex_count * sizeof(Vector2));
						}

						if (side_tangents.size() > 0) {
							const int append_index = arrays.tangents.size();
							arrays.tangents.resize(arrays.tangents.size() + vertex_count * 4);
							memcpy(arrays.tangents.data() + append_index, side_tangents.data(),
									(vertex_count * 4) * sizeof(float));
						}

						{
							const int append_index = arrays.normals.size();
							arrays.normals.resize(arrays.normals.size() + vertex_count);
							Vector3 *w = arrays.normals.data() + append_index;
							for (unsigned int i = 0; i < vertex_count; ++i) {
								w[i] = Cube::g_side_normals[side].to_vec3();
							}
						}

						{
							const int append_index = arrays.colors.size();
							arrays.colors.resize(arrays.colors.size() + vertex_count);
							Color *w = arrays.colors.data() + append_index;
							const Color modulate_color = voxel.color;

							if (bake_occlusion) {
								for (unsigned int i = 0; i < vertex_count; ++i) {
									Vector3 v = side_positions[i];

									// General purpose occlusion colouring.
									// TODO Optimize for cubes
									// TODO Fix occlusion inconsistency caused by triangles orientation? Not sure if worth it
									float shade = 0;
									for (unsigned int j = 0; j < 4; ++j) {
										unsigned int corner = Cube::g_side_corners[side][j];
										if (shaded_corner[corner]) {
											float s = baked_occlusion_darkness * static_cast<float>(shaded_corner[corner]);
											//float k = 1.f - Cube::g_corner_position[corner].distance_to(v);
											float k = 1.f - Cube::g_corner_position[corner].distance_squared_to(v);
											if (k < 0.0) {
												k = 0.0;
											}
											s *= k;
											if (s > shade) {
												shade = s;
											}
										}
									}
									const float gs = 1.0 - shade;
									w[i] = Color(gs, gs, gs) * modulate_color;
								}

							} else {
								for (unsigned int i = 0; i < vertex_count; ++i) {
									w[i] = modulate_color;
								}
							}
						}

						const std::vector<int> &side_indices = voxel.model.side_indices[side];
						const unsigned int index_count = side_indices.size();

						{
							int i = arrays.indices.size();
							arrays.indices.resize(arrays.indices.size() + index_count);
							int *w = arrays.indices.data();
							for (unsigned int j = 0; j < index_count; ++j) {
								w[i++] = index_offset + side_indices[j];
							}
						}

						index_offset += vertex_count;
					}

					// Inside
					if (voxel.model.positions.size() != 0) {
						// TODO Get rid of push_backs

						const std::vector<Vector3> &positions = voxel.model.positions;
						const unsigned int vertex_count = positions.size();
						const Color modulate_color = voxel.color;

						const std::vector<Vector3> &normals = voxel.model.normals;
						const std::vector<Vector2> &uvs = voxel.model.uvs;
						const std::vector<float> &tangents = voxel.model.tangents;

						const Vector3 pos(x - 1, y - 1, z - 1);

						if (tangents.size() > 0) {
							const int append_index = arrays.tangents.size();
							arrays.tangents.resize(arrays.tangents.size() + vertex_count * 4);
							memcpy(arrays.tangents.data() + append_index, tangents.data(),
									(vertex_count * 4) * sizeof(float));
						}

						for (unsigned int i = 0; i < vertex_count; ++i) {
							arrays.normals.push_back(normals[i]);
							arrays.uvs.push_back(uvs[i]);
							arrays.positions.push_back(positions[i] + pos);
							// TODO handle ambient occlusion on inner parts
							arrays.colors.push_back(modulate_color);
						}

						const std::vector<int> &indices = voxel.model.indices;
						const unsigned int index_count = indices.size();

						for (unsigned int i = 0; i < index_count; ++i) {
							arrays.indices.push_back(index_offset + indices[i]);
						}

						index_offset += vertex_count;
					}
				}
			}
		}
	}
}

thread_local VoxelMesherBlocky::Cache VoxelMesherBlocky::_cache;

VoxelMesherBlocky::VoxelMesherBlocky() {
	set_padding(PADDING, PADDING);

	// Default library, less steps to setup in editor
	Ref<VoxelLibrary> library;
	library.instance();
	library->load_default();
	_parameters.library = library;
}

VoxelMesherBlocky::~VoxelMesherBlocky() {
}

void VoxelMesherBlocky::set_library(Ref<VoxelLibrary> library) {
	RWLockWrite wlock(_parameters_lock);
	_parameters.library = library;
}

Ref<VoxelLibrary> VoxelMesherBlocky::get_library() const {
	RWLockRead rlock(_parameters_lock);
	return _parameters.library;
}

void VoxelMesherBlocky::set_occlusion_darkness(float darkness) {
	RWLockWrite wlock(_parameters_lock);
	_parameters.baked_occlusion_darkness = clamp(darkness, 0.0f, 1.0f);
}

float VoxelMesherBlocky::get_occlusion_darkness() const {
	RWLockRead rlock(_parameters_lock);
	return _parameters.baked_occlusion_darkness;
}

void VoxelMesherBlocky::set_occlusion_enabled(bool enable) {
	RWLockWrite wlock(_parameters_lock);
	_parameters.bake_occlusion = enable;
}

bool VoxelMesherBlocky::get_occlusion_enabled() const {
	RWLockRead rlock(_parameters_lock);
	return _parameters.bake_occlusion;
}

void VoxelMesherBlocky::build(VoxelMesher::Output &output, const VoxelMesher::Input &input) {
	const int channel = VoxelBuffer::CHANNEL_TYPE;
	Parameters params;
	{
		RWLockRead rlock(_parameters_lock);
		params = _parameters;
	}

	ERR_FAIL_COND(params.library.is_null());

	Cache &cache = _cache;

	for (unsigned int i = 0; i < cache.arrays_per_material.size(); ++i) {
		Arrays &a = cache.arrays_per_material[i];
		a.clear();
	}

	float baked_occlusion_darkness = 0;
	if (params.bake_occlusion) {
		baked_occlusion_darkness = params.baked_occlusion_darkness / 3.0f;
	}

	// The technique is Culled faces.
	// Could be improved with greedy meshing: https://0fps.net/2012/06/30/meshing-in-a-minecraft-game/
	// However I don't feel it's worth it yet:
	// - Not so much gain for organic worlds with lots of texture variations
	// - Works well with cubes but not with any shape
	// - Slower
	// => Could be implemented in a separate class?

	const VoxelBufferInternal &voxels = input.voxels;
#ifdef TOOLS_ENABLED
	if (input.lod != 0) {
		WARN_PRINT("VoxelMesherBlocky received lod != 0, it is not supported");
	}
#endif

	// Iterate 3D padded data to extract voxel faces.
	// This is the most intensive job in this class, so all required data should be as fit as possible.

	// The buffer we receive MUST be dense (i.e not compressed, and channels allocated).
	// That means we can use raw pointers to voxel data inside instead of using the higher-level getters,
	// and then save a lot of time.

	if (voxels.get_channel_compression(channel) == VoxelBufferInternal::COMPRESSION_UNIFORM) {
		// All voxels have the same type.
		// If it's all air, nothing to do. If it's all cubes, nothing to do either.
		// TODO Handle edge case of uniform block with non-cubic voxels!
		// If the type of voxel still produces geometry in this situation (which is an absurd use case but not an error),
		// decompress into a backing array to still allow the use of the same algorithm.
		return;

	} else if (voxels.get_channel_compression(channel) != VoxelBufferInternal::COMPRESSION_NONE) {
		// No other form of compression is allowed
		ERR_PRINT("VoxelMesherBlocky received unsupported voxel compression");
		return;
	}

	Span<uint8_t> raw_channel;
	if (!voxels.get_channel_raw(channel, raw_channel)) {
		/*       _
		//      | \
		//     /\ \\
		//    / /|\\\
		//    | |\ \\\
		//    | \_\ \\|
		//    |    |  )
		//     \   |  |
		//      \    /
		*/
		// Case supposedly handled before...
		ERR_PRINT("Something wrong happened");
		return;
	}

	const Vector3i block_size = voxels.get_size();
	const VoxelBufferInternal::Depth channel_depth = voxels.get_channel_depth(channel);

	{
		// We can only access baked data. Only this data is made for multithreaded access.
		RWLockRead lock(params.library->get_baked_data_rw_lock());
		const VoxelLibrary::BakedData &library_baked_data = params.library->get_baked_data();

		switch (channel_depth) {
			case VoxelBufferInternal::DEPTH_8_BIT:
				if (input.greedy) {
					generate_blocky_greedy_pass(cache.arrays_per_material, raw_channel,
							block_size, library_baked_data, params.bake_occlusion,
							baked_occlusion_darkness, input.cull_down_faces);
				}
				generate_blocky_mesh(cache.arrays_per_material, raw_channel,
						block_size, library_baked_data, params.bake_occlusion, baked_occlusion_darkness,
						input.cull_down_faces, input.greedy);
				break;

			case VoxelBufferInternal::DEPTH_16_BIT:
				if (input.greedy) {
					generate_blocky_greedy_pass(cache.arrays_per_material, raw_channel.reinterpret_cast_to<uint16_t>(),
							block_size, library_baked_data, params.bake_occlusion,
							baked_occlusion_darkness, input.cull_down_faces);
				}
				generate_blocky_mesh(cache.arrays_per_material, raw_channel.reinterpret_cast_to<uint16_t>(),
						block_size, library_baked_data, params.bake_occlusion, baked_occlusion_darkness,
						input.cull_down_faces, input.greedy);
				break;

			default:
				ERR_PRINT("Unsupported voxel depth");
				return;
		}
	}

	// TODO We could return a single byte array and use Mesh::add_surface down the line?

	for (unsigned int i = 0; i < MAX_MATERIALS; ++i) {
		const Arrays &arrays = cache.arrays_per_material[i];
		if (arrays.positions.size() != 0) {
			Array mesh_arrays;
			mesh_arrays.resize(Mesh::ARRAY_MAX);

			{
				PoolVector<Vector3> positions;
				PoolVector<Vector2> uvs;
				PoolVector<Vector3> normals;
				PoolVector<Color> colors;
				PoolVector<int> indices;

				raw_copy_to(positions, arrays.positions);
				raw_copy_to(uvs, arrays.uvs);
				raw_copy_to(normals, arrays.normals);
				raw_copy_to(colors, arrays.colors);
				raw_copy_to(indices, arrays.indices);

				mesh_arrays[Mesh::ARRAY_VERTEX] = positions;
				mesh_arrays[Mesh::ARRAY_TEX_UV] = uvs;
				mesh_arrays[Mesh::ARRAY_NORMAL] = normals;
				mesh_arrays[Mesh::ARRAY_COLOR] = colors;
				mesh_arrays[Mesh::ARRAY_INDEX] = indices;
				if (arrays.tangents.size() > 0) {
					PoolVector<float> tangents;
					raw_copy_to(tangents, arrays.tangents);
					mesh_arrays[Mesh::ARRAY_TANGENT] = tangents;
				}
			}

			output.surfaces.push_back(mesh_arrays);

		} else {
			// Empty
			output.surfaces.push_back(Array());
		}
	}

	output.primitive_type = Mesh::PRIMITIVE_TRIANGLES;
}

Ref<Resource> VoxelMesherBlocky::duplicate(bool p_subresources) const {
	Parameters params;
	{
		RWLockRead rlock(_parameters_lock);
		params = _parameters;
	}

	if (p_subresources && params.library.is_valid()) {
		params.library = params.library->duplicate(true);
	}

	VoxelMesherBlocky *c = memnew(VoxelMesherBlocky);
	c->_parameters = params;
	return c;
}

int VoxelMesherBlocky::get_used_channels_mask() const {
	return (1 << VoxelBuffer::CHANNEL_TYPE);
}

void VoxelMesherBlocky::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_library", "voxel_library"), &VoxelMesherBlocky::set_library);
	ClassDB::bind_method(D_METHOD("get_library"), &VoxelMesherBlocky::get_library);

	ClassDB::bind_method(D_METHOD("set_occlusion_enabled", "enable"), &VoxelMesherBlocky::set_occlusion_enabled);
	ClassDB::bind_method(D_METHOD("get_occlusion_enabled"), &VoxelMesherBlocky::get_occlusion_enabled);

	ClassDB::bind_method(D_METHOD("set_occlusion_darkness", "value"), &VoxelMesherBlocky::set_occlusion_darkness);
	ClassDB::bind_method(D_METHOD("get_occlusion_darkness"), &VoxelMesherBlocky::get_occlusion_darkness);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "library", PROPERTY_HINT_RESOURCE_TYPE, "VoxelLibrary"),
			"set_library", "get_library");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "occlusion_enabled"), "set_occlusion_enabled", "get_occlusion_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "occlusion_darkness", PROPERTY_HINT_RANGE, "0,1,0.01"),
			"set_occlusion_darkness", "get_occlusion_darkness");
}
