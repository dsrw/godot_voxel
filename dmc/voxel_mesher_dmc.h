#ifndef VOXEL_MESHER_DMC_H
#define VOXEL_MESHER_DMC_H

#include "../voxel_buffer.h"
#include "hermite_value.h"
#include "mesh_builder.h"
#include "scene/resources/mesh.h"

namespace dmc {

struct DualCell {
	Vector3 corners[8];
	HermiteValue values[8];
	bool has_values;

	DualCell() :
			has_values(false) {}

	inline void set_corner(int i, Vector3 vertex, HermiteValue value) {
		CRASH_COND(i < 0 || i >= 8);
		corners[i] = vertex;
		values[i] = value;
	}
};

struct DualGrid {
	std::vector<DualCell> cells;
};

} // namespace dmc

class VoxelMesherDMC : public Reference {
	GDCLASS(VoxelMesherDMC, Reference)
public:
	enum Mode {
		MODE_NORMAL,
		MODE_WIREFRAME,
		MODE_DEBUG_OCTREE,
		MODE_DEBUG_DUAL_GRID
	};

	Ref<ArrayMesh> build_mesh(const VoxelBuffer &voxels, real_t geometric_error, Mode mode);
	Dictionary get_stats() const;

protected:
	static void _bind_methods();
	Ref<ArrayMesh> _build_mesh_b(Ref<VoxelBuffer> voxels, real_t geometric_error, Mode mode);

private:
	dmc::MeshBuilder _mesh_builder;
	dmc::DualGrid _dual_grid;

	struct Stats {
		real_t octree_build_time;
		real_t dualgrid_derivation_time;
		real_t meshing_time;
		real_t commit_time;
	};

	Stats _stats;
};

VARIANT_ENUM_CAST(VoxelMesherDMC::Mode)

#endif // VOXEL_MESHER_DMC_H