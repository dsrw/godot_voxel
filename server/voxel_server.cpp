#include "voxel_server.h"

#include <limits>
#include "../constants/voxel_constants.h"
#include "../storage/voxel_memory_pool.h"
#include "../util/funcs.h"
#include "../util/godot/funcs.h"
#include "../util/macros.h"
#include "../util/profiling.h"
#include "voxel_async_dependency_tracker.h"

#include <core/os/memory.h>
#include <scene/main/viewport.h>
#include <thread>

namespace {
VoxelServer *g_voxel_server = nullptr;
// Could be atomics, but it's for debugging so I don't bother for now
int g_debug_generate_tasks_count = 0;
int g_debug_stream_tasks_count = 0;
int g_debug_mesh_tasks_count = 0;
} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VoxelTimeSpreadTaskRunner::~VoxelTimeSpreadTaskRunner() {
	flush();
}

void VoxelTimeSpreadTaskRunner::push(IVoxelTimeSpreadTask *task) {
	_tasks.push(task);
}

void VoxelTimeSpreadTaskRunner::process(uint64_t time_budget_usec) {
	VOXEL_PROFILE_SCOPE();
	const OS &os = *OS::get_singleton();

	if (_tasks.size() > 0) {
		const uint64_t time_before = os.get_ticks_usec();

		// Do at least one task
		do {
			IVoxelTimeSpreadTask *task = _tasks.front();
			_tasks.pop();
			task->run();
			// TODO Call recycling function instead?
			memdelete(task);

		} while (_tasks.size() > 0 && os.get_ticks_usec() - time_before < time_budget_usec);
	}
}

void VoxelTimeSpreadTaskRunner::flush() {
	while (!_tasks.empty()) {
		IVoxelTimeSpreadTask *task = _tasks.front();
		_tasks.pop();
		task->run();
		memdelete(task);
	}
}

unsigned int VoxelTimeSpreadTaskRunner::get_pending_count() const {
	return _tasks.size();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VoxelServer *VoxelServer::get_singleton() {
	CRASH_COND_MSG(g_voxel_server == nullptr, "Accessing singleton while it's null");
	return g_voxel_server;
}

void VoxelServer::create_singleton() {
	CRASH_COND_MSG(g_voxel_server != nullptr, "Creating singleton twice");
	g_voxel_server = memnew(VoxelServer);
}

void VoxelServer::destroy_singleton() {
	CRASH_COND_MSG(g_voxel_server == nullptr, "Destroying singleton twice");
	memdelete(g_voxel_server);
	g_voxel_server = nullptr;
}

VoxelServer::VoxelServer() {
	const int hw_threads_hint = std::thread::hardware_concurrency();
	PRINT_VERBOSE(String("Voxel: HW threads hint: {0}").format(varray(hw_threads_hint)));

	// Compute thread count for general pool.
	// Note that the I/O thread counts as one used thread and will always be present.

	// "RST" means changing the property requires an editor restart (or game restart)
	GLOBAL_DEF_RST("voxel/threads/count/minimum", 1);
	ProjectSettings::get_singleton()->set_custom_property_info("voxel/threads/count/minimum",
			PropertyInfo(Variant::INT, "voxel/threads/count/minimum", PROPERTY_HINT_RANGE, "1,64"));

	GLOBAL_DEF_RST("voxel/threads/count/margin_below_max", 1);
	ProjectSettings::get_singleton()->set_custom_property_info("voxel/threads/count/margin_below_max",
			PropertyInfo(Variant::INT, "voxel/threads/count/margin_below_max", PROPERTY_HINT_RANGE, "1,64"));

	GLOBAL_DEF_RST("voxel/threads/count/ratio_over_max", 0.5f);
	ProjectSettings::get_singleton()->set_custom_property_info("voxel/threads/count/ratio_over_max",
			PropertyInfo(Variant::REAL, "voxel/threads/count/ratio_over_max", PROPERTY_HINT_RANGE, "0,1,0.1"));

	GLOBAL_DEF_RST("voxel/threads/main/time_budget_ms", 8);
	ProjectSettings::get_singleton()->set_custom_property_info("voxel/threads/main/time_budget_ms",
			PropertyInfo(Variant::INT, "voxel/threads/main/time_budget_ms", PROPERTY_HINT_RANGE, "0,1000"));

	_main_thread_time_budget_usec =
			1000 * int(ProjectSettings::get_singleton()->get("voxel/threads/main/time_budget_ms"));

	const int minimum_thread_count = max(1, int(ProjectSettings::get_singleton()->get("voxel/threads/count/minimum")));

	// How many threads below available count on the CPU should we set as limit
	const int thread_count_margin =
			max(1, int(ProjectSettings::get_singleton()->get("voxel/threads/count/margin_below_max")));

	// Portion of available CPU threads to attempt using
	const float threads_ratio =
			clamp(float(ProjectSettings::get_singleton()->get("voxel/threads/count/ratio_over_max")), 0.f, 1.f);

	const int maximum_thread_count = max(hw_threads_hint - thread_count_margin, minimum_thread_count);
	// `-1` is for the stream thread
	const int thread_count_by_ratio = int(Math::round(float(threads_ratio) * hw_threads_hint)) - 1;
	const int thread_count = clamp(thread_count_by_ratio, minimum_thread_count, maximum_thread_count);
	PRINT_VERBOSE(String("Voxel: automatic thread count set to {0}").format(varray(thread_count)));

	if (thread_count > hw_threads_hint) {
		WARN_PRINT("Configured thread count exceeds hardware thread count. Performance may not be optimal");
	}

	// I/O can't be more than 1 thread. File access with more threads isn't worth it.
	// This thread isn't configurable at the moment.
	_streaming_thread_pool.set_name("Voxel streaming");
	_streaming_thread_pool.set_thread_count(1);
	_streaming_thread_pool.set_priority_update_period(300);
	// Batching is only to give a chance for file I/O tasks to be grouped and reduce open/close calls.
	// But in the end it might be better to move this idea to the tasks themselves?
	_streaming_thread_pool.set_batch_count(16);

	_general_thread_pool.set_name("Voxel general");
	_general_thread_pool.set_thread_count(thread_count);
	_general_thread_pool.set_priority_update_period(200);
	_general_thread_pool.set_batch_count(1);

	// Init world
	_world.shared_priority_dependency = gd_make_shared<PriorityDependencyShared>();

	PRINT_VERBOSE(String("Size of BlockDataRequest: {0}").format(varray((int)sizeof(BlockDataRequest))));
	PRINT_VERBOSE(String("Size of BlockMeshRequest: {0}").format(varray((int)sizeof(BlockMeshRequest))));
}

VoxelServer::~VoxelServer() {
	// The GDScriptLanguage singleton can get destroyed before ours, so any script referenced by tasks
	// cannot be freed. To work this around, tasks are cleared when the scene tree autoload is destroyed.
	// So normally there should not be any task left to clear here,
	// but doing it anyways for correctness, it's how it should have been...
	// See https://github.com/Zylann/godot_voxel/issues/189
	wait_and_clear_all_tasks(true);
}

void VoxelServer::wait_and_clear_all_tasks(bool warn) {
	_streaming_thread_pool.wait_for_all_tasks();
	_general_thread_pool.wait_for_all_tasks();

	// Wait a second time because the generation pool can generate streaming requests
	_streaming_thread_pool.wait_for_all_tasks();

	_streaming_thread_pool.dequeue_completed_tasks([warn](IVoxelTask *task) {
		if (warn) {
			WARN_PRINT("Streaming tasks remain on module cleanup, "
					   "this could become a problem if they reference scripts");
		}
		memdelete(task);
	});

	_general_thread_pool.dequeue_completed_tasks([warn](IVoxelTask *task) {
		if (warn) {
			WARN_PRINT("General tasks remain on module cleanup, "
					   "this could become a problem if they reference scripts");
		}
		memdelete(task);
	});
}

int VoxelServer::get_priority(const PriorityDependency &dep, uint8_t lod_index, float *out_closest_distance_sq) {
	const std::vector<Vector3> &viewer_positions = dep.shared->viewers;
	const Vector3 block_position = dep.world_position;

	float closest_distance_sq = 99999.f;
	if (viewer_positions.size() == 0) {
		// Assume origin
		closest_distance_sq = block_position.length_squared();
	} else {
		for (size_t i = 0; i < viewer_positions.size(); ++i) {
			float d = viewer_positions[i].distance_squared_to(block_position);
			if (d < closest_distance_sq) {
				closest_distance_sq = d;
			}
		}
	}

	if (out_closest_distance_sq != nullptr) {
		*out_closest_distance_sq = closest_distance_sq;
	}

	// TODO Any way to optimize out the sqrt?
	// I added it because the LOD modifier was not working with squared distances,
	// which led blocks to subdivide too much compared to their neighbors, making cracks more likely to happen
	int priority = static_cast<int>(Math::sqrt(closest_distance_sq));

	// TODO Prioritizing LOD makes generation slower... but not prioritizing makes cracks more likely to appear...
	// This could be fixed by allowing the volume to preemptively request blocks of the next LOD?
	//
	// Higher lod indexes come first to allow the octree to subdivide.
	// Then comes distance, which is modified by how much in view the block is
	priority += (VoxelConstants::MAX_LOD - lod_index) * 10000;

	return priority;
}

uint32_t VoxelServer::add_volume(VolumeCallbacks callbacks, VolumeType type) {
	CRASH_COND(!callbacks.check_callbacks());
	Volume volume;
	volume.type = type;
	volume.callbacks = callbacks;
	volume.meshing_dependency = gd_make_shared<MeshingDependency>();
	return _world.volumes.create(volume);
}

void VoxelServer::set_volume_transform(uint32_t volume_id, Transform t) {
	Volume &volume = _world.volumes.get(volume_id);
	volume.transform = t;
}

void VoxelServer::set_volume_render_block_size(uint32_t volume_id, uint32_t block_size) {
	Volume &volume = _world.volumes.get(volume_id);
	volume.render_block_size = block_size;
}

void VoxelServer::set_volume_data_block_size(uint32_t volume_id, uint32_t block_size) {
	Volume &volume = _world.volumes.get(volume_id);
	volume.data_block_size = block_size;
}

void VoxelServer::set_volume_stream(uint32_t volume_id, Ref<VoxelStream> stream) {
	Volume &volume = _world.volumes.get(volume_id);
	volume.stream = stream;

	// Commit a new dependency to process requests with
	if (volume.stream_dependency != nullptr) {
		volume.stream_dependency->valid = false;
	}

	volume.stream_dependency = gd_make_shared<StreamingDependency>();
	volume.stream_dependency->generator = volume.generator;
	volume.stream_dependency->stream = volume.stream;
}

void VoxelServer::set_volume_generator(uint32_t volume_id, Ref<VoxelGenerator> generator) {
	Volume &volume = _world.volumes.get(volume_id);
	volume.generator = generator;

	// Commit a new dependency to process requests with
	if (volume.stream_dependency != nullptr) {
		volume.stream_dependency->valid = false;
	}

	volume.stream_dependency = gd_make_shared<StreamingDependency>();
	volume.stream_dependency->generator = volume.generator;
	volume.stream_dependency->stream = volume.stream;

	if (volume.meshing_dependency != nullptr) {
		volume.meshing_dependency->valid = false;
	}

	volume.meshing_dependency = gd_make_shared<MeshingDependency>();
	volume.meshing_dependency->mesher = volume.mesher;
	volume.meshing_dependency->generator = volume.generator;
}

void VoxelServer::set_volume_mesher(uint32_t volume_id, Ref<VoxelMesher> mesher) {
	Volume &volume = _world.volumes.get(volume_id);
	volume.mesher = mesher;

	if (volume.meshing_dependency != nullptr) {
		volume.meshing_dependency->valid = false;
	}

	volume.meshing_dependency = gd_make_shared<MeshingDependency>();
	volume.meshing_dependency->mesher = volume.mesher;
	volume.meshing_dependency->generator = volume.generator;
}

void VoxelServer::set_volume_octree_lod_distance(uint32_t volume_id, float lod_distance) {
	Volume &volume = _world.volumes.get(volume_id);
	volume.octree_lod_distance = lod_distance;
}

void VoxelServer::invalidate_volume_mesh_requests(uint32_t volume_id) {
	Volume &volume = _world.volumes.get(volume_id);
	volume.meshing_dependency->valid = false;
	volume.meshing_dependency = gd_make_shared<MeshingDependency>();
	volume.meshing_dependency->mesher = volume.mesher;
	volume.meshing_dependency->generator = volume.generator;
}

static inline Vector3i get_block_center(Vector3i pos, int bs, int lod) {
	return (pos << lod) * bs + Vector3i(bs / 2);
}

void VoxelServer::init_priority_dependency(
		VoxelServer::PriorityDependency &dep, Vector3i block_position, uint8_t lod, const Volume &volume, int block_size) {
	const Vector3i voxel_pos = get_block_center(block_position, block_size, lod);
	const float block_radius = (block_size << lod) / 2;
	dep.shared = _world.shared_priority_dependency;
	dep.world_position = volume.transform.xform(voxel_pos.to_vec3());
	const float transformed_block_radius =
			volume.transform.basis.xform(Vector3(block_radius, block_radius, block_radius)).length();

	switch (volume.type) {
		case VOLUME_SPARSE_GRID:
			// Never drop by distance: these are bounded volumes whose data is
			// already loaded, so completing the mesh costs little — while a
			// dropped mesh update leaves the block displaying its previous
			// mesh forever (apply marks it in-flight and nothing re-queues
			// it), which shows up as stale geometry stranded at the edge of
			// view range after bulk edits. Distance still drives priority,
			// just not cancellation.
			dep.drop_distance_squared = std::numeric_limits<float>::max();
			break;

		case VOLUME_SPARSE_OCTREE:
			// Distance beyond which it is safe to drop a block without risking to block LOD subdivision.
			// This does not depend on viewer's view distance, but on LOD precision instead.
			dep.drop_distance_squared =
					squared(2.f * transformed_block_radius *
							get_octree_lod_block_region_extent(volume.octree_lod_distance, block_size));
			break;

		default:
			CRASH_NOW_MSG("Unexpected type");
			break;
	}
}

void VoxelServer::request_block_mesh(uint32_t volume_id, const BlockMeshInput &input) {
	const Volume &volume = _world.volumes.get(volume_id);
	ERR_FAIL_COND(volume.meshing_dependency == nullptr);
	ERR_FAIL_COND(volume.meshing_dependency->mesher.is_null());
	ERR_FAIL_COND(volume.data_block_size > 255);

	BlockMeshRequest *r = memnew(BlockMeshRequest);
	r->volume_id = volume_id;
	r->blocks = input.data_blocks;
	r->blocks_count = input.data_blocks_count;
	r->position = input.render_block_position;
	r->lod = input.lod;
	r->cull_down_faces = input.cull_down_faces;
	r->greedy = input.greedy;
	r->meshing_dependency = volume.meshing_dependency;
	r->data_block_size = volume.data_block_size;

	init_priority_dependency(
			r->priority_dependency, input.render_block_position, input.lod, volume, volume.render_block_size);

	// We'll allocate this quite often. If it becomes a problem, it should be easy to pool.
	_general_thread_pool.enqueue(r);
}

void VoxelServer::request_frame_mesh(uint32_t volume_id, Vector3i render_block_position,
		std::shared_ptr<VoxelBufferInternal> voxels, int64_t tag, bool cull_down_faces, bool greedy,
		std::vector<Ref<Material>> materials, bool bake_collision) {
	const Volume &volume = _world.volumes.get(volume_id);
	ERR_FAIL_COND(volume.meshing_dependency == nullptr);
	ERR_FAIL_COND(volume.meshing_dependency->mesher.is_null());

	BlockMeshRequest *r = memnew(BlockMeshRequest);
	r->volume_id = volume_id;
	r->blocks_count = 0;
	r->position = render_block_position;
	r->lod = 0;
	r->cull_down_faces = cull_down_faces;
	r->greedy = greedy;
	r->explicit_voxels = voxels;
	r->materials = std::move(materials);
	r->bake_collision = bake_collision;
	r->tag = tag;
	r->meshing_dependency = volume.meshing_dependency;
	r->data_block_size = volume.data_block_size;

	init_priority_dependency(
			r->priority_dependency, render_block_position, 0, volume, volume.render_block_size);

	_general_thread_pool.enqueue(r);
}

void VoxelServer::request_block_load(uint32_t volume_id, Vector3i block_pos, int lod, bool request_instances,
		std::vector<uint8_t> enu_chunk, std::shared_ptr<std::vector<uint16_t>> palette_slots) {
	const Volume &volume = _world.volumes.get(volume_id);
	ERR_FAIL_COND(volume.stream_dependency == nullptr);
	ERR_FAIL_COND(volume.data_block_size > 255);

	// Enu prefill: the receiver handed us this block's real bytes. Expand
	// them on the streaming thread instead of streaming/generating an empty block.
	// (`run()` short-circuits before touching the stream when `enu_chunk` is set.)
	if (!enu_chunk.empty()) {
		BlockDataRequest *r = memnew(BlockDataRequest);
		r->volume_id = volume_id;
		r->position = block_pos;
		r->lod = lod;
		r->type = BlockDataRequest::TYPE_LOAD;
		r->block_size = volume.data_block_size;
		r->stream_dependency = volume.stream_dependency;
		r->request_instances = request_instances;
		r->enu_chunk = std::move(enu_chunk);
		r->enu_palette_slots = std::move(palette_slots);

		init_priority_dependency(r->priority_dependency, block_pos, lod, volume, volume.data_block_size);

		_streaming_thread_pool.enqueue(r);
		return;
	}

	if (volume.stream_dependency->stream.is_valid()) {
		BlockDataRequest *r = memnew(BlockDataRequest);
		r->volume_id = volume_id;
		r->position = block_pos;
		r->lod = lod;
		r->type = BlockDataRequest::TYPE_LOAD;
		r->block_size = volume.data_block_size;
		r->stream_dependency = volume.stream_dependency;
		r->request_instances = request_instances;

		init_priority_dependency(r->priority_dependency, block_pos, lod, volume, volume.data_block_size);

		_streaming_thread_pool.enqueue(r);

	} else {
		// Directly generate the block without checking the stream
		ERR_FAIL_COND(volume.stream_dependency->generator.is_null());

		BlockGenerateRequest *r = memnew(BlockGenerateRequest);
		r->volume_id = volume_id;
		r->position = block_pos;
		r->lod = lod;
		r->block_size = volume.data_block_size;
		r->stream_dependency = volume.stream_dependency;

		init_priority_dependency(r->priority_dependency, block_pos, lod, volume, volume.data_block_size);

		_general_thread_pool.enqueue(r);
	}
}

void VoxelServer::request_block_generate(uint32_t volume_id, Vector3i block_pos, int lod,
		std::shared_ptr<VoxelAsyncDependencyTracker> tracker) {
	//
	const Volume &volume = _world.volumes.get(volume_id);
	ERR_FAIL_COND(volume.stream_dependency->generator.is_null());

	BlockGenerateRequest *r = memnew(BlockGenerateRequest);
	r->volume_id = volume_id;
	r->position = block_pos;
	r->lod = lod;
	r->block_size = volume.data_block_size;
	r->stream_dependency = volume.stream_dependency;
	r->tracker = tracker;
	r->drop_beyond_max_distance = false;

	init_priority_dependency(r->priority_dependency, block_pos, lod, volume, volume.data_block_size);

	_general_thread_pool.enqueue(r);
}

void VoxelServer::request_all_stream_blocks(uint32_t volume_id) {
	PRINT_VERBOSE(String("Request all blocks for volume {0}").format(varray(volume_id)));
	const Volume &volume = _world.volumes.get(volume_id);
	ERR_FAIL_COND(volume.stream.is_null());
	CRASH_COND(volume.stream_dependency == nullptr);

	AllBlocksDataRequest *r = memnew(AllBlocksDataRequest);
	r->volume_id = volume_id;
	r->stream_dependency = volume.stream_dependency;

	_general_thread_pool.enqueue(r);
}

void VoxelServer::request_voxel_block_save(uint32_t volume_id, std::shared_ptr<VoxelBufferInternal> voxels,
		Vector3i block_pos, int lod) {
	//
	const Volume &volume = _world.volumes.get(volume_id);
	ERR_FAIL_COND(volume.stream.is_null());
	CRASH_COND(volume.stream_dependency == nullptr);

	BlockDataRequest *r = memnew(BlockDataRequest);
	r->voxels = voxels;
	r->volume_id = volume_id;
	r->position = block_pos;
	r->lod = lod;
	r->type = BlockDataRequest::TYPE_SAVE;
	r->block_size = volume.data_block_size;
	r->stream_dependency = volume.stream_dependency;
	r->request_instances = false;
	r->request_voxels = true;

	// No priority data, saving doesnt need sorting

	_streaming_thread_pool.enqueue(r);
}

void VoxelServer::request_instance_block_save(uint32_t volume_id, std::unique_ptr<VoxelInstanceBlockData> instances,
		Vector3i block_pos, int lod) {
	const Volume &volume = _world.volumes.get(volume_id);
	ERR_FAIL_COND(volume.stream.is_null());
	CRASH_COND(volume.stream_dependency == nullptr);

	BlockDataRequest *r = memnew(BlockDataRequest);
	r->instances = std::move(instances);
	r->volume_id = volume_id;
	r->position = block_pos;
	r->lod = lod;
	r->type = BlockDataRequest::TYPE_SAVE;
	r->block_size = volume.data_block_size;
	r->stream_dependency = volume.stream_dependency;
	r->request_instances = true;
	r->request_voxels = false;

	// No priority data, saving doesnt need sorting

	_streaming_thread_pool.enqueue(r);
}

void VoxelServer::request_block_generate_from_data_request(BlockDataRequest &src) {
	// This can be called from another thread

	BlockGenerateRequest *r = memnew(BlockGenerateRequest);
	r->voxels = src.voxels;
	r->volume_id = src.volume_id;
	r->position = src.position;
	r->lod = src.lod;
	r->block_size = src.block_size;
	r->stream_dependency = src.stream_dependency;
	r->priority_dependency = src.priority_dependency;

	_general_thread_pool.enqueue(r);
}

void VoxelServer::request_block_save_from_generate_request(BlockGenerateRequest &src) {
	// This can be called from another thread

	PRINT_VERBOSE(String("Requesting save of generator output for block {0} lod {1}")
						  .format(varray(src.position.to_vec3(), src.lod)));

	BlockDataRequest *r = memnew(BlockDataRequest());
	// TODO Optimization: `r->voxels` doesnt actually need to be shared
	r->voxels = gd_make_shared<VoxelBufferInternal>();
	src.voxels->duplicate_to(*r->voxels, true);
	r->volume_id = src.volume_id;
	r->position = src.position;
	r->lod = src.lod;
	r->type = BlockDataRequest::TYPE_SAVE;
	r->block_size = src.block_size;
	r->stream_dependency = src.stream_dependency;

	// No instances, generators are not designed to produce them at this stage yet.
	// No priority data, saving doesnt need sorting

	_streaming_thread_pool.enqueue(r);
}

void VoxelServer::remove_volume(uint32_t volume_id) {
	{
		Volume &volume = _world.volumes.get(volume_id);
		if (volume.stream_dependency != nullptr) {
			volume.stream_dependency->valid = false;
		}
		if (volume.meshing_dependency != nullptr) {
			volume.meshing_dependency->valid = false;
		}
	}

	_world.volumes.destroy(volume_id);
	// TODO How to cancel meshing tasks?

	if (_world.volumes.count() == 0) {
		// To workaround https://github.com/Zylann/godot_voxel/issues/189
		// When the last remaining volume got destroyed (as in game exit)
		wait_and_clear_all_tasks(false);
	}
}

bool VoxelServer::is_volume_valid(uint32_t volume_id) const {
	return _world.volumes.is_valid(volume_id);
}

uint32_t VoxelServer::add_viewer() {
	return _world.viewers.create(Viewer());
}

void VoxelServer::remove_viewer(uint32_t viewer_id) {
	_world.viewers.destroy(viewer_id);
}

void VoxelServer::set_viewer_position(uint32_t viewer_id, Vector3 position) {
	Viewer &viewer = _world.viewers.get(viewer_id);
	viewer.world_position = position;
}

void VoxelServer::set_viewer_distance(uint32_t viewer_id, unsigned int distance) {
	Viewer &viewer = _world.viewers.get(viewer_id);
	viewer.view_distance = distance;
}

unsigned int VoxelServer::get_viewer_distance(uint32_t viewer_id) const {
	const Viewer &viewer = _world.viewers.get(viewer_id);
	return viewer.view_distance;
}

void VoxelServer::set_viewer_requires_visuals(uint32_t viewer_id, bool enabled) {
	Viewer &viewer = _world.viewers.get(viewer_id);
	viewer.require_visuals = enabled;
}

bool VoxelServer::is_viewer_requiring_visuals(uint32_t viewer_id) const {
	const Viewer &viewer = _world.viewers.get(viewer_id);
	return viewer.require_visuals;
}

void VoxelServer::set_viewer_requires_collisions(uint32_t viewer_id, bool enabled) {
	Viewer &viewer = _world.viewers.get(viewer_id);
	viewer.require_collisions = enabled;
}

bool VoxelServer::is_viewer_requiring_collisions(uint32_t viewer_id) const {
	const Viewer &viewer = _world.viewers.get(viewer_id);
	return viewer.require_collisions;
}

bool VoxelServer::viewer_exists(uint32_t viewer_id) const {
	return _world.viewers.is_valid(viewer_id);
}

void VoxelServer::push_time_spread_task(IVoxelTimeSpreadTask *task) {
	_time_spread_task_runner.push(task);
}

int VoxelServer::get_main_thread_time_budget_usec() const {
	return _main_thread_time_budget_usec;
}

void VoxelServer::push_async_task(IVoxelTask *task) {
	_general_thread_pool.enqueue(task);
}

void VoxelServer::push_async_tasks(Span<IVoxelTask *> tasks) {
	_general_thread_pool.enqueue(tasks);
}

void VoxelServer::process() {
	// Note, this shouldn't be here. It should normally done just after SwapBuffers.
	// Godot does not have any C++ profiler usage anywhere, so when using Tracy Profiler I have to put it somewhere...
	// TODO Could connect to VisualServer end_frame_draw signal? How to make sure the singleton is available?
	//VOXEL_PROFILE_MARK_FRAME();
	VOXEL_PROFILE_SCOPE();
	VOXEL_PROFILE_PLOT("Static memory usage", int64_t(OS::get_singleton()->get_static_memory_usage()));

	// Receive data updates
	_streaming_thread_pool.dequeue_completed_tasks([](IVoxelTask *task) {
		task->apply_result();
		memdelete(task);
	});

	// Receive generation and meshing results
	_general_thread_pool.dequeue_completed_tasks([](IVoxelTask *task) {
		task->apply_result();
		memdelete(task);
	});

	// Run this after dequeueing threaded tasks, because they can add some to this runner,
	// which could in turn complete right away (we avoid 1-frame delays this way).
	_time_spread_task_runner.process(_main_thread_time_budget_usec);

	// Update viewer dependencies
	{
		const size_t viewer_count = _world.viewers.count();
		if (_world.shared_priority_dependency->viewers.size() != viewer_count) {
			// TODO We can avoid the invalidation by using an atomic size or memory barrier?
			_world.shared_priority_dependency = gd_make_shared<PriorityDependencyShared>();
			_world.shared_priority_dependency->viewers.resize(viewer_count);
		}
		size_t i = 0;
		unsigned int max_distance = 0;
		_world.viewers.for_each([&i, &max_distance, this](Viewer &viewer) {
			_world.shared_priority_dependency->viewers[i] = viewer.world_position;
			if (viewer.view_distance > max_distance) {
				max_distance = viewer.view_distance;
			}
			++i;
		});
		// Cancel distance is increased because of two reasons:
		// - Some volumes use a cubic area which has higher distances on their corners
		// - Hysteresis is needed to reduce ping-pong
		_world.shared_priority_dependency->highest_view_distance = max_distance * 2;
	}
}

static unsigned int debug_get_active_thread_count(const VoxelThreadPool &pool) {
	unsigned int active_count = 0;
	for (unsigned int i = 0; i < pool.get_thread_count(); ++i) {
		VoxelThreadPool::State s = pool.get_thread_debug_state(i);
		if (s == VoxelThreadPool::STATE_RUNNING) {
			++active_count;
		}
	}
	return active_count;
}

static VoxelServer::Stats::ThreadPoolStats debug_get_pool_stats(const VoxelThreadPool &pool) {
	VoxelServer::Stats::ThreadPoolStats d;
	d.tasks = pool.get_debug_remaining_tasks();
	d.active_threads = debug_get_active_thread_count(pool);
	d.thread_count = pool.get_thread_count();
	return d;
}

Dictionary VoxelServer::Stats::to_dict() {
	Dictionary pools;
	pools["streaming"] = streaming.to_dict();
	pools["general"] = general.to_dict();

	Dictionary tasks;
	tasks["streaming"] = streaming_tasks;
	tasks["generation"] = generation_tasks;
	tasks["meshing"] = meshing_tasks;
	tasks["main_thread"] = main_thread_tasks;

	// This part is additional for scripts because VoxelMemoryPool is not exposed
	Dictionary mem;
	mem["voxel_total"] = SIZE_T_TO_VARIANT(VoxelMemoryPool::get_singleton()->debug_get_total_memory());
	mem["voxel_used"] = SIZE_T_TO_VARIANT(VoxelMemoryPool::get_singleton()->debug_get_used_memory());
	mem["block_count"] = VoxelMemoryPool::get_singleton()->debug_get_used_blocks();

	Dictionary d;
	d["thread_pools"] = pools;
	d["tasks"] = tasks;
	d["memory_pools"] = mem;
	return d;
}

VoxelServer::Stats VoxelServer::get_stats() const {
	Stats s;
	s.streaming = debug_get_pool_stats(_streaming_thread_pool);
	s.general = debug_get_pool_stats(_general_thread_pool);
	s.generation_tasks = g_debug_generate_tasks_count;
	s.meshing_tasks = g_debug_mesh_tasks_count;
	s.streaming_tasks = g_debug_stream_tasks_count;
	s.main_thread_tasks = _time_spread_task_runner.get_pending_count();
	return s;
}

Dictionary VoxelServer::_b_get_stats() {
	return get_stats().to_dict();
}

void VoxelServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_stats"), &VoxelServer::_b_get_stats);
}

//----------------------------------------------------------------------------------------------------------------------

VoxelServer::BlockDataRequest::BlockDataRequest() {
	++g_debug_stream_tasks_count;
}

VoxelServer::BlockDataRequest::~BlockDataRequest() {
	--g_debug_stream_tasks_count;
}

namespace {

// Enu prefill chunk expansion. These decoders mirror the wire format in
// `src/models/voxels/codec.nim` (the source of truth) so the streaming thread can
// turn a chunk's raw compressed snapshot bytes into resolved engine voxel ids
// without touching Nim. Only the snapshot formats `encode_chunk` can emit are
// handled (empty / RLE8 / RLE16 / sparse-full 8+16); deltas stay on Enu's main
// paint path. A parity test guards against drift (see test_voxel_codec.nim).

enum EnuChunkFormat {
	ENU_FMT_RLE = 0x00, // legacy 8-bit
	ENU_FMT_SPARSE_FULL = 0x01, // legacy 8-bit
	ENU_FMT_EMPTY = 0x03,
	ENU_FMT_RLE16 = 0x04,
	ENU_FMT_SPARSE_FULL16 = 0x05,
};

const int ENU_CHUNK_DIM = 16;
const int ENU_CHUNK_VOLUME = ENU_CHUNK_DIM * ENU_CHUNK_DIM * ENU_CHUNK_DIM; // 4096
const uint32_t ENU_STATIC_COLOR_BASE = 64;
const uint8_t ENU_CMD_REPEAT = 241; // legacy 8-bit RLE escape

// SQLite-style varint (Nim std/varints readVu64), reading from a zero-padded
// 9-byte window so a truncated tail decodes like Nim's fixed-size buffer.
inline uint64_t enu_read_varint(const uint8_t *data, size_t len, size_t &i) {
	uint8_t z[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	const size_t avail = (i < len) ? (len - i) : 0;
	for (size_t j = 0; j < 9 && j < avail; ++j) {
		z[j] = data[i + j];
	}
	uint64_t r = 0;
	int used;
	if (z[0] <= 240) {
		r = z[0];
		used = 1;
	} else if (z[0] <= 248) {
		r = (uint64_t(z[0]) - 241) * 256 + uint64_t(z[1]) + 240;
		used = 2;
	} else if (z[0] == 249) {
		r = 2288u + 256u * uint64_t(z[1]) + uint64_t(z[2]);
		used = 3;
	} else if (z[0] == 250) {
		r = (uint64_t(z[1]) << 16) + (uint64_t(z[2]) << 8) + uint64_t(z[3]);
		used = 4;
	} else {
		const uint64_t x = (uint64_t(z[1]) << 24) + (uint64_t(z[2]) << 16) +
				(uint64_t(z[3]) << 8) + uint64_t(z[4]);
		if (z[0] == 251) {
			r = x;
			used = 5;
		} else if (z[0] == 252) {
			r = (x << 8) + uint64_t(z[5]);
			used = 6;
		} else if (z[0] == 253) {
			r = (x << 16) + (uint64_t(z[5]) << 8) + uint64_t(z[6]);
			used = 7;
		} else if (z[0] == 254) {
			r = (x << 24) + (uint64_t(z[5]) << 16) + (uint64_t(z[6]) << 8) + uint64_t(z[7]);
			used = 8;
		} else {
			r = (x << 32) + (0xffffffffu & ((uint64_t(z[5]) << 24) +
					(uint64_t(z[6]) << 16) + (uint64_t(z[7]) << 8) + uint64_t(z[8])));
			used = 9;
		}
	}
	i += used;
	return r;
}

inline uint16_t enu_read_u16(const uint8_t *data, size_t &i) {
	const uint16_t v = uint16_t(data[i]) | (uint16_t(data[i + 1]) << 8);
	i += 2;
	return v;
}

// Decode `data` into `cells` (4096 packed voxels, linear order z + y*16 + x*256).
// `cells` must be pre-zeroed (air).
void enu_decode_chunk(const uint8_t *data, size_t len, uint16_t *cells) {
	if (len == 0) {
		return;
	}
	const uint8_t format = data[0];
	switch (format) {
		case ENU_FMT_EMPTY:
			break;

		case ENU_FMT_RLE: {
			int out_idx = 0;
			size_t i = 1;
			while (i < len && out_idx < ENU_CHUNK_VOLUME) {
				const uint8_t b = data[i];
				if (b == ENU_CMD_REPEAT) {
					if (i + 2 >= len) {
						break;
					}
					const int count = int(data[i + 1]) + 3;
					const uint16_t value = data[i + 2];
					for (int k = 0; k < count && out_idx < ENU_CHUNK_VOLUME; ++k) {
						cells[out_idx++] = value;
					}
					i += 3;
				} else {
					cells[out_idx++] = b;
					++i;
				}
			}
		} break;

		case ENU_FMT_RLE16: {
			int out_idx = 0;
			size_t i = 1;
			while (i < len && out_idx < ENU_CHUNK_VOLUME) {
				const uint8_t tag = data[i];
				++i;
				const int count = int(enu_read_varint(data, len, i));
				if (tag == 1) {
					if (i + 1 >= len) {
						break;
					}
					const uint16_t value = enu_read_u16(data, i);
					for (int k = 0; k < count && out_idx < ENU_CHUNK_VOLUME; ++k) {
						cells[out_idx++] = value;
					}
				} else {
					for (int k = 0; k < count; ++k) {
						if (i + 1 >= len || out_idx >= ENU_CHUNK_VOLUME) {
							break;
						}
						cells[out_idx++] = enu_read_u16(data, i);
					}
				}
			}
		} break;

		case ENU_FMT_SPARSE_FULL:
		case ENU_FMT_SPARSE_FULL16: {
			const bool wide = (format == ENU_FMT_SPARSE_FULL16);
			size_t i = 1;
			const int count = int(enu_read_varint(data, len, i));
			for (int k = 0; k < count; ++k) {
				const uint64_t pos = enu_read_varint(data, len, i);
				if (wide) {
					if (i + 1 >= len) {
						break;
					}
					const uint16_t voxel = enu_read_u16(data, i);
					if (pos < uint64_t(ENU_CHUNK_VOLUME)) {
						cells[pos] = voxel;
					}
				} else {
					if (i >= len) {
						break;
					}
					const uint16_t voxel = data[i];
					++i;
					if (pos < uint64_t(ENU_CHUNK_VOLUME)) {
						cells[pos] = voxel;
					}
				}
			}
		} break;

		default:
			ERR_PRINT(String("Unknown enu chunk format {0}").format(varray(int(format))));
			break;
	}
}

// A packed color index resolves to an engine voxel slot: named colors are
// identity (they are their own library slots), static-RGB colors index the
// per-build palette snapshot. Mirrors `renderer.nim` `library_slot`.
inline uint64_t enu_resolve_slot(uint16_t packed, const std::vector<uint16_t> *palette_slots) {
	if (packed == 0) {
		return 0;
	}
	const uint32_t c = (uint32_t(packed) - 1) / 3;
	if (c < ENU_STATIC_COLOR_BASE) {
		return c;
	}
	const uint32_t pi = c - ENU_STATIC_COLOR_BASE;
	if (palette_slots != nullptr && pi < palette_slots->size()) {
		return (*palette_slots)[pi];
	}
	return 5; // WHITE ordinal — visible fallback for an un-synced palette entry
}

void enu_expand_chunk(VoxelBufferInternal &vb, const std::vector<uint8_t> &data,
		const std::vector<uint16_t> *palette_slots) {
	uint16_t cells[ENU_CHUNK_VOLUME] = { 0 };
	enu_decode_chunk(data.data(), data.size(), cells);

	// Air stays uniform; only non-empty cells de-uniform the channel — matches a
	// normally-generated block's memory profile.
	vb.fill(0, VoxelBufferInternal::CHANNEL_TYPE);
	for (int x = 0; x < ENU_CHUNK_DIM; ++x) {
		for (int y = 0; y < ENU_CHUNK_DIM; ++y) {
			for (int z = 0; z < ENU_CHUNK_DIM; ++z) {
				const uint16_t packed = cells[z + y * ENU_CHUNK_DIM + x * ENU_CHUNK_DIM * ENU_CHUNK_DIM];
				if (packed != 0) {
					vb.set_voxel(enu_resolve_slot(packed, palette_slots), x, y, z,
							VoxelBufferInternal::CHANNEL_TYPE);
				}
			}
		}
	}
}

} // namespace

void VoxelServer::BlockDataRequest::run(VoxelTaskContext ctx) {
	VOXEL_PROFILE_SCOPE();

	CRASH_COND(stream_dependency == nullptr);

	// Enu prefill: the receiver supplied this block's real bytes on main.
	// Expand them into the buffer and skip the stream entirely (the volume may
	// have no stream at all — Enu drives loads through a flat generator).
	if (type == TYPE_LOAD && !enu_chunk.empty()) {
		ERR_FAIL_COND(voxels != nullptr);
		voxels = gd_make_shared<VoxelBufferInternal>();
		voxels->create(block_size, block_size, block_size);
		enu_expand_chunk(*voxels, enu_chunk, enu_palette_slots.get());
		has_run = true;
		return;
	}

	Ref<VoxelStream> stream = stream_dependency->stream;
	CRASH_COND(stream.is_null());

	const Vector3i origin_in_voxels = (position << lod) * block_size;

	switch (type) {
		case TYPE_LOAD: {
			ERR_FAIL_COND(voxels != nullptr);
			voxels = gd_make_shared<VoxelBufferInternal>();
			voxels->create(block_size, block_size, block_size);

			// TODO We should consider batching this again, but it needs to be done carefully.
			// Each task is one block, and priority depends on distance to closest viewer.
			// If we batch blocks, we have to do it by distance too.

			// TODO Assign max_lod_hint when available

			const VoxelStream::Result voxel_result = stream->emerge_block(*voxels, origin_in_voxels, lod);

			if (voxel_result == VoxelStream::RESULT_ERROR) {
				ERR_PRINT("Error loading voxel block");

			} else if (voxel_result == VoxelStream::RESULT_BLOCK_NOT_FOUND) {
				Ref<VoxelGenerator> generator = stream_dependency->generator;
				if (generator.is_valid()) {
					VoxelServer::get_singleton()->request_block_generate_from_data_request(*this);
					type = TYPE_FALLBACK_ON_GENERATOR;
				} else {
					// If there is no generator... what do we do? What defines the format of that empty block?
					// If the user leaves the defaults it's fine, but otherwise blocks of inconsistent format can
					// end up in the volume and that can cause errors.
					// TODO Define format on volume?
				}
			}

			if (request_instances && stream->supports_instance_blocks()) {
				ERR_FAIL_COND(instances != nullptr);

				VoxelStreamInstanceDataRequest instance_data_request;
				instance_data_request.lod = lod;
				instance_data_request.position = position;
				VoxelStream::Result instances_result;
				stream->load_instance_blocks(
						Span<VoxelStreamInstanceDataRequest>(&instance_data_request, 1),
						Span<VoxelStream::Result>(&instances_result, 1));

				if (instances_result == VoxelStream::RESULT_ERROR) {
					ERR_PRINT("Error loading instance block");

				} else if (voxel_result == VoxelStream::RESULT_BLOCK_FOUND) {
					instances = std::move(instance_data_request.data);
				}
				// If not found, instances will return null,
				// which means it can be generated by the instancer after the meshing process
			}
		} break;

		case TYPE_SAVE: {
			if (request_voxels) {
				ERR_FAIL_COND(voxels == nullptr);
				VoxelBufferInternal voxels_copy;
				{
					RWLockRead lock(voxels->get_lock());
					// TODO Optimization: is that copy necessary? It's possible it was already done while issuing the request
					voxels->duplicate_to(voxels_copy, true);
				}
				voxels = nullptr;
				stream->immerge_block(voxels_copy, origin_in_voxels, lod);
			}

			if (request_instances && stream->supports_instance_blocks()) {
				// If the provided data is null, it means this instance block was never modified.
				// Since we are in a save request, the saved data will revert to unmodified.
				// On the other hand, if we want to represent the fact that "everything was deleted here",
				// this should not be null.

				PRINT_VERBOSE(String("Saving instance block {0} lod {1} with data {2}")
									  .format(varray(position.to_vec3(), lod, ptr2s(instances.get()))));

				VoxelStreamInstanceDataRequest instance_data_request;
				instance_data_request.lod = lod;
				instance_data_request.position = position;
				instance_data_request.data = std::move(instances);
				stream->save_instance_blocks(Span<VoxelStreamInstanceDataRequest>(&instance_data_request, 1));
			}
		} break;

		default:
			CRASH_NOW_MSG("Invalid type");
	}

	has_run = true;
}

int VoxelServer::BlockDataRequest::get_priority() {
	if (type == TYPE_SAVE) {
		return 0;
	}
	float closest_viewer_distance_sq;
	const int p = VoxelServer::get_priority(priority_dependency, lod, &closest_viewer_distance_sq);
	too_far = closest_viewer_distance_sq > priority_dependency.drop_distance_squared;
	return p;
}

bool VoxelServer::BlockDataRequest::is_cancelled() {
	return type == TYPE_LOAD && (!stream_dependency->valid || too_far);
}

void VoxelServer::BlockDataRequest::apply_result() {
	Volume *volume = VoxelServer::get_singleton()->_world.volumes.try_get(volume_id);

	if (volume != nullptr) {
		// TODO Comparing pointer may not be guaranteed
		// The request response must match the dependency it would have been requested with.
		// If it doesn't match, we are no longer interested in the result.
		if (stream_dependency == volume->stream_dependency && type != BlockDataRequest::TYPE_FALLBACK_ON_GENERATOR) {
			BlockDataOutput o;
			o.voxels = voxels;
			o.instances = std::move(instances);
			o.position = position;
			o.lod = lod;
			o.dropped = !has_run;
			o.max_lod_hint = max_lod_hint;
			o.initial_load = false;

			switch (type) {
				case BlockDataRequest::TYPE_SAVE:
					o.type = BlockDataOutput::TYPE_SAVE;
					break;

				case BlockDataRequest::TYPE_LOAD:
					o.type = BlockDataOutput::TYPE_LOAD;
					break;

				default:
					CRASH_NOW_MSG("Unexpected data request response type");
			}

			CRASH_COND(volume->callbacks.data_output_callback == nullptr);
			volume->callbacks.data_output_callback(volume->callbacks.data, o);
		}

	} else {
		// This can happen if the user removes the volume while requests are still about to return
		PRINT_VERBOSE("Stream data request response came back but volume wasn't found");
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VoxelServer::AllBlocksDataRequest::AllBlocksDataRequest() {
}

VoxelServer::AllBlocksDataRequest::~AllBlocksDataRequest() {
}

void VoxelServer::AllBlocksDataRequest::run(VoxelTaskContext ctx) {
	VOXEL_PROFILE_SCOPE();

	CRASH_COND(stream_dependency == nullptr);
	Ref<VoxelStream> stream = stream_dependency->stream;
	CRASH_COND(stream.is_null());

	stream->load_all_blocks(result);

	PRINT_VERBOSE(String("Loaded {0} blocks for volume {1}").format(varray(SIZE_T_TO_VARIANT(result.blocks.size()), volume_id)));
}

int VoxelServer::AllBlocksDataRequest::get_priority() {
	return 0;
}

bool VoxelServer::AllBlocksDataRequest::is_cancelled() {
	return !stream_dependency->valid;
}

void VoxelServer::AllBlocksDataRequest::apply_result() {
	Volume *volume = VoxelServer::get_singleton()->_world.volumes.try_get(volume_id);

	if (volume != nullptr) {
		// TODO Comparing pointer may not be guaranteed
		// The request response must match the dependency it would have been requested with.
		// If it doesn't match, we are no longer interested in the result.
		if (stream_dependency == volume->stream_dependency) {
			ERR_FAIL_COND(volume->callbacks.data_output_callback == nullptr);

			for (auto it = result.blocks.begin(); it != result.blocks.end(); ++it) {
				VoxelStream::FullLoadingResult::Block &rb = *it;

				BlockDataOutput o;
				o.voxels = rb.voxels;
				o.instances = std::move(rb.instances_data);
				o.position = rb.position;
				o.lod = rb.lod;
				o.dropped = false;
				o.max_lod_hint = false;
				o.initial_load = true;

				volume->callbacks.data_output_callback(volume->callbacks.data, o);
			}
		}

	} else {
		// This can happen if the user removes the volume while requests are still about to return
		PRINT_VERBOSE("Stream data request response came back but volume wasn't found");
	}
}

//----------------------------------------------------------------------------------------------------------------------

VoxelServer::BlockGenerateRequest::BlockGenerateRequest() {
	++g_debug_generate_tasks_count;
}

VoxelServer::BlockGenerateRequest::~BlockGenerateRequest() {
	--g_debug_generate_tasks_count;
}

void VoxelServer::BlockGenerateRequest::run(VoxelTaskContext ctx) {
	VOXEL_PROFILE_SCOPE();

	CRASH_COND(stream_dependency == nullptr);
	Ref<VoxelGenerator> generator = stream_dependency->generator;
	ERR_FAIL_COND(generator.is_null());

	const Vector3i origin_in_voxels = (position << lod) * block_size;

	if (voxels == nullptr) {
		voxels = gd_make_shared<VoxelBufferInternal>();
		voxels->create(block_size, block_size, block_size);
	}

	VoxelBlockRequest r{ *voxels, origin_in_voxels, lod };
	const VoxelGenerator::Result result = generator->generate_block(r);
	max_lod_hint = result.max_lod_hint;

	if (stream_dependency->valid) {
		Ref<VoxelStream> stream = stream_dependency->stream;
		if (stream.is_valid() && stream->get_save_generator_output()) {
			VoxelServer::get_singleton()->request_block_save_from_generate_request(*this);
		}
	}

	has_run = true;
}

int VoxelServer::BlockGenerateRequest::get_priority() {
	float closest_viewer_distance_sq;
	const int p = VoxelServer::get_priority(priority_dependency, lod, &closest_viewer_distance_sq);
	too_far = drop_beyond_max_distance && closest_viewer_distance_sq > priority_dependency.drop_distance_squared;
	return p;
}

bool VoxelServer::BlockGenerateRequest::is_cancelled() {
	return !stream_dependency->valid || too_far; // || stream_dependency->stream->get_fallback_generator().is_null();
}

void VoxelServer::BlockGenerateRequest::apply_result() {
	Volume *volume = VoxelServer::get_singleton()->_world.volumes.try_get(volume_id);

	bool aborted = true;

	if (volume != nullptr) {
		// TODO Comparing pointer may not be guaranteed
		// The request response must match the dependency it would have been requested with.
		// If it doesn't match, we are no longer interested in the result.
		if (stream_dependency == volume->stream_dependency) {
			BlockDataOutput o;
			o.voxels = voxels;
			o.position = position;
			o.lod = lod;
			o.dropped = !has_run;
			o.type = BlockDataOutput::TYPE_LOAD;
			o.max_lod_hint = max_lod_hint;
			o.initial_load = false;

			ERR_FAIL_COND(volume->callbacks.data_output_callback == nullptr);
			volume->callbacks.data_output_callback(volume->callbacks.data, o);

			aborted = !has_run;
		}

	} else {
		// This can happen if the user removes the volume while requests are still about to return
		PRINT_VERBOSE("Gemerated data request response came back but volume wasn't found");
	}

	// TODO We could complete earlier inside run() if we had access to the data structure to write the block into.
	// This would reduce latency a little. The rest of things the terrain needs to do with the generated block could
	// run later.
	if (tracker != nullptr) {
		if (aborted) {
			tracker->abort();
		} else {
			tracker->post_complete();
		}
	}
}

//----------------------------------------------------------------------------------------------------------------------

// Takes a list of blocks and interprets it as a cube of blocks centered around the area we want to create a mesh from.
// Voxels from central blocks are copied, and part of side blocks are also copied so we get a temporary buffer
// which includes enough neighbors for the mesher to avoid doing bound checks.
static void copy_block_and_neighbors(Span<std::shared_ptr<VoxelBufferInternal>> blocks, VoxelBufferInternal &dst,
		int min_padding, int max_padding, int channels_mask, Ref<VoxelGenerator> generator, int data_block_size,
		int lod_index, Vector3i mesh_block_pos) {
	VOXEL_PROFILE_SCOPE();

	// Extract wanted channels in a list
	unsigned int channels_count = 0;
	VoxelFixedArray<uint8_t, VoxelBuffer::MAX_CHANNELS> channels =
			VoxelBufferInternal::mask_to_channels_list(channels_mask, channels_count);

	// Determine size of the cube of blocks
	int edge_size;
	int mesh_block_size_factor;
	switch (blocks.size()) {
		case 3 * 3 * 3:
			edge_size = 3;
			mesh_block_size_factor = 1;
			break;
		case 4 * 4 * 4:
			edge_size = 4;
			mesh_block_size_factor = 2;
			break;
		default:
			ERR_FAIL_MSG("Unsupported block count");
	}

	// Pick anchor block, usually within the central part of the cube (that block must be valid)
	const unsigned int anchor_buffer_index = edge_size * edge_size + edge_size + 1;

	std::shared_ptr<VoxelBufferInternal> &central_buffer = blocks[anchor_buffer_index];
	ERR_FAIL_COND_MSG(central_buffer == nullptr && generator.is_null(), "Central buffer must be valid");
	if (central_buffer != nullptr) {
		ERR_FAIL_COND_MSG(central_buffer->get_size().all_members_equal() == false, "Central buffer must be cubic");
	}
	const int mesh_block_size = data_block_size * mesh_block_size_factor;
	const int padded_mesh_block_size = mesh_block_size + min_padding + max_padding;

	dst.create(padded_mesh_block_size, padded_mesh_block_size, padded_mesh_block_size);

	// TODO Need to provide format
	// for (unsigned int ci = 0; ci < channels.size(); ++ci) {
	// 	dst.set_channel_depth(ci, central_buffer->get_channel_depth(ci));
	// }

	const Vector3i min_pos = -Vector3i(min_padding);
	const Vector3i max_pos = Vector3i(mesh_block_size + max_padding);

	std::vector<Box3i> boxes_to_generate;
	const Box3i mesh_data_box = Box3i::from_min_max(min_pos, max_pos);
	if (generator.is_valid()) {
		boxes_to_generate.push_back(mesh_data_box);
	}

	// Using ZXY as convention to reconstruct positions with thread locking consistency
	unsigned int block_index = 0;
	for (int z = -1; z < edge_size - 1; ++z) {
		for (int x = -1; x < edge_size - 1; ++x) {
			for (int y = -1; y < edge_size - 1; ++y) {
				const Vector3i offset = data_block_size * Vector3i(x, y, z);
				const std::shared_ptr<VoxelBufferInternal> &src = blocks[block_index];
				++block_index;

				if (src == nullptr) {
					continue;
				}

				const Vector3i src_min = min_pos - offset;
				const Vector3i src_max = max_pos - offset;

				{
					RWLockRead read(src->get_lock());
					for (unsigned int ci = 0; ci < channels_count; ++ci) {
						dst.copy_from(*src, src_min, src_max, Vector3i(), channels[ci]);
					}
				}

				if (generator.is_valid()) {
					// Subtract edited box from the area to generate
					// TODO This approach allows to batch boxes if necessary,
					// but is it just better to do it anyways for every clipped box?
					VOXEL_PROFILE_SCOPE_NAMED("Box subtract");
					unsigned int count = boxes_to_generate.size();
					Box3i block_box = Box3i(offset, Vector3i(data_block_size)).clipped(mesh_data_box);

					for (unsigned int box_index = 0; box_index < count; ++box_index) {
						Box3i box = boxes_to_generate[box_index];
						box.difference(block_box, boxes_to_generate);
#ifdef DEBUG_ENABLED
						CRASH_COND(box_index >= boxes_to_generate.size());
#endif
						boxes_to_generate[box_index] = boxes_to_generate.back();
						boxes_to_generate.pop_back();
					}
				}
			}
		}
	}

	if (generator.is_valid()) {
		// Complete data with generated voxels
		VOXEL_PROFILE_SCOPE_NAMED("Generate");
		VoxelBufferInternal generated_voxels;

		const Vector3i origin_in_voxels = mesh_block_pos * (mesh_block_size_factor * data_block_size << lod_index);

		for (unsigned int i = 0; i < boxes_to_generate.size(); ++i) {
			const Box3i &box = boxes_to_generate[i];
			//print_line(String("size={0}").format(varray(box.size.to_vec3())));
			generated_voxels.create(box.size);
			//generated_voxels.set_voxel_f(2.0f, box.size.x / 2, box.size.y / 2, box.size.z / 2, VoxelBufferInternal::CHANNEL_SDF);
			VoxelBlockRequest r{ generated_voxels, (box.pos << lod_index) + origin_in_voxels, lod_index };
			generator->generate_block(r);

			for (unsigned int ci = 0; ci < channels_count; ++ci) {
				dst.copy_from(generated_voxels, Vector3i(), generated_voxels.get_size(),
						box.pos + Vector3i(min_padding), channels[ci]);
			}
		}
	}
}

VoxelServer::BlockMeshRequest::BlockMeshRequest() {
	++g_debug_mesh_tasks_count;
}

VoxelServer::BlockMeshRequest::~BlockMeshRequest() {
	--g_debug_mesh_tasks_count;
}

void VoxelServer::BlockMeshRequest::run(VoxelTaskContext ctx) {
	VOXEL_PROFILE_SCOPE();
	CRASH_COND(meshing_dependency == nullptr);

	Ref<VoxelMesher> mesher = meshing_dependency->mesher;
	CRASH_COND(mesher.is_null());
	const unsigned int min_padding = mesher->get_minimum_padding();
	const unsigned int max_padding = mesher->get_maximum_padding();

	if (explicit_voxels != nullptr) {
		// Frame bake: the receiver supplied the exact padded content — a
		// pure function of data, independent of any world state.
		const VoxelMesher::Input input = { *explicit_voxels, lod, cull_down_faces, greedy };
		mesher->build(surfaces_output, input);

		// Assemble the render mesh here on the worker: the VisualServer runs
		// multi-threaded (thread_model=2), so mesh creation is queue-safe
		// off-main. Collision faces are pure CPU. Main receives finished
		// products and only instantiates the physics shape (PhysicsServer
		// isn't thread-safe in debug builds). This keeps warm-up bake floods
		// off the main thread.
		Ref<ArrayMesh> mesh;
		mesh.instance();
		int surface_index = 0;
		for (int i = 0; i < surfaces_output.surfaces.size(); ++i) {
			Array surface = surfaces_output.surfaces[i];
			if (surface.empty() || !is_surface_triangulated(surface)) {
				continue;
			}
			mesh->add_surface_from_arrays(
					surfaces_output.primitive_type, surface, Array(), surfaces_output.compression_flags);
			if (i < (int)materials.size()) {
				mesh->surface_set_material(surface_index, materials[i]);
			}
			++surface_index;
		}
		if (surface_index > 0) {
			baked_mesh = mesh;
			if (bake_collision) {
				collision_faces = concave_polygon_faces(surfaces_output.surfaces);
			}
		}

		has_run = true;
		return;
	}

	// TODO Cache?
	VoxelBufferInternal voxels;
	copy_block_and_neighbors(to_span(blocks, blocks_count),
			voxels, min_padding, max_padding, mesher->get_used_channels_mask(),
			meshing_dependency->generator, data_block_size, lod, position);

	const VoxelMesher::Input input = { voxels, lod, cull_down_faces, greedy };
	mesher->build(surfaces_output, input);

	has_run = true;
}

int VoxelServer::BlockMeshRequest::get_priority() {
	float closest_viewer_distance_sq;
	const int p = VoxelServer::get_priority(priority_dependency, lod, &closest_viewer_distance_sq);
	too_far = closest_viewer_distance_sq > priority_dependency.drop_distance_squared;
	return p;
}

bool VoxelServer::BlockMeshRequest::is_cancelled() {
	return !meshing_dependency->valid || too_far;
}

void VoxelServer::BlockMeshRequest::apply_result() {
	Volume *volume = VoxelServer::get_singleton()->_world.volumes.try_get(volume_id);

	if (volume != nullptr) {
		// TODO Comparing pointer may not be guaranteed
		// The request response must match the dependency it would have been requested with.
		// If it doesn't match, we are no longer interested in the result.
		if (volume->meshing_dependency == meshing_dependency) {
			BlockMeshOutput o;
			// TODO Check for invalidation due to property changes

			if (has_run) {
				o.type = BlockMeshOutput::TYPE_MESHED;
			} else {
				o.type = BlockMeshOutput::TYPE_DROPPED;
			}

			o.position = position;
			o.lod = lod;
			o.frame_bake = explicit_voxels != nullptr;
			o.tag = tag;
			o.surfaces = surfaces_output;
			o.baked_mesh = baked_mesh;
			o.collision_faces = collision_faces;

			ERR_FAIL_COND(volume->callbacks.mesh_output_callback == nullptr);
			ERR_FAIL_COND(volume->callbacks.data == nullptr);
			volume->callbacks.mesh_output_callback(volume->callbacks.data, o);
		}

	} else {
		// This can happen if the user removes the volume while requests are still about to return
		PRINT_VERBOSE("Mesh request response came back but volume wasn't found");
	}
}

//----------------------------------------------------------------------------------------------------------------------

namespace {
bool g_updater_created = false;
}

VoxelServerUpdater::VoxelServerUpdater() {
	PRINT_VERBOSE("Creating VoxelServerUpdater");
	set_process(true);
	g_updater_created = true;
}

VoxelServerUpdater::~VoxelServerUpdater() {
	g_updater_created = false;
}

void VoxelServerUpdater::ensure_existence(SceneTree *st) {
	if (st == nullptr) {
		return;
	}
	if (g_updater_created) {
		return;
	}
	Viewport *root = st->get_root();
	for (int i = 0; i < root->get_child_count(); ++i) {
		VoxelServerUpdater *u = Object::cast_to<VoxelServerUpdater>(root->get_child(i));
		if (u != nullptr) {
			return;
		}
	}
	VoxelServerUpdater *u = memnew(VoxelServerUpdater);
	u->set_name("VoxelServerUpdater_dont_touch_this");
	root->add_child(u);
}

void VoxelServerUpdater::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PROCESS:
			// To workaround the absence of API to have a custom server processing in the main loop
			VoxelServer::get_singleton()->process();
			break;

		case NOTIFICATION_PREDELETE:
			PRINT_VERBOSE("Deleting VoxelServerUpdater");
			break;

		default:
			break;
	}
}
