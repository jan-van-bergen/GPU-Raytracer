#include "Integrator.h"

#include <GL/glew.h>

#include <Imgui/imgui.h>

#include "BVH/Converters/BVH4Converter.h"
#include "BVH/Converters/BVH8Converter.h"

#include "Core/Allocators/PinnedAllocator.h"

#include "Util/BlueNoise.h"

void Integrator::init_globals() {
	global_camera      = cuda_module.get_global("camera");
	global_config      = cuda_module.get_global("config");
	global_pixel_query = cuda_module.get_global("pixel_query");
	global_aovs        = cuda_module.get_global("aovs");
}

void Integrator::init_materials() {
	ptr_material_types = CUDAMemory::malloc<Material::Type>(scene.asset_manager.materials.size());
	ptr_materials      = CUDAMemory::malloc<CUDAMaterial>  (scene.asset_manager.materials.size());
	cuda_module.get_global("material_types").set_value(ptr_material_types);
	cuda_module.get_global("materials")     .set_value(ptr_materials);

	scene.asset_manager.wait_until_loaded();

	ptr_media = CUDAMemory::malloc<CUDAMedium>(scene.asset_manager.media.size());
	cuda_module.get_global("media").set_value(ptr_media);

	// Set global Texture table
	size_t texture_count = scene.asset_manager.textures.size();
	if (texture_count > 0) {
		textures      .resize(texture_count);
		texture_arrays.resize(texture_count);

		// Get maximum anisotropy from OpenGL
		int max_aniso;
		glGetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &max_aniso);

		for (int i = 0; i < texture_count; i++) {
			const Texture & texture = scene.asset_manager.textures[i];

			// Create mipmapped CUDA array
			texture_arrays[i] = CUDAMemory::create_array_mipmap(
				texture.width,
				texture.height,
				texture.channels,
				texture.get_cuda_array_format(),
				texture.mip_levels()
			);

			// Upload each level of the mipmap
			for (int level = 0; level < texture.mip_levels(); level++) {
				CUarray level_array;
				CUDACALL(cuMipmappedArrayGetLevel(&level_array, texture_arrays[i], level));

				int level_width_in_bytes = texture.get_width_in_bytes(level);
				int level_height         = Math::max(texture.height >> level, 1);

				CUDAMemory::copy_array(level_array, level_width_in_bytes, level_height, texture.data.data() + texture.mip_offsets[level]);
			}

			// Describe the Array to read from
			CUDA_RESOURCE_DESC res_desc = { };
			res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
			res_desc.res.mipmap.hMipmappedArray = texture_arrays[i];

			// Describe how to sample the Texture
			CUDA_TEXTURE_DESC tex_desc = { };
			tex_desc.addressMode[0] = CUaddress_mode::CU_TR_ADDRESS_MODE_WRAP;
			tex_desc.addressMode[1] = CUaddress_mode::CU_TR_ADDRESS_MODE_WRAP;
			tex_desc.addressMode[2] = CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP;
			tex_desc.filterMode       = CUfilter_mode::CU_TR_FILTER_MODE_LINEAR;
			tex_desc.mipmapFilterMode = CUfilter_mode::CU_TR_FILTER_MODE_LINEAR;
			tex_desc.mipmapLevelBias = 0.0f;
			tex_desc.maxAnisotropy = max_aniso;
			tex_desc.minMipmapLevelClamp = 0.0f;
			tex_desc.maxMipmapLevelClamp = float(texture.mip_levels() - 1);
			tex_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;

			// Describe the Texture View
			CUDA_RESOURCE_VIEW_DESC view_desc = { };
			view_desc.format = texture.get_cuda_resource_view_format();
			view_desc.width  = texture.get_cuda_resource_view_width();
			view_desc.height = texture.get_cuda_resource_view_height();
			view_desc.firstMipmapLevel = 0;
			view_desc.lastMipmapLevel  = texture.mip_levels() - 1;

			CUDACALL(cuTexObjectCreate(&textures[i].texture, &res_desc, &tex_desc, &view_desc));

			textures[i].lod_bias = 0.5f * log2f(float(texture.width * texture.height));
		}

		ptr_textures = CUDAMemory::malloc(textures);
		cuda_module.get_global("textures").set_value(ptr_textures);
	}
}

void Integrator::init_geometry() {
	for (size_t i = 0; i < scene.meshes.size(); i++) {
		scene.meshes[i].calc_aabb(scene);
	}

	size_t mesh_data_count = scene.asset_manager.mesh_datas.size();

	mesh_data_bvh_offsets     .resize(mesh_data_count);
	mesh_data_triangle_offsets.resize(mesh_data_count);

	Array<int> mesh_data_index_offsets(mesh_data_count);

	size_t aggregated_bvh_node_count = 2 * scene.meshes.size(); // Reserve 2 times Mesh count for TLAS
	size_t aggregated_triangle_count = 0;
	size_t aggregated_index_count    = 0;

	for (size_t i = 0; i < mesh_data_count; i++) {
		mesh_data_bvh_offsets     [i] = aggregated_bvh_node_count;
		mesh_data_triangle_offsets[i] = aggregated_triangle_count;
		mesh_data_index_offsets   [i] = aggregated_index_count;

		aggregated_bvh_node_count += scene.asset_manager.mesh_datas[i].bvh->node_count();
		aggregated_triangle_count += scene.asset_manager.mesh_datas[i].triangles.size();
		aggregated_index_count    += scene.asset_manager.mesh_datas[i].bvh->indices.size();;
	}

	Array<CUDATriangle> aggregated_triangles(aggregated_index_count);
	reverse_indices.resize(aggregated_triangle_count);

	for (int m = 0; m < mesh_data_count; m++) {
		const MeshData & mesh_data = scene.asset_manager.mesh_datas[m];

		for (size_t i = 0; i < mesh_data.bvh->indices.size(); i++) {
			int index = mesh_data.bvh->indices[i];
			const Triangle & triangle = mesh_data.triangles[index];

			aggregated_triangles[mesh_data_index_offsets[m] + i].position_0      = triangle.position_0;
			aggregated_triangles[mesh_data_index_offsets[m] + i].position_edge_1 = triangle.position_1 - triangle.position_0;
			aggregated_triangles[mesh_data_index_offsets[m] + i].position_edge_2 = triangle.position_2 - triangle.position_0;

			aggregated_triangles[mesh_data_index_offsets[m] + i].normal_0      = triangle.normal_0;
			aggregated_triangles[mesh_data_index_offsets[m] + i].normal_edge_1 = triangle.normal_1 - triangle.normal_0;
			aggregated_triangles[mesh_data_index_offsets[m] + i].normal_edge_2 = triangle.normal_2 - triangle.normal_0;

			aggregated_triangles[mesh_data_index_offsets[m] + i].tex_coord_0      = triangle.tex_coord_0;
			aggregated_triangles[mesh_data_index_offsets[m] + i].tex_coord_edge_1 = triangle.tex_coord_1 - triangle.tex_coord_0;
			aggregated_triangles[mesh_data_index_offsets[m] + i].tex_coord_edge_2 = triangle.tex_coord_2 - triangle.tex_coord_0;

			reverse_indices[mesh_data_triangle_offsets[m] + index] = mesh_data_index_offsets[m] + i;
		}
	}

	ptr_triangles = CUDAMemory::malloc(aggregated_triangles);
	cuda_module.get_global("triangles").set_value(ptr_triangles);

	pinned_mesh_bvh_root_indices             = CUDAMemory::malloc_pinned<int>      (scene.meshes.size());
	pinned_mesh_material_ids                 = CUDAMemory::malloc_pinned<int>      (scene.meshes.size());
	pinned_mesh_transforms                   = CUDAMemory::malloc_pinned<Matrix3x4>(scene.meshes.size());
	pinned_mesh_transforms_inv               = CUDAMemory::malloc_pinned<Matrix3x4>(scene.meshes.size());
	pinned_mesh_transforms_prev              = CUDAMemory::malloc_pinned<Matrix3x4>(scene.meshes.size());
	pinned_light_mesh_cumulative_probability = CUDAMemory::malloc_pinned<float>    (scene.meshes.size());
	pinned_light_mesh_triangle_span          = CUDAMemory::malloc_pinned<int2>     (scene.meshes.size());
	pinned_light_mesh_transform_indices      = CUDAMemory::malloc_pinned<int>      (scene.meshes.size());

	ptr_mesh_bvh_root_indices = CUDAMemory::malloc<int>      (scene.meshes.size());
	ptr_mesh_material_ids     = CUDAMemory::malloc<int>      (scene.meshes.size());
	ptr_mesh_transforms       = CUDAMemory::malloc<Matrix3x4>(scene.meshes.size());
	ptr_mesh_transforms_inv   = CUDAMemory::malloc<Matrix3x4>(scene.meshes.size());
	ptr_mesh_transforms_prev  = CUDAMemory::malloc<Matrix3x4>(scene.meshes.size());

	cuda_module.get_global("mesh_bvh_root_indices").set_value(ptr_mesh_bvh_root_indices);
	cuda_module.get_global("mesh_material_ids")    .set_value(ptr_mesh_material_ids);
	cuda_module.get_global("mesh_transforms")      .set_value(ptr_mesh_transforms);
	cuda_module.get_global("mesh_transforms_inv")  .set_value(ptr_mesh_transforms_inv);
	cuda_module.get_global("mesh_transforms_prev") .set_value(ptr_mesh_transforms_prev);

	tlas_raw.indices.resize(scene.meshes.size());
	tlas_raw.nodes  .resize(scene.meshes.size() * 2);
	tlas_builder = make_owned<SAHBuilder>(tlas_raw, scene.meshes.size());

	switch (cpu_config.bvh_type) {
		case BVHType::BVH:
		case BVHType::SBVH: {
			Array<BVHNode2> aggregated_bvh_nodes(aggregated_bvh_node_count);

			// Each individual BVH needs to put its Nodes in a shared aggregated array of BVH Nodes before being upload to the GPU
			// The procedure to do this is different for each BVH type
			for (int m = 0; m < mesh_data_count; m++) {
				const MeshData & mesh_data = scene.asset_manager.mesh_datas[m];
				const BVH2 * bvh = static_cast<const BVH2 *>(mesh_data.bvh.get());

				int index_offset = mesh_data_index_offsets[m];
				int bvh_offset   = mesh_data_bvh_offsets[m];

				BVHNode2 * dst = aggregated_bvh_nodes.data() + bvh_offset;

				for (size_t n = 0; n < bvh->nodes.size(); n++) {
					BVHNode2 & node = dst[n];
					node = bvh->nodes[n];

					if (node.is_leaf()) {
						node.first += index_offset;
					} else {
						node.left += bvh_offset;
					}
				}
			}

			ptr_bvh_nodes_2 = CUDAMemory::malloc<BVHNode2>(aggregated_bvh_nodes);
			cuda_module.get_global("bvh2_nodes").set_value(ptr_bvh_nodes_2);

			tlas           = make_owned<BVH2>(PinnedAllocator::instance());
			tlas_converter = make_owned<BVH2Converter>(static_cast<BVH2 &>(*tlas.get()), tlas_raw);
			break;
		}
		case BVHType::BVH4: {
			Array<BVHNode4> aggregated_bvh_nodes(aggregated_bvh_node_count);

			// Each individual BVH needs to put its Nodes in a shared aggregated array of BVH Nodes before being upload to the GPU
			// The procedure to do this is different for each BVH type
			for (int m = 0; m < mesh_data_count; m++) {
				const MeshData & mesh_data = scene.asset_manager.mesh_datas[m];
				const BVH4 * bvh = static_cast<const BVH4 *>(mesh_data.bvh.get());

				int index_offset = mesh_data_index_offsets[m];
				int bvh_offset   = mesh_data_bvh_offsets[m];

				BVHNode4 * dst = aggregated_bvh_nodes.data() + bvh_offset;

				for (size_t n = 0; n < bvh->nodes.size(); n++) {
					BVHNode4 & node = dst[n];
					node = bvh->nodes[n];

					int child_count = node.get_child_count();
					for (int c = 0; c < child_count; c++) {
						if (node.is_leaf(c)) {
							node.get_index(c) += index_offset;
						} else {
							node.get_index(c) += bvh_offset;
						}
					}
				}
			}

			ptr_bvh_nodes_4 = CUDAMemory::malloc<BVHNode4>(aggregated_bvh_nodes);
			cuda_module.get_global("bvh4_nodes").set_value(ptr_bvh_nodes_4);

			tlas           = make_owned<BVH4>(PinnedAllocator::instance());
			tlas_converter = make_owned<BVH4Converter>(static_cast<BVH4 &>(*tlas.get()), tlas_raw);
			break;
		}
		case BVHType::BVH8: {
			Array<BVHNode8> aggregated_bvh_nodes(aggregated_bvh_node_count);

			// Each individual BVH needs to put its Nodes in a shared aggregated array of BVH Nodes before being upload to the GPU
			// The procedure to do this is different for each BVH type
			for (int m = 0; m < mesh_data_count; m++) {
				const MeshData & mesh_data = scene.asset_manager.mesh_datas[m];
				const BVH8 * bvh = static_cast<const BVH8 *>(mesh_data.bvh.get());

				int index_offset = mesh_data_index_offsets[m];
				int bvh_offset   = mesh_data_bvh_offsets[m];

				BVHNode8 * dst = aggregated_bvh_nodes.data() + bvh_offset;

				for (size_t n = 0; n < bvh->nodes.size(); n++) {
					BVHNode8 & node = dst[n];
					node = bvh->nodes[n];

					node.base_index_triangle += index_offset;
					node.base_index_child    += bvh_offset;
				}
			}

			ptr_bvh_nodes_8 = CUDAMemory::malloc<BVHNode8>(aggregated_bvh_nodes);
			cuda_module.get_global("bvh8_nodes").set_value(ptr_bvh_nodes_8);

			tlas           = make_owned<BVH8>(PinnedAllocator::instance());
			tlas_converter = make_owned<BVH8Converter>(static_cast<BVH8 &>(*tlas.get()), tlas_raw);
			break;
		}
	}
}

void Integrator::init_sky() {
	sky_array = CUDAMemory::create_array(scene.sky.width, scene.sky.height, 4, CU_AD_FORMAT_FLOAT);
	CUDAMemory::Ptr<Vector4> ptr_sky_data = CUDAMemory::malloc(scene.sky.data);
	CUDAMemory::copy_array(sky_array, scene.sky.width * sizeof(float4), scene.sky.height, ptr_sky_data.ptr);
	CUDAMemory::free(ptr_sky_data);

	sky_texture = CUDAMemory::create_texture(sky_array, CU_TR_FILTER_MODE_LINEAR, CU_TR_ADDRESS_MODE_CLAMP);
	cuda_module.get_global("sky_texture").set_value(sky_texture);

	global_sky_scale = cuda_module.get_global("sky_scale");
	global_sky_scale.set_value(scene.sky.scale);
}

void Integrator::init_rng() {
	ptr_pmj_samples = CUDAMemory::malloc<PMJ::Point>(PMJ::samples, PMJ_NUM_SEQUENCES * PMJ_NUM_SAMPLES_PER_SEQUENCE);
	cuda_module.get_global("pmj_samples").set_value(ptr_pmj_samples);

	ptr_blue_noise_textures = CUDAMemory::malloc<unsigned short>(&BlueNoise::textures[0][0][0], BLUE_NOISE_NUM_TEXTURES * BLUE_NOISE_TEXTURE_DIM * BLUE_NOISE_TEXTURE_DIM);
	cuda_module.get_global("blue_noise_textures").set_value(ptr_blue_noise_textures);
}

void Integrator::init_aovs() {
	for (size_t i = 0; i < size_t(AOVType::COUNT); i++) {
		aovs[i].framebuffer = CUDAMemory::Ptr<float4>(NULL);
		aovs[i].accumulator = CUDAMemory::Ptr<float4>(NULL);
	}
}

void Integrator::free_materials() {
	CUDAMemory::free(ptr_material_types);
	CUDAMemory::free(ptr_materials);

	CUDAMemory::free(ptr_media);

	if (scene.asset_manager.textures.size() > 0) {
		CUDAMemory::free(ptr_textures);

		for (int i = 0; i < scene.asset_manager.textures.size(); i++) {
			CUDAMemory::free_array(texture_arrays[i]);
			CUDAMemory::free_texture(textures[i].texture);
		}

		textures      .clear();
		texture_arrays.clear();
	}
}

void Integrator::free_geometry() {
	CUDAMemory::free_pinned(pinned_mesh_bvh_root_indices);
	CUDAMemory::free_pinned(pinned_mesh_material_ids);
	CUDAMemory::free_pinned(pinned_mesh_transforms);
	CUDAMemory::free_pinned(pinned_mesh_transforms_inv);
	CUDAMemory::free_pinned(pinned_mesh_transforms_prev);
	CUDAMemory::free_pinned(pinned_light_mesh_cumulative_probability);
	CUDAMemory::free_pinned(pinned_light_mesh_triangle_span);
	CUDAMemory::free_pinned(pinned_light_mesh_transform_indices);

	CUDAMemory::free(ptr_mesh_bvh_root_indices);
	CUDAMemory::free(ptr_mesh_material_ids);
	CUDAMemory::free(ptr_mesh_transforms);
	CUDAMemory::free(ptr_mesh_transforms_inv);
	CUDAMemory::free(ptr_mesh_transforms_prev);

	switch (cpu_config.bvh_type) {
		case BVHType::BVH:
		case BVHType::SBVH: CUDAMemory::free(ptr_bvh_nodes_2); break;
		case BVHType::BVH4: CUDAMemory::free(ptr_bvh_nodes_4); break;
		case BVHType::BVH8: CUDAMemory::free(ptr_bvh_nodes_8); break;
	}

	CUDAMemory::free(ptr_triangles);
}

void Integrator::free_sky() {
	CUDAMemory::free_array(sky_array);
	CUDAMemory::free_texture(sky_texture);
}

void Integrator::free_rng() {
	CUDAMemory::free(ptr_pmj_samples);
	CUDAMemory::free(ptr_blue_noise_textures);
}

void Integrator::free_aovs() {
	for (size_t i = 0; i < size_t(AOVType::COUNT); i++) {
		if (aov_is_enabled(AOVType(i))) {
			CUDAMemory::free(aovs[i].framebuffer);
			CUDAMemory::free(aovs[i].accumulator);
		}
	}
}

void Integrator::aovs_clear_to_zero() {
	for (size_t i = 0; i < size_t(AOVType::COUNT); i++) {
		if (aov_is_enabled(AOVType(i))) {
			CUDAMemory::memset_async(aovs[i].framebuffer, 0, screen_pitch * screen_height, memory_stream);
		}
	}
}

bool Integrator::aov_render_gui_checkbox(AOVType aov_type, const char * aov_name) {
	bool enabled = aov_is_enabled(aov_type);
	if (ImGui::Checkbox(aov_name, &enabled)) {
		if (enabled) {
			aov_enable(aov_type);
		} else {
			aov_disable(aov_type);
		}
		return true;
	}
	return false;
};

// Construct Top Level Acceleration Structure (TLAS) over the Meshes in the Scene
void Integrator::build_tlas() {
	tlas_builder->build(scene.meshes);
	tlas_converter->convert();

	switch (cpu_config.bvh_type) {
		case BVHType::BVH:
		case BVHType::SBVH: CUDAMemory::memcpy_async(ptr_bvh_nodes_2, static_cast<BVH2 *>(tlas.get())->nodes.data(), tlas->node_count(), memory_stream); break;
		case BVHType::BVH4: CUDAMemory::memcpy_async(ptr_bvh_nodes_4, static_cast<BVH4 *>(tlas.get())->nodes.data(), tlas->node_count(), memory_stream); break;
		case BVHType::BVH8: CUDAMemory::memcpy_async(ptr_bvh_nodes_8, static_cast<BVH8 *>(tlas.get())->nodes.data(), tlas->node_count(), memory_stream); break;
		default: ASSERT_UNREACHABLE();
	}
	ASSERT(tlas->indices.data());

	for (int i = 0; i < scene.meshes.size(); i++) {
		const Mesh & mesh = scene.meshes[tlas->indices[i]];

		pinned_mesh_bvh_root_indices[i] = mesh_data_bvh_offsets[mesh.mesh_data_handle.handle] | (mesh.has_identity_transform() << 31);

		ASSERT(mesh.material_handle.handle != INVALID);
		pinned_mesh_material_ids[i] = mesh.material_handle.handle;

		memcpy(pinned_mesh_transforms     [i].cells, mesh.transform     .cells, sizeof(Matrix3x4));
		memcpy(pinned_mesh_transforms_inv [i].cells, mesh.transform_inv .cells, sizeof(Matrix3x4));
		memcpy(pinned_mesh_transforms_prev[i].cells, mesh.transform_prev.cells, sizeof(Matrix3x4));
	}

	CUDAMemory::memcpy_async(ptr_mesh_bvh_root_indices,  pinned_mesh_bvh_root_indices, scene.meshes.size(), memory_stream);
	CUDAMemory::memcpy_async(ptr_mesh_material_ids,      pinned_mesh_material_ids,     scene.meshes.size(), memory_stream);
	CUDAMemory::memcpy_async(ptr_mesh_transforms,        pinned_mesh_transforms,       scene.meshes.size(), memory_stream);
	CUDAMemory::memcpy_async(ptr_mesh_transforms_inv,    pinned_mesh_transforms_inv,   scene.meshes.size(), memory_stream);
	CUDAMemory::memcpy_async(ptr_mesh_transforms_prev,   pinned_mesh_transforms_prev,  scene.meshes.size(), memory_stream);
}

void Integrator::update(float delta, Allocator * frame_allocator) {
	if (invalidated_gpu_config && gpu_config.enable_svgf && scene.camera.aperture_radius > 0.0f) {
		IO::print("WARNING: SVGF and DoF cannot simultaneously be enabled!\n"_sv);
		scene.camera.aperture_radius = 0.0f;
		invalidated_camera = true;
	}

	if (cpu_config.enable_scene_update) {
		scene.update(delta);
		invalidated_scene = true;
	} else if (gpu_config.enable_svgf || invalidated_scene) {
		scene.camera.update(0.0f);
		scene.update(0.0f);
	}

	if (invalidated_scene) {
		invalidated_scene = false;
		build_tlas();
	}

	scene.camera.update(delta);

	if (scene.camera.moved || invalidated_camera) {
		// Upload Camera
		struct CUDACamera {
			Vector3 position;
			Vector3 bottom_left_corner;
			Vector3 x_axis;
			Vector3 y_axis;
			float pixel_spread_angle;
			float aperture_radius;
			float focal_distance;
		} cuda_camera;

		cuda_camera.position           = scene.camera.position;
		cuda_camera.bottom_left_corner = scene.camera.bottom_left_corner_rotated;
		cuda_camera.x_axis             = scene.camera.x_axis_rotated;
		cuda_camera.y_axis             = scene.camera.y_axis_rotated;
		cuda_camera.pixel_spread_angle = scene.camera.pixel_spread_angle;
		cuda_camera.aperture_radius    = scene.camera.aperture_radius;
		cuda_camera.focal_distance     = scene.camera.focal_distance;

		global_camera.set_value_async(cuda_camera, memory_stream);

		if (!gpu_config.enable_svgf) {
			sample_index = 0;
		}

		invalidated_camera = false;
	}

	if (pixel_query_status == PixelQueryStatus::OUTPUT_READY) {
		CUDAMemory::memcpy_async(&pixel_query, CUDAMemory::Ptr<PixelQuery>(global_pixel_query.ptr), 1, memory_stream);

		if (pixel_query.mesh_id != INVALID) {
			pixel_query.mesh_id = tlas->indices[pixel_query.mesh_id];
		}

		// Reset pixel query
		pixel_query.pixel_index = INVALID;
		CUDAMemory::memcpy_async(CUDAMemory::Ptr<PixelQuery>(global_pixel_query.ptr), &pixel_query, 1, memory_stream);

		pixel_query_status = PixelQueryStatus::INACTIVE;
	}

	if (invalidated_aovs) {
		invalidated_aovs = false;

		for (size_t i = 0; i < size_t(AOVType::COUNT); i++) {
			bool is_enabled   = aov_is_enabled(AOVType(i));
			bool is_allocated = aovs[i].framebuffer.ptr != NULL;

			if (is_enabled && !is_allocated) {
				aovs[i].framebuffer = CUDAMemory::malloc<float4>(screen_pitch * screen_height);
				aovs[i].accumulator = CUDAMemory::malloc<float4>(screen_pitch * screen_height);
			} else if (!is_enabled && is_allocated) {
				CUDAMemory::free(aovs[i].framebuffer);
				CUDAMemory::free(aovs[i].accumulator);
			}
		}

		global_aovs.set_value_async(aovs, memory_stream);

		invalidated_gpu_config = true;
	}

	if (invalidated_gpu_config) {
		invalidated_gpu_config = false;
		sample_index = 0;
		global_config.set_value_async(gpu_config, memory_stream);
	} else if (scene.camera.moved && !gpu_config.enable_svgf) {
		sample_index = 0;
	} else {
		sample_index++;
	}

}
