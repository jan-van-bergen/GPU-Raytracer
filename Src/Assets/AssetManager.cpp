#include "AssetManager.h"

#include "Math/Vector4.h"

#include "BVHLoader.h"
#include "TextureLoader.h"

#include "BVH/Builders/BVHBuilder.h"
#include "BVH/Builders/SBVHBuilder.h"
#include "BVH/BVHCollapser.h"
#include "BVH/BVHOptimizer.h"

#include "Util/Util.h"
#include "Util/StringUtil.h"
#include "Util/ScopeTimer.h"
#include "Util/ThreadPool.h"

BVH AssetManager::build_bvh(const Triangle * triangles, int triangle_count) {
	printf("Constructing BVH...\r");
	BVH bvh;

	// Only the SBVH uses SBVH as its starting point,
	// all other BVH types use the standard BVH as their starting point
	if (config.bvh_type == BVHType::SBVH) {
		ScopeTimer timer("SBVH Construction");

		SBVHBuilder sbvh_builder = { };
		sbvh_builder.init(&bvh, triangle_count);
		sbvh_builder.build(triangles, triangle_count);
		sbvh_builder.free();
	} else  {
		ScopeTimer timer("BVH Construction");

		BVHBuilder bvh_builder = { };
		bvh_builder.init(&bvh, triangle_count);
		bvh_builder.build(triangles, triangle_count);
		bvh_builder.free();
	}

	if (config.enable_bvh_optimization) {
		BVHOptimizer::optimize(bvh);
	}

	return bvh;
}

void AssetManager::init() {
	thread_pool = new ThreadPool();
	thread_pool->init();

	textures_mutex.init();
	mesh_datas_mutex.init();

	Material default_material = { };
	default_material.name    = "Default";
	default_material.diffuse = Vector3(1.0f, 0.0f, 1.0f);
	add_material(default_material);

	Medium default_medium = { };
	default_medium.name = "Default";
	add_medium(default_medium);
}

MeshDataHandle AssetManager::add_mesh_data(const MeshData & mesh_data) {
	MeshDataHandle mesh_data_id = { int(mesh_datas.size()) };
	mesh_datas.push_back(mesh_data);

	return mesh_data_id;
}

MeshDataHandle AssetManager::add_mesh_data(Triangle * triangles, int triangle_count) {
	BVH bvh = build_bvh(triangles, triangle_count);

	MeshData mesh_data = { };
	mesh_data.triangles      = triangles;
	mesh_data.triangle_count = triangle_count;
	mesh_data.init_bvh(bvh);

	return add_mesh_data(mesh_data);
}

MaterialHandle AssetManager::add_material(const Material & material) {
	MaterialHandle material_id = { int(materials.size()) };
	materials.push_back(material);

	return material_id;
}

MediumHandle AssetManager::add_medium(const Medium & medium) {
	MediumHandle medium_id = { int(media.size()) };
	media.push_back(medium);

	return medium_id;
}

TextureHandle AssetManager::add_texture(const String & filename) {
	TextureHandle & texture_id = texture_cache[filename];

	// If the cache already contains this Texture simply return its index
	if (texture_id.handle != INVALID) return texture_id;

	// Otherwise, create new Texture and load it from disk
	textures_mutex.lock();
	texture_id.handle = textures.size();
	textures.emplace_back();
	textures_mutex.unlock();

	thread_pool->submit([this, filename, texture_id]() {
		StringView name = Util::remove_directory(filename.view());

		Texture texture = { };
		texture.name = name;

		bool success = false;

		StringView file_extension = Util::get_file_extension(filename.view());
		if (!file_extension.is_empty()) {
			if (file_extension == "dds") {
				success = TextureLoader::load_dds(filename, texture); // DDS is loaded using custom code
			} else {
				success = TextureLoader::load_stb(filename, texture); // other file formats use stb_image
			}
		}

		if (!success) {
			printf("WARNING: Failed to load Texture '%.*s'!\n", FMT_STRING(filename));

			if (texture.data) delete [] texture.data;

			// Use a default 1x1 pink Texture
			texture.data = reinterpret_cast<const unsigned char *>(new Vector4(1.0f, 0.0f, 1.0f, 1.0f));
			texture.format = Texture::Format::RGBA;
			texture.width  = 1;
			texture.height = 1;
			texture.channels = 4;
			texture.mip_levels  = 1;
			texture.mip_offsets = new int(0);
		}

		textures_mutex.lock();
		textures[texture_id.handle] = texture;
		textures_mutex.unlock();
	});

	return texture_id;
}

void AssetManager::wait_until_loaded() {
	if (assets_loaded) return; // Only necessary (and valid) to do this once

	thread_pool->sync();
	thread_pool->free();
	delete thread_pool;

	textures_mutex.free();
	mesh_datas_mutex.free();

	mesh_data_cache.clear();
	texture_cache  .clear();

	assets_loaded = true;
}
