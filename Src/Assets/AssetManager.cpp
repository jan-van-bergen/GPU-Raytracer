#include "AssetManager.h"

#include "Core/IO.h"
#include "Core/Timer.h"

#include "Math/Vector4.h"

#include "BVHLoader.h"
#include "TextureLoader.h"

#include "Util/Util.h"
#include "Util/StringUtil.h"
#include "Util/ThreadPool.h"
#include "Util/Geometry.h"

AssetManager::AssetManager(Allocator * allocator) : mesh_datas(allocator), materials(allocator), media(allocator), textures(allocator), mesh_data_cache(allocator), texture_cache(allocator), thread_pool(make_owned<ThreadPool>()) {
	Material default_material = { };
	default_material.name    = "Default";
	default_material.diffuse = Vector3(1.0f, 0.0f, 1.0f);
	add_material(std::move(default_material));

	Medium default_medium = { };
	default_medium.name = "Default";
	add_medium(std::move(default_medium));
}

// NOTE: Seemingly pointless desctructor needed here since ThreadPool is
// forward declared, so its destructor is not available in the header file
AssetManager::~AssetManager() = default;

MeshDataHandle AssetManager::new_mesh_data() {
	MutexLock lock(mesh_datas_mutex);
	MeshDataHandle mesh_data_handle = { mesh_datas.size() };
	mesh_datas.emplace_back();
	return mesh_data_handle;
}

TextureHandle AssetManager::new_texture() {
	MutexLock lock(textures_mutex);
	TextureHandle texture_handle = { textures.size() };
	textures.emplace_back();
	return texture_handle;
}

MeshDataHandle AssetManager::add_mesh_data(String filename, FallbackLoader fallback_loader) {
	String bvh_filename = BVHLoader::get_bvh_filename(filename.view(), nullptr);
	return add_mesh_data(std::move(filename), std::move(bvh_filename), std::move(fallback_loader));
}

MeshDataHandle AssetManager::add_mesh_data(String filename, String bvh_filename, FallbackLoader fallback_loader) {
	MeshDataHandle & mesh_data_handle = mesh_data_cache[filename];

	if (mesh_data_handle.handle != INVALID) return mesh_data_handle;

	mesh_data_handle = new_mesh_data();

	thread_pool->submit([this, filename = std::move(filename), bvh_filename = std::move(bvh_filename), fallback_loader = std::move(fallback_loader), mesh_data_handle]() mutable {
		BVH2     bvh       = { };
		MeshData mesh_data = { };

		bool bvh_loaded = BVHLoader::try_to_load(filename, bvh_filename, &mesh_data, &bvh);
		if (!bvh_loaded) {
			mesh_data.triangles = fallback_loader(filename, nullptr);

			if (mesh_data.triangles.size() == 0) {
				// FIXME: Right now empty MeshData is handled by inserting a dummy Triangle
				Triangle triangle = { };
				triangle.position_0 = Vector3(-1.0f, -1.0f, 0.0f);
				triangle.position_1 = Vector3( 0.0f, +1.0f, 0.0f);
				triangle.position_2 = Vector3(+1.0f, -1.0f, 0.0f);
				triangle.tex_coord_0 = Vector2(0.0f, 1.0f);
				triangle.tex_coord_1 = Vector2(0.5f, 0.0f);
				triangle.tex_coord_2 = Vector2(1.0f, 1.0f);
				triangle.normal_0 = Vector3(0.0f, 0.0f, 1.0f);
				triangle.normal_1 = Vector3(0.0f, 0.0f, 1.0f);
				triangle.normal_2 = Vector3(0.0f, 0.0f, 1.0f);
				triangle.init();
				mesh_data.triangles = { triangle };
			}

			bvh = BVH::create_from_triangles(mesh_data.triangles);
			BVHLoader::save(bvh_filename, mesh_data, bvh);
		}

		if (cpu_config.bvh_type != BVHType::BVH8) {
			BVHCollapser::collapse(bvh);
		}

		mesh_data.bvh = BVH::create_from_bvh2(std::move(bvh));

		{
			MutexLock lock(mesh_datas_mutex);
			get_mesh_data(mesh_data_handle) = std::move(mesh_data);
		}
	});

	return mesh_data_handle;
}

MeshDataHandle AssetManager::add_mesh_data(Array<Triangle> triangles) {
	MeshDataHandle mesh_data_handle = new_mesh_data();

	thread_pool->submit([this, triangles = std::move(triangles), mesh_data_handle]() mutable {
		BVH2 bvh = BVH::create_from_triangles(triangles);

		MeshData mesh_data = { };
		mesh_data.triangles = std::move(triangles);
		mesh_data.bvh = BVH::create_from_bvh2(std::move(bvh));

		{
			MutexLock mutex(mesh_datas_mutex);
			get_mesh_data(mesh_data_handle) = std::move(mesh_data);
		}
	});

	return mesh_data_handle;
}

MaterialHandle AssetManager::add_material(Material material) {
	MaterialHandle material_handle = { int(materials.size()) };
	materials.emplace_back(std::move(material));

	return material_handle;
}

MediumHandle AssetManager::add_medium(Medium medium) {
	MediumHandle medium_handle = { int(media.size()) };
	media.emplace_back(std::move(medium));

	return medium_handle;
}

TextureHandle AssetManager::add_texture(String filename, String name) {
	TextureHandle & texture_handle = texture_cache[filename];

	// If the cache already contains this Texture simply return its index
	if (texture_handle.handle != INVALID) return texture_handle;

	// Otherwise, create new Texture and load it from disk
	texture_handle = new_texture();

	thread_pool->submit([this, filename = std::move(filename), name = std::move(name), texture_handle]() mutable {
		Texture texture = { };
		texture.name = std::move(name);

		bool success = false;

		StringView file_extension = Util::get_file_extension(filename.view());
		if (!file_extension.is_empty()) {
			if (file_extension == "dds") {
				success = TextureLoader::load_dds(filename, &texture); // DDS is loaded using custom code
			} else {
				success = TextureLoader::load_stb(filename, &texture); // other file formats use stb_image
			}
		}

		if (!success) {
			IO::print("WARNING: Failed to load Texture '{}'!\n"_sv, filename);

			// Use a default 1x1 pink Texture
			texture.data.resize(sizeof(Vector4));
			new (texture.data.data()) Vector4(1.0f, 0.0f, 1.0f, 1.0f);

			texture.format = Texture::Format::RGBA;
			texture.width  = 1;
			texture.height = 1;
			texture.channels = 4;
			texture.mip_offsets = { 0 };
		}

		{
			MutexLock lock(textures_mutex);
			get_texture(texture_handle) = std::move(texture);
		}
	});

	return texture_handle;
}

void AssetManager::wait_until_loaded() {
	if (assets_loaded) return; // Only necessary (and valid) to do this once

	thread_pool->sync();
	thread_pool.release();

	mesh_data_cache.clear();
	texture_cache  .clear();

	assets_loaded = true;
}
