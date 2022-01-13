#pragma once
#include "Core/Array.h"
#include "Core/HashMap.h"
#include "Core/String.h"
#include "Core/Mutex.h"
#include "Core/OwnPtr.h"

#include "Pathtracer/MeshData.h"
#include "Pathtracer/Material.h"
#include "Pathtracer/Medium.h"
#include "Pathtracer/Texture.h"

#include "BVHLoader.h"
#include "BVH/BVHCollapser.h"

struct ThreadPool;

struct AssetManager {
	Array<MeshData> mesh_datas;
	Array<Material> materials;
	Array<Medium>   media;
	Array<Texture>  textures;

	AssetManager();
	~AssetManager();

private:
	HashMap<String, MeshDataHandle> mesh_data_cache;
	HashMap<String, TextureHandle>  texture_cache;

	Mutex mesh_datas_mutex;
	Mutex textures_mutex;

	OwnPtr<ThreadPool> thread_pool;

	bool assets_loaded = false;

public:
	template<typename FallbackLoader>
	MeshDataHandle add_mesh_data(const String & filename, FallbackLoader fallback_loader) {
		String bvh_filename = BVHLoader::get_bvh_filename(filename.view());
		MeshDataHandle mesh_data_handle = add_mesh_data(filename, bvh_filename, fallback_loader);

		return mesh_data_handle;
	}

	template<typename FallbackLoader>
	MeshDataHandle add_mesh_data(const String & filename, const String & bvh_filename, FallbackLoader fallback_loader) {
		MeshDataHandle & mesh_data_handle = mesh_data_cache[filename];

		if (mesh_data_handle.handle != INVALID) return mesh_data_handle;

		BVH2     bvh       = { };
		MeshData mesh_data = { };

		bool bvh_loaded = BVHLoader::try_to_load(filename, bvh_filename, mesh_data, bvh);
		if (!bvh_loaded) {
			mesh_data.triangles = fallback_loader(filename);

			if (mesh_data.triangles.size() == 0) {
				return { INVALID };
			}

			bvh = BVH::create_from_triangles(mesh_data.triangles);
			BVHLoader::save(bvh_filename, mesh_data, bvh);
		}

		if (config.bvh_type != BVHType::BVH8) {
			BVHCollapser::collapse(bvh);
		}

		mesh_data.bvh = BVH::create_from_bvh2(std::move(bvh));

		mesh_data_handle.handle = mesh_datas.size();
		mesh_datas.push_back(std::move(mesh_data));

		return mesh_data_handle;
	}

	MeshDataHandle add_mesh_data(MeshData mesh_data);
	MeshDataHandle add_mesh_data(Array<Triangle> triangles);

	MaterialHandle add_material(const Material & material);

	MediumHandle add_medium(const Medium & medium);

	TextureHandle add_texture(const String & filename);

	void wait_until_loaded();

	MeshData & get_mesh_data(MeshDataHandle handle) { return mesh_datas[handle.handle]; }
	Material & get_material (MaterialHandle handle) { return materials [handle.handle]; }
	Medium   & get_medium   (MediumHandle   handle) { return media     [handle.handle]; }
	Texture  & get_texture  (TextureHandle  handle) { return textures  [handle.handle]; }

	const MeshData & get_mesh_data(MeshDataHandle handle) const { return mesh_datas[handle.handle]; }
	const Material & get_material (MaterialHandle handle) const { return materials [handle.handle]; }
	const Medium   & get_medium   (MediumHandle   handle) const { return media     [handle.handle]; }
	const Texture  & get_texture  (TextureHandle  handle) const { return textures  [handle.handle]; }
};
