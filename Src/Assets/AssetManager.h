#pragma once
#include "MeshData.h"
#include "Material.h"
#include "Texture.h"

#include "BVHLoader.h"

#include "Util/Array.h"
#include "Util/HashMap.h"
#include "Util/String.h"
#include "Util/ThreadPool.h"

struct AssetManager {
	Array<MeshData> mesh_datas;
	Array<Material> materials;
	Array<Texture>  textures;

private:
	HashMap<String, MeshDataHandle, StringHash> mesh_data_cache;
	HashMap<String, TextureHandle,  StringHash> texture_cache;

	std::mutex mesh_datas_mutex;
	std::mutex textures_mutex;

	ThreadPool thread_pool;

	bool assets_loaded = false;

	BVH build_bvh(const Triangle * triangles, int triangle_count);

public:
	void init();

	template<typename FallbackLoader>
	MeshDataHandle add_mesh_data(const char * filename, FallbackLoader fallback_loader) {
		const char * bvh_filename = BVHLoader::get_bvh_filename(filename);
		MeshDataHandle mesh_data_handle = add_mesh_data(filename, bvh_filename, fallback_loader);

		delete [] bvh_filename;
		return mesh_data_handle;
	}

	template<typename FallbackLoader>
	MeshDataHandle add_mesh_data(const char * filename, const char * bvh_filename, FallbackLoader fallback_loader) {
		MeshDataHandle & mesh_data_handle = mesh_data_cache[filename];

		if (mesh_data_handle.handle != INVALID) return mesh_data_handle;

		BVH      bvh       = { };
		MeshData mesh_data = { };

		bool bvh_loaded = BVHLoader::try_to_load(filename, bvh_filename, mesh_data, bvh);
		if (!bvh_loaded) {
			fallback_loader(filename, mesh_data.triangles, mesh_data.triangle_count);

			bvh = build_bvh(mesh_data.triangles, mesh_data.triangle_count);
			BVHLoader::save(bvh_filename, mesh_data, bvh);
		}

		mesh_data.init_bvh(bvh);

		mesh_data_handle.handle = mesh_datas.size();
		mesh_datas.push_back(mesh_data);

		return mesh_data_handle;
	}

	MeshDataHandle add_mesh_data(const MeshData & mesh_data);
	MeshDataHandle add_mesh_data(Triangle * triangles, int triangle_count);

	MaterialHandle add_material(const Material & material);

	TextureHandle add_texture(const char * filename);

	void wait_until_loaded();

	inline MeshData & get_mesh_data(MeshDataHandle handle) { return mesh_datas[handle.handle]; }
	inline Material & get_material (MaterialHandle handle) { return materials [handle.handle]; }
	inline Texture  & get_texture  (TextureHandle  handle) { return textures  [handle.handle]; }
};
