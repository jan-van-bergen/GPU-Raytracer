#pragma once
#include <vector>
#include <unordered_map>

#include "../CUDA_Source/Common.h"

#include "MeshData.h"
#include "Material.h"
#include "Texture.h"

#include "Util/ThreadPool.h"

struct AssetManager {
	std::vector<MeshData> mesh_datas;
	std::vector<Material> materials;
	std::vector<Texture>  textures;

private:
	std::unordered_map<std::string, MeshDataHandle> mesh_data_cache;
	std::unordered_map<std::string, TextureHandle>  texture_cache;

	std::mutex mesh_datas_mutex;
	std::mutex textures_mutex;

	ThreadPool thread_pool;

	bool assets_loaded = false;

public:
	void init();

	MeshDataHandle add_mesh_data(const char * filename);
	MeshDataHandle add_mesh_data(Triangle * triangles, int triangle_count);

	MaterialHandle add_material(const Material & material);

	TextureHandle add_texture(const char * filename);

	void wait_until_loaded();
	
	inline MeshData & get_mesh_data(MeshDataHandle handle) { return mesh_datas[handle.handle]; }
	inline Material & get_material (MaterialHandle handle) { return materials [handle.handle]; }
	inline Texture  & get_texture  (TextureHandle  handle) { return textures  [handle.handle]; }
};
