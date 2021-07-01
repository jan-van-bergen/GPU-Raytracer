#pragma once
#include <mutex>
#include <atomic>

#include <vector>
#include <unordered_map>

#include "../CUDA_Source/Common.h"

#include "MeshData.h"
#include "Material.h"
#include "Texture.h"

struct AssetManager {
	std::vector<MeshData> mesh_datas;
	std::vector<Material> materials;
	std::vector<Texture>  textures;
	
	std::mutex       textures_mutex;
	std::atomic<int> num_textures_finished;

private:
	std::unordered_map<std::string, MeshDataHandle> mesh_data_cache;
	std::unordered_map<std::string, TextureHandle>  texture_cache;

public:
	MeshDataHandle add_mesh_data(const char * filename);
	MeshDataHandle add_mesh_data(Triangle * triangles, int triangle_count);

	MaterialHandle add_material(const Material & material);

	TextureHandle add_texture(const char * filename);

	void wait_until_textures_loaded();
	
	inline MeshData & get_mesh_data(MeshDataHandle handle) { return mesh_datas[handle.handle]; }
	inline Material & get_material (MaterialHandle handle) { return materials [handle.handle]; }
	inline Texture  & get_texture  (TextureHandle  handle) { return textures  [handle.handle]; }
};
