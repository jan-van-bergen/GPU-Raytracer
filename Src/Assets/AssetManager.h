#pragma once
#include "Core/Array.h"
#include "Core/HashMap.h"
#include "Core/String.h"
#include "Core/Mutex.h"
#include "Core/Function.h"
#include "Core/OwnPtr.h"
#include "Core/Allocators/Allocator.h"

#include "Renderer/MeshData.h"
#include "Renderer/Material.h"
#include "Renderer/Medium.h"
#include "Renderer/Texture.h"

#include "BVHLoader.h"
#include "BVH/BVHCollapser.h"

struct ThreadPool;

struct AssetManager {
	Array<MeshData> mesh_datas;
	Array<Material> materials;
	Array<Medium>   media;
	Array<Texture>  textures;

	AssetManager(Allocator * allocator);
	~AssetManager();

private:
	HashMap<String, MeshDataHandle> mesh_data_cache;
	HashMap<String, TextureHandle>  texture_cache;

	Mutex mesh_datas_mutex;
	Mutex textures_mutex;

	OwnPtr<ThreadPool> thread_pool;

	bool assets_loaded = false;

	MeshDataHandle new_mesh_data();
	TextureHandle  new_texture();

public:
	using FallbackLoader = Function<Array<Triangle>(const String & filename, Allocator * allocator)>;

	MeshDataHandle add_mesh_data(String filename,                      FallbackLoader fallback_loader);
	MeshDataHandle add_mesh_data(String filename, String bvh_filename, FallbackLoader fallback_loader);
	MeshDataHandle add_mesh_data(Array<Triangle> triangles);

	MaterialHandle add_material(Material material);

	MediumHandle add_medium(Medium medium);

	TextureHandle add_texture(String filename, String name);

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
