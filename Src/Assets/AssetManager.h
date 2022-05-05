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

#include "Util/ThreadPool.h"

struct AssetManager {
	Array<MeshData> mesh_datas;
	Array<Material> materials;
	Array<Medium>   media;
	Array<Texture>  textures;

	AssetManager(Allocator * allocator);
	~AssetManager();

private:
	HashMap<String, Handle<MeshData>> mesh_data_cache;
	HashMap<String, Handle<Texture>>  texture_cache;

	Mutex mesh_datas_mutex;
	Mutex textures_mutex;

	bool assets_loaded = false;

	Handle<MeshData> new_mesh_data();
	Handle<Texture>  new_texture();

public:
	using FallbackLoader = Function<Array<Triangle>(const String & filename, Allocator * allocator)>;

	Handle<MeshData> add_mesh_data(String filename,                      FallbackLoader fallback_loader);
	Handle<MeshData> add_mesh_data(String filename, String bvh_filename, FallbackLoader fallback_loader);
	Handle<MeshData> add_mesh_data(Array<Triangle> triangles);

	Handle<Material> add_material(Material material);

	Handle<Medium> add_medium(Medium medium);

	Handle<Texture> add_texture(String filename, String name);

	void wait_until_loaded();

	MeshData & get_mesh_data(Handle<MeshData> handle) { return mesh_datas[handle.handle]; }
	Material & get_material (Handle<Material> handle) { return materials [handle.handle]; }
	Medium   & get_medium   (Handle<Medium>   handle) { return media     [handle.handle]; }
	Texture  & get_texture  (Handle<Texture>  handle) { return textures  [handle.handle]; }

	const MeshData & get_mesh_data(Handle<MeshData> handle) const { return mesh_datas[handle.handle]; }
	const Material & get_material (Handle<Material> handle) const { return materials [handle.handle]; }
	const Medium   & get_medium   (Handle<Medium>   handle) const { return media     [handle.handle]; }
	const Texture  & get_texture  (Handle<Texture>  handle) const { return textures  [handle.handle]; }
};
