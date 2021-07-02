#include "AssetManager.h"

#include "BVHLoader.h"
#include "OBJLoader.h"
#include "TextureLoader.h"

#include "BVH/Builders/BVHBuilder.h"
#include "BVH/Builders/SBVHBuilder.h"
#include "BVH/BVHOptimizer.h"

#include "Util/ScopeTimer.h"

static BVH build_bvh(const Triangle * triangles, int triangle_count) {
	printf("Constructing BVH...\r");

	BVH bvh;
#if BVH_TYPE == BVH_SBVH
	{
		ScopeTimer timer("SBVH Construction");

		SBVHBuilder sbvh_builder;
		sbvh_builder.init(&bvh, triangle_count, BVHLoader::MAX_PRIMITIVES_IN_LEAF);
		sbvh_builder.build(triangles, triangle_count);
		sbvh_builder.free();
	}
#else // All other BVH types use standard BVH as a starting point
	{
		ScopeTimer timer("BVH Construction");
			
		BVHBuilder bvh_builder;
		bvh_builder.init(&bvh, triangle_count, BVHLoader::MAX_PRIMITIVES_IN_LEAF);
		bvh_builder.build(triangles, triangle_count);
		bvh_builder.free();
	}
#endif

#if BVH_ENABLE_OPTIMIZATION
	BVHOptimizer::optimize(bvh);
#endif

	return bvh;
}

MeshDataHandle AssetManager::add_mesh_data(const char * filename) {
	MeshDataHandle & mesh_data_handle = mesh_data_cache[filename];

	if (mesh_data_handle.handle != INVALID) return mesh_data_handle;
	
	MeshData mesh_data = { };

	BVH bvh;
	bool bvh_loaded = BVHLoader::try_to_load(filename, mesh_data, bvh);
	if (!bvh_loaded) {
		// Unable to load disk cached BVH, load model from source and construct BVH
		OBJLoader::load(filename, mesh_data.triangles, mesh_data.triangle_count);
		
		bvh = build_bvh(mesh_data.triangles, mesh_data.triangle_count);
		BVHLoader::save(filename, mesh_data, bvh);
	}
	
	mesh_data.init_bvh(bvh);
	
	mesh_data_handle.handle = mesh_datas.size();
	mesh_datas.push_back(mesh_data);
	
	return mesh_data_handle;
}

MeshDataHandle AssetManager::add_mesh_data(Triangle * triangles, int triangle_count) {
	BVH bvh = build_bvh(triangles, triangle_count);

	MeshData mesh_data = { };
	mesh_data.triangles      = triangles;
	mesh_data.triangle_count = triangle_count;
	mesh_data.init_bvh(bvh);
	
	MeshDataHandle mesh_data_id = { mesh_datas.size() };
	mesh_datas.push_back(mesh_data);
	
	return mesh_data_id;
}

MaterialHandle AssetManager::add_material(const Material & material) {
	MaterialHandle material_id = { materials.size() };
	materials.push_back(material);

	return material_id;
}

static void load_texture(char * filename, int texture_id, AssetManager & asset_manager) {
	Texture texture = { };
	bool success = false;
	
	const char * file_extension = Util::file_get_extension(filename);
	if (file_extension) {
		if (strcmp(file_extension, "dds") == 0) {
			success = TextureLoader::load_dds(filename, texture); // DDS is loaded using custom code
		} else {
			success = TextureLoader::load_stb(filename, texture); // other file formats use stb_image
		}
	}

	if (!success) {
		printf("WARNING: Failed to load Texture '%s'!\n", filename);

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
	
	{
		std::lock_guard<std::mutex> lock(asset_manager.textures_mutex);
		asset_manager.textures[texture_id] = texture;
	}
	asset_manager.num_textures_finished++;

	free(filename); // Allocated by strdup
}

TextureHandle AssetManager::add_texture(const char * filename) {
	TextureHandle & texture_id = texture_cache[filename];

	// If the cache already contains this Texture simply return its index
	if (texture_id.handle != INVALID) return texture_id;
	
	// Otherwise, create new Texture and load it from disk
	{
		std::lock_guard<std::mutex> lock(textures_mutex);

		texture_id.handle = textures.size();
		textures.emplace_back();
	}

	std::thread loader(load_texture, _strdup(filename), texture_id.handle, std::ref(*this));
	loader.detach();

	return texture_id;
}

void AssetManager::wait_until_textures_loaded() {
	using namespace std::chrono_literals;

	while (num_textures_finished < textures.size()) {
		std::this_thread::sleep_for(100ms);
	}

	mesh_data_cache.clear();
	texture_cache.clear();
}
