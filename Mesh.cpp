#include "Mesh.h"

#include <unordered_map>

#include "OBJLoader.h"
#include "BVHBuilders.h"

#include "Util.h"
#include "ScopeTimer.h"

static std::unordered_map<std::string, Mesh *> cache;

static void bvh_save_to_disk(const BVH & bvh, const char * filename, const char * file_extension) {
	assert(file_extension[0] == '.');

	int    bvh_filename_length = strlen(filename) + strlen(file_extension) + 1;
	char * bvh_filename        = reinterpret_cast<char *>(_malloca(bvh_filename_length));

	strcpy_s(bvh_filename, bvh_filename_length, filename);
	strcat_s(bvh_filename, bvh_filename_length, file_extension);

	FILE * file;
	fopen_s(&file, bvh_filename, "wb");

	if (file == nullptr) {
		printf("WARNING: Unable to save BVH to file %s!\n", bvh_filename);

		_freea(bvh_filename);

		return;
	}

	fwrite(reinterpret_cast<const char *>(&bvh.triangle_count), sizeof(int),      1,                  file);
	fwrite(reinterpret_cast<const char *>( bvh.triangles),      sizeof(Triangle), bvh.triangle_count, file);

	fwrite(reinterpret_cast<const char *>(&bvh.node_count), sizeof(int),     1,              file);
	fwrite(reinterpret_cast<const char *>( bvh.nodes),      sizeof(BVHNode), bvh.node_count, file);

	fwrite(reinterpret_cast<const char *>(&bvh.index_count), sizeof(int), 1,               file);
	fwrite(reinterpret_cast<const char *>( bvh.indices),     sizeof(int), bvh.index_count, file);

	fclose(file);

	_freea(bvh_filename);
}

static bool bvh_try_to_load_from_disk(BVH & bvh, const char * filename, const char * file_extension) {
	assert(file_extension[0] == '.');

	int    bvh_filename_length = strlen(filename) + strlen(file_extension) + 1;
	char * bvh_filename        = reinterpret_cast<char *>(_malloca(bvh_filename_length));

	strcpy_s(bvh_filename, bvh_filename_length, filename);
	strcat_s(bvh_filename, bvh_filename_length, file_extension);

	if (!Util::file_exists(bvh_filename) || !Util::file_is_newer(filename, bvh_filename)) {
		_freea(bvh_filename);

		return false;
	}

	FILE * file;
	fopen_s(&file, bvh_filename, "rb");

	if (!file) {
		_freea(bvh_filename);

		return false;
	}

	fread(reinterpret_cast<char *>(&bvh.triangle_count), sizeof(int), 1, file);

	bvh.triangles = new Triangle[bvh.triangle_count];
	fread(reinterpret_cast<char *>(bvh.triangles), sizeof(Triangle), bvh.triangle_count, file);
		
	fread(reinterpret_cast<char *>(&bvh.node_count), sizeof(int), 1, file);

	bvh.nodes = new BVHNode[bvh.node_count];
	fread(reinterpret_cast<char *>(bvh.nodes), sizeof(BVHNode), bvh.node_count, file);

	fread(reinterpret_cast<char *>(&bvh.index_count), sizeof(int), 1, file);
			
	bvh.indices = new int[bvh.index_count];
	fread(reinterpret_cast<char *>(bvh.indices), sizeof(int), bvh.index_count, file);

	fclose(file);

	printf("Loaded BVH  %s from disk\n", bvh_filename);

	_freea(bvh_filename);

	return true;
}

const Mesh * Mesh::load(const char * filename) {
	Mesh *& mesh = cache[filename];

	// If the cache already contains this Model Data simply return it
	if (mesh) return mesh;

#if BVH_TYPE == BVH_BVH
	const char * file_extension = ".bvh";
#else
	const char * file_extension = ".sbvh";
#endif

	mesh = new Mesh();

	bool bvh_loaded = bvh_try_to_load_from_disk(mesh->bvh, filename, file_extension);

	if (bvh_loaded) {
		// If the BVH loaded successfully we only need to load the Materials
		// of the Mesh, because the geometry is already included in the BVH

		// Replace ".obj" in the filename with ".mtl"
		int filename_length = strlen(filename);
		char * mtl_filename = reinterpret_cast<char *>(_malloca(filename_length + 1));
		strcpy_s(mtl_filename, filename_length + 1, filename);
		memcpy(mtl_filename + filename_length - 4, ".mtl", 4);

		OBJLoader::load_mtl(mtl_filename, mesh);

		_freea(mtl_filename);
	} else {
		OBJLoader::load_obj(filename, mesh);

#if BVH_TYPE == BVH_BVH
		{
			ScopeTimer timer("BVH Construction");

			BVHBuilders::build_bvh(mesh->bvh);
		}
#else
		{
			ScopeTimer timer("SBVH Construction");

			BVHBuilders::build_sbvh(mesh->bvh);
		}
#endif

		bvh_save_to_disk(mesh->bvh, filename, file_extension);
	}

	return mesh;
}
