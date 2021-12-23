#include "MitsubaLoader.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "Assets/BVHLoader.h"
#include "Assets/OBJLoader.h"
#include "Assets/PLYLoader.h"

#include "BVH/Builders/BVHBuilder.h"
#include "BVH/Builders/CWBVHBuilder.h"

#include "Pathtracer/Scene.h"
#include "Pathtracer/MeshData.h"

#include "Util/Util.h"
#include "Util/Array.h"
#include "Util/HashMap.h"
#include "Util/Parser.h"
#include "Util/Geometry.h"
#include "Util/StringView.h"

#include "XMLParser.h"
#include "MitshairLoader.h"
#include "SerializedLoader.h"

struct ShapeGroup {
	MeshDataHandle mesh_data_handle;
	MaterialHandle material_handle;
};

using ShapeGroupMap = HashMap<StringView, ShapeGroup,     StringViewHash>;
using SerializedMap = HashMap<StringView, Serialized,     StringViewHash>;
using MaterialMap   = HashMap<StringView, MaterialHandle, StringViewHash>;
using TextureMap    = HashMap<StringView, TextureHandle,  StringViewHash>;

static const char * get_absolute_filename(const char * path, int len_path, const char * filename, int len_filename) {
	char * filename_abs = new char[len_path + len_filename + 1];

	memcpy(filename_abs,            path,     len_path);
	memcpy(filename_abs + len_path, filename, len_filename);
	filename_abs[len_path + len_filename] = '\0';

	return filename_abs;
}

static void parse_texture(const XMLNode * node, TextureMap & texture_map, const char * path, Scene & scene, TextureHandle & texture) {
	StringView type = node->get_attribute_value("type");

	if (type == "bitmap") {
		const StringView & filename_rel = node->get_child_value<StringView>("filename");
		const char       * filename_abs = get_absolute_filename(path, strlen(path), filename_rel.start, filename_rel.length());

		texture = scene.asset_manager.add_texture(filename_abs);

		StringView texture_id = { };
		const XMLAttribute * id = node->get_attribute("id");
		if (id) {
			texture_id = id->value;
		} else {
			texture_id = filename_rel;
		}
		texture_map[texture_id] = texture;

		delete [] filename_abs;
		return;
	} else {
		WARNING(node->location, "Only bitmap textures are supported!\n");
	}
}

static void parse_rgb_or_texture(const XMLNode * node, const char * name, TextureMap & texture_map, const char * path, Scene & scene, Vector3 & rgb, TextureHandle & texture) {
	const XMLNode * reflectance = node->get_child_by_name(name);
	if (reflectance) {
		if (reflectance->tag == "rgb") {
			rgb = reflectance->get_attribute_optional("value", Vector3(1.0f));
		} else if (reflectance->tag == "srgb") {
			rgb = reflectance->get_attribute_optional("value", Vector3(1.0f));
			rgb.x = Math::gamma_to_linear(rgb.x);
			rgb.y = Math::gamma_to_linear(rgb.y);
			rgb.z = Math::gamma_to_linear(rgb.z);
		} else if (reflectance->tag == "texture") {
			parse_texture(reflectance, texture_map, path, scene, texture);

			const XMLNode * scale = reflectance->get_child_by_name("scale");
			if (scale) {
				rgb = scale->get_attribute_optional("value", Vector3(1.0f));
			}
		} else if (reflectance->tag == "ref") {
			const StringView & texture_name = reflectance->get_attribute_value<StringView>("id");
			bool found = texture_map.try_get(texture_name, texture);
			if (!found) {
				WARNING(reflectance->location, "Invalid texture ref '%.*s'!\n", unsigned(texture_name.length()), texture_name.start);
			}
		}
	} else {
		rgb = Vector3(1.0f);
	}
}

static void parse_transform(const XMLNode * node, Vector3 * position, Quaternion * rotation, float * scale, const Vector3 & forward = Vector3(0.0f, 0.0f, 1.0f)) {
	const XMLNode * transform = node->get_child_by_tag("transform");
	if (transform) {
		const XMLNode * matrix = transform->get_child_by_tag("matrix");
		if (matrix) {
			Matrix4 world = matrix->get_attribute_value<Matrix4>("value");
			Matrix4::decompose(world, position, rotation, scale, forward);
			return;
		}

		const XMLNode * lookat = transform->get_child_by_tag("lookat");
		if (lookat) {
			Vector3 origin = lookat->get_attribute_optional("origin", Vector3(0.0f, 0.0f,  0.0f));
			Vector3 target = lookat->get_attribute_optional("target", Vector3(0.0f, 0.0f, -1.0f));
			Vector3 up     = lookat->get_attribute_optional("up",     Vector3(0.0f, 1.0f,  0.0f));

			Vector3 forward = Vector3::normalize(target - origin);

			if (position) *position = origin;
			if (rotation) *rotation = Quaternion::look_rotation(forward, up);
		}

		const XMLNode * scale_node = transform->get_child_by_tag("scale");
		if (scale_node && scale) {
			const XMLAttribute * scale_value = scale_node->get_attribute("value");
			if (scale_value) {
				*scale = scale_value->get_value<float>();
			} else {
				float x = scale_node->get_attribute_optional("x", 1.0f);
				float y = scale_node->get_attribute_optional("y", 1.0f);
				float z = scale_node->get_attribute_optional("z", 1.0f);

				*scale = cbrtf(x * y * z);
			}
		}

		const XMLNode * rotate = transform->get_child_by_tag("rotate");
		if (rotate && rotation) {
			float x = rotate->get_attribute_optional("x", 0.0f);
			float y = rotate->get_attribute_optional("y", 0.0f);
			float z = rotate->get_attribute_optional("z", 0.0f);

			if (x == 0.0f && y == 0.0f && z == 0.0f) {
				WARNING(rotate->location, "WARNING: Rotation without axis specified!\n");
				*rotation = Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
			} else {
				float angle = rotate->get_attribute_optional("angle", 0.0f);
				*rotation = Quaternion::axis_angle(Vector3(x, y, z), Math::deg_to_rad(angle));
			}
		}

		const XMLNode * translate = transform->get_child_by_tag("translate");
		if (translate && position) {
			*position = Vector3(
				translate->get_attribute_optional("x", 0.0f),
				translate->get_attribute_optional("y", 0.0f),
				translate->get_attribute_optional("z", 0.0f)
			);
		}
	} else {
		if (position) *position = Vector3(0.0f);
		if (rotation) *rotation = Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
		if (scale)    *scale    = 1.0f;
	}
}

static Matrix4 parse_transform_matrix(const XMLNode * node) {
	const XMLNode * transform = node->get_child_by_tag("transform");
	if (transform) {
		const XMLNode * matrix = transform->get_child_by_tag("matrix");
		if (matrix) {
			return matrix->get_attribute_value<Matrix4>("value");
		}
	}

	Vector3    translation;
	Quaternion rotation;
	float      scale = 1.0f;
	parse_transform(node, &translation, &rotation, &scale);

	return Matrix4::create_translation(translation) * Matrix4::create_rotation(rotation) * Matrix4::create_scale(scale);
}

static MaterialHandle parse_material(const XMLNode * node, Scene & scene, const MaterialMap & material_map, TextureMap & texture_map, const char * path) {
	Material material = { };

	const XMLNode * bsdf;

	if (node->tag != "bsdf") {
		// Check if there is an emitter defined
		const XMLNode * emitter = node->get_child_by_tag("emitter");
		if (emitter) {
			material.type = Material::Type::LIGHT;
			material.name = "emitter";
			material.emission = emitter->get_child_value<Vector3>("radiance");

			return scene.asset_manager.add_material(material);
		}

		// Check if an existing Material is referenced
		const XMLNode * ref = node->get_child_by_tag("ref");
		if (ref) {
			const StringView & material_name = ref->get_attribute_value<StringView>("id");

			MaterialHandle material_id;
			bool found = material_map.try_get(material_name, material_id);
			if (!found) {
				WARNING(ref->location, "Invalid material Ref '%.*s'!\n", unsigned(material_name.length()), material_name.start);

				return MaterialHandle::get_default();
			}

			return material_id;
		}

		// Otherwise, parse BSDF
		bsdf = node->get_child_by_tag("bsdf");
		if (bsdf == nullptr) {
			WARNING(node->location, "Unable to parse BSDF!\n");
			return MaterialHandle::get_default();
		}
	} else {
		bsdf = node;
	}

	const XMLAttribute * name = bsdf->get_attribute("id");

	const XMLNode * inner_bsdf = bsdf;
	StringView inner_bsdf_type = inner_bsdf->get_attribute_value<StringView>("type");

	// Keep peeling back nested BSDFs, we only care about the innermost one
	while (
		inner_bsdf_type == "twosided" ||
		inner_bsdf_type == "mask" ||
		inner_bsdf_type == "bumpmap" ||
		inner_bsdf_type == "coating"
	) {
		const XMLNode * inner_bsdf_child = inner_bsdf->get_child_by_tag("bsdf");
		if (inner_bsdf_child) {
			inner_bsdf = inner_bsdf_child;
		} else {
			const XMLNode * ref = inner_bsdf->get_child_by_tag("ref");
			if (ref) {
				StringView id = ref->get_attribute_value<StringView>("id");

				MaterialHandle material_handle;
				if (material_map.try_get(id, material_handle)) {
					return material_handle;
				} else {
					WARNING(ref->location, "Invalid material Ref '%.*s'!\n", unsigned(id.length()), id.start);
					return MaterialHandle::get_default();
				}
			} else {
				return MaterialHandle::get_default();
			}
		}

		inner_bsdf_type = inner_bsdf->get_attribute_value<StringView>("type");

		if (name == nullptr) {
			name = inner_bsdf->get_attribute("id");
		}
	}
	if (name) {
		material.name = name->value.c_str();
	} else {
		material.name = "Material";
	}

	if (inner_bsdf_type == "diffuse") {
		material.type = Material::Type::DIFFUSE;

		parse_rgb_or_texture(inner_bsdf, "reflectance", texture_map, path, scene, material.diffuse, material.texture_id);
	} else if (inner_bsdf_type == "conductor" || inner_bsdf_type == "roughconductor") {
		material.type = Material::Type::CONDUCTOR;

		if (inner_bsdf_type == "conductor") {
			material.linear_roughness = 0.0f;
		} else {
			material.linear_roughness = sqrtf(inner_bsdf->get_child_value_optional("alpha", 0.25f));
		}

		const XMLNode * material_str = inner_bsdf->get_child_by_name("material");
		if (material_str && material_str->get_attribute_value<StringView>("value") == "none") {
			material.eta = Vector3(0.0f);
			material.k   = Vector3(1.0f);
		} else {
			material.eta = inner_bsdf->get_child_value_optional("eta", Vector3(1.33f));
			material.k   = inner_bsdf->get_child_value_optional("k",   Vector3(1.0f));
		}
	} else if (inner_bsdf_type == "plastic" || inner_bsdf_type == "roughplastic" || inner_bsdf_type == "roughdiffuse") {
		material.type = Material::Type::PLASTIC;

		parse_rgb_or_texture(inner_bsdf, "diffuseReflectance", texture_map, path, scene, material.diffuse, material.texture_id);

		if (inner_bsdf_type == "plastic") {
			material.linear_roughness = 0.0f;
		} else {
			material.linear_roughness = sqrtf(inner_bsdf->get_child_value_optional("alpha", 0.25f));
		}
	} else if (inner_bsdf_type == "phong") {
		material.type = Material::Type::PLASTIC;

		parse_rgb_or_texture(inner_bsdf, "diffuseReflectance", texture_map, path, scene, material.diffuse, material.texture_id);

		float exponent = inner_bsdf->get_child_value_optional("exponent", 1.0f);
		material.linear_roughness = powf(0.5f * exponent + 1.0f, 0.25f);

	} else if (inner_bsdf_type == "thindielectric" || inner_bsdf_type == "dielectric" || inner_bsdf_type == "roughdielectric") {
		float int_ior = 0.0f;
		float ext_ior = 0.0f;

		auto lookup_known_ior = [](StringView name, float & ior) {
			// Based on: https://www.mitsuba-renderer.org/releases/0.5.0/documentation.pdf (page 58)
			struct IOR {
				const char * name;
				float        ior;
			};
			static IOR known_iors[] = {
				{ "vacuum",               1.0f },
				{ "helium",               1.00004f },
				{ "hydrogen",             1.00013f },
				{ "air",                  1.00028f },
				{ "carbon dioxide",       1.00045f },
				{ "water",                1.3330f },
				{ "acetone",              1.36f },
				{ "ethanol",              1.361f },
				{ "carbon tetrachloride", 1.461f },
				{ "glycerol",             1.4729f },
				{ "benzene",              1.501f },
				{ "silicone oil",         1.52045f },
				{ "bromine",              1.661f },
				{ "water ice",            1.31f },
				{ "fused quartz",         1.458f },
				{ "pyrex",                1.470f },
				{ "acrylic glass",        1.49f },
				{ "polypropylene",        1.49f },
				{ "bk7",                  1.5046f },
				{ "sodium chloride",      1.544f },
				{ "amber",                1.55f },
				{ "pet",                  1.575f },
				{ "diamond",              2.419f }
			};

			for (int i = 0; i < Util::array_count(known_iors); i++) {
				if (name == known_iors[i].name) {
					ior = known_iors[i].ior;
					return true;
				}
			}

			return false;
		};

		const XMLNode * child_int_ior = inner_bsdf->get_child_by_name("intIOR");
		if (child_int_ior && child_int_ior->tag == "string") {
			StringView int_ior_name = child_int_ior->get_attribute_value("value");
			if (!lookup_known_ior(int_ior_name, int_ior)) {
				ERROR(child_int_ior->location, "Index of refraction not known for '%.*s'\n", unsigned(int_ior_name.length()), int_ior_name.start);
			}
		} else {
			int_ior = inner_bsdf->get_child_value_optional("intIOR", 1.33f);
		}

		const XMLNode * child_ext_ior = inner_bsdf->get_child_by_name("extIOR");
		if (child_ext_ior && child_ext_ior->tag == "string") {
			StringView ext_ior_name = child_ext_ior->get_attribute_value("value");
			if (!lookup_known_ior(ext_ior_name, ext_ior)) {
				ERROR(child_ext_ior->location, "Index of refraction not known for '%.*s'\n", unsigned(ext_ior_name.length()), ext_ior_name.start);
			}
		} else {
			ext_ior = inner_bsdf->get_child_value_optional("extIOR", 1.0f);
		}

		material.type = Material::Type::DIELECTRIC;
		material.index_of_refraction = ext_ior == 0.0f ? int_ior : int_ior / ext_ior;

		if (inner_bsdf_type == "roughdielectric") {
			material.linear_roughness = sqrtf(inner_bsdf->get_child_value_optional("alpha", 0.25f));
		} else {
			material.linear_roughness = 0.0f;
		}

		const XMLNode * xml_medium = node->get_child_by_tag("medium");
		if (xml_medium) {
			StringView medium_type = xml_medium->get_attribute_value("type");

			if (medium_type == "homogeneous") {
				Medium medium = { };

				if (const XMLAttribute * name = xml_medium->get_attribute("name")) {
					medium.name = name->value.c_str();
				}

				const XMLNode * xml_sigma_a = xml_medium->get_child_by_name("sigmaA");
				const XMLNode * xml_sigma_s = xml_medium->get_child_by_name("sigmaS");
				const XMLNode * xml_sigma_t = xml_medium->get_child_by_name("sigmaT");
				const XMLNode * xml_albedo  = xml_medium->get_child_by_name("albedo"); // Single scatter albedo

				Vector3 sigma_a = { };
				Vector3 sigma_s = { };

				if (!((xml_sigma_a && xml_sigma_s) ^ (xml_sigma_t && xml_albedo))) {
					WARNING(xml_medium->location, "WARNING: Incorrect configuration of Medium properties\nPlease provide EITHER sigmaA and sigmaS OR sigmaT and albedo\n");
				} else if (xml_sigma_a && xml_sigma_s) {
					sigma_a = xml_sigma_a->get_attribute_value<Vector3>("value");
					sigma_s = xml_sigma_s->get_attribute_value<Vector3>("value");
				} else {
					Vector3 sigma_t = xml_sigma_t->get_attribute_value<Vector3>("value");
					Vector3 albedo  = xml_albedo ->get_attribute_value<Vector3>("value");

					sigma_s = albedo  * sigma_t;
					sigma_a = sigma_t - sigma_s;
				}

				float scale = xml_medium->get_child_value_optional("scale", 1.0f);

				medium.set_A_and_d(scale * sigma_a, scale * sigma_s);

				if (const XMLNode * phase = xml_medium->get_child_by_tag("phase")) {
					StringView phase_type = phase->get_attribute_value("type");

					if (phase_type == "isotropic") {
						medium.g = 0.0f;
					} else if (phase_type == "hg") {
						medium.g = phase->get_child_value_optional("g", 0.0f);
					} else {
						WARNING(xml_medium->location, "WARNING: Phase function type '%.*s' not supported!\n", unsigned(phase_type.length()), phase_type.start);
					}
				}

				material.medium_handle = scene.asset_manager.add_medium(medium);
			} else {
				WARNING(xml_medium->location, "WARNING: Medium type '%.*s' not supported!\n", unsigned(medium_type.length()), medium_type.start);
			}
		}
	} else if (inner_bsdf_type == "difftrans") {
		material.type = Material::Type::DIFFUSE;

		parse_rgb_or_texture(inner_bsdf, "transmittance", texture_map, path, scene, material.diffuse, material.texture_id);
	} else {
		WARNING(inner_bsdf->location, "WARNING: BSDF type '%.*s' not supported!\n", unsigned(inner_bsdf_type.length()), inner_bsdf_type.start);

		return MaterialHandle::get_default();
	}

	return scene.asset_manager.add_material(material);
}

static MeshDataHandle parse_shape(const XMLNode * node, Scene & scene, SerializedMap & serialized_map, const char * path, const char *& name) {
	StringView type = node->get_attribute_value<StringView>("type");

	if (type == "obj" || type == "ply") {
		const StringView & filename_rel = node->get_child_value<StringView>("filename");
		const char       * filename_abs = get_absolute_filename(path, strlen(path), filename_rel.start, filename_rel.length());

		MeshDataHandle mesh_data_handle;
		if (type == "obj") {
			mesh_data_handle = scene.asset_manager.add_mesh_data(filename_abs, OBJLoader::load);
		} else {
			mesh_data_handle = scene.asset_manager.add_mesh_data(filename_abs, PLYLoader::load);
		}
		delete [] filename_abs;

		name = filename_rel.c_str();

		return mesh_data_handle;
	} else if (type == "rectangle" || type == "cube" || type == "disk" || type == "cylinder" || type == "sphere") {
		Matrix4 transform = parse_transform_matrix(node);

		Triangle * triangles = nullptr;
		int        triangle_count = 0;

		if (type == "rectangle") {
			Geometry::rectangle(triangles, triangle_count, transform);
		} else if (type == "cube") {
			Geometry::cube(triangles, triangle_count, transform);
		} else if (type == "disk") {
			Geometry::disk(triangles, triangle_count, transform);
		} else if (type == "cylinder") {
			Vector3 p0     = node->get_child_value_optional("p0", Vector3(0.0f, 0.0f, 0.0f));
			Vector3 p1     = node->get_child_value_optional("p1", Vector3(0.0f, 0.0f, 1.0f));
			float   radius = node->get_child_value_optional("radius", 1.0f);

			Geometry::cylinder(triangles, triangle_count, transform, p0, p1, radius);
		} else if (type == "sphere") {
			float   radius = node->get_child_value_optional("radius", 1.0f);
			Vector3 center = Vector3(0.0f);

			const XMLNode * xml_center = node->get_child_by_name("center");
			if (xml_center) {
				center = Vector3(
					xml_center->get_attribute_optional("x", 0.0f),
					xml_center->get_attribute_optional("y", 0.0f),
					xml_center->get_attribute_optional("z", 0.0f)
				);
			}

			transform = transform * Matrix4::create_translation(center) * Matrix4::create_scale(radius);

			Geometry::sphere(triangles, triangle_count, transform);
		} else {
			abort(); // Unreachable
		}

		name = type.c_str();

		return scene.asset_manager.add_mesh_data(triangles, triangle_count);
	} else if (type == "serialized") {
		const StringView & filename_rel = node->get_child_value<StringView>("filename");
		const char       * filename_abs = get_absolute_filename(path, strlen(path), filename_rel.start, filename_rel.length());

		int shape_index = node->get_child_value_optional("shapeIndex", 0);

		char bvh_filename[512] = { };
		sprintf_s(bvh_filename, "%s.shape_%i.bvh", filename_abs, shape_index);

		auto fallback_loader = [&](const char * filename, Triangle *& triangles, int & triangle_count) {
			Serialized serialized;
			bool found = serialized_map.try_get(filename_rel, serialized);
			if (!found) {
				serialized = SerializedLoader::load(filename_abs, node->location);
				serialized_map[filename_rel] = serialized;
			}

			triangles      = serialized.triangles     [shape_index];
			triangle_count = serialized.triangle_count[shape_index];
		};
		MeshDataHandle mesh_data_handle = scene.asset_manager.add_mesh_data(bvh_filename, bvh_filename, fallback_loader);

		char * shape_name = new char[filename_rel.length() + 32];
		sprintf_s(shape_name, filename_rel.length() + 32, "%.*s_%i", unsigned(filename_rel.length()), filename_rel.start, shape_index);
		name = shape_name;

		delete [] filename_abs;
		return mesh_data_handle;
	} else if (type == "hair") {
		const StringView & filename_rel = node->get_child_value<StringView>("filename");
		const char       * filename_abs = get_absolute_filename(path, strlen(path), filename_rel.start, filename_rel.length());

		float radius = node->get_child_value_optional("radius", 0.0025f);

		auto fallback_loader = [&](const char * filename, Triangle *& triangles, int & triangle_count) {
			MitshairLoader::load(filename, node->location, triangles, triangle_count, radius);
		};
		MeshDataHandle mesh_data_handle = scene.asset_manager.add_mesh_data(filename_abs, fallback_loader);

		name = filename_rel.c_str();

		delete [] filename_abs;
		return mesh_data_handle;
	} else {
		WARNING(node->location, "WARNING: Shape type '%.*s' not supported!\n", unsigned(type.length()), type.start);
		return MeshDataHandle { INVALID };
	}
}

static void walk_xml_tree(const XMLNode * node, Scene & scene, ShapeGroupMap & shape_group_map, SerializedMap & serialized_map, MaterialMap & material_map, TextureMap & texture_map, const char * path) {
	if (node->tag == "bsdf") {
		MaterialHandle   material_handle = parse_material(node, scene, material_map, texture_map, path);
		const Material & material = scene.asset_manager.get_material(material_handle);

		StringView str = { material.name, material.name + strlen(material.name) };
		material_map[str] = material_handle;
	} else if (node->tag == "texture") {
		TextureHandle texture;
		parse_texture(node, texture_map, path, scene, texture);
	} else if (node->tag == "shape") {
		StringView type = node->get_attribute_value<StringView>("type");
		if (type == "shapegroup") {
			if (node->children.size() > 0) {
				const XMLNode * shape = node->get_child_by_tag("shape");
				if (!shape) {
					ERROR(node->location, "Shapegroup needs a <shape> child!\n");
				}

				const char * name = nullptr;
				MeshDataHandle mesh_data_handle = parse_shape(shape, scene, serialized_map, path, name);
				MaterialHandle material_handle  = parse_material(shape, scene, material_map, texture_map, path);

				const StringView & id = node->get_attribute_value<StringView>("id");
				shape_group_map[id] = { mesh_data_handle, material_handle };
			}
		} else if (type == "instance") {
			const XMLNode * ref = node->get_child_by_tag("ref");
			if (!ref) {
				WARNING(node->location, "Instance without ref!\n");
				return;
			}
			StringView id = ref->get_attribute_value<StringView>("id");

			ShapeGroup shape_group;
			if (shape_group_map.try_get(id, shape_group) && shape_group.mesh_data_handle.handle != INVALID) {
				Mesh & mesh = scene.add_mesh(id.c_str(), shape_group.mesh_data_handle, shape_group.material_handle);
				parse_transform(node, &mesh.position, &mesh.rotation, &mesh.scale);
			}
		} else {
			char const * name = nullptr;
			MeshDataHandle mesh_data_handle = parse_shape(node, scene, serialized_map, path, name);
			MaterialHandle material_handle  = parse_material(node, scene, material_map, texture_map, path);

			if (mesh_data_handle.handle != INVALID) {
				Mesh & mesh = scene.add_mesh(name, mesh_data_handle, material_handle);

				// Do not apply transform to primitive shapes, since they have the transform baked into their vertices
				bool type_is_primitive = type == "rectangle" || type == "cube" || type == "disk" || type == "cylinder" || type == "sphere";
				if (!type_is_primitive) {
					parse_transform(node, &mesh.position, &mesh.rotation, &mesh.scale);
				}
			}
		}
	} else if (node->tag == "sensor") {
		const StringView & camera_type = node->get_attribute_value<StringView>("type");

		if (camera_type == "perspective" || camera_type == "perspective_rdist" || camera_type == "thinlens") {
			float fov = node->get_child_value_optional("fov", 110.0f);
			scene.camera.set_fov(Math::deg_to_rad(fov));

			if (camera_type == "perspective") {
				scene.camera.aperture_radius = 0.0f;
			} else {
				scene.camera.aperture_radius = node->get_child_value_optional("apertureRadius", 0.05f);
				scene.camera.focal_distance  = node->get_child_value_optional("focusDistance", 10.0f);
			}

			float scale = 1.0f;
			parse_transform(node, &scene.camera.position, &scene.camera.rotation, &scale, Vector3(0.0f, 0.0f, -1.0f));

			if (scale < 0.0f) {
//				scene.camera.rotation = Quaternion::conjugate(scene.camera.rotation);
			}
		} else {
			WARNING(node->location, "WARNING: Camera type '%.*s' not supported!\n", unsigned(camera_type.length()), camera_type.start);
		}
	} else if (node->tag == "emitter") {
		const StringView & emitter_type = node->get_attribute_value<StringView>("type");

		if (emitter_type == "area") {
			WARNING(node->location, "Area emitter defined without geometry!\n");
		} else if (emitter_type == "envmap") {
			const char * filename_rel = node->get_child_value<StringView>("filename").c_str();

			const char * extension = Util::find_last(filename_rel, ".");
			if (!extension) {
				WARNING(node->location, "Environment Map '%s' has no file extension!\n", filename_rel);
			} else if (strcmp(extension, "hdr") != 0) {
				WARNING(node->location, "Environment Map '%s' has unsupported file extension. Only HDR Environment Maps are supported!\n", filename_rel);
			} else {
				scene_config.sky = get_absolute_filename(path, strlen(path), filename_rel, strlen(filename_rel));
			}

			delete [] filename_rel;
		} else if (emitter_type == "point") {
			Material material = { };
			material.type = Material::Type::LIGHT;
			material.emission = node->get_child_value_optional<Vector3>("intensity", Vector3(1.0f));

			MaterialHandle material_handle = scene.asset_manager.add_material(material);

			// Make small area light
			constexpr float RADIUS = 0.0001f;
			Matrix4 transform = parse_transform_matrix(node) * Matrix4::create_scale(RADIUS);

			Triangle * triangles;
			int        triangle_count;
			Geometry::sphere(triangles, triangle_count, transform, 0);

			MeshDataHandle mesh_data_handle = scene.asset_manager.add_mesh_data(triangles, triangle_count);
			scene.add_mesh("PointLight", mesh_data_handle, material_handle);
		} else {
			WARNING(node->location, "Emitter type '%.*s' is not supported!\n", unsigned(emitter_type.length()), emitter_type.start);
		}
	} else if (node->tag == "include") {
		const StringView & filename_rel = node->get_attribute_value<StringView>("filename");
		const char       * filename_abs = get_absolute_filename(path, strlen(path), filename_rel.start, filename_rel.length());

		MitsubaLoader::load(filename_abs, scene);

		delete [] filename_abs;
	} else for (int i = 0; i < node->children.size(); i++) {
		walk_xml_tree(&node->children[i], scene, shape_group_map, serialized_map, material_map, texture_map, path);
	}
}

void MitsubaLoader::load(const char * filename, Scene & scene) {
	XMLParser xml_parser = { };
	xml_parser.init(filename);

	XMLNode root = xml_parser.parse_root();

	ShapeGroupMap shape_group_map;
	SerializedMap serialized_map;
	MaterialMap   material_map;
	TextureMap    texture_map;
	char path[512];	Util::get_path(filename, path);
	walk_xml_tree(&root, scene, shape_group_map, serialized_map, material_map, texture_map, path);

	xml_parser.free();
}
