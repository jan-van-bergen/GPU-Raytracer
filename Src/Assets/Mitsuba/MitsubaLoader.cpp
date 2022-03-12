#include "MitsubaLoader.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "Core/Array.h"
#include "Core/HashMap.h"
#include "Core/Format.h"
#include "Core/Parser.h"
#include "Core/StringView.h"

#include "Assets/BVHLoader.h"
#include "Assets/OBJLoader.h"
#include "Assets/PLYLoader.h"

#include "Renderer/Scene.h"
#include "Renderer/MeshData.h"

#include "Util/Util.h"
#include "Util/Geometry.h"

#include "XMLParser.h"
#include "MitshairLoader.h"
#include "SerializedLoader.h"

struct ShapeGroup {
	Handle<MeshData> mesh_data_handle;
	Handle<Material> material_handle;
};

using ShapeGroupMap = HashMap<String, ShapeGroup>;
using MaterialMap   = HashMap<String, Handle<Material>>;
using TextureMap    = HashMap<String, Handle<Texture>>;

static Handle<Texture> parse_texture(const XMLNode * node, TextureMap & texture_map, StringView path, Scene & scene, Vector3 * rgb) {
	StringView type = node->get_attribute_value("type");

	if (type == "scale") {
		if (const XMLNode * scale = node->get_child_by_name("scale")) {
			if (scale->tag == "float") {
				*rgb *= scale->get_attribute_value<float>("value");
			} else if (scale->tag == "rgb") {
				*rgb *= scale->get_attribute_value<Vector3>("value");
			} else {
				WARNING(scale->location, "Invalid scale tag <{}>!\n", scale->tag);
			}
		}

		node = node->get_child_by_tag("texture");
		type = node->get_attribute_value("type");
	}

	if (type == "bitmap") {
		StringView filename_rel = node->get_child_value<StringView>("filename");
		String     filename_abs = Util::combine_stringviews(path, filename_rel, scene.allocator);

		String        texture_name   = String(Util::remove_directory(filename_abs.view()), scene.allocator);
		Handle<Texture> texture_handle = scene.asset_manager.add_texture(std::move(filename_abs), std::move(texture_name));

		if (const XMLAttribute * id = node->get_attribute("id")) {
			texture_map.insert(id->value, texture_handle);
		}

		return texture_handle;
	} else {
		WARNING(node->location, "Only bitmap textures are supported!\n");
	}

	return Handle<Texture> { INVALID };
}

static void parse_rgb_or_texture(const XMLNode * node, const char * name, TextureMap & texture_map, StringView path, Scene & scene, Vector3 * rgb, Handle<Texture> * texture_handle) {
	const XMLNode * colour = node->get_child_by_name(name);
	if (colour) {
		if (colour->tag == "rgb") {
			*rgb = colour->get_attribute_optional("value", Vector3(1.0f));
		} else if (colour->tag == "srgb") {
			*rgb = colour->get_attribute_optional("value", Vector3(1.0f));
			rgb->x = Math::gamma_to_linear(rgb->x);
			rgb->y = Math::gamma_to_linear(rgb->y);
			rgb->z = Math::gamma_to_linear(rgb->z);
		} else if (colour->tag == "texture") {
			*texture_handle = parse_texture(colour, texture_map, path, scene, rgb);

			const XMLNode * scale = colour->get_child_by_name("scale");
			if (scale) {
				*rgb = scale->get_attribute_optional("value", Vector3(1.0f));
			}
		} else if (colour->tag == "ref") {
			StringView texture_name = colour->get_attribute_value<StringView>("id");
			Handle<Texture> * ref_handle = texture_map.try_get(texture_name);
			if (ref_handle) {
				*texture_handle = *ref_handle;
			} else {
				WARNING(colour->location, "Invalid texture ref '{}'!\n", texture_name);
			}
		}
	} else {
		*rgb = Vector3(1.0f);
	}
}

static Matrix4 parse_transform_matrix(const XMLNode * node) {
	Matrix4 world = { };

	const XMLNode * transform = node->get_child_by_tag("transform");
	if (!transform) {
		return world;
	}

	for (size_t i = 0; i < transform->children.size(); i++) {
		const XMLNode & transformation = transform->children[i];

		if (transformation.tag == "matrix") {
			world = transformation.get_attribute_value<Matrix4>("value") * world;
		} else if (transformation.tag == "lookat") {
			Vector3 origin = transformation.get_attribute_optional("origin", Vector3(0.0f, 0.0f,  0.0f));
			Vector3 target = transformation.get_attribute_optional("target", Vector3(0.0f, 0.0f, -1.0f));
			Vector3 up     = transformation.get_attribute_optional("up",     Vector3(0.0f, 1.0f,  0.0f));

			world = Matrix4::create_translation(origin) * Matrix4::create_rotation(Quaternion::look_rotation(target - origin, up)) * world;
		} else if (transformation.tag == "scale") {
			const XMLAttribute * scale = transformation.get_attribute("value");
			if (scale) {
				world = Matrix4::create_scale(scale->get_value<float>()) * world;
			} else {
				float x = transformation.get_attribute_optional("x", 1.0f);
				float y = transformation.get_attribute_optional("y", 1.0f);
				float z = transformation.get_attribute_optional("z", 1.0f);

				world = Matrix4::create_scale(x, y, z) * world;
			}
		} else if (transformation.tag == "rotate") {
			float x = transformation.get_attribute_optional("x", 0.0f);
			float y = transformation.get_attribute_optional("y", 0.0f);
			float z = transformation.get_attribute_optional("z", 0.0f);

			if (x == 0.0f && y == 0.0f && z == 0.0f) {
				WARNING(transformation.location, "WARNING: Rotation without axis specified!\n");
			} else {
				float angle = transformation.get_attribute_optional("angle", 0.0f);
				world = Matrix4::create_rotation(Quaternion::axis_angle(Vector3(x, y, z), Math::deg_to_rad(angle))) * world;
			}
		} else if (transformation.tag == "translate") {
			world = Matrix4::create_translation(Vector3(
				transformation.get_attribute_optional("x", 0.0f),
				transformation.get_attribute_optional("y", 0.0f),
				transformation.get_attribute_optional("z", 0.0f)
			)) * world;
		} else {
			WARNING(transformation.location, "Node <{}> is not a valid transformation!\n", transformation.tag);
		}
	}

	return world;
}

static void parse_transform(const XMLNode * node, Vector3 * position, Quaternion * rotation, float * scale, const Vector3 & forward = Vector3(0.0f, 0.0f, 1.0f)) {
	Matrix4 world = parse_transform_matrix(node);
	Matrix4::decompose(world, position, rotation, scale, forward);
}

static Handle<Material> parse_material(const XMLNode * node, Scene & scene, const MaterialMap & material_map, TextureMap & texture_map, StringView path) {
	Material material = { };

	const XMLNode * bsdf;

	if (node->tag != "bsdf") {
		// Check if there is an emitter defined
		const XMLNode * emitter = node->get_child_by_tag("emitter");
		if (emitter) {
			material.type = Material::Type::LIGHT;
			material.name = "emitter";
			material.emission = emitter->get_child_value<Vector3>("radiance");

			return scene.asset_manager.add_material(std::move(material));
		}

		// Check if an existing Material is referenced
		const XMLNode * ref = node->get_child_by_tag("ref");
		if (ref) {
			StringView material_name = ref->get_attribute_value<StringView>("id");

			if (Handle<Material> * material_id = material_map.try_get(material_name)) {
				return *material_id;
			} else {
				WARNING(ref->location, "Invalid material Ref '{}'!\n", material_name);
				return Handle<Material>::get_default();
			}
		}

		// Otherwise, parse BSDF
		bsdf = node->get_child_by_tag("bsdf");
		if (bsdf == nullptr) {
			WARNING(node->location, "Unable to parse BSDF!\n");
			return Handle<Material>::get_default();
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

				if (Handle<Material> * material_handle = material_map.try_get(id)) {
					return *material_handle;
				} else {
					WARNING(ref->location, "Invalid material Ref '{}'!\n", id);
					return Handle<Material>::get_default();
				}
			} else {
				return Handle<Material>::get_default();
			}
		}

		inner_bsdf_type = inner_bsdf->get_attribute_value<StringView>("type");

		if (name == nullptr) {
			name = inner_bsdf->get_attribute("id");
		}
	}
	if (name) {
		material.name = String(name->value, scene.allocator);
	} else {
		material.name = "Material";
	}

	if (inner_bsdf_type == "diffuse") {
		material.type = Material::Type::DIFFUSE;

		parse_rgb_or_texture(inner_bsdf, "reflectance", texture_map, path, scene, &material.diffuse, &material.texture_handle);
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

		parse_rgb_or_texture(inner_bsdf, "diffuseReflectance", texture_map, path, scene, &material.diffuse, &material.texture_handle);

		if (inner_bsdf_type == "plastic") {
			material.linear_roughness = 0.0f;
		} else {
			material.linear_roughness = sqrtf(inner_bsdf->get_child_value_optional("alpha", 0.25f));
		}
	} else if (inner_bsdf_type == "phong") {
		material.type = Material::Type::PLASTIC;

		parse_rgb_or_texture(inner_bsdf, "diffuseReflectance", texture_map, path, scene, &material.diffuse, &material.texture_handle);

		float exponent = inner_bsdf->get_child_value_optional("exponent", 1.0f);
		material.linear_roughness = powf(0.5f * exponent + 1.0f, 0.25f);

	} else if (inner_bsdf_type == "thindielectric" || inner_bsdf_type == "dielectric" || inner_bsdf_type == "roughdielectric") {
		float int_ior = 0.0f;
		float ext_ior = 0.0f;

		auto lookup_known_ior = [](StringView name, float * ior) {
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
					*ior = known_iors[i].ior;
					return true;
				}
			}

			return false;
		};

		const XMLNode * child_int_ior = inner_bsdf->get_child_by_name("intIOR");
		if (child_int_ior && child_int_ior->tag == "string") {
			StringView int_ior_name = child_int_ior->get_attribute_value("value");
			if (!lookup_known_ior(int_ior_name, &int_ior)) {
				ERROR(child_int_ior->location, "Index of refraction not known for '{}'\n", int_ior_name);
			}
		} else {
			int_ior = inner_bsdf->get_child_value_optional("intIOR", 1.33f);
		}

		const XMLNode * child_ext_ior = inner_bsdf->get_child_by_name("extIOR");
		if (child_ext_ior && child_ext_ior->tag == "string") {
			StringView ext_ior_name = child_ext_ior->get_attribute_value("value");
			if (!lookup_known_ior(ext_ior_name, &ext_ior)) {
				ERROR(child_ext_ior->location, "Index of refraction not known for '{}'\n", ext_ior_name);
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
	} else if (inner_bsdf_type == "difftrans") {
		material.type = Material::Type::DIFFUSE;

		parse_rgb_or_texture(inner_bsdf, "transmittance", texture_map, path, scene, &material.diffuse, &material.texture_handle);
	} else {
		WARNING(inner_bsdf->location, "WARNING: BSDF type '{}' not supported!\n", inner_bsdf_type);

		return Handle<Material>::get_default();
	}

	return scene.asset_manager.add_material(std::move(material));
}

static Handle<Medium> parse_medium(const XMLNode * node, Scene & scene) {
	const XMLNode * xml_medium = node->get_child_by_tag("medium");
	if (!xml_medium) {
		return Handle<Medium> { INVALID };
	}

	StringView medium_type = xml_medium->get_attribute_value("type");

	if (medium_type == "homogeneous") {
		Medium medium = { };

		if (const XMLAttribute * name = xml_medium->get_attribute("name")) {
			medium.name = String(name->value, scene.allocator);
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
				WARNING(xml_medium->location, "WARNING: Phase function type '{}' not supported!\n", phase_type);
			}
		}

		return scene.asset_manager.add_medium(medium);
	} else {
		WARNING(xml_medium->location, "WARNING: Medium type '{}' not supported!\n", medium_type);
	}

	return Handle<Medium> { INVALID };
}

static Handle<MeshData> parse_shape(const XMLNode * node, Allocator * allocator, Scene & scene, StringView path, String * name) {
	StringView type = node->get_attribute_value<StringView>("type");

	if (type == "obj" || type == "ply") {
		String filename = Util::combine_stringviews(path, node->get_child_value<StringView>("filename"), scene.allocator);

		Handle<MeshData> mesh_data_handle;
		if (type == "obj") {
			mesh_data_handle = scene.asset_manager.add_mesh_data(filename, OBJLoader::load);
		} else {
			mesh_data_handle = scene.asset_manager.add_mesh_data(filename, PLYLoader::load);
		}

		*name = String(Util::remove_directory(filename.view()), scene.allocator);

		return mesh_data_handle;
	} else if (type == "rectangle" || type == "cube" || type == "disk" || type == "cylinder" || type == "sphere") {
		Matrix4 transform = parse_transform_matrix(node);

		Array<Triangle> triangles;

		if (type == "rectangle") {
			triangles = Geometry::rectangle(transform);
		} else if (type == "cube") {
			triangles = Geometry::cube(transform);
		} else if (type == "disk") {
			triangles = Geometry::disk(transform);
		} else if (type == "cylinder") {
			Vector3 p0     = node->get_child_value_optional("p0", Vector3(0.0f, 0.0f, 0.0f));
			Vector3 p1     = node->get_child_value_optional("p1", Vector3(0.0f, 0.0f, 1.0f));
			float   radius = node->get_child_value_optional("radius", 1.0f);

			triangles = Geometry::cylinder(transform, p0, p1, radius);
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

			triangles = Geometry::sphere(transform);
		} else {
			ASSERT_UNREACHABLE();
		}

		*name = String(type, scene.allocator);

		return scene.asset_manager.add_mesh_data(std::move(triangles));
	} else if (type == "serialized") {
		StringView filename_rel = node->get_child_value<StringView>("filename");
		String     filename_abs = Util::combine_stringviews(path, filename_rel);

		int shape_index = node->get_child_value_optional("shapeIndex", 0);

		*name = Format(scene.allocator).format("{}_{}"_sv, filename_rel, shape_index);

		String bvh_filename = Format().format("{}.shape_{}.bvh"_sv, filename_abs, shape_index);

		auto fallback_loader = [filename_abs = std::move(filename_abs), location = node->location, shape_index](const String & filename, Allocator * allocator) {
			return SerializedLoader::load(filename_abs, allocator, location, shape_index);
		};
		return scene.asset_manager.add_mesh_data(bvh_filename, bvh_filename, fallback_loader);
	} else if (type == "hair") {
		StringView filename_rel = node->get_child_value<StringView>("filename");
		String     filename_abs = Util::combine_stringviews(path, filename_rel, scene.allocator);

		*name = String(filename_rel, scene.allocator);

		float radius = node->get_child_value_optional("radius", 0.0025f);

		auto fallback_loader = [location = node->location, radius](const String & filename, Allocator * allocator) {
			return MitshairLoader::load(filename, allocator, location, radius);
		};
		return scene.asset_manager.add_mesh_data(filename_abs, fallback_loader);
	} else {
		WARNING(node->location, "WARNING: Shape type '{}' not supported!\n", type);
		return Handle<MeshData> { INVALID };
	}
}

static void walk_xml_tree(const XMLNode * node, Allocator * allocator, Scene & scene, ShapeGroupMap & shape_group_map, MaterialMap & material_map, TextureMap & texture_map, StringView path) {
	if (node->tag == "bsdf") {
		Handle<Material> material_handle = parse_material(node, scene, material_map, texture_map, path);
		const Material & material = scene.asset_manager.get_material(material_handle);

		material_map.insert(material.name, material_handle);
	} else if (node->tag == "texture") {
		Vector3 scale = 1.0f;
		parse_texture(node, texture_map, path, scene, &scale);
	} else if (node->tag == "shape") {
		StringView type = node->get_attribute_value<StringView>("type");
		if (type == "shapegroup") {
			if (node->children.size() > 0) {
				const XMLNode * shape = node->get_child_by_tag("shape");
				if (!shape) {
					ERROR(node->location, "Shapegroup needs a <shape> child!\n");
				}

				String name = { };

				Handle<MeshData> mesh_data_handle = parse_shape(shape, allocator, scene, path, &name);
				Handle<Material> material_handle  = parse_material(shape, scene, material_map, texture_map, path);

				StringView id = node->get_attribute_value<StringView>("id");
				shape_group_map[id] = { mesh_data_handle, material_handle };
			}
		} else if (type == "instance") {
			const XMLNode * ref = node->get_child_by_tag("ref");
			if (!ref) {
				WARNING(node->location, "Instance without ref!\n");
				return;
			}
			StringView id = ref->get_attribute_value<StringView>("id");

			const ShapeGroup * shape_group = shape_group_map.try_get(id);
			if (shape_group && shape_group->mesh_data_handle.handle != INVALID) {
				Mesh & mesh = scene.add_mesh(id, shape_group->mesh_data_handle, shape_group->material_handle);
				parse_transform(node, &mesh.position, &mesh.rotation, &mesh.scale);
			}
		} else {
			String name = { };

			Handle<MeshData> mesh_data_handle = parse_shape(node, allocator, scene, path, &name);
			Handle<Material> material_handle  = parse_material(node, scene, material_map, texture_map, path);
			Handle<Medium>   medium_handle    = parse_medium(node, scene);

			if (material_handle.handle != INVALID) {
				Material & material = scene.asset_manager.get_material(material_handle);

				if (material.medium_handle.handle != INVALID && material.medium_handle.handle != medium_handle.handle) {
					// This Material is already used with a different Medium
					// Make a copy of the Material and add it as a new Material to the Scene
					Material material_copy = material;
					material_copy.medium_handle = medium_handle;

					material_handle = scene.asset_manager.add_material(std::move(material_copy));
				} else {
					material.medium_handle = medium_handle;
				}
			}

			if (mesh_data_handle.handle != INVALID) {
				Mesh & mesh = scene.add_mesh(std::move(name), mesh_data_handle, material_handle);

				// Do not apply transform to primitive shapes, since they have the transform baked into their vertices
				bool type_is_primitive = type == "rectangle" || type == "cube" || type == "disk" || type == "cylinder" || type == "sphere";
				if (!type_is_primitive) {
					parse_transform(node, &mesh.position, &mesh.rotation, &mesh.scale);
				}
			}
		}
	} else if (node->tag == "sensor") {
		StringView camera_type = node->get_attribute_value<StringView>("type");

		if (camera_type == "perspective" || camera_type == "perspective_rdist" || camera_type == "thinlens") {
			if (const XMLNode * fov = node->get_child_by_name("fov")) {
				scene.camera.set_fov(Math::deg_to_rad(fov->get_attribute_value<float>("value")));
			}

			if (camera_type == "perspective") {
				scene.camera.aperture_radius = 0.0f;
			} else {
				scene.camera.aperture_radius = node->get_child_value_optional("apertureRadius", 0.05f);
				scene.camera.focal_distance  = node->get_child_value_optional("focusDistance", 10.0f);
			}

			parse_transform(node, &scene.camera.position, &scene.camera.rotation, nullptr, Vector3(0.0f, 0.0f, -1.0f));
		} else {
			WARNING(node->location, "WARNING: Camera type '{}' not supported!\n", camera_type);
		}

		if (const XMLNode * film = node->get_child_by_tag("film")) {
			cpu_config.initial_width  = film->get_child_value_optional("width",  cpu_config.initial_width);
			cpu_config.initial_height = film->get_child_value_optional("height", cpu_config.initial_height);
			scene.camera.resize(cpu_config.initial_width, cpu_config.initial_height);
		}
	} else if (node->tag == "integrator") {
		gpu_config.num_bounces = node->get_child_value_optional("maxDepth", gpu_config.num_bounces);
	} else if (node->tag == "emitter") {
		StringView emitter_type = node->get_attribute_value<StringView>("type");

		if (emitter_type == "area") {
			const XMLAttribute * id = node->get_attribute("id");
			if (id) {
				StringView emitter_id = id->get_value<StringView>();

				Material material = { };
				material.type = Material::Type::LIGHT;
				material.name = String(emitter_id, allocator);
				material.emission = node->get_child_value<Vector3>("radiance");

				material_map.insert(emitter_id, scene.asset_manager.add_material(std::move(material)));
			} else {
				WARNING(node->location, "Emitter defined without an id that is also not attached to any geometry!");
			}
		} else if (emitter_type == "envmap") {
			StringView filename_rel = node->get_child_value<StringView>("filename");

			StringView extension = Util::get_file_extension(filename_rel);
			if (extension.is_empty()) {
				WARNING(node->location, "Environment Map '{}' has no file extension!\n", filename_rel);
			} else if (extension != "hdr") {
				WARNING(node->location, "Environment Map '{}' has unsupported file extension. Only HDR Environment Maps are supported!\n", filename_rel);
			} else {
				cpu_config.sky_filename = Util::combine_stringviews(path, filename_rel, scene.allocator);
			}
		} else if (emitter_type == "point") {
			// Make small area light
			constexpr float RADIUS = 0.0001f;
			Matrix4 transform = parse_transform_matrix(node) * Matrix4::create_scale(RADIUS);

			Array<Triangle> triangles = Geometry::sphere(transform, 0);
			Handle<MeshData> mesh_data_handle = scene.asset_manager.add_mesh_data(std::move(triangles));

			Material material = { };
			material.type = Material::Type::LIGHT;
			material.emission = node->get_child_value_optional<Vector3>("intensity", Vector3(1.0f));

			Handle<Material> material_handle = scene.asset_manager.add_material(std::move(material));

			scene.add_mesh("PointLight", mesh_data_handle, material_handle);
		} else {
			WARNING(node->location, "Emitter type '{}' is not supported!\n", emitter_type);
		}
	} else if (node->tag == "include") {
		StringView filename_rel = node->get_attribute_value<StringView>("filename");
		String     filename_abs = Util::combine_stringviews(path, filename_rel, allocator);

		MitsubaLoader::load(filename_abs, allocator, scene);
	} else for (int i = 0; i < node->children.size(); i++) {
		walk_xml_tree(&node->children[i], allocator, scene, shape_group_map, material_map, texture_map, path);
	}
}

void MitsubaLoader::load(const String & filename, Allocator * allocator, Scene & scene) {
	XMLParser xml_parser(filename, allocator);

	XMLNode root = xml_parser.parse_root();

	const XMLNode * scene_node = root.get_child_by_tag("scene");
	if (!scene_node) {
		ERROR(root.location, "File does not contain a <scene> tag!\n");
	}

	{
		StringView version = scene_node->get_attribute_value<StringView>("version");
		Parser version_parser(version);

		int major = version_parser.parse_int(); version_parser.expect('.');
		int minor = version_parser.parse_int(); version_parser.expect('.');
		int patch = version_parser.parse_int();

		int version_number = major * 100 + minor * 10 + patch;
		if (version_number >= 200) {
			ERROR(scene_node->location, "Mitsuba 2 files are not supported!\n");
		}
	}

	ShapeGroupMap shape_group_map(allocator);
	MaterialMap   material_map   (allocator);
	TextureMap    texture_map    (allocator);
	walk_xml_tree(scene_node, allocator, scene, shape_group_map, material_map, texture_map, Util::get_directory(filename.view()));
}
