#include "Shader.h"

#include <stdio.h>
#include <string.h>

#include "Core/IO.h"
#include "Util.h"

static GLuint load_shader(const char * source, int source_len, GLuint shader_type) {
	GLuint shader = glCreateShader(shader_type);

	glShaderSource(shader, 1, &source, &source_len);
	glCompileShader(shader);

	GLint success; glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

	if (!success) {
		char info_log[1024] = { };
		glGetShaderInfoLog(shader, sizeof(info_log), nullptr, info_log);

		IO::print("Error compiling shader type {}: '{}'\n"sv, shader_type, info_log);
 		__debugbreak();
	}

	return shader;
}

Shader Shader::load(const char * vertex_filename, int vertex_len, const char * fragment_filename, int fragment_len) {
	// Create Program
	Shader shader;
	shader.program_id = glCreateProgram();

	// Load Shader sources
	shader.vertex_id   = load_shader(vertex_filename,   vertex_len,   GL_VERTEX_SHADER);
	shader.fragment_id = load_shader(fragment_filename, fragment_len, GL_FRAGMENT_SHADER);

	// Attach Vertex and Fragment Shaders to the Program
	glAttachShader(shader.program_id, shader.vertex_id);
	glAttachShader(shader.program_id, shader.fragment_id);

	// Link the Program
	glLinkProgram(shader.program_id);

	// Check if linking succeeded
	GLint success;
	glGetProgramiv(shader.program_id, GL_LINK_STATUS, &success);

	if (!success) {
		char info_log[1024] = { };
		glGetProgramInfoLog(shader.program_id, sizeof(info_log), nullptr, info_log);

		IO::print("Error linking shader program: '{}'\n"sv, info_log);
		__debugbreak();
	}

	// Validate Program
	glValidateProgram(shader.program_id);

	// Check if the Program is valid
	GLint valid; glGetProgramiv(shader.program_id, GL_VALIDATE_STATUS, &valid);

	if (!valid) {
		char info_log[1024] = { };
		glGetProgramInfoLog(shader.program_id, sizeof(info_log), nullptr, info_log);

		IO::print("Error validating shader program: '{}'\n"sv, info_log);
		__debugbreak();
	}

	return shader;
}
