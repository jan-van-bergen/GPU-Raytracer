#include "Shader.h"

#include <stdio.h>
#include <string.h>

#include "Util/Util.h"

static GLuint load_shader(const char * filename, GLuint shader_type) {
	GLuint shader = glCreateShader(shader_type);

	const char * source = Util::file_read(filename);

	const GLchar * srcs[] = { source };
	const GLint    lens[] = { (int)strlen(source) };

	glShaderSource(shader, 1, srcs, lens);
	glCompileShader(shader);

	GLint success; glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

	if (!success) {
		GLchar info_log[1024]; glGetShaderInfoLog(shader, sizeof(info_log), nullptr, info_log);
		
		printf("Error compiling shader type %d: '%s'\n", shader_type, info_log);
		__debugbreak();
	}

	delete [] source;

	return shader;
}

Shader Shader::load(const char * vertex_filename, const char * fragment_filename) {
	// Create Program
	Shader shader;
	shader.program_id = glCreateProgram();

	// Load Shader sources
	shader.vertex_id   = load_shader(vertex_filename,   GL_VERTEX_SHADER);
	shader.fragment_id = load_shader(fragment_filename, GL_FRAGMENT_SHADER);

	// Attach Vertex and Fragment Shaders to the Program
	glAttachShader(shader.program_id, shader.vertex_id);
	glAttachShader(shader.program_id, shader.fragment_id);

	// Link the Program
	glLinkProgram(shader.program_id);

	// Check if linking succeeded
	GLint success; glGetProgramiv(shader.program_id, GL_LINK_STATUS, &success);

	if (!success) {
		GLchar info_log[1024]; glGetProgramInfoLog(shader.program_id, sizeof(info_log), nullptr, info_log);

		printf("Error linking shader program: '%s'\n", info_log);
		__debugbreak();
	}

	// Validate Program
	glValidateProgram(shader.program_id);

	// Check if the Program is valid
	GLint valid; glGetProgramiv(shader.program_id, GL_VALIDATE_STATUS, &valid);

	if (!valid) {
		GLchar info_log[1024]; glGetProgramInfoLog(shader.program_id, sizeof(info_log), nullptr, info_log);
		
		printf("Error validating shader program: '%s'\n", info_log);
		__debugbreak();
	}

	return shader;
}
