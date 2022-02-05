#pragma once
#include <GL/glew.h>

#include "Core/StringView.h"

struct Shader {
private:
	GLuint program_id;
	GLuint vertex_id;
	GLuint fragment_id;

public:
	inline void bind() const {
		glUseProgram(program_id);
	}

	inline void unbind() const {
		glUseProgram(0);
	}

	inline GLuint get_uniform(const char * name) const {
		return glGetUniformLocation(program_id, name);
	}

	static Shader load(StringView source_vertex, StringView source_fragment);
};
