"""
Blender Script to export camera to mitsuba XML
"""

import bpy
import math
import mathutils

camera = bpy.context.scene.camera
matrix = mathutils.Matrix.Rotation(math.radians(-90.0), 4, 'X') @ camera.matrix_world

with open('camera.xml', 'w') as file:
    file.write(f'<transform name="toWorld">\n')
    file.write(f'\t<matrix value="\n')
    file.write(f'\t\t{matrix[0][0]} {matrix[0][1]} {-matrix[0][2]} {matrix[0][3]}\n')
    file.write(f'\t\t{matrix[1][0]} {matrix[1][1]} {-matrix[1][2]} {matrix[1][3]}\n')
    file.write(f'\t\t{matrix[2][0]} {matrix[2][1]} {-matrix[2][2]} {matrix[2][3]}\n')
    file.write(f'\t\t{matrix[3][0]} {matrix[3][1]} {-matrix[3][2]} {matrix[3][3]}\n')
    file.write(f'\t"/>\n')
    file.write(f'</transform>\n')
