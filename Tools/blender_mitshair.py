"""
Blender Script to export hair geometry to the Mitsuba hair format
"""

import bpy

me = bpy.context.object.data

file = open("hair.mitshair", "w")

prev_index_b = -1

for edge in me.edges:
    index_a, index_b = edge.vertices

    if (prev_index_b != -1 and index_a != prev_index_b):
        coord = me.vertices[prev_index_b].co
        file.write(f"{coord[0]} {coord[2]} {-coord[1]}\n\n")

    coord = me.vertices[index_a].co
    file.write(f"{coord[0]} {coord[2]} {-coord[1]}\n")

    prev_index_b = index_b

file.close()
