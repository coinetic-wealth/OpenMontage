import math
import bpy
from mathutils import Vector

scene = bpy.context.scene
bpy.context.preferences.edit.keyframe_new_interpolation_type = 'BEZIER'

def mat(name, color, emission=0.0, rough=0.35, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Emission Color"].default_value = (*color, 1)
    b.inputs["Emission Strength"].default_value = emission
    return m

world = bpy.data.worlds.new("W"); scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.01, 0.014, 0.03, 1)

values = OM_DATA["values"]; labels = OM_DATA["labels"]
n = len(values); vmax = max(values); gap = 1.5
green = mat("Up", (0.05, 0.75, 0.42), 0.6); gold = mat("Gold", (1.0, 0.72, 0.2), 0.8, 0.25, 0.6)
floor_m = mat("Floor", (0.03, 0.04, 0.07), 0.0, 0.25)
white = mat("Text", (0.95, 0.96, 1.0), 1.0)

bpy.ops.mesh.primitive_plane_add(size=60); bpy.context.object.data.materials.append(floor_m)

for i, v in enumerate(values):
    x = (i - (n - 1) / 2) * gap
    h = 5.0 * v / vmax
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, 0, 0.5))
    bar = bpy.context.object
    # origin to the base so scaling grows upward
    bpy.context.scene.cursor.location = (x, 0, 0)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    bar.data.materials.append(gold if i == n - 1 else green)
    f0 = 1 + i * 6
    bar.scale = (0.9, 0.9, 0.001); bar.keyframe_insert("scale", frame=f0)
    bar.scale = (0.9, 0.9, h * 1.08); bar.keyframe_insert("scale", frame=f0 + 18)
    bar.scale = (0.9, 0.9, h); bar.keyframe_insert("scale", frame=f0 + 26)

    c = bpy.data.curves.new(f"L{i}", type="FONT"); c.body = labels[i]; c.align_x = 'CENTER'; c.size = 0.42
    t = bpy.data.objects.new(f"L{i}", c); scene.collection.objects.link(t)
    t.location = (x, -0.9, 0.02); t.rotation_euler = (math.radians(60), 0, 0)
    t.data.materials.append(white)

key = bpy.data.lights.new("Key", type='AREA'); key.energy = 2500; key.size = 8
ko = bpy.data.objects.new("Key", key); scene.collection.objects.link(ko)
ko.location = (5, -7, 10); ko.rotation_euler = (math.radians(40), 0, math.radians(30))

target = bpy.data.objects.new("Target", None); scene.collection.objects.link(target); target.location = (0, 0, 2.2)
cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam")); scene.collection.objects.link(cam)
cam.data.lens = 32
con = cam.constraints.new('TRACK_TO'); con.target = target
cam.location = (-7, -15, 5); cam.keyframe_insert("location", frame=OM_START_FRAME)
cam.location = (6, -14, 7); cam.keyframe_insert("location", frame=OM_END_FRAME)
scene.camera = cam
