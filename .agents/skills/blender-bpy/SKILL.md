---
name: blender-bpy
description: |
  Trigger when: (1) calling the `blender_scene` tool, (2) writing Blender Python (`import bpy`) for a video shot, (3) a brief asks for 3D charts, coin stacks, candlesticks, a lit 3D object, or a transparent 3D overlay.

  How to author a bpy script that `blender_scene` renders headless into a still or a clip, and how that clip fits into a Remotion or HyperFrames composition. NOT for terrain or world fly-throughs (use `blender_world` and `threejs-world-generation`).
---

# Blender scenes from agent-authored bpy

`blender_scene` runs your script inside headless Blender and renders it. It is local and free. You write the scene; the tool owns resolution, fps, frame range, engine, and output encoding.

## When 3D earns its place

Use Blender when depth, lighting, or a camera move carries the idea: money stacking up, a bar race seen from a moving camera, a candlestick you fly past. Use the Remotion or HyperFrames chart stack for flat charts, axes with many labels, and anything text-heavy. A 3D chart is harder to read exactly, so pair it with a 2D number overlay added in the composition.

## The script contract

- The scene starts empty. Create everything, including the world, lights, and camera.
- Assign `bpy.context.scene.camera`. The tool fails without it.
- Do not set resolution, fps, frame range, or engine, and do not call render operators. The tool applies them before and after your script.
- These globals are defined: `OM_DATA` (the tool's `data` input), `OM_FPS`, `OM_WIDTH`, `OM_HEIGHT`, `OM_START_FRAME`, `OM_END_FRAME`. Derive timing from them so one script serves any duration and aspect ratio.
- Pass numbers and labels through `data`. The safety scan rejects file, network, and process access (`os`, `pathlib`, `open`, `subprocess`, `getattr`, dunder attributes). `bpy`, `bmesh`, `mathutils`, `math`, `random`, and `colorsys` are fine.

## Workflow

1. Write the script to `projects/<project-id>/assets/blender/<shot>.py`.
2. Render a still at the busiest frame (`operation: render_still`, `.png`) and look at it. Check framing, legibility at phone size, and colour.
3. Render the animation (`operation: render_animation`) to `projects/<project-id>/assets/video/<shot>.mp4`.
4. Place the clip in the composition like any other video asset.

```python
from tools.tool_registry import registry
registry.discover()
result = registry._tools["blender_scene"].execute({
    "operation": "render_animation",
    "script_path": "projects/sip-explainer/assets/blender/growth_bars.py",
    "data": {"values": [12, 18, 15, 27, 34, 48], "labels": ["2020", "2021", "2022", "2023", "2024", "2025"]},
    "width": 1080, "height": 960, "fps": 30, "duration_seconds": 4,
    "output_path": "projects/sip-explainer/assets/video/growth_bars.mp4",
})
```

## Sizes for vertical shorts

| Shot | width x height |
|---|---|
| Full frame 9:16 | 1080 x 1920 |
| Top half of a split layout | 1080 x 960 |
| Overlay over the presenter | 1080 x 1920, `transparent: true`, `.mov` or `.webm` |

Render at the size of the slot the clip fills. A panel scaled up from a smaller render looks soft, and a landscape render cropped to a tall slot loses the subject.

## Engines

- `eevee` (default): fast, needs a GPU or display driver. Use it on a Mac or a workstation. About 0.1 to 0.5 s per frame for simple scenes.
- `cycles`: path-traced on CPU, runs anywhere, including a headless Linux container with no GPU. Much slower. Keep `samples` at 16 to 32 and the clip short.
- `workbench`: flat preview shading, for checking motion only.

## Patterns that work on Blender 4.2 through 5.x

`examples/bar_chart_3d.py` is a complete, tested script. The pieces:

```python
import math
import bpy

scene = bpy.context.scene

# Smooth easing for every keyframe inserted after this line.
bpy.context.preferences.edit.keyframe_new_interpolation_type = 'BEZIER'

# Material. These Principled BSDF input names are stable since 4.0.
m = bpy.data.materials.new("Up")
m.use_nodes = True
bsdf = m.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.05, 0.75, 0.42, 1)
bsdf.inputs["Roughness"].default_value = 0.35
bsdf.inputs["Emission Color"].default_value = (0.05, 0.75, 0.42, 1)
bsdf.inputs["Emission Strength"].default_value = 0.6

# World background colour.
world = bpy.data.worlds.new("World"); scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.01, 0.014, 0.03, 1)

# A bar that grows from its base: move the origin to the bottom, then key scale.z.
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
bar = bpy.context.object
scene.cursor.location = (0, 0, 0)
bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
bar.scale = (0.9, 0.9, 0.001); bar.keyframe_insert("scale", frame=1)
bar.scale = (0.9, 0.9, 4.3);   bar.keyframe_insert("scale", frame=19)   # overshoot
bar.scale = (0.9, 0.9, 4.0);   bar.keyframe_insert("scale", frame=27)   # settle

# Text.
curve = bpy.data.curves.new("Label", type="FONT")
curve.body = "2025"; curve.align_x = 'CENTER'; curve.size = 0.42
label = bpy.data.objects.new("Label", curve); scene.collection.objects.link(label)

# Light and a camera that tracks a target while it moves.
light = bpy.data.lights.new("Key", type='AREA'); light.energy = 2500; light.size = 8
light_obj = bpy.data.objects.new("Key", light); scene.collection.objects.link(light_obj)
light_obj.location = (5, -7, 10)

target = bpy.data.objects.new("Target", None); scene.collection.objects.link(target)
cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam")); scene.collection.objects.link(cam)
cam.constraints.new('TRACK_TO').target = target
cam.location = (-7, -15, 5); cam.keyframe_insert("location", frame=OM_START_FRAME)
cam.location = (6, -14, 7);  cam.keyframe_insert("location", frame=OM_END_FRAME)
scene.camera = cam
```

## Things that go wrong

- **Washed-out colours.** The default AgX view transform desaturates bright emissive colours. For brand-exact colour set `scene.view_settings.view_transform = 'Standard'`, and keep emission strength under 1.
- **Subject too small.** Frame for the slot. In a 1080 x 960 panel the chart should fill about 70% of the width. Move the camera closer or raise `cam.data.lens`.
- **Version-specific API.** Reading F-curves through `action.fcurves` and naming the EEVEE engine both differ between Blender 4 and 5. Set easing through the preference above and leave the engine to the tool.
- **Linear, robotic motion.** Add an overshoot key and a settle key, and stagger starts by a few frames per object.
- **Numbers the viewer must read.** Put exact figures in the 2D composition layer, where type is crisp, and let the 3D shot carry the shape of the story.
