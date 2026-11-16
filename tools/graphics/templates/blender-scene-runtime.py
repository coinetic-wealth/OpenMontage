"""Blender-side runner for the blender_scene tool.

Invoked as: blender --background --python-exit-code 1 --python <this> -- <job.json>

Applies render settings, executes the agent-authored bpy script, then renders a
still or a PNG sequence. Video encoding happens outside Blender (FFmpeg) so the
runner does not depend on Blender's version-specific video output settings.
"""

import json
import sys
from pathlib import Path

import bpy

# Blender 4.2-4.5 call the realtime engine BLENDER_EEVEE_NEXT; 5.x renamed it
# back to BLENDER_EEVEE. Try both so one tool works across versions.
_ENGINE_IDS = {
    "eevee": ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"),
    "cycles": ("CYCLES",),
    "workbench": ("BLENDER_WORKBENCH",),
}


def _set_engine(scene, engine: str) -> str:
    for engine_id in _ENGINE_IDS[engine]:
        try:
            scene.render.engine = engine_id
            return engine_id
        except TypeError:
            continue
    raise RuntimeError(f"Render engine '{engine}' is not available in Blender {bpy.app.version_string}")


def _apply_render_settings(scene, job: dict) -> str:
    engine_id = _set_engine(scene, job["engine"])
    if engine_id == "CYCLES":
        scene.cycles.samples = job["samples"]
        scene.cycles.device = "CPU"
    elif engine_id.startswith("BLENDER_EEVEE"):
        scene.eevee.taa_render_samples = job["samples"]
    render = scene.render
    render.resolution_x = job["width"]
    render.resolution_y = job["height"]
    render.resolution_percentage = 100
    render.fps = job["fps"]
    render.film_transparent = job["transparent"]
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA" if job["transparent"] else "RGB"
    render.use_overwrite = not job["resume"]
    scene.frame_start = job["start_frame"]
    scene.frame_end = job["end_frame"]
    return engine_id


def main() -> None:
    job = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text(encoding="utf-8"))

    if job.get("base_blend"):
        bpy.ops.wm.open_mainfile(filepath=job["base_blend"])
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    # Settings go in before the script so it can read frame range and aspect.
    _apply_render_settings(scene, job)

    namespace = {
        "__name__": "__main__",
        "OM_DATA": job.get("data"),
        "OM_FPS": job["fps"],
        "OM_WIDTH": job["width"],
        "OM_HEIGHT": job["height"],
        "OM_START_FRAME": job["start_frame"],
        "OM_END_FRAME": job["end_frame"],
    }
    script_path = job["script_path"]
    source = Path(script_path).read_text(encoding="utf-8")
    exec(compile(source, script_path, "exec"), namespace)  # noqa: S102 - the tool's purpose

    scene = bpy.context.scene
    # Re-apply: the tool's inputs, not the script, own resolution/range/engine.
    engine_id = _apply_render_settings(scene, job)
    if scene.camera is None:
        raise RuntimeError("The script must create a camera and assign it to bpy.context.scene.camera")

    if job.get("blend_path"):
        bpy.ops.wm.save_as_mainfile(filepath=job["blend_path"])

    scene.render.filepath = job["render_path"]
    if job["operation"] == "render_still":
        scene.frame_set(job["frame"])
        bpy.ops.render.render(write_still=True)
    elif job["operation"] == "render_animation":
        bpy.ops.render.render(animation=True)

    print("OPENMONTAGE_BLENDER_SCENE=" + json.dumps({
        "blender_version": bpy.app.version_string,
        "engine": engine_id,
        "objects": len(bpy.data.objects),
    }))


main()
