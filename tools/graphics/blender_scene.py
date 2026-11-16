"""General-purpose Blender rendering from an agent-authored bpy script.

`blender_world` builds terrain worlds from a fixed spec. This tool is the open
counterpart: the agent writes the Blender Python for any 3D scene (animated
charts, coin stacks, candlesticks, product shots, logo stings) and the tool runs
it headless and returns a still, a PNG sequence, or an encoded video. Local,
free, no API key.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)
from tools.graphics.blender_world import find_blender
from tools.graphics.math_animate import MathAnimate

_RUNTIME_SCRIPT = Path(__file__).resolve().parent / "templates" / "blender-scene-runtime.py"
_STILL_EXTS = {".png"}
# Encoder args per container. Alpha survives only in .mov (ProRes 4444) and .webm (VP9).
_VIDEO_ENCODERS: dict[str, dict[bool, list[str]]] = {
    ".mp4": {
        False: ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-movflags", "+faststart"],
    },
    ".mov": {
        False: ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"],
        True: ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"],
    },
    ".webm": {
        False: ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-crf", "24", "-b:v", "0"],
        True: ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-crf", "24", "-b:v", "0"],
    },
}


class BlenderScene(BaseTool):
    name = "blender_scene"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "3d_scene_rendering"
    provider = "blender"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL
    dependencies: list[str] = []
    install_instructions = (
        "Install Blender 4.2+ (macOS: brew install --cask blender; Linux: download the "
        "official tarball from blender.org and put `blender` on PATH), or set BLENDER_PATH "
        "to the binary. FFmpeg is needed to encode videos."
    )
    agent_skills = ["blender-bpy"]
    capabilities = [
        "render_from_bpy_script", "render_still", "render_animation",
        "transparent_overlay", "blend_project_export",
    ]
    supports = {
        "still": True,
        "animation": True,
        "transparent_background": True,
        "vertical_9_16": True,
        "eevee": True,
        "cycles_cpu": True,
    }
    best_for = [
        "3D data visuals for explainers: animated bar/candlestick charts, coin stacks, pie/donut solids",
        "Hero 3D shots with real lighting, depth of field, and camera moves",
        "Transparent 3D overlays composited over a presenter or a 2D composition",
    ]
    not_good_for = [
        "Flat 2D charts and text cards (use the Remotion/HyperFrames scene stack)",
        "Terrain or world fly-throughs (use blender_world)",
    ]
    input_schema = {
        "type": "object",
        "required": ["operation"],
        "properties": {
            "operation": {"type": "string", "enum": ["doctor", "render_still", "render_animation"]},
            "script": {
                "type": "string",
                "description": (
                    "Blender Python (bpy) that builds the scene and assigns scene.camera. It runs "
                    "in an empty scene with render size, fps, and frame range already applied; do "
                    "not call render operators. Globals: OM_DATA, OM_FPS, OM_WIDTH, OM_HEIGHT, "
                    "OM_START_FRAME, OM_END_FRAME. SECURITY: executed on the host; scanned for "
                    "system/network/file access and rejected by default."
                ),
            },
            "script_path": {"type": "string", "description": "Path to a .py file, instead of script."},
            "data": {"description": "JSON value exposed to the script as OM_DATA (chart series, labels)."},
            "output_path": {
                "type": "string",
                "description": ".png for a still; .mp4, .mov, or .webm for an animation.",
            },
            "width": {"type": "integer", "minimum": 64, "maximum": 7680, "default": 1920},
            "height": {"type": "integer", "minimum": 64, "maximum": 7680, "default": 1080},
            "fps": {"type": "integer", "minimum": 1, "maximum": 120, "default": 30},
            "duration_seconds": {"type": "number", "minimum": 0.1, "maximum": 600, "default": 5},
            "start_frame": {"type": "integer", "minimum": 1, "default": 1},
            "end_frame": {"type": "integer", "minimum": 1},
            "frame": {"type": "integer", "minimum": 1, "description": "Frame for render_still."},
            "engine": {
                "type": "string", "enum": ["eevee", "cycles", "workbench"], "default": "eevee",
                "description": "eevee needs a GPU/display driver; use cycles (CPU) on headless servers.",
            },
            "samples": {"type": "integer", "minimum": 1, "maximum": 4096, "default": 32},
            "transparent": {
                "type": "boolean", "default": False,
                "description": "Transparent film. Alpha is kept for .png, .mov, and .webm outputs.",
            },
            "base_blend": {"type": "string", "description": "Open this .blend before running the script."},
            "blend_path": {"type": "string", "description": "Save the built scene here as a .blend."},
            "keep_frames": {"type": "boolean", "default": False},
            "resume": {
                "type": "boolean", "default": False,
                "description": "Skip frames already rendered by an earlier interrupted run.",
            },
            "timeout_seconds": {"type": "integer", "minimum": 30, "default": 3600},
            "allow_unsafe_code": {
                "type": "boolean", "default": False,
                "description": "Bypass the script safety scan. Only for code you fully trust.",
            },
        },
    }
    output_schema = {"type": "object"}
    resource_profile = ResourceProfile(cpu_cores=4, ram_mb=4096, vram_mb=0, disk_mb=2000, network_required=False)
    idempotency_key_fields = [
        "operation", "script", "script_path", "data", "width", "height", "fps",
        "duration_seconds", "start_frame", "end_frame", "frame", "engine", "samples", "transparent",
    ]
    side_effects = [
        "EXECUTES caller-supplied Python inside Blender on the host; the script is scanned and "
        "rejected by default unless allow_unsafe_code=true",
        "writes an image, a video, and optionally a .blend and a frame directory",
    ]
    user_visible_verification = [
        "Render one still at the busiest frame before committing to an animation render",
        "Check framing for the target aspect ratio and that labels are legible at phone size",
    ]

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if find_blender() and _RUNTIME_SCRIPT.is_file() else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        if inputs.get("operation") != "render_animation":
            return 20.0
        start, end = self._frame_range(inputs)
        per_frame = 8.0 if inputs.get("engine") == "cycles" else 1.0
        return (end - start + 1) * per_frame

    @staticmethod
    def _frame_range(inputs: dict[str, Any]) -> tuple[int, int]:
        start = int(inputs.get("start_frame", 1))
        end = inputs.get("end_frame")
        if end is None:
            frames = round(float(inputs.get("duration_seconds", 5)) * int(inputs.get("fps", 30)))
            end = start + max(1, frames) - 1
        return start, int(end)

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        blender = find_blender()
        if not blender:
            return ToolResult(success=False, error="Blender not found. " + self.install_instructions)
        operation = str(inputs.get("operation") or "")
        if operation == "doctor":
            return self._doctor(blender)
        if operation not in {"render_still", "render_animation"}:
            return ToolResult(success=False, error=f"Unknown operation: {operation}")

        script = inputs.get("script")
        if script is None and inputs.get("script_path"):
            script = Path(str(inputs["script_path"])).expanduser().read_text(encoding="utf-8")
        if not script:
            return ToolResult(success=False, error="script or script_path is required")
        if not inputs.get("allow_unsafe_code", False):
            violations = MathAnimate._scan_scene_code(script)
            if violations:
                return ToolResult(success=False, error=(
                    "script blocked by the blender_scene safety scan. This tool executes "
                    "caller-supplied Python on the host; these constructs are disallowed by "
                    "default:\n  - " + "\n  - ".join(violations)
                    + "\nPass values in through `data` (OM_DATA) instead of reading files. If you "
                    "fully trust this code, pass allow_unsafe_code=true."
                ))

        if not inputs.get("output_path"):
            return ToolResult(success=False, error="output_path is required")
        output = Path(str(inputs["output_path"])).expanduser().resolve()
        ext = output.suffix.lower()
        animation = operation == "render_animation"
        transparent = bool(inputs.get("transparent", False))
        if animation and ext not in _VIDEO_ENCODERS:
            return ToolResult(success=False, error="render_animation output_path must end in .mp4, .mov, or .webm")
        if not animation and ext not in _STILL_EXTS:
            return ToolResult(success=False, error="render_still output_path must end in .png")
        if animation and transparent and True not in _VIDEO_ENCODERS[ext]:
            return ToolResult(success=False, error="transparent animations need a .mov or .webm output_path")
        if animation and not shutil.which("ffmpeg"):
            return ToolResult(success=False, error="FFmpeg is required to encode the rendered frames")
        output.parent.mkdir(parents=True, exist_ok=True)

        start_frame, end_frame = self._frame_range(inputs)
        fps = int(inputs.get("fps", 30))
        frames_dir = output.parent / f"{output.stem}_frames"
        if animation:
            if not inputs.get("resume"):
                shutil.rmtree(frames_dir, ignore_errors=True)
            frames_dir.mkdir(parents=True, exist_ok=True)

        work_dir = Path(tempfile.mkdtemp(prefix="blender_scene_"))
        started = time.time()
        try:
            script_file = work_dir / "scene.py"
            script_file.write_text(script, encoding="utf-8")
            job = {
                "operation": operation,
                "script_path": str(script_file),
                "data": inputs.get("data"),
                "width": int(inputs.get("width", 1920)),
                "height": int(inputs.get("height", 1080)),
                "fps": fps,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame": int(inputs.get("frame", start_frame)),
                "engine": inputs.get("engine", "eevee"),
                "samples": int(inputs.get("samples", 32)),
                "transparent": transparent,
                "resume": bool(inputs.get("resume", False)),
                "base_blend": self._resolved(inputs.get("base_blend")),
                "blend_path": self._resolved(inputs.get("blend_path")),
                "render_path": str(frames_dir / "frame_") if animation else str(output),
            }
            job_file = work_dir / "job.json"
            job_file.write_text(json.dumps(job), encoding="utf-8")
            try:
                process = subprocess.run(
                    [str(blender), "--background", "--python-exit-code", "1",
                     "--python", str(_RUNTIME_SCRIPT), "--", str(job_file)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=int(inputs.get("timeout_seconds", 3600)),
                )
            except subprocess.TimeoutExpired:
                return ToolResult(success=False, error=(
                    "Blender render timed out. Frames rendered so far are kept; rerun with "
                    "resume=true, or lower samples/resolution."
                ))
            if process.returncode != 0:
                return ToolResult(success=False, error="Blender scene failed:\n" + self._error_tail(process))
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        report = self._report(process.stdout)
        artifacts: list[str] = []
        if animation:
            encode = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                 "-start_number", str(start_frame), "-i", str(frames_dir / "frame_%04d.png"),
                 *_VIDEO_ENCODERS[ext][transparent], str(output)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if encode.returncode != 0:
                return ToolResult(success=False, error="FFmpeg encode failed: " + encode.stderr[-2000:])
            if inputs.get("keep_frames") or inputs.get("resume"):
                artifacts.append(str(frames_dir))
            else:
                shutil.rmtree(frames_dir, ignore_errors=True)
        if not output.is_file():
            return ToolResult(success=False, error="Blender finished but wrote no output:\n" + self._error_tail(process))
        artifacts.insert(0, str(output))
        if job["blend_path"]:
            artifacts.append(job["blend_path"])
        return ToolResult(
            success=True,
            data={
                "output": str(output),
                "width": job["width"],
                "height": job["height"],
                "fps": fps,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "transparent": transparent,
                "blender_path": str(blender),
                **report,
            },
            artifacts=artifacts,
            duration_seconds=round(time.time() - started, 2),
            model=f"blender-{report.get('blender_version', 'unknown')}",
        )

    def _doctor(self, blender: Path) -> ToolResult:
        process = subprocess.run(
            [str(blender), "--background", "--python-expr",
             "import bpy; print('OPENMONTAGE_BLENDER=' + bpy.app.version_string)"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        )
        version = next(
            (line.split("=", 1)[1] for line in process.stdout.splitlines() if line.startswith("OPENMONTAGE_BLENDER=")),
            "",
        )
        return ToolResult(
            success=bool(version),
            data={"blender_path": str(blender), "blender_version": version, "ffmpeg": bool(shutil.which("ffmpeg"))},
            error=None if version else (process.stderr or process.stdout)[-1000:],
        )

    @staticmethod
    def _resolved(path: Any) -> str | None:
        return str(Path(str(path)).expanduser().resolve()) if path else None

    @staticmethod
    def _report(stdout: str) -> dict[str, Any]:
        for line in stdout.splitlines():
            if line.startswith("OPENMONTAGE_BLENDER_SCENE="):
                return json.loads(line.split("=", 1)[1])
        return {}

    @staticmethod
    def _error_tail(process: subprocess.CompletedProcess) -> str:
        text = (process.stderr or "") + "\n" + (process.stdout or "")
        marker = text.rfind("Traceback (most recent call last)")
        return text[marker:][-3000:] if marker != -1 else text[-3000:]
