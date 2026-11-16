"""Contracts for the general-purpose Blender scene tool."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tools.base_tool import ToolStatus
from tools.graphics import blender_scene
from tools.graphics.blender_scene import BlenderScene
from tools.tool_registry import ToolRegistry

_SCRIPT = "import bpy\nbpy.context.scene.camera = None\n"


def _fake_blender(monkeypatch, tmp_path, jobs):
    """Stand in for Blender: record the job and write the frames it asks for."""
    binary = tmp_path / "blender"
    binary.write_text("")
    monkeypatch.setattr(blender_scene, "find_blender", lambda: binary)
    monkeypatch.setattr(blender_scene.shutil, "which", lambda name: f"/usr/bin/{name}")

    def run(command, **kwargs):
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"video")
            jobs.append({"ffmpeg": command})
            return subprocess.CompletedProcess(command, 0, "", "")
        job = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        jobs.append(job)
        if job["operation"] == "render_still":
            Path(job["render_path"]).write_bytes(b"png")
        else:
            for frame in range(job["start_frame"], job["end_frame"] + 1):
                Path(f"{job['render_path']}{frame:04d}.png").write_bytes(b"png")
        report = {"blender_version": "5.3.0", "engine": "BLENDER_EEVEE", "objects": 1}
        return subprocess.CompletedProcess(command, 0, "OPENMONTAGE_BLENDER_SCENE=" + json.dumps(report), "")

    monkeypatch.setattr(blender_scene.subprocess, "run", run)


def test_registry_discovers_blender_scene():
    registry = ToolRegistry()
    registry.discover("tools")
    assert {tool.name for tool in registry.get_by_capability("3d_scene_rendering")} == {"blender_scene"}


def test_unavailable_without_blender(monkeypatch):
    monkeypatch.setattr(blender_scene, "find_blender", lambda: None)
    tool = BlenderScene()
    assert tool.get_status() == ToolStatus.UNAVAILABLE
    result = tool.execute({"operation": "render_still", "script": _SCRIPT, "output_path": "x.png"})
    assert not result.success and "Blender not found" in result.error


def test_unsafe_script_is_rejected_before_blender_runs(monkeypatch, tmp_path):
    jobs: list = []
    _fake_blender(monkeypatch, tmp_path, jobs)
    result = BlenderScene().execute({
        "operation": "render_still",
        "script": "import os\nos.system('id')\n",
        "output_path": str(tmp_path / "out.png"),
    })
    assert not result.success and "safety scan" in result.error
    assert jobs == []


def test_still_passes_data_and_vertical_size_to_the_job(monkeypatch, tmp_path):
    jobs: list = []
    _fake_blender(monkeypatch, tmp_path, jobs)
    output = tmp_path / "still.png"
    result = BlenderScene().execute({
        "operation": "render_still", "script": _SCRIPT, "data": {"values": [1, 2]},
        "width": 1080, "height": 1920, "frame": 40, "output_path": str(output),
    })
    assert result.success, result.error
    assert output.is_file()
    job = jobs[0]
    assert (job["width"], job["height"], job["frame"]) == (1080, 1920, 40)
    assert job["data"] == {"values": [1, 2]}
    assert result.model == "blender-5.3.0"


def test_animation_encodes_frames_and_removes_them(monkeypatch, tmp_path):
    jobs: list = []
    _fake_blender(monkeypatch, tmp_path, jobs)
    output = tmp_path / "bars.mp4"
    result = BlenderScene().execute({
        "operation": "render_animation", "script": _SCRIPT,
        "fps": 30, "duration_seconds": 2, "output_path": str(output),
    })
    assert result.success, result.error
    assert (jobs[0]["start_frame"], jobs[0]["end_frame"]) == (1, 60)
    ffmpeg = jobs[1]["ffmpeg"]
    assert ffmpeg[ffmpeg.index("-framerate") + 1] == "30"
    assert "libx264" in ffmpeg
    assert output.is_file()
    assert not (tmp_path / "bars_frames").exists()


def test_transparent_animation_requires_an_alpha_container(monkeypatch, tmp_path):
    jobs: list = []
    _fake_blender(monkeypatch, tmp_path, jobs)
    tool = BlenderScene()
    rejected = tool.execute({
        "operation": "render_animation", "script": _SCRIPT, "transparent": True,
        "output_path": str(tmp_path / "overlay.mp4"),
    })
    assert not rejected.success and ".mov or .webm" in rejected.error
    accepted = tool.execute({
        "operation": "render_animation", "script": _SCRIPT, "transparent": True,
        "duration_seconds": 1, "output_path": str(tmp_path / "overlay.mov"),
    })
    assert accepted.success, accepted.error
    assert "yuva444p10le" in jobs[-1]["ffmpeg"]


def test_script_failure_surfaces_the_traceback(monkeypatch, tmp_path):
    binary = tmp_path / "blender"
    binary.write_text("")
    monkeypatch.setattr(blender_scene, "find_blender", lambda: binary)
    trace = "Traceback (most recent call last):\n  File \"scene.py\", line 2\nNameError: name 'bar' is not defined"
    monkeypatch.setattr(
        blender_scene.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "Blender 5.3.0\n", trace),
    )
    result = BlenderScene().execute({
        "operation": "render_still", "script": _SCRIPT, "output_path": str(tmp_path / "x.png"),
    })
    assert not result.success
    assert "NameError: name 'bar' is not defined" in result.error
