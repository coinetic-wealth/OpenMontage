"""GPT Image generation through a locally signed-in Codex CLI.

`openai_image` calls the OpenAI Images API with OPENAI_API_KEY. This tool
reaches the same family of OpenAI image models through the `codex` command
instead: `codex exec` runs Codex's built-in image generation tool, which a
ChatGPT-plan sign-in (`codex login`) pays for from the plan's own limits, so
no API key and no per-image bill are involved.

The pipeline treats it exactly like `openai_image`: the prompt the pipeline
wrote is handed to Codex verbatim, with an instruction not to rewrite it. Only
the transport differs. The tool is available only on a machine where `codex`
is installed and signed in; elsewhere (CI, cloud sandboxes) it reports
UNAVAILABLE and the selector falls back to the other image providers.
"""

from __future__ import annotations

import os
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
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

INSTALL_INSTRUCTIONS = (
    "Install the Codex CLI (npm install -g @openai/codex, or see "
    "https://github.com/openai/codex), then sign in once with your ChatGPT "
    "account: codex login. Image generation needs a paid ChatGPT plan and "
    "counts against that plan's limits. Set CODEX_CLI_PATH if `codex` is not "
    "on PATH."
)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_FORMAT_SUFFIX = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}
_PIL_FORMAT = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}


def find_codex() -> str | None:
    """Path of the codex command: CODEX_CLI_PATH, PATH, then ~/.local/bin."""
    override = os.environ.get("CODEX_CLI_PATH")
    if override:
        return override if Path(override).is_file() else None
    found = shutil.which("codex")
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / "codex"
    return str(fallback) if fallback.is_file() else None


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def build_instruction(
    prompt: str,
    target: Path,
    size: str,
    background: str | None,
    reference_count: int,
) -> str:
    """The message handed to Codex. The pipeline's prompt goes in verbatim."""
    lines = [
        "$imagegen",
        "Generate exactly one image with your built-in image generation tool.",
        "Pass the prompt between the PROMPT markers to the tool exactly as written.",
        "Do not rewrite, shorten, expand, translate or 'improve' it, and do not add",
        "style words of your own. It was written by a production pipeline and",
        "already says everything the image needs.",
    ]
    if size and size != "auto":
        lines.append(f"Image size: {size} pixels (width x height).")
    if background == "transparent":
        lines.append("Use a transparent background.")
    if reference_count:
        lines.append(
            f"The {reference_count} attached image(s) are the source/reference "
            "images for this generation; use them as the prompt describes."
        )
    lines += [
        f"Save the final image file to this exact path: {target}",
        "Do nothing else: no other files, no follow-up edits, no commentary.",
        "",
        "PROMPT_START",
        prompt,
        "PROMPT_END",
    ]
    return "\n".join(lines)


class CodexImage(BaseTool):
    name = "codex_image"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "openai_codex"
    hosting_provider = "openai_codex"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []  # checked dynamically: the codex command
    install_instructions = INSTALL_INSTRUCTIONS
    agent_skills = ["provider-model-refresh"]
    fallback_tools = ["openai_image", "openai_fal_image"]

    capabilities = ["generate_image", "generate_illustration", "text_to_image"]
    supports = {
        "complex_instructions": True,
        "text_in_image": True,
        "multiple_outputs": True,
        "image_edit": True,
        "mask": False,
        "transparent_background": True,
        "subscription_billing": True,
    }
    best_for = [
        "complex multi-element compositions",
        "images with text/labels",
        "following detailed instructions accurately",
        "GPT Image quality without an API key (ChatGPT plan via Codex CLI)",
    ]
    not_good_for = [
        "offline generation",
        "machines without a signed-in Codex CLI (cloud sandboxes, CI)",
        "large batches (each image takes about a minute and uses plan limits)",
        "masked inpainting",
    ]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "size": {
                "type": "string",
                "enum": ["1024x1024", "1536x1024", "1024x1536", "auto"],
                "default": "1024x1024",
            },
            "quality": {
                "type": "string",
                "enum": ["low", "medium", "high", "auto"],
                "default": "high",
                "description": "Accepted for parity with openai_image; Codex picks quality itself.",
            },
            "output_format": {
                "type": "string",
                "enum": ["png", "jpeg", "webp"],
                "default": "png",
            },
            "n": {"type": "integer", "default": 1, "minimum": 1, "maximum": 4},
            "output_path": {"type": "string"},
            "generation_mode": {"type": "string", "enum": ["generate", "edit"]},
            "image_path": {"type": "string"},
            "image_paths": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 16,
            },
            "background": {"type": "string", "enum": ["auto", "opaque", "transparent"]},
            "timeout_seconds": {"type": "integer", "minimum": 60, "default": 600},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=100, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=0)
    idempotency_key_fields = [
        "prompt",
        "size",
        "generation_mode",
        "image_path",
        "image_paths",
        "background",
        "output_format",
        "n",
    ]
    side_effects = [
        "writes image file to output_path",
        "runs the local codex CLI, which uses the signed-in ChatGPT plan's limits",
    ]
    user_visible_verification = ["Inspect generated image for relevance and quality"]

    # ------------------------------------------------------------------ status

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if find_codex() else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # Paid from the ChatGPT plan, not per image.
        return 0.0

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _output_paths(output_path: str | None, count: int, fmt: str) -> list[Path]:
        ext = _FORMAT_SUFFIX[fmt]
        if not output_path:
            return [Path(f"generated_image_{idx + 1}{ext}") for idx in range(count)]
        path = Path(output_path)
        if count == 1:
            return [path if path.suffix else path.with_suffix(ext)]
        base = path.with_suffix("") if path.suffix else path
        suffix = path.suffix or ext
        return [base.parent / f"{base.name}_{idx + 1}{suffix}" for idx in range(count)]

    @staticmethod
    def _approval_flags(exe: str) -> list[str]:
        """Flags that let `codex exec` act without prompting, by CLI version."""
        try:
            help_text = subprocess.run(
                [exe, "exec", "--help"], capture_output=True, text=True,
                timeout=30, check=False,
            ).stdout
        except Exception:
            help_text = ""
        if "--approve-for-me" in help_text:
            return ["--approve-for-me"]
        if "--full-auto" in help_text:
            return ["--full-auto"]
        return []

    @staticmethod
    def _login_status(exe: str) -> str:
        try:
            proc = subprocess.run(
                [exe, "login", "status"], capture_output=True, text=True,
                timeout=30, check=False,
            )
        except Exception as exc:
            return f"unknown ({exc})"
        return ((proc.stdout or "") + (proc.stderr or "")).strip()

    @staticmethod
    def _newest_generated(since: float) -> Path | None:
        """Codex keeps every generated image under ~/.codex/generated_images."""
        root = codex_home() / "generated_images"
        if not root.is_dir():
            return None
        fresh = [
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
            and p.stat().st_mtime >= since
        ]
        return max(fresh, key=lambda p: p.stat().st_mtime) if fresh else None

    @staticmethod
    def _finalize(src: Path, dest: Path, fmt: str) -> tuple[int, int] | None:
        """Write src to dest in the requested format; return its pixel size."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
        except ImportError:
            shutil.copyfile(src, dest)
            return None
        with Image.open(src) as img:
            size = img.size
            current = (img.format or "").upper()
            if current == _PIL_FORMAT[fmt]:
                if src.resolve() != dest.resolve():
                    shutil.copyfile(src, dest)
            else:
                out = img.convert("RGB") if fmt == "jpeg" else img
                out.save(dest, _PIL_FORMAT[fmt])
        return size

    def _generate_one(
        self,
        exe: str,
        flags: list[str],
        inputs: dict[str, Any],
        sources: list[str],
        dest: Path,
        fmt: str,
    ) -> dict[str, Any]:
        timeout = int(inputs.get("timeout_seconds", 600))
        with tempfile.TemporaryDirectory(prefix="codex_image_") as work:
            workdir = Path(work)
            target = workdir / f"image{_FORMAT_SUFFIX['png']}"
            instruction = build_instruction(
                inputs["prompt"], target, inputs.get("size", "1024x1024"),
                inputs.get("background"), len(sources),
            )
            # The instruction goes in on stdin ("-"), placed before --image,
            # because --image takes several values and would swallow it.
            cmd = [exe, "exec", "--skip-git-repo-check", *flags, "-C", str(workdir), "-"]
            for src in sources:
                cmd += ["--image", str(Path(src).expanduser().resolve())]

            started = time.time() - 1
            proc = subprocess.run(
                cmd, input=instruction, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout,
                check=False, cwd=workdir,
            )
            log_tail = ((proc.stdout or "") + (proc.stderr or ""))[-2000:]

            produced = None
            if target.is_file():
                produced = target
            else:
                saved = [
                    p for p in workdir.rglob("*")
                    if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
                ]
                produced = saved[0] if saved else self._newest_generated(started)
            if produced is None:
                raise RuntimeError(
                    f"codex exec produced no image (exit {proc.returncode}). "
                    f"Output tail: {log_tail}"
                )
            pixel_size = self._finalize(produced, dest, fmt)
        return {
            "path": str(dest),
            "pixel_size": list(pixel_size) if pixel_size else None,
            "exit_code": proc.returncode,
            "log_tail": log_tail,
        }

    # ----------------------------------------------------------------- execute

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        exe = find_codex()
        if not exe:
            return ToolResult(
                success=False,
                error="codex CLI not found. " + self.install_instructions,
            )

        start = time.time()
        try:
            from jsonschema import validate

            validate(inputs, self.input_schema)
            if inputs.get("mask_path"):
                raise ValueError("codex_image does not support masks; use openai_image")
            if inputs.get("image_url") or inputs.get("image_urls"):
                raise ValueError("codex_image needs local image_path/image_paths, not URLs")
            fmt = inputs.get("output_format", "png")
            if inputs.get("background") == "transparent" and fmt == "jpeg":
                raise ValueError("Transparent output requires PNG or WebP")
            sources = list(inputs.get("image_paths") or [])
            if inputs.get("image_path"):
                sources.insert(0, inputs["image_path"])
            mode = inputs.get("generation_mode", "edit" if sources else "generate")
            if (mode == "edit") != bool(sources):
                raise ValueError(
                    "Edit requires source images; generate cannot accept source images"
                )
            for src in sources:
                if not Path(src).expanduser().is_file():
                    raise ValueError(f"Source image not found: {src}")

            login = self._login_status(exe)
            if "not logged in" in login.lower():
                raise RuntimeError("Codex CLI is not signed in. Run: codex login")
            flags = self._approval_flags(exe)

            n = inputs.get("n", 1)
            dests = self._output_paths(inputs.get("output_path"), n, fmt)
            results = [
                self._generate_one(exe, flags, inputs, sources, dest, fmt)
                for dest in dests
            ]
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                success=False, error=f"Codex image generation timed out after {exc.timeout}s"
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Codex image generation failed: {exc}")

        outputs = [r["path"] for r in results]
        return ToolResult(
            success=True,
            data={
                "provider": "openai_codex",
                "model": "codex-imagegen",
                "prompt": inputs["prompt"],
                "output": outputs[0],
                "outputs": outputs,
                "images_generated": len(outputs),
                "operation": mode,
                "requested_size": inputs.get("size", "1024x1024"),
                "pixel_sizes": [r["pixel_size"] for r in results],
                "auth": login.splitlines()[0] if login else "",
                "cost_status": "subscription",
                "codex_log_tail": results[-1]["log_tail"],
            },
            artifacts=outputs,
            cost_usd=0.0,
            duration_seconds=round(time.time() - start, 2),
            model="codex-imagegen",
        )
