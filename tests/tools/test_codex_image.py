"""codex_image: GPT Image through a signed-in Codex CLI instead of an API key.

A fake `codex` executable stands in for the real CLI so the tests run anywhere.
"""

import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.base_tool import ToolStatus  # noqa: E402
from tools.graphics.codex_image import CodexImage, build_instruction  # noqa: E402

FAKE_CODEX = textwrap.dedent(
    """\
    #!{python}
    import json, sys
    from pathlib import Path
    args = sys.argv[1:]
    log = Path({log!r})
    if args[:2] == ["exec", "--help"]:
        print("Usage: codex exec [OPTIONS] [PROMPT]\\n  --approve-for-me  ...\\n  -i, --image <FILE>...")
        sys.exit(0)
    if args[:2] == ["login", "status"]:
        print({login!r})
        sys.exit(0)
    if args and args[0] == "exec":
        text = sys.stdin.read() if "-" in args else args[-1]
        with log.open("a") as fh:
            fh.write(json.dumps({{"args": args, "stdin": text}}) + "\\n")
        if {mode!r} == "nothing":
            sys.exit(1)
        target = next(
            line.split(": ", 1)[1] for line in text.splitlines()
            if line.startswith("Save the final image file to this exact path: ")
        )
        from PIL import Image
        Image.new("RGB", (1024, 1536), (0, 90, 90)).save(target, "PNG")
        sys.exit(0)
    sys.exit(2)
    """
)


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    def make(login="Logged in using ChatGPT", mode="ok"):
        log = tmp_path / "calls.jsonl"
        exe = tmp_path / "codex"
        exe.write_text(
            FAKE_CODEX.format(python=sys.executable, log=str(log), login=login, mode=mode)
        )
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("CODEX_CLI_PATH", str(exe))
        monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex_home"))
        return log

    return make


def _calls(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_unavailable_without_codex(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_CLI_PATH", str(tmp_path / "missing"))
    tool = CodexImage()
    assert tool.get_status() == ToolStatus.UNAVAILABLE
    result = tool.execute({"prompt": "x"})
    assert not result.success and "codex CLI not found" in result.error


def test_prompt_is_passed_verbatim_and_image_saved(fake_codex, tmp_path):
    log = fake_codex()
    prompt = "Flat illustration of a rupee coin stack, teal background, no text.\nSecond line stays."
    out = tmp_path / "assets" / "hero.png"
    result = CodexImage().execute(
        {"prompt": prompt, "size": "1024x1536", "output_path": str(out)}
    )
    assert result.success, result.error
    assert out.is_file()
    assert result.cost_usd == 0.0
    assert result.data["pixel_sizes"] == [[1024, 1536]]
    assert result.data["auth"] == "Logged in using ChatGPT"

    (call,) = _calls(log)
    assert "--approve-for-me" in call["args"]
    assert "--skip-git-repo-check" in call["args"]
    body = call["stdin"].split("PROMPT_START\n", 1)[1].rsplit("\nPROMPT_END", 1)[0]
    assert body == prompt
    assert "1024x1536" in call["stdin"]


def test_multiple_outputs_and_format_conversion(fake_codex, tmp_path):
    log = fake_codex()
    out = tmp_path / "frame.jpg"
    result = CodexImage().execute(
        {"prompt": "p", "n": 2, "output_format": "jpeg", "output_path": str(out)}
    )
    assert result.success, result.error
    assert result.data["outputs"] == [
        str(tmp_path / "frame_1.jpg"),
        str(tmp_path / "frame_2.jpg"),
    ]
    from PIL import Image

    for path in result.data["outputs"]:
        with Image.open(path) as img:
            assert img.format == "JPEG"
    assert len(_calls(log)) == 2


def test_reference_images_go_after_stdin_marker(fake_codex, tmp_path):
    log = fake_codex()
    ref = tmp_path / "ref.png"
    from PIL import Image

    Image.new("RGB", (8, 8)).save(ref)
    result = CodexImage().execute(
        {"prompt": "restyle", "image_path": str(ref), "output_path": str(tmp_path / "o.png")}
    )
    assert result.success, result.error
    args = _calls(log)[0]["args"]
    assert args.index("-") < args.index("--image")
    assert args[args.index("--image") + 1] == str(ref.resolve())


def test_fails_cleanly_when_codex_makes_nothing(fake_codex, tmp_path):
    fake_codex(mode="nothing")
    result = CodexImage().execute({"prompt": "p", "output_path": str(tmp_path / "o.png")})
    assert not result.success
    assert "produced no image" in result.error


def test_not_signed_in(fake_codex, tmp_path):
    fake_codex(login="Not logged in")
    result = CodexImage().execute({"prompt": "p", "output_path": str(tmp_path / "o.png")})
    assert not result.success and "codex login" in result.error


def test_rejects_masks():
    result = CodexImage().execute({"prompt": "p", "mask_path": "m.png"})
    assert not result.success


def test_instruction_forbids_rewriting():
    text = build_instruction("P", Path("/tmp/x.png"), "1024x1024", "transparent", 0)
    assert "exactly as written" in text and "transparent" in text


def test_wrong_shape_is_cropped_to_requested_size(fake_codex, tmp_path):
    fake_codex()  # fake always returns 1024x1536
    out = tmp_path / "sq.png"
    result = CodexImage().execute({"prompt": "p", "size": "1024x1024", "output_path": str(out)})
    assert result.success, result.error
    assert result.data["pixel_sizes"] == [[1024, 1024]]
    assert result.data["generated_sizes"] == [[1024, 1536]]
    assert "instead of 1024x1024" in result.data["warnings"][0]


def test_fit_none_keeps_generated_shape(fake_codex, tmp_path):
    fake_codex()
    result = CodexImage().execute(
        {"prompt": "p", "size": "1536x1024", "fit": "none", "output_path": str(tmp_path / "w.png")}
    )
    assert result.success, result.error
    assert result.data["pixel_sizes"] == [[1024, 1536]]
    assert result.data["warnings"]


def test_fit_to_size_handles_real_codex_shape():
    from PIL import Image
    from tools.graphics.codex_image import fit_to_size

    assert fit_to_size(Image.new("RGB", (1199, 1312)), (1024, 1536)).size == (1024, 1536)
    assert fit_to_size(Image.new("RGB", (1312, 1199)), (1024, 1536)).size == (1024, 1536)


def test_instruction_names_the_aspect():
    text = build_instruction("P", Path("/tmp/x.png"), "1024x1536", None, 0)
    assert "portrait 2:3" in text
