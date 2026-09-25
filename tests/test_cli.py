"""The command line, on a committed example (no model needed)."""

import json
from pathlib import Path

import pytest

from panoptic_prelabel.cli import main

SCENE = Path(__file__).resolve().parents[1] / "examples" / "bay"


def test_resolve_export_compare(tmp_path, capsys):
    override = tmp_path / "cfg.yaml"
    override.write_text("resolve:\n  fill: nearest\n")
    main(["resolve", str(SCENE / "cvat_corrected.zip"), "--out", str(tmp_path / "pan"), "--config", str(override)])
    main(
        [
            "export-web",
            str(tmp_path / "pan" / "panoptic.json"),
            "--image",
            str(SCENE / "image.jpg"),
            "--name",
            "bay",
            "--out",
            str(tmp_path / "web"),
        ]
    )
    main(["compare", str(SCENE / "reference_2026-07" / "panoptic.json"), str(tmp_path / "pan" / "panoptic.json")])
    out = capsys.readouterr().out
    pixel_acc = float(next(line for line in out.splitlines() if line.startswith("pixel accuracy")).split()[2])
    assert pixel_acc > 99.5
    boxes = json.loads((tmp_path / "web" / "bay_boxes.json").read_text())
    assert {b["cls"] for b in boxes["boxes"]} == {"PERSON", "BACKPACK"}


def test_config_command_prints_the_defaults(capsys):
    main(["config"])
    assert "priority:" in capsys.readouterr().out


def test_config_typos_are_errors(tmp_path):
    override = tmp_path / "cfg.yaml"
    override.write_text("resolve:\n  fill: neareast\n")
    with pytest.raises(ValueError, match="not one of"):
        main(["resolve", str(SCENE / "cvat_corrected.zip"), "--out", str(tmp_path / "p"), "--config", str(override)])
    override.write_text("stuff:\n  min_regoin_area: 5\n")
    with pytest.raises(ValueError, match="unknown key"):
        main(["resolve", str(SCENE / "cvat_corrected.zip"), "--out", str(tmp_path / "p"), "--config", str(override)])


def test_run_skip_prelabel_without_work_and_without_corrected_export_fails_cleanly(tmp_path):
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "image.jpg").write_bytes((SCENE / "image.jpg").read_bytes())
    with pytest.raises(SystemExit, match="earlier run"):
        main(["run", str(scene), "--out", str(tmp_path / "out"), "--skip-prelabel"])


def test_run_auto_only_ignores_scene_groups(tmp_path):
    # scene.yaml names CVAT annotation ids; they must not be applied to the pre-annotation
    main(["run", str(SCENE.parent / "wharf"), "--out", str(tmp_path / "w"), "--skip-prelabel", "--auto-only"])
    assert (tmp_path / "w" / "web" / "wharf_pan.png").exists()
