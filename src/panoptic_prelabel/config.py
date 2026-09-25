"""Configuration: the packaged default YAML, optionally overridden by a user file."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Category:
    id: int
    name: str
    isthing: bool
    color: str  # "#rrggbb"

    @property
    def rgb(self) -> tuple[int, int, int]:
        return hex_to_rgb(self.color)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


ENUMS: dict[tuple[str, ...], set[str]] = {("resolve", "fill"): {"nearest", "void"}}
# mappings whose keys are open (any COCO class name); their values are checked in Config
OPEN_MAPPINGS = {("things", "coco_to_ontology")}
# lists whose items must be class names of the ontology
CLASS_LISTS = [("resolve", "priority"), ("resolve", "split_classes"), ("web", "box_exclude")]


def _validate(default: Any, override: Any, path: tuple[str, ...] = ()) -> None:
    """Reject keys the default config does not have, and values of the wrong type.

    A typo such as ``fill: neareast`` or ``min_regoin_area`` must fail loudly
    instead of silently falling back to a default or a different behaviour.
    """
    where = ".".join(path) or "<root>"
    if isinstance(default, dict):
        if not isinstance(override, dict):
            raise ValueError(f"config {where}: expected a mapping, got {type(override).__name__}")
        if path in OPEN_MAPPINGS:
            for key, value in override.items():
                if not isinstance(key, str) or not (value is None or isinstance(value, str)):
                    raise ValueError(f"config {where}: expected 'coco class: ontology class' pairs, got {key!r}")
            return
        for key, value in override.items():
            if key not in default:
                raise ValueError(f"config {where}: unknown key {key!r} (known: {sorted(default)})")
            _validate(default[key], value, (*path, key))
        return
    if isinstance(default, list):
        ok = isinstance(override, list)
        if ok and default and not isinstance(default[0], dict):
            for item in override:
                _validate(default[0], item, path)  # every item has the type of the default's items
    else:
        ok = _same_type(default, override)
    if not ok:
        raise ValueError(f"config {where}: expected {type(default).__name__}, got {override!r}")
    allowed = ENUMS.get(path)
    if allowed is not None and override not in allowed:
        raise ValueError(f"config {where}: {override!r} is not one of {sorted(allowed)}")


def _same_type(default: Any, value: Any) -> bool:
    if isinstance(default, bool) or isinstance(value, bool):
        return isinstance(default, bool) and isinstance(value, bool)
    if isinstance(default, float):
        return isinstance(value, (int, float))  # 1 is a valid 1.0
    if isinstance(default, int):
        return isinstance(value, int)  # but 32.5 is not a valid point count
    return isinstance(value, type(default))


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def default_config_text() -> str:
    return resources.files("panoptic_prelabel").joinpath("configs/default.yaml").read_text("utf-8")


class Config:
    """Thin, validated view over the YAML dictionary."""

    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.categories = [
            Category(int(c["id"]), str(c["name"]), bool(c["isthing"]), str(c["color"])) for c in data["categories"]
        ]
        self.by_name = {c.name: c for c in self.categories}
        self.by_id = {c.id: c for c in self.categories}
        if len(self.by_name) != len(self.categories) or len(self.by_id) != len(self.categories):
            raise ValueError("category names and ids must be unique")
        fill = data["resolve"]["fill"]
        if fill not in ENUMS[("resolve", "fill")]:
            raise ValueError(f"config resolve.fill: {fill!r} is not one of {sorted(ENUMS[('resolve', 'fill')])}")
        self._check_priority()
        for section, key in CLASS_LISTS:
            unknown = sorted(set(data[section][key]) - set(self.by_name))
            if unknown:
                raise ValueError(f"config {section}.{key}: unknown classes {unknown}")
        mapping = data["things"]["coco_to_ontology"]
        bad = sorted(v for v in mapping.values() if v is not None and v not in self.by_name)
        if bad:
            raise ValueError(f"config things.coco_to_ontology maps to unknown classes {bad}")

    # sections, returned as plain dicts
    def __getitem__(self, section: str) -> dict[str, Any]:
        return self.data[section]

    def _check_priority(self) -> None:
        prio = self.data["resolve"]["priority"]
        unknown = sorted(set(prio) - set(self.by_name))
        missing = sorted(set(self.by_name) - set(prio))
        if unknown:
            raise ValueError(f"resolve.priority names unknown classes: {unknown}")
        if missing:
            raise ValueError(f"resolve.priority must list every class; missing: {missing}")

    def priority_rank(self, name: str) -> int:
        return self.data["resolve"]["priority"].index(name)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        data = yaml.safe_load(default_config_text())
        if path is not None:
            with open(path, encoding="utf-8") as f:
                override = yaml.safe_load(f) or {}
            _validate(data, override)
            data = _deep_merge(data, override)
        return cls(data)
