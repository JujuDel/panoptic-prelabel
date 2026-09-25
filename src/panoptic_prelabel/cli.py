"""Command-line interface: one sub-command per stage, plus `run` for all of them.

panoptic-prelabel download-weights
panoptic-prelabel prelabel   IMAGE --out WORK
panoptic-prelabel refine     WORK [--labels regions.yaml]
#   ... correct WORK/coco_for_cvat.zip in CVAT, export "COCO 1.0" ...
panoptic-prelabel resolve    ANNOTATIONS.zip --out DIR [--scene scene.yaml]
panoptic-prelabel export-web DIR/panoptic.json --image IMAGE --name NAME --out WEB
panoptic-prelabel run        SCENE_DIR --out OUT
panoptic-prelabel compare    A/panoptic.json B/panoptic.json
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import yaml

from .config import Config, default_config_text


def _cfg(args) -> Config:
    cfg = Config.load(args.config)
    wd = Path(args.weights_dir) if getattr(args, "weights_dir", None) else None
    if wd is not None:
        cfg.data["things"]["model"] = str(wd / Path(cfg["things"]["model"]).name)
        cfg.data["stuff"]["checkpoint"] = str(wd / Path(cfg["stuff"]["checkpoint"]).name)
    return cfg


def _read_groups(scene: str | None) -> list[list[int]]:
    if not scene:
        return []
    data = yaml.safe_load(Path(scene).read_text("utf-8")) or {}
    return [list(map(int, g)) for g in data.get("groups", [])]


def _timed(label: str):
    class _T:
        def __enter__(self):
            self.t = time.perf_counter()

        def __exit__(self, *exc):
            if exc[0] is None:
                print(f"[{label}] {time.perf_counter() - self.t:.1f} s")

    return _T()


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_download_weights(args) -> None:
    from .weights import download

    download(args.dir, force=args.force)


def cmd_prelabel(args) -> None:
    from .pipeline import LABELS, REGIONS_JPG, prelabel

    cfg = _cfg(args)
    with _timed("prelabel"):
        work = prelabel(args.image, args.out, cfg)
    print(f"wrote {work}. Label the regions of {work / REGIONS_JPG} in {work / LABELS}, then run refine.")


def cmd_refine(args) -> None:
    from .pipeline import refine

    cfg = _cfg(args)
    with _timed("refine"):
        out = refine(args.work, cfg, labels=args.labels, allow_unlabeled=args.allow_unlabeled, image_path=args.image)
    print(f"wrote {out}. Import it into CVAT as 'COCO 1.0', correct it, export 'COCO 1.0'.")


def _resolve_to(annotations: str, out: str | Path, cfg: Config, scene: str | None, image: str | None):
    from .coco_io import read_coco
    from .panoptic import write_panoptic
    from .pipeline import load_rgb
    from .resolve import resolve
    from .viz import panoptic_overlay, save_jpeg

    pan = resolve(read_coco(annotations), cfg, groups=_read_groups(scene))
    _, js = write_panoptic(pan, cfg, out)
    if image:
        save_jpeg(panoptic_overlay(load_rgb(image), pan, cfg), Path(out) / "panoptic_preview.jpg")
    things = sum(s.isthing for s in pan.segments)
    print(f"wrote {js}: {len(pan.segments)} segments ({things} things, {len(pan.segments) - things} stuff)")
    return pan, js


def cmd_resolve(args) -> None:
    cfg = _cfg(args)
    with _timed("resolve"):
        _resolve_to(args.annotations, args.out, cfg, args.scene, args.image)


def cmd_export_web(args) -> None:
    from .panoptic import read_panoptic
    from .web import export_web

    cfg = _cfg(args)
    payload = export_web(read_panoptic(args.panoptic), args.image, args.name, args.out, cfg)
    print(
        f"wrote {args.out}/{args.name}_{{raw.jpg,pan.png,inst.png,boxes.json,scene.js}} ({len(payload['boxes'])} boxes)"
    )


def _find_image(scene_dir: Path) -> Path:
    for pattern in ("image.jpg", "image.jpeg", "image.png", "*.jpg", "*.jpeg", "*.png"):
        hits = sorted(scene_dir.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no image in {scene_dir}")


def cmd_run(args) -> None:
    """End to end on a scene directory (see examples/): prelabel -> refine -> resolve -> export-web.

    The scene directory holds image.jpg and, optionally, regions.yaml (region
    labels), cvat_corrected.zip (the human-corrected CVAT export) and
    scene.yaml (instance groups). With a corrected export, the final panoptic
    map is resolved from it; without one, from the automatic pre-annotation.
    """
    from .panoptic import read_panoptic
    from .pipeline import LABELS, THINGS, UnlabeledRegions, prelabel, refine
    from .web import export_web

    cfg = _cfg(args)
    scene_dir = Path(args.scene_dir)
    out = Path(args.out)
    work = out / "work"
    name = args.name or scene_dir.name
    image = _find_image(scene_dir)
    labels = scene_dir / LABELS

    corrected = scene_dir / "cvat_corrected.zip"
    use_corrected = corrected.exists() and not args.auto_only
    recorded = scene_dir / "prelabel"
    if not args.skip_prelabel:
        with _timed("prelabel"):
            prelabel(image, work, cfg)
    elif not (work / THINGS).exists() and (recorded / THINGS).exists():
        shutil.copytree(recorded, work, dirs_exist_ok=True)  # replay the recorded model outputs
        print(f"reusing the recorded model outputs in {recorded}")
    elif not (work / THINGS).exists() and not use_corrected:
        sys.exit(f"--skip-prelabel reuses {work} from an earlier run, and there is none")

    # refine is cheap and always re-run, so an edited regions.yaml takes effect
    # without re-running the models (--skip-prelabel)
    if (work / THINGS).exists():
        # When the final map comes from a corrected CVAT export, the pre-annotation is
        # informative only: a region the labels file does not cover (other hardware or
        # library versions can shift SAM's output slightly) is dropped, not fatal.
        try:
            with _timed("refine"):
                refine(
                    work,
                    cfg,
                    labels=labels if labels.exists() else None,
                    allow_unlabeled=args.allow_unlabeled or use_corrected,
                    image_path=image,
                )
        except UnlabeledRegions as e:
            sys.exit(f"stopped before refine: {e}")

    source = corrected if use_corrected else work / "coco_for_cvat.zip"
    print(f"resolving {source}")
    # scene.yaml groups name annotation ids of the CVAT export: meaningless for the pre-annotation
    scene = scene_dir / "scene.yaml"
    groups = str(scene) if scene.exists() and use_corrected else None
    with _timed("resolve"):
        _, js = _resolve_to(str(source), out / "panoptic", cfg, groups, str(image))
    export_web(read_panoptic(js), image, name, out / "web", cfg)
    print(f"wrote {out / 'web'}")


def cmd_compare(args) -> None:
    from .compare import evaluate, format_report
    from .panoptic import read_panoptic

    files = args.pairs
    if len(files) % 2:
        sys.exit("compare expects pairs: REF1 PRED1 [REF2 PRED2 ...]")
    pairs = [(read_panoptic(files[i]), read_panoptic(files[i + 1])) for i in range(0, len(files), 2)]
    print(format_report(evaluate(pairs), per_class=args.per_class))


def cmd_config(args) -> None:
    """The default config, or the effective one when --config is given."""
    if args.config:
        sys.stdout.write(yaml.safe_dump(Config.load(args.config).data, sort_keys=False))
    else:
        sys.stdout.write(default_config_text())


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="panoptic-prelabel", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    # --config is accepted after any sub-command (a parent parser, not a global flag:
    # argparse would let the sub-command's default overwrite a global value)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="YAML file overriding the default config (see `panoptic-prelabel config`)")

    def add(name: str, **kw) -> argparse.ArgumentParser:
        return sub.add_parser(name, parents=[common], **kw)

    s = add("download-weights", help="fetch the two model files and check their sha256")
    s.add_argument("--dir", default="weights")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_download_weights)

    s = add("prelabel", help="Mask R-CNN things + MobileSAM regions")
    s.add_argument("image")
    s.add_argument("--out", required=True, help="work directory")
    s.add_argument("--weights-dir")
    s.set_defaults(func=cmd_prelabel)

    s = add("refine", help="labelled regions + things -> clean COCO for CVAT")
    s.add_argument("work")
    s.add_argument("--labels", help="region labels (default: WORK/regions.yaml)")
    s.add_argument("--image", help="the photo, for refine_preview.jpg")
    s.add_argument("--allow-unlabeled", action="store_true", help="drop unlabelled regions instead of failing")
    s.set_defaults(func=cmd_refine)

    s = add("resolve", help="overlapping COCO annotations -> COCO panoptic PNG + JSON")
    s.add_argument("annotations", help="CVAT COCO 1.0 zip, or an instances JSON")
    s.add_argument("--out", required=True)
    s.add_argument("--scene", help="scene.yaml with instance groups")
    s.add_argument("--image", help="the photo, for panoptic_preview.jpg")
    s.set_defaults(func=cmd_resolve)

    s = add("export-web", help="panoptic -> the overlays and boxes the website uses")
    s.add_argument("panoptic", help="panoptic.json written by resolve")
    s.add_argument("--image", required=True)
    s.add_argument("--name", required=True, help="scene name, used in file names")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_export_web)

    s = add("run", help="all stages on a scene directory")
    s.add_argument("scene_dir")
    s.add_argument("--out", required=True)
    s.add_argument("--name")
    s.add_argument("--weights-dir")
    s.add_argument("--auto-only", action="store_true", help="ignore cvat_corrected.zip, resolve the pre-annotation")
    s.add_argument(
        "--skip-prelabel",
        action="store_true",
        help="no model inference: reuse OUT/work, or the scene's recorded prelabel/ outputs",
    )
    s.add_argument("--allow-unlabeled", action="store_true")
    s.set_defaults(func=cmd_run)

    s = add("compare", help="PQ / mIoU of predictions against references, pooled over image pairs")
    s.add_argument("pairs", nargs="+", metavar="REF PRED", help="panoptic.json files: REF1 PRED1 [REF2 PRED2 ...]")
    s.add_argument("--per-class", action="store_true")
    s.set_defaults(func=cmd_compare)

    s = add("config", help="print the default configuration")
    s.set_defaults(func=cmd_config)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
