from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import auto_split_asset, inspect_asset, render_asset, split_asset


def _parts(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("parts must contain at least one comma-separated name")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="split3d", description="Split a mesh into coarse semantic parts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect source mesh geometry")
    inspect_parser.add_argument("source", type=Path)

    render_parser = subparsers.add_parser("render", help="render RGB and face-id views")
    render_parser.add_argument("source", type=Path)
    render_parser.add_argument("--output", type=Path, required=True)
    render_parser.add_argument("--views", type=int, default=12)
    render_parser.add_argument("--resolution", type=int, default=512)

    split_parser = subparsers.add_parser("split", help="split a source mesh")
    split_parser.add_argument("source", type=Path)
    split_parser.add_argument("--output", type=Path, required=True)
    split_parser.add_argument("--face-labels", type=Path)
    split_parser.add_argument("--parts", type=_parts)
    split_parser.add_argument("--no-individual", action="store_true", help="only write the multi-node split.glb")

    auto_parser = subparsers.add_parser("auto", help="detect, segment, and split semantic parts")
    auto_parser.add_argument("source", type=Path)
    auto_parser.add_argument("--output", type=Path, required=True)
    auto_parser.add_argument("--parts", type=_parts, required=True)
    auto_parser.add_argument("--detector-model", default="IDEA-Research/grounding-dino-base")
    auto_parser.add_argument("--segmenter-model", default="models/sam2-hiera-tiny")
    auto_parser.add_argument("--views", type=int, default=12)
    auto_parser.add_argument("--resolution", type=int, default=512)
    auto_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    auto_parser.add_argument("--allow-download", action="store_true")
    auto_parser.add_argument("--no-individual", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        print(json.dumps(inspect_asset(args.source), ensure_ascii=False, indent=2))
        return 0
    if args.command == "split":
        manifest = split_asset(
            args.source,
            args.output,
            part_names=args.parts,
            face_labels_path=args.face_labels,
            individual=not args.no_individual,
        )
        print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "render":
        result = render_asset(args.source, args.output, view_count=args.views, resolution=args.resolution)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "auto":
        manifest = auto_split_asset(
            args.source,
            args.output,
            part_names=args.parts,
            detector_model=args.detector_model,
            segmenter_model=args.segmenter_model,
            view_count=args.views,
            resolution=args.resolution,
            device=args.device,
            local_files_only=not args.allow_download,
            individual=not args.no_individual,
        )
        print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
