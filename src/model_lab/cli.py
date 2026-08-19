from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .adapters.meta_sam3 import MetaSam3Adapter
from .adapters.sam3_cpp import Sam3CppAdapter, parse_boxes, parse_points
from .compare import compare_image
from .config import DEFAULT_CONFIG, load_config
from .doctor import doctor_report
from .downloader import download_models, model_status
from .rendering import render_sam_manifest, render_video_manifest


def _split_records(value: str) -> list[str]:
    return [item.strip() for item in value.splitlines() if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-lab", description="YOLO, RF-DETR, and SAM 3 lab")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)

    models = sub.add_parser("models", help="Download or inspect model files")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser("status")
    download = models_sub.add_parser("download")
    download.add_argument(
        "--model",
        choices=("all", "yolo", "rfdetr", "sam3", "sam3-official", "sam3-q8"),
        default="all",
    )

    doctor = sub.add_parser("doctor", help="Inspect server/runtime readiness")
    doctor.add_argument("--strict", action="store_true")

    compare = sub.add_parser("compare-image", help="Run comparable inference on one image")
    compare.add_argument("--input", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument("--models", default="yolo,rfdetr,sam3")
    compare.add_argument("--sam-text", default="object")
    compare.add_argument("--sam-backend", choices=("official", "q8"), default=None)

    image = sub.add_parser("sam-image", help="Use every supported SAM 3 image prompt")
    image.add_argument("--input", required=True, type=Path)
    image.add_argument("--output", required=True, type=Path)
    image.add_argument("--mode", choices=("text", "visual"), default="text")
    image.add_argument("--backend", choices=("official", "q8"), default="official")
    image.add_argument("--text", default="")
    image.add_argument("--positive", default="", help="Semicolon-separated x,y points")
    image.add_argument("--negative", default="", help="Semicolon-separated x,y points")
    image.add_argument("--box", default="", help="x0,y0,x1,y1")
    image.add_argument("--positive-exemplars", default="", help="Semicolon-separated boxes")
    image.add_argument("--negative-exemplars", default="", help="Semicolon-separated boxes")
    image.add_argument("--multimask", action="store_true")
    image.add_argument("--mask-input", type=Path, default=None, help="Official backend .npy logits from a prior run")
    image.add_argument("--cpu", action="store_true")

    video = sub.add_parser("sam-video", help="Text or visual-prompt video tracking")
    video.add_argument("--input", required=True, type=Path)
    video.add_argument("--output", required=True, type=Path)
    video.add_argument("--mode", choices=("text", "visual"), default="text")
    video.add_argument("--backend", choices=("official", "q8"), default="official")
    video.add_argument("--text", default="")
    video.add_argument(
        "--object",
        action="append",
        default=[],
        help="Repeatable: p:x,y;p:x,y;n:x,y;b:x0,y0,x1,y1",
    )
    video.add_argument(
        "--refine",
        action="append",
        default=[],
        help="Repeatable: frame:10;id:0;p:x,y;n:x,y",
    )
    video.add_argument("--remove", action="append", default=[], help="Repeatable: frame:object_id")
    video.add_argument("--start-frame", type=int, default=0)
    video.add_argument("--max-frames", type=int, default=0)
    video.add_argument("--direction", choices=("forward", "backward", "both"), default="forward")
    video.add_argument("--offload-video-to-cpu", action="store_true")
    video.add_argument("--offload-state-to-cpu", action="store_true")
    video.add_argument("--cpu", action="store_true")

    playground = sub.add_parser("playground", help="Launch the browser playground")
    playground.add_argument("--host", default=None)
    playground.add_argument("--port", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "models":
        if args.models_command == "status":
            print(json.dumps(model_status(config), indent=2))
        else:
            paths = download_models(config, args.model)
            print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
        return
    if args.command == "doctor":
        report = doctor_report(config)
        print(json.dumps(report, indent=2))
        if args.strict:
            ready = all(item["ready"] for item in report["models"])
            packages = all(report["packages"].values())
            commands = report["commands"]
            if not ready or not packages or not commands["cmake"] or not commands["ffmpeg"] or not report["gpu"]:
                raise SystemExit(2)
        return
    if args.command == "compare-image":
        names = [name.strip() for name in args.models.split(",") if name.strip()]
        payload = compare_image(
            config,
            args.input,
            args.output,
            models=names,
            sam_text=args.sam_text,
            sam_backend=args.sam_backend,
        )
        print(json.dumps(payload, indent=2))
        return
    if args.command == "sam-image":
        adapter = MetaSam3Adapter(config) if args.backend == "official" else Sam3CppAdapter(config)
        box_values = parse_boxes(args.box)
        image_kwargs = {
            "mode": args.mode,
            "text": args.text,
            "positive_points": parse_points(args.positive),
            "negative_points": parse_points(args.negative),
            "box": box_values[0] if box_values else None,
            "positive_exemplars": parse_boxes(args.positive_exemplars),
            "negative_exemplars": parse_boxes(args.negative_exemplars),
            "multimask": args.multimask,
        }
        if args.backend == "official":
            image_kwargs["mask_input"] = args.mask_input
        else:
            image_kwargs["use_gpu"] = not args.cpu
        manifest, payload = adapter.run_image(
            args.input.expanduser().resolve(), args.output.expanduser().resolve(), **image_kwargs
        )
        annotated = render_sam_manifest(args.input, manifest, args.output / "annotated.jpg")
        print(json.dumps({"manifest": str(manifest), "annotated": str(annotated), "result": payload}, indent=2))
        return
    if args.command == "sam-video":
        adapter = MetaSam3Adapter(config) if args.backend == "official" else Sam3CppAdapter(config)
        video_kwargs = {
            "mode": args.mode,
            "text": args.text,
            "objects": args.object,
            "refinements": args.refine,
            "start_frame": args.start_frame,
            "max_frames": args.max_frames,
        }
        if args.backend == "official":
            video_kwargs.update(
                removals=args.remove,
                propagation_direction=args.direction,
                offload_video_to_cpu=args.offload_video_to_cpu,
                offload_state_to_cpu=args.offload_state_to_cpu,
            )
        else:
            if args.remove or args.direction != "forward":
                raise ValueError("Removal and direction controls require the official backend")
            video_kwargs["use_gpu"] = not args.cpu
        manifest, payload = adapter.run_video(
            args.input.expanduser().resolve(), args.output.expanduser().resolve(), **video_kwargs
        )
        rendered = render_video_manifest(args.input, manifest, args.output / "annotated.mp4")
        print(json.dumps({"manifest": str(manifest), "annotated": str(rendered), "result": payload}, indent=2))
        return
    if args.command == "playground":
        try:
            from .playground.app import APP_CSS, create_app
        except ImportError as exc:
            raise RuntimeError("Install playground dependencies: pip install -e '.[playground]'") from exc
        settings = config.raw["playground"]
        host = args.host or os.environ.get("MODEL_LAB_HOST") or settings["host"]
        port = args.port or int(os.environ.get("MODEL_LAB_PORT", settings["port"]))
        auth = None
        user, password = os.environ.get("MODEL_LAB_USER"), os.environ.get("MODEL_LAB_PASSWORD")
        if bool(user) != bool(password):
            raise ValueError("Set both MODEL_LAB_USER and MODEL_LAB_PASSWORD, or neither")
        if user and password:
            auth = (user, password)
        create_app(config).queue(default_concurrency_limit=1).launch(
            server_name=host,
            server_port=port,
            auth=auth,
            show_error=True,
            css=APP_CSS,
        )
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main(sys.argv[1:])
