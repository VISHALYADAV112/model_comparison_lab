from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import load_ground_truth, load_prediction_report, match_detections
from .physics import ImagingInputs, imaging_report
from .pipeline import LongRangePipeline, draw_report, load_config, save_report, stress_test
from .video import VideoSettings, run_video


def _csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Long-range tiny-object detection benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    physics = subparsers.add_parser("physics", help="Calculate sampling, diffraction and turbulence limits")
    physics.add_argument("--sensor-height-px", type=int, required=True)
    physics.add_argument("--vfov-deg", type=float, required=True)
    physics.add_argument("--target-height-m", type=float, default=1.7)
    physics.add_argument("--range-m", type=float, required=True)
    physics.add_argument("--aperture-mm", type=float)
    physics.add_argument("--wavelength-nm", type=float, default=550.0)
    physics.add_argument("--fried-parameter-cm", type=float)
    physics.add_argument("--focal-length-mm", type=float)
    physics.add_argument("--pixel-pitch-um", type=float)
    physics.add_argument("--exposure-ms", type=float)
    physics.add_argument("--angular-rate-mrad-s", type=float)
    physics.add_argument("--output")

    run = subparsers.add_parser("run-image", help="Run enabled pretrained models on overlapping tiles")
    run.add_argument("--config", default="configs/research.toml")
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--visualization")

    video = subparsers.add_parser(
        "run-video",
        help="Detect on keyframes, track every frame, and emit track-level evidence",
    )
    video.add_argument("--config", default="configs/video_general.toml")
    video.add_argument("--input", required=True)
    video.add_argument("--output-dir", required=True)
    video.add_argument("--detection-interval", type=int, default=5)
    video.add_argument("--min-hits", type=int, default=2)
    video.add_argument("--max-missed-keyframes", type=int, default=2)
    video.add_argument("--association-iou", type=float, default=0.2)
    video.add_argument("--start-frame", type=int, default=0)
    video.add_argument("--max-frames", type=int)
    video.add_argument("--no-keyframe-reports", action="store_true")
    video.add_argument(
        "--appearance-encoder",
        choices=("none", "histogram", "mobilenet_v3_small"),
        default="none",
        help="ROI token encoder for recurrent appearance-aware object memory",
    )
    video.add_argument("--appearance-device", default="auto")
    video.add_argument("--appearance-batch-size", type=int, default=64)
    video.add_argument("--appearance-roi-padding", type=float, default=0.35)
    video.add_argument("--appearance-weight", type=float, default=0.35)
    video.add_argument("--appearance-min-similarity", type=float, default=0.25)
    video.add_argument("--appearance-cross-label-similarity", type=float, default=0.90)
    video.add_argument("--appearance-momentum", type=float, default=0.85)
    video.add_argument(
        "--replay-keyframes",
        help="Reuse frame_XXXXXXXX.json detector reports from an earlier run for a controlled tracker ablation",
    )

    stress = subparsers.add_parser("stress-test", help="Measure detection survival as real image detail is removed")
    stress.add_argument("--config", default="configs/research.toml")
    stress.add_argument("--input", required=True)
    stress.add_argument("--output-dir", required=True)
    stress.add_argument("--scales", type=_csv_floats, default=[1.0, 0.75, 0.5, 0.35, 0.25, 0.15])

    evaluate = subparsers.add_parser("evaluate-report", help="Score a prediction report against compact or COCO boxes")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--ground-truth", required=True)
    evaluate.add_argument("--image-name")
    evaluate.add_argument("--image-id", type=int)
    evaluate.add_argument("--iou", type=float, default=0.5)
    evaluate.add_argument(
        "--all-classes",
        action="store_true",
        help="Evaluate every COCO category instead of retaining only person-like categories",
    )
    evaluate.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "physics":
        report = imaging_report(
            ImagingInputs(
                sensor_height_px=args.sensor_height_px,
                vertical_fov_deg=args.vfov_deg,
                target_height_m=args.target_height_m,
                range_m=args.range_m,
                aperture_mm=args.aperture_mm,
                wavelength_nm=args.wavelength_nm,
                fried_parameter_cm=args.fried_parameter_cm,
                focal_length_mm=args.focal_length_mm,
                pixel_pitch_um=args.pixel_pitch_um,
                exposure_ms=args.exposure_ms,
                angular_rate_mrad_s=args.angular_rate_mrad_s,
            )
        )
        if args.output:
            save_report(report, args.output)
        print(json.dumps(report, indent=2))
        return
    if args.command == "run-image":
        pipeline = LongRangePipeline(load_config(args.config))
        report = pipeline.run_image(args.input)
        save_report(report, args.output)
        if args.visualization:
            draw_report(args.input, report, args.visualization)
        print(f"Wrote {len(report['detections'])} fused detections to {Path(args.output).resolve()}")
        return
    if args.command == "run-video":
        pipeline = None if args.replay_keyframes else LongRangePipeline(load_config(args.config))
        report = run_video(
            pipeline,
            args.input,
            args.output_dir,
            VideoSettings(
                detection_interval=args.detection_interval,
                min_hits=args.min_hits,
                max_missed_keyframes=args.max_missed_keyframes,
                association_iou=args.association_iou,
                start_frame=args.start_frame,
                max_frames=args.max_frames,
                save_keyframe_reports=not args.no_keyframe_reports,
                appearance_encoder=args.appearance_encoder,
                appearance_device=args.appearance_device,
                appearance_batch_size=args.appearance_batch_size,
                appearance_roi_padding=args.appearance_roi_padding,
                appearance_weight=args.appearance_weight,
                appearance_min_similarity=args.appearance_min_similarity,
                appearance_cross_label_similarity=args.appearance_cross_label_similarity,
                appearance_momentum=args.appearance_momentum,
                replay_keyframe_dir=args.replay_keyframes,
            ),
        )
        print(
            f"Wrote {report['confirmed_track_count']} confirmed tracks and "
            f"{report['tentative_track_count']} tentative tracks to {Path(args.output_dir).resolve()}"
        )
        return
    if args.command == "stress-test":
        pipeline = LongRangePipeline(load_config(args.config))
        report = stress_test(pipeline, args.input, args.output_dir, args.scales)
        print(json.dumps(report, indent=2))
        return
    if args.command == "evaluate-report":
        predictions = load_prediction_report(args.predictions)
        ground_truth = load_ground_truth(
            args.ground_truth,
            image_name=args.image_name,
            image_id=args.image_id,
            person_only=not args.all_classes,
        )
        report = match_detections(predictions, ground_truth, iou_threshold=args.iou)
        report.update(
            prediction_count=len(predictions),
            ground_truth_count=len(ground_truth),
            iou_threshold=args.iou,
            class_scope="all" if args.all_classes else "person",
        )
        if args.output:
            save_report(report, args.output)
        print(json.dumps(report, indent=2))
