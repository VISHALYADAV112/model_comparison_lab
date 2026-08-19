from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters.meta_sam3 import MetaSam3Adapter
from .config import LabConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated SAM 3.1 bounded-chunk worker")
    parser.add_argument("--worker-config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-frames", required=True, type=int)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--grounding-batch-size", required=True, type=int)
    parser.add_argument("--max-num-objects", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(
        f"[bounded-worker] starting SAM 3.1 chunk: "
        f"{args.input.name}, {args.max_frames} frames, batch "
        f"{args.grounding_batch_size}, object cap {args.max_num_objects}",
        flush=True,
    )
    serialized = json.loads(args.worker_config.read_text())
    config = LabConfig(root=Path(serialized["root"]), raw=serialized["raw"])
    adapter = MetaSam3Adapter(config)
    manifest, _ = adapter.run_video(
        args.input,
        args.output,
        mode="text",
        text=args.text,
        start_frame=0,
        max_frames=args.max_frames,
        propagation_direction="forward",
        output_prob_threshold=args.threshold,
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        grounding_batch_size=args.grounding_batch_size,
        max_num_objects=args.max_num_objects,
    )
    if not manifest.is_file():
        raise RuntimeError("SAM 3.1 worker finished without a manifest")
    print(f"[bounded-worker] completed chunk: {manifest}", flush=True)


if __name__ == "__main__":
    main()
