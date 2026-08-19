# Model matrix

| Model | Current selection | Main job | Prompt/classes | Strength here | Important limit |
|---|---|---|---|---|---|
| YOLO | YOLO26 Large, `yolo26l.pt` | Real-time boxes | 80 COCO classes | Fast baseline, mature export/tracking ecosystem, small-target-aware training | Closed vocabulary unless changed to a YOLOE model; AGPL/enterprise licensing |
| RF-DETR | RF-DETR Large 2026, `rf-detr-large-2026.pth` | Transformer boxes | COCO classes | Independent DETR/DINOv2 detector, strong accuracy and fine-tuning path | Closed vocabulary in this checkpoint; slower than small YOLO variants |
| SAM 3 image | Official `facebook/sam3` | Open-vocabulary boxes + masks | Text, positive/negative exemplars, points, boxes, mask logits | Exhaustive concept segmentation and precise interactive masks | Gated checkpoint, CUDA-first runtime, prompt quality matters |
| SAM 3.1 video | Official `facebook/sam3.1` | Discovery + multi-object masks/tracks | Text, points, boxes, refinements | Object Multiplex shared-memory tracking; native temporal evidence | Heavy server runtime; not an 8-bit checkpoint |
| SAM 3 Q8 | Public `PABannier/sam3.cpp` | Quantized images and memory-bank video tracking | Text, exemplars, points, boxes | Non-gated 1.1 GB Q8 weight file; Linux CPU / Apple Metal bridge; lowest weight-memory text-video option in this lab | Community conversion of base SAM 3; CPU-only and slow on Linux; not official SAM 3.1 or Object Multiplex |

## Why YOLO26 Large

YOLO26 is the current Ultralytics generation. The default Large checkpoint prioritizes detail over speed, while `image_size = 1280` avoids forcing every comparison through the 640-pixel default. Change to `yolo26n.pt` for an edge-speed baseline or fine-tune a P2 architecture for tiny targets. Ultralytics states that P2/P6 definitions are architecture-only and do not have scale-specific pretrained P2 weights.

## Why RF-DETR Large 2026

The current RF-DETR package names `rf-detr-large-2026.pth` as the Large default. It uses a query-based detector rather than YOLO's head, so agreement is stronger evidence and disagreement is diagnostically useful. The adapter sets `RF_HOME` to this workspace so weights do not disappear into an unrelated user cache.

## Why two SAM runtimes

The full and quantized experiments answer different questions:

- Official SAM 3/3.1 tells us the maximum supported quality and feature set on appropriate CUDA hardware.
- Q8 tells us what accuracy and latency survive a roughly 1.1 GB community conversion and a portable runtime.

They must be reported separately. A Q8 success cannot be attributed to Object Multiplex, and an official full-precision result cannot be used to claim the Q8 model has the same fidelity.

The August 2026 Hugging Face audit found community SAM 3.1 INT8/INT4 and FP8
files, but no drop-in CUDA/PyTorch checkpoint with the full text-video path and
broad tracking validation. See `QUANTIZED_VIDEO_RESEARCH.md` before adding one.

## Licensing checklist

- Ultralytics YOLO code/models: review AGPL-3.0 or obtain an enterprise license for incompatible commercial deployment.
- RF-DETR core: Apache-2.0 according to the official repository; re-check any optional platform checkpoint separately.
- Official Meta SAM code/checkpoints: follow the license and acceptable-use files shipped by Meta and the Hugging Face access terms.
- `sam3.cpp` code: MIT. Its converted weights derive from SAM weights; do not assume the code license replaces the upstream model-weight terms.
