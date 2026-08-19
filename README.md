# Three-model vision lab

This is the isolated workspace for the current research decision: compare **YOLO26**, **RF-DETR Large 2026**, and **SAM 3**, then make official **SAM 3.1 Object Multiplex** the advanced video-segmentation path.

It does not replace the earlier `long_range_vision` image/video pipeline. It gives the three selected model families one reproducible download command, one normalized report format, command-line entry points, and a browser playground that can run on a CUDA server and be viewed from a laptop.

## The two SAM runtimes are one model family

- `official`: Meta SAM 3 for images and SAM 3.1 Object Multiplex for video. This is the full-feature, accuracy-first CUDA backend. Its authenticated checkpoints are downloaded from `facebook/sam3` and `facebook/sam3.1`.
- `q8`: the public, non-gated `PABannier/sam3.cpp` Q8_0 community conversion. It is used to measure the memory/quality trade-off and provides a CPU, Metal, or CUDA-capable GGML fallback. It is base SAM 3, not SAM 3.1 Object Multiplex.

The authenticated Hugging Face account on this Mac was successfully checked against both Meta repositories with a dry run on 18 August 2026. Authentication is machine-specific: run `hf auth login` again inside a new SSH server.

## Fast server setup

On an Ubuntu NVIDIA server with CUDA and Python 3.12:

```bash
cd model_comparison_lab
hf auth login
./scripts/bootstrap_server.sh
./scripts/run_playground_lan.sh
```

The bootstrap script creates `.venv`, installs the CUDA PyTorch wheel and all three Python model stacks, installs pinned official Meta SAM 3 source, builds the pinned Q8 C++ bridge, and downloads all checkpoints. Downloads are resumable. Expect several gigabytes.

After the one-time installation, use [DAILY_START.md](DAILY_START.md) for the short server, SSH-tunnel, browser, and shutdown command list.

Inside the browser, [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md) explains the simple image and video workflows before the advanced SAM controls.

To install everything but postpone model downloads:

```bash
DOWNLOAD_MODELS=0 ./scripts/bootstrap_server.sh
.venv/bin/model-lab models download --model all
```

## Open the playground from the laptop

For a trusted same-LAN connection, run the LAN script and open:

```text
http://SERVER_LAN_IP:7860
```

The safer option does not expose the port on the LAN:

```bash
# Server
MODEL_LAB_HOST=127.0.0.1 ./scripts/run_playground_lan.sh

# Laptop
ssh -N -L 7860:127.0.0.1:7860 USER@SERVER_LAN_IP
```

Then open `http://127.0.0.1:7860`. See [SERVER_AND_SSH.md](SERVER_AND_SSH.md) for firewall, authentication, `tmux`, and troubleshooting details.

## Playground coverage

The browser exposes:

- image text/concept prompts;
- positive and negative exemplar boxes;
- positive/negative point prompts and visual boxes;
- multimask output and official low-resolution mask-logit refinement;
- video text discovery and tracking;
- video multi-object point/box initialization;
- corrections on arbitrary frames, object removal, and forward/backward/both propagation with the official backend;
- a persistent official SAM 3.1 session panel exposing start, add/refine, propagate, remove, reset, cancel, and close operations;
- SAM 3.1 Object Multiplex capacity, offload, threshold, and propagation controls;
- per-image comparison of YOLO, RF-DETR, and either SAM backend, with a shared target filter for comparable output;
- structured JSON, individual PNG masks, annotated images/video, and downloadable result archives;
- model status and resumable downloads.

SAM 3 Agent is deliberately excluded because it introduces an MLLM. That contradicts this project's goal of extracting native perception abilities without the extra general-language-model runtime.

## Command-line examples

Inspect readiness and download models:

```bash
.venv/bin/model-lab doctor
.venv/bin/model-lab models status
.venv/bin/model-lab models download --model sam3
```

Compare all three on an image:

```bash
.venv/bin/model-lab compare-image \
  --input /data/frame.jpg \
  --output outputs/frame_comparison \
  --sam-text "distant vehicle" \
  --sam-backend official
```

Official SAM 3 concept segmentation with positive and negative exemplars:

```bash
.venv/bin/model-lab sam-image \
  --backend official \
  --mode text \
  --input /data/frame.jpg \
  --output outputs/sam_text \
  --text "small aircraft" \
  --positive-exemplars "410,220,460,270" \
  --negative-exemplars "700,210,755,275"
```

Official interactive image segmentation:

```bash
.venv/bin/model-lab sam-image \
  --backend official \
  --mode visual \
  --input /data/frame.jpg \
  --output outputs/sam_visual \
  --positive "530,300;545,315" \
  --negative "600,340" \
  --box "480,240,620,390" \
  --multimask
```

Use `outputs/sam_visual/low_res_logits.npy` as `--mask-input` on the next run for iterative mask refinement.

Official SAM 3.1 text-prompted video tracking:

```bash
.venv/bin/model-lab sam-video \
  --backend official \
  --mode text \
  --input /data/video.mp4 \
  --output outputs/sam_video_text \
  --text "vehicle" \
  --start-frame 0 \
  --max-frames 300 \
  --direction forward
```

`--max-frames` is the maximum number of unique output frames. Use `0` to
process the full propagation range. The adapter applies this limit outside
Meta's finite-window implementation to avoid its final-frame cache boundary
bug.

Multi-object visual tracking with a later correction:

```bash
.venv/bin/model-lab sam-video \
  --backend official \
  --mode visual \
  --input /data/video.mp4 \
  --output outputs/sam_video_visual \
  --object "id:0;p:430,260;b:390,210,490,330" \
  --object "id:1;p:820,310" \
  --refine "frame:45;id:0;p:455,275;n:510,300" \
  --remove "120:1" \
  --direction both
```

Run the public Q8 backend by changing `--backend official` to `--backend q8`. Object removal, mask-logit input, and backward/both propagation are official-only controls.

## Project structure

```text
model_comparison_lab/
├── configs/models.toml              # one source of model/runtime settings
├── cpp/
│   ├── CMakeLists.txt                # builds against pinned sam3.cpp
│   └── sam3_bridge.cpp               # headless Q8 image/video JSON bridge
├── scripts/
│   ├── bootstrap_server.sh           # complete CUDA server setup
│   ├── install_meta_sam3.sh          # pinned official Meta runtime
│   ├── build_sam3_cpp.sh             # Q8 CPU/Metal/CUDA bridge
│   └── run_playground_lan.sh         # LAN/SSH-friendly Gradio launch
├── src/model_lab/
│   ├── adapters/                     # YOLO, RF-DETR, official SAM, Q8 SAM
│   ├── playground/                   # browser UI and service layer
│   ├── compare.py                    # normalized three-model runner
│   ├── contracts.py                  # common detection/result schema
│   ├── downloader.py                 # automatic/resumable downloads
│   ├── rendering.py                  # overlays and annotated MP4
│   ├── doctor.py                     # server diagnostics
│   └── cli.py                        # `model-lab` command
├── tests/                            # tests that do not need model weights
├── models/                           # downloaded weights; git-ignored
├── runtime/                          # pinned cloned runtimes/builds; ignored
└── outputs/                          # reports, masks, images, video; ignored
```

Read [ARCHITECTURE.md](ARCHITECTURE.md), [MODEL_MATRIX.md](MODEL_MATRIX.md), and [RESEARCH_AND_DECISIONS.md](RESEARCH_AND_DECISIONS.md) before interpreting comparisons.

## Comparison rule

A raw count or attractive overlay is not an accuracy result. YOLO and RF-DETR are closed-set COCO detectors, while SAM 3 is prompt-conditioned open-vocabulary segmentation. Meaningful ranking requires the same labeled test set, the same resolution/tiling policy, latency warmups, and task-appropriate box/mask/tracking metrics.
