# New-chat handoff: long-range vision and three-model lab

Last updated: 20 August 2026. Project release: 0.6.0.

## Copy this into the new chat

```text
Continue the long-range vision project from the existing workspace. Do not
rebuild it from scratch. First read model_comparison_lab/NEW_CHAT_HANDOFF.md,
then inspect git status, the latest commit, and
/Users/vishalyadav/Desktop/Practice/long_range_vision/error.txt. The active
repository is model_comparison_lab on branch main and changes must be tested
and pushed to GitHub so the Rocky Linux server can pull them. Current priority:
verify release 0.5.3's continuous native SAM 3.1 engine on the server. First
run two 60-frame windows to exercise the repaired frame-59/frame-60 boundary,
then 300+ frames while checking that peak VRAM and rolling-state counts
plateau. Compare native IDs/masks against ordinary whole-video output. Test
RTSP only with a user-authorized camera reachable from the server. Secondary
priority once that passes: validate release 0.6.0's image cascade mode
(tile -> propose -> fuse -> padded ROI crop -> SAM verify) on the server with
real model weights, since it was only unit/static-tested on the Mac. Keep the
architecture lean: YOLO, RF-DETR and SAM; no LLM or VLM runtime.
```

## Goal and architecture decision

The research goal is detection, segmentation, and tracking of any object,
especially very small or distant objects. The project deliberately uses
specialized perception models instead of a frontier LLM/VLM:

1. YOLO26-L provides the fast closed-set detector baseline.
2. RF-DETR Large 2026 provides an independent transformer detector baseline.
3. SAM 3 provides promptable masks on images; SAM 3.1 Object Multiplex provides
   the official text/visual video tracking path.
4. The public SAM 3 Q8 GGML conversion is a separate low-weight-memory
   experiment. It is base SAM 3, not SAM 3.1.

The likely production architecture is a cascade: tiled/multiscale YOLO and
RF-DETR proposals, disagreement/uncertainty selection, SAM mask refinement,
and temporal confirmation. No model can recover details that optics and sensor
sampling did not record.

## Workspace and repositories

- Main Mac workspace:
  `/Users/vishalyadav/Desktop/Practice/long_range_vision`
- Active three-model Git repository:
  `/Users/vishalyadav/Desktop/Practice/long_range_vision/model_comparison_lab`
- GitHub: <https://github.com/VISHALYADAV112/model_comparison_lab>
- Branch: `main`
- Latest pasted server log:
  `/Users/vishalyadav/Desktop/Practice/long_range_vision/error.txt`
- Test video on Mac:
  `/Users/vishalyadav/Desktop/Practice/test_video.mp4`
- Server checkout: `/home/vishal/model_comparison_lab`

The parent `long_range_vision` project is the earlier general image/video
pipeline. Its key records are `PROJECT_STATUS.md`, `IMPLEMENTED_ARCHITECTURES.md`,
`ARCHITECTURE_PROGRESSION.md`, `VIDEO_ARCHITECTURE.md`, `RESEARCH.md`, and
`RUNBOOK.md`. The `model_comparison_lab` subdirectory is the active isolated
YOLO/RF-DETR/SAM comparison and playground repository.

## Server facts

- Host: Rocky Linux 8 at `192.168.1.216`.
- User: `vishal`; there is no sudo/root installation workflow.
- GPU: NVIDIA L40S with about 46 GB VRAM.
- Bootstrap Conda environment: `model-lab-bootstrap`.
- Project Python environment: `/home/vishal/model_comparison_lab/.venv`.
- Python 3.12, PyTorch 2.10 CUDA 12.8, CUDA available.
- FFmpeg and CMake come from Miniforge/Conda. The installed FFmpeg rejects the
  legacy `-vsync` option.
- The Hugging Face CLI is authenticated for Meta's gated repositories.
- Models and environments are already downloaded. Do not rerun the complete
  bootstrap or redownload checkpoints during normal updates.

## Installed models

- YOLO: `models/yolo/yolo26l.pt` (~53 MB).
- RF-DETR: `models/rfdetr/rf-detr-large-2026.pth` (~136 MB).
- Official SAM 3 image: `models/sam3/official/sam3.pt` (~3.45 GB).
- Official SAM 3.1 video: `models/sam3/official/sam3.1_multiplex.pt`
  (~3.5 GB).
- Public SAM 3 Q8: `models/sam3/q8/sam3-q8_0.ggml` (~1.1 GB).
- Q8 bridge: `runtime/sam3_cpp/build/sam3_bridge`.

Official SAM uses CUDA. The pinned `PABannier/sam3.cpp` revision initializes
CPU on Linux and Metal on Apple; it does not initialize CUDA. Q8 therefore
saves weight memory but is expected to be slow on the Rocky Linux server.

## Dashboard and connection

The Gradio dashboard contains simple image comparison, simple video tracking,
advanced SAM image/video prompts, live video correction, and model status.
It binds to localhost and is viewed through an SSH tunnel.

Server:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate model-lab-bootstrap
cd "$HOME/model_comparison_lab"
.venv/bin/model-lab doctor --strict
MODEL_LAB_HOST=127.0.0.1 MODEL_LAB_PORT=7861 ./scripts/run_playground_lan.sh
```

Mac, in a second terminal:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 7861:127.0.0.1:7861 vishal@192.168.1.216
```

Open `http://127.0.0.1:7861`. A successful `ssh -N` terminal remains blank.

## How to update the server after a pushed fix

Stop the dashboard with `Ctrl-C`, then run:

```bash
cd "$HOME/model_comparison_lab"
git pull --ff-only
.venv/bin/python -m pip install -e ".[all]"
SAM3_CPP_BACKEND=cpu ./scripts/build_sam3_cpp.sh
.venv/bin/model-lab doctor --strict
```

The C++ rebuild is mandatory after `cpp/sam3_bridge.cpp` changes. No `sudo` is
needed.

## Important completed fixes

- Added dependencies missing from Meta's package metadata: compatible
  Setuptools/pkg_resources, einops, pycocotools, and psutil.
- Converted returned BF16 tensors to FP32 before NumPy serialization.
- Filtered unsupported SAM 3.1 session/state-offload arguments.
- Added YOLO/RF-DETR target filtering for fairer image comparisons.
- Added official batch 1/4/16 profiles and decoded-frame CPU offload.
- Rendered annotated output as browser-compatible H.264/yuv420p.
- Added explicit 60, 300, and whole-video choices and accurate output-duration
  reporting.
- Added Q8 image/video bridge, model download, diagnostics, and dashboard
  integration.
- Corrected the documentation: Q8 is CPU-only on Linux at the pinned revision.
- Added bounded long-file and RTSP processing with isolated per-chunk CUDA
  workers, overlap ID handoff, incremental JSONL/MP4 outputs, RTSP reconnects,
  UTC capture timestamps, and a fixed two-chunk capture queue.
- Added continuous native SAM 3.1 processing: one model/session and native ID
  space, lazy frame decode, Meta's stateful production batching, conservative
  rolling-state pruning, one final MP4, bounded RTSP frame capture, and a
  SQLite track/best-crop catalogue. The isolated-chunk path remains a fallback.

## Release 0.3.3 fixes being verified

The newest pasted Q8 failure was:

```text
Unrecognized option 'vsync'.
Error splitting the argument list: Option not found
sam3_bridge: Failed to decode frame 0
```

Release 0.3.3 removes `-vsync` and retains one persistent sequential FFmpeg
raw-RGB decoder. The bridge must be rebuilt on the server before retesting.

The official memory regression had two causes in recent code:

1. The simple dashboard changed its default from 60 frames to the whole video.
2. The finite-frame workaround stopped the returned stream at 60 but sent no
   internal limit to Meta, potentially preparing the entire video.

Release 0.3.3 restores the 60-frame safe default. It also aligns Meta's two
inconsistent finite-range interpretations: the propagation loop uses an
inclusive endpoint while the detector cache uses an exclusive endpoint. A
requested N-frame run now sends N-1 to propagation and N to the detector, so
the result is exact and internally bounded.

Whole-video official tracking can still exceed VRAM because temporal state and
objects grow across frames. Start with official batch 4 at 60 frames; if that
fails, restart the dashboard, confirm free VRAM with `nvidia-smi`, and test
batch 1 at 60 frames. The exact OOM traceback was not present in the latest
`error.txt`; obtain it before claiming a remaining model or allocator bug.

## Release 0.5.3 continuous native SAM 3.1 path

Release 0.5.3 is pushed to `origin/main` at commit `1697a0a` (`Keep continuous
SAM prompt state synchronized`). Pull that commit before the next server test;
no C++ rebuild is required because this release changes only Python and
documentation.

Dashboard tab **Long video and RTSP surveillance** defaults to **Continuous
native SAM 3.1**. It keeps one predictor and tracking state for the complete
file/feed. The 60-frame control is only a progress/pruning window: SAM is not
reset, IDs are not stitched, and frames are decoded lazily through a compact
global-index adapter.

The runner invokes the pinned runtime's `Sam3MultiplexTrackingProd` batching
method so the 15-frame hot-start state survives window boundaries. It retains
32 frames of tracker history, explicit point prompts, the latest four detector
conditioning frames per object, at most 96 decoded pending originals, and a
bounded 64-frame RTSP capture queue. Masks,
JSONL, track crops, and video frames are written incrementally. File outputs
are one final H.264 MP4 with source audio; RTSP outputs are video-only.
`live_preview.jpg` is atomically replaced as results arrive and Gradio refreshes
it about once per second. The first visible result follows Meta's native
15-frame hot-start confirmation delay.

Release 0.5.1 also fixes the server traceback `Inplace update to inference
tensor outside InferenceMode` by performing manual rolling-session prompt and
language-cache initialization under `torch.inference_mode()`, matching Meta's
decorated prompt path.

Release 0.5.2 fixes the subsequent server `KeyError:
'multistep_point_inputs'`. The lab had enabled Meta's
`trim_past_non_cond_mem_for_eval`, an optimization documented for VOS where
only the first frame is prompted. Text-based object multiplexing adds later
detector mask prompts, and direct-mask outputs do not satisfy that trimmer's
state contract. The flag now remains disabled; the existing 32-frame
window-boundary pruner still bounds retained state. Empty partial runs also
skip FFmpeg finalization instead of producing a secondary no-video-stream
warning.

Release 0.5.3 fixes the next-window `AssertionError` from Meta's
`propagate_in_video_preflight`. Periodic text-detector mask prompts become
non-conditioning inputs after tracking starts. The boundary pruner now retains
the latest four mask-prompt anchors per object irrespective of their output
classification and synchronizes both consolidated-frame sets to the remaining
point/mask inputs. The server evidence for 0.5.2 showed window 1 completing at
about 7.9 GB peak CUDA allocation before the old bookkeeping mismatch appeared
four frames into window 2.

`track_identities.sqlite3` stores first/last frame, best confidence, and best
crop for each SAM ID, with nullable verified-identity/embedding fields. It
supports later human review but is not automatic ReID. A numeric SAM ID alone
cannot recognize a person who returns after track retirement; that requires a
separately evaluated face or person-ReID embedding model.

This path has unit/static validation on the Mac but still requires model-weight
validation on the L40S. Do not claim exact whole-video equivalence or a flat
VRAM curve until the server comparison passes. Use `--engine chunked` if the
pinned Meta runtime still leaks or a process-exit cleanup boundary is needed.

## Release 0.4.0 isolated-chunk fallback

The fallback decodes a rolling CPU clip, retains only overlap frames, runs each
finite SAM 3.1 chunk in a separate Python/CUDA process, writes results
incrementally, and lets the worker process exit before the next chunk. Defaults
are 60 frames, 8 overlap frames, batch 1, and 16 active objects.

SAM IDs are local to each chunk. A bounded CPU registry assigns global IDs from
box IoU on overlap frames and records the original ID as `chunk_instance_id`.
This is not appearance re-identification; identities can change after long
occlusion, camera cuts, or dropped RTSP chunks.

RTSP credentials and query parameters are redacted from manifests. Capture and
inference run concurrently with at most two waiting chunks. If inference falls
behind, the oldest pending chunk is deleted and `dropped_rtsp_chunks` is
incremented instead of allowing memory and latency to grow without bound.

## Release 0.6.0 image cascade mode

`model_comparison_lab`'s image comparison previously ran YOLO, RF-DETR, and
SAM 3 each once on the whole frame — good for apples-to-apples model
comparison, but SAM 3/3.1 internally resizes every frame to a fixed 1008x1008
(confirmed in the pinned `facebookresearch/sam3` source at
`sam3/model_builder.py`, `image_size=1008` in five places), so a small or
distant object in a large source image can be squashed away before any model
sees it. The sibling root project's `src/long_range_vision/pipeline.py`
already solved this for its own image architecture with a tile -> propose ->
fuse -> padded-ROI-crop -> verify cascade; that pattern did not exist in
`model_comparison_lab` until now.

New modules:

- `src/model_lab/tiling.py`: overlapping source-resolution tiles
  (`generate_tiles`), coordinate mapping (`translate_box`, `clip_box`), and
  padded verification crops (`padded_crop_box`). Ported from
  `long_range_vision.tiling`, adapted to model_lab's tuple-based `Detection`.
- `src/model_lab/fusion.py`: `iou`, `non_max_suppression`, and
  `weighted_box_fusion`. Ported from `long_range_vision.fusion`; provenance is
  read from `detection.metadata["source_model"]` since model_lab's `Detection`
  has no dedicated model field.
- `src/model_lab/cascade.py`: `compare_image_cascade` orchestrates the whole
  pipeline. YOLO/RF-DETR are file-path adapters, so each tile crop is written
  to a scratch temp file, detected, then deleted; per-model NMS runs on the
  tile results, then cross-model weighted box fusion. Each fused box becomes a
  padded ROI crop (default padding 2.0, matching root) cut from the original
  full-resolution image, and SAM (official or Q8, whichever backend is
  selected) verifies only that crop. A crop-sized mask is pasted into a
  full-image-sized canvas before saving so the existing mask-overlay/rendering
  code works unmodified.

Wired in three places: `model-lab compare-image --cascade` (CLI, with
`--tile-size`, `--tile-overlap`, `--roi-padding`, `--detector-target`), a new
"Cascade mode" checkbox on the dashboard's first tab
(`quick_cascade` in `playground/app.py`, threaded through
`service.quick_compare`/`service.compare`), and a `[cascade]` section in
`configs/models.toml` (`tile_size=1008`, `tile_overlap=0.2`,
`roi_padding=2.0`, `per_model_nms_iou=0.45`, `ensemble_iou=0.5`,
`[cascade.weights]`).

Status: 23 new tests (`tests/test_tiling.py`, `tests/test_fusion.py`,
`tests/test_cascade.py`) pass on the Mac using fake detector/SAM adapters — no
real model weights were exercised. This mode is slower than whole-image
comparison (one detector call per tile, one SAM call per fused proposal) and
has not been run against real YOLO/RF-DETR/SAM weights or compared for actual
accuracy improvement. Validate on the server before treating it as the
recommended default, and expect proportionally longer runs on large/high-tile-
count images.

## Video cascade research (0.6.0, no code changes)

The same question was asked for video: can the tile/ROI-crop pattern protect
small objects in the continuous or bounded SAM 3.1 video paths the way it now
does for images? Investigation (grep across `src/model_lab/` plus a read-only
clone of the pinned `facebookresearch/sam3` commit `8f0b7f4d4e7eda2ed606ebde
6702c93359ad01da`) found:

- Neither `continuous_video.py` nor `bounded_video.py`/`bounded_worker.py`
  ever calls the YOLO or RF-DETR adapters. Once SAM 3.1 is given a prompt, its
  own `Sam3MultiplexTrackingProd.propagate_in_video()` generator owns frame
  decoding and tracking end to end; our code only consumes
  `(frame_index, outputs)` and manages rolling state/pruning around it.
- Meta's runtime already supports injecting box prompts into a *live* video
  session: `Sam3BasePredictor.add_prompt` (`sam3_base_predictor.py`) accepts
  `bounding_boxes`/`bounding_box_labels` at any `frame_idx` mid-session. This
  is not a gap that needs new upstream capability.
- `model_comparison_lab` already exercises that exact mechanism today, but
  only through `MetaSam3Adapter.run_video(mode="visual")` and its
  `_prompt_request` helper (`adapters/meta_sam3.py`), reachable via
  `model-lab sam-video --mode visual --object "b:x0,y0,x1,y1"` and the
  **bounded/chunked fallback engine** (`bounded_worker.py`). Prompts there are
  manually typed strings, never auto-generated from a detector.
- The **continuous native engine** (`continuous_video.py`, the dashboard's
  current default) does **not** go through `run_video`/`add_prompt`/
  `handle_request` at all — it drives `Sam3MultiplexTrackingProd` directly
  against a hand-managed `inference_state`. That custom loop was only just
  stabilized through releases 0.5.1-0.5.3 by carefully mirroring Meta's exact
  prompt/state initialization order.

Conclusion for a future session: wiring tiled YOLO/RF-DETR keyframe detection
into the **bounded/chunked engine** to auto-generate box prompts (feeding
`objects=["b:...", ...]` into `run_video(mode="visual")`) is a contained,
moderate-risk change because that engine already speaks the box-prompt API.
Doing the same for the **continuous engine** means new surgery on the same
rolling-window/pruning state machine that took three point releases to
stabilize, and should wait until after the 0.5.3 server verification below has
passed and there is capacity to test carefully on the L40S. No code was
written for either video path this round — image cascade only.

## Immediate server verification checklist

1. Stop the dashboard, pull commit `1697a0a`, reinstall editable Python, run
   `model-lab doctor --strict`, and restart. Do not rebuild the C++ bridge.
2. In tab 5 select **Continuous native SAM 3.1** and set **Maximum chunks =
   2**. Confirm it crosses into window 2 and completes 120 frames without the
   `propagate_in_video_preflight` assertion.
3. Confirm the live annotated preview refreshes after Meta's initial 15-frame
   confirmation delay, then confirm one final MP4, `frames.jsonl`,
   `index.json`, and `track_identities.sqlite3`.
4. Check that object IDs remain native and continuous across frames 59/60; no
   overlap-based ID stitching should appear.
5. Run five or more windows (300+ frames) and confirm
   `rolling_state.cached_output_frames`, allocated VRAM, and reserved VRAM
   plateau. Compare IDs and mask IoU with an ordinary whole-video run on the
   same source.
6. If another exception occurs, replace `error.txt` with the complete terminal
   output including the first `[continuous-sam31] window` line and the final
   traceback. Also capture `nvidia-smi` if it is a CUDA/OOM failure.
7. For RTSP, use a short authorized test feed and finite duration. Confirm the
   stored source contains no credentials, Stop works, and disconnect/reconnect
   does not create an unbounded pending queue.

## Test commands on the Mac

The local parent environment has had NumPy conflicts, so use the isolated test
environment:

```bash
UV_CACHE_DIR=/tmp/model_lab_uv_cache \
uv run --no-project --isolated --python 3.12 \
  --with pytest --with numpy==1.26.4 --with Pillow \
  --with huggingface-hub --with opencv-python-headless==4.11.0.86 \
  python -m pytest -q

UV_CACHE_DIR=/tmp/model_lab_uv_cache \
uv run --no-project --isolated --python 3.12 --with ruff \
  ruff check src/model_lab/__init__.py \
  src/model_lab/adapters/meta_sam3.py src/model_lab/bounded_video.py \
  src/model_lab/bounded_worker.py src/model_lab/continuous_video.py \
  src/model_lab/cli.py src/model_lab/cascade.py src/model_lab/tiling.py \
  src/model_lab/fusion.py \
  src/model_lab/playground/app.py src/model_lab/playground/service.py \
  src/model_lab/playground/sessions.py \
  tests/test_bounded_video.py tests/test_continuous_video.py tests/test_config.py \
  tests/test_meta_sam3_adapter.py \
  tests/test_cpp_bridge_contract.py \
  tests/test_tiling.py tests/test_fusion.py tests/test_cascade.py
```

Model-weight inference cannot be fully validated on the Mac. Previous local
checks compiled the Q8 bridge against its exact pinned source, validated the
H.264 test video's raw frame byte count, and tested rendering with H.264. The
image cascade mode (0.6.0) was tested only with fake detector/SAM adapters
standing in for YOLO/RF-DETR/SAM, since real inference needs the server.
The local suite reports 81 passed and one Torch-specific regression skipped
because the isolated macOS runner has no Torch. A repository-wide Ruff run
still reports four pre-existing import-order/unused-import findings in
untouched files; changed files are clean.

## Documentation map

- `README.md`: setup, CLI examples, and project structure.
- `DAILY_START.md`: routine server/tunnel/start/stop commands.
- `DASHBOARD_GUIDE.md`: how to use each dashboard tab.
- `SERVER_AND_SSH.md`: installation and troubleshooting.
- `ARCHITECTURE.md`: component and data-flow design.
- `MODEL_MATRIX.md`: model roles and limitations.
- `RESEARCH_AND_DECISIONS.md`: research rationale and physical limits.
- `QUANTIZED_VIDEO_RESEARCH.md`: community quantization audit and next steps.
- `LONG_VIDEO_AND_RTSP.md`: bounded-memory long-file and surveillance runbook.

## Working rules for the next chat

- Inspect the latest log and code before changing dependencies or model files.
- Preserve `models/`, `runtime/`, `.venv/`, and user outputs.
- Do not use sudo and do not expose Gradio publicly.
- Keep official SAM 3.1 and Q8 results labeled separately.
- Raw detection count is not accuracy; proper comparison needs labeled box,
  mask, and tracking ground truth.
- Test changes, commit them, and push `main`; the server is updated through
  GitHub.
