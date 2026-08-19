# Architecture

## System view

```text
                         ┌─ YOLO26 adapter ───────────────┐
image / video / prompts ─┼─ RF-DETR adapter ─────────────┼─ normalized JSON
                         └─ SAM adapter selector ─────────┘       │
                              ├─ official Meta CUDA               ├─ masks
                              │   ├─ SAM 3 image                  ├─ overlays
                              │   └─ SAM 3.1 Object Multiplex     └─ metrics input
                              └─ public Q8 sam3.cpp

browser on laptop ── HTTP over trusted LAN ── Gradio on server ── adapters above
                 └── or SSH local-port tunnel (recommended)
```

## Why three adapters instead of one transformer doing everything

The three models have different jobs and operational strengths:

- YOLO26 is the fast closed-set baseline. Its convolutional/attention feature hierarchy and end-to-end detection head provide high throughput and strong deployment support.
- RF-DETR is the transformer detector baseline. Its DINOv2 feature encoder and query-based DETR decoder provide a different accuracy/latency trade-off and a useful independent error pattern.
- SAM 3 is the promptable segmentation and tracking system. Its detector and tracker share a vision encoder while remaining decoupled, supporting open-vocabulary concepts, geometry prompts, masks, and temporal memory.

Putting them behind a common result contract gives comparable artifacts without pretending their raw outputs are semantically identical. It also lets us fuse or cascade them later: detector proposals can become SAM box prompts, disagreement can trigger high-resolution reprocessing, and video memory can confirm weak single-frame detections.

## SAM backend policy

`official` is the default on the server:

- official Meta source pinned to commit `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`;
- authenticated `facebook/sam3/sam3.pt` for image concept and visual prompting;
- authenticated `facebook/sam3.1/sam3.1_multiplex.pt` for video;
- Python 3.12, PyTorch 2.10 CUDA 12.8 wheel, and NVIDIA GPU;
- complete supported inference API, including Object Multiplex.

`q8` is the experiment/fallback backend:

- `PABannier/sam3.cpp` pinned to commit `01832ef85fcc8eb6488f1d01cd247f07e96ff5a9`;
- public `PABannier/sam3.cpp/sam3-q8_0.ggml`, downloaded with authentication explicitly disabled;
- headless C++ bridge instead of its SDL desktop UI;
- CPU on Linux or Apple Metal selected at build time; the pinned runtime does
  not initialize its compiled GGML CUDA backend;
- text PCS, visual PVS, multimask, correction points, and memory-bank video tracking;
- no SAM 3.1 Object Multiplex.

The headless bridge decodes video through one persistent FFmpeg raw-RGB stream.
This avoids the upstream helper's fragile process-per-frame decoder and reads
each frame fully before inference. It improves I/O reliability and overhead,
but it does not turn Linux Q8 inference into GPU inference.

This distinction matters: an 8-bit community conversion is not evidence that Meta's current official video checkpoint has been quantized faithfully.

## Data contract

Each model result records:

- source path and image dimensions;
- model/runtime identity and elapsed time;
- zero or more detections with `xyxy` pixel box, score, label, class ID, instance ID, and optional mask path;
- model-specific metadata;
- errors without discarding other models' successful results.

SAM image and video use a shared manifest structure. Each frame contains detections, and each detection references a grayscale PNG mask relative to the manifest. This makes large binary masks streamable and keeps JSON readable.

## Server execution and safety

- The UI queue has concurrency `1` by default because simultaneously loading large model stacks can exhaust VRAM.
- Official video defaults to CPU-decoded-frame storage and a configurable grounding batch of `4`; `1` minimizes peak VRAM and `16` maximizes throughput.
- A positive official frame limit is aligned across Meta's inclusive tracking
  loop and exclusive detector cache, so finite tests remain internally bounded
  and do not prepare the full source video.
- Official SAM adapters release Python references and empty the CUDA cache after a completed job.
- Model, runtime, and output directories are local to this workspace and git-ignored.
- The launcher supports optional `MODEL_LAB_USER` and `MODEL_LAB_PASSWORD` authentication.
- `0.0.0.0` is only for a trusted LAN. An SSH local-port tunnel with the server bound to `127.0.0.1` is preferred.
- No Gradio public share link is enabled.

## Next architecture steps

After the standalone comparison is measured, the likely production cascade is:

1. Run tiled YOLO and RF-DETR at detector-friendly scales.
2. Fuse or retain their proposals with source-model provenance.
3. Convert selected boxes into SAM visual prompts.
4. Track confirmed masks through video with SAM 3.1 memory.
5. Re-run high-resolution crops only on small/uncertain/disagreeing regions.
6. Evaluate against labeled box, mask, and track ground truth.

An additional speed-first experiment is a sparse-discovery cascade: run full
text discovery only on keyframes, initialize EdgeTAM Q8 or SAM 2.1 Q8 from the
resulting boxes, and propagate with the tiny visual tracker between keyframes.
This can be much lighter than full SAM on every frame, but must be benchmarked
for distant-object recall, occlusion recovery, and identity switches.

That cascade spends the expensive SAM computation where it adds information while preserving the fast detectors' broad search ability.
