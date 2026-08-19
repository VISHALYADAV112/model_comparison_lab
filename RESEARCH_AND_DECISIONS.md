# Research findings and decisions

Last verified: 18 August 2026.

## Current research position

The field is converging on specialized transformer-based perception systems, not a single language model inside every camera pipeline. Large multimodal models can reason over images and video, but their language generation, broad world knowledge, token decoding, and general-purpose context machinery add cost that this task does not need. The useful abilities can be recovered more directly:

- multi-scale representations and small-target-aware training from modern detectors;
- object queries and global attention from DETR-style transformers;
- open-vocabulary concept conditioning from SAM 3's text and geometry detector;
- dense masks, temporal memory, and interactive correction from SAM's tracker;
- cross-frame confidence through track persistence rather than language reasoning;
- selective high-resolution crops/tiles instead of processing the entire frame at the largest scale.

This is why the project keeps YOLO, RF-DETR, and SAM as modular perception components.

## Latest selections

### YOLO26

Ultralytics documents YOLO26 as its January 2026 generation. It uses native end-to-end inference, a lighter DFL-free box head, and a small-target-aware label assignment component. The family covers detection, instance/semantic segmentation, pose, depth, classification, and oriented boxes. The lab uses the Large detection checkpoint as an accuracy-first baseline, not because it is universally best.

Primary source: <https://docs.ultralytics.com/models/yolo26/>

### RF-DETR

RF-DETR is the transformer detector baseline. The official repository describes detection, segmentation, and other variants, and the current package's Large configuration points at `rf-detr-large-2026.pth`. Its Apache-licensed core and fine-tuning support make it valuable as both a benchmark and a future domain-adaptation path.

Primary source: <https://github.com/roboflow/rf-detr>

### SAM 3 and SAM 3.1

Meta describes SAM 3 as an 848M-parameter detector/tracker system with a shared vision encoder, DETR-based detector conditioned on text/geometry/exemplars, and a tracker inheriting SAM 2's transformer encoder-decoder design. It supports text and visual prompts in images and videos.

SAM 3.1 was released 27 March 2026. Object Multiplex groups tracked objects into shared-memory buckets, and Meta reports about a seven-times speedup at 128 objects on one H100 versus its November 2025 SAM 3 release. It also adds optimized inference and new checkpoints.

Primary sources:

- <https://github.com/facebookresearch/sam3>
- <https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md>

## Long-video and streaming decision

The default duration-independent engine is one persistent rolling SAM 3.1
session, not reset-per-clip tracking. The pinned source already limits native
attention to seven mask-memory positions, four conditioning frames, and sixteen
object pointers. It also contains a production batching class that preserves
hot-start/generator state across smaller propagation calls. The lab supplies
the missing public runtime pieces: lazy sequential decode, compact global-frame
mapping, sparse per-frame bookkeeping, conservative state pruning, incremental
output, and a bounded RTSP frame queue.

Hugging Face exposes a supported sequential `init_video_session` workflow for
base SAM 3, confirming the persistent-session pattern. Its documentation also
warns that zero-lookahead streaming disables future-frame hot-start heuristics
and can increase false positives or duplicate tracks. The lab therefore keeps
SAM 3.1's 15-frame hot-start state across rolling windows. RTSP has a small
bounded delay; removing that delay would not be equivalent to offline
processing.

Primary sources:

- <https://huggingface.co/docs/transformers/model_doc/sam3_video>
- <https://github.com/facebookresearch/sam3/issues/481>
- <https://github.com/facebookresearch/sam3/issues/514>

A database containing only numeric SAM IDs cannot recognize a person after a
track has ended. Release 0.5.1 archives each track's best crop and reserves
fields for a verified identity and embedding. Human review can label those
records. Automatic cross-visit matching still requires a separately evaluated
face or person-ReID embedding model, thresholds, camera-domain validation, and
appropriate privacy/access controls.

## Weight-access finding

Meta's official checkpoints are gated. The user's existing Hugging Face login was verified with non-downloading dry runs against both `facebook/sam3` and `facebook/sam3.1`, so the server architecture now uses those official weights by default. A different SSH server must authenticate separately or receive an `HF_TOKEN`; this project never copies or prints the token.

The requested non-gated 8-bit file also exists as `PABannier/sam3.cpp/sam3-q8_0.ggml`. The public repository reports text PCS, point/box PVS, multimask, refinement, and memory-bank video tracking. It is a community conversion, so the lab treats it as a separate quantization experiment and pins both model filename and runtime commit.

Primary sources:

- <https://huggingface.co/facebook/sam3.1>
- <https://huggingface.co/PABannier/sam3.cpp/tree/main>
- <https://github.com/PABannier/sam3.cpp>

## What “all SAM features” means here

The playground covers the stable inference surface relevant to this project: concept text, positive/negative exemplars, positive/negative points, boxes, multimask, mask-logit refinement, image batch-style repeated jobs, text video tracking, multiple prompted objects, arbitrary-frame correction, removal, forward/backward/both propagation, offload controls, mask export, and Object Multiplex settings. An advanced persistent-session tab directly exposes the official start, add/refine, propagate, remove, reset, cancel, and close operations.

It does not claim to cover training, fine-tuning, every evaluation notebook, or SAM 3 Agent. The Agent adds an MLLM and is intentionally outside the lean perception architecture.

## Physical-limit reminder

No model can recover arbitrary detail that the sensor did not record. Once an object occupies too few independent pixels, multiple real scenes map to the same sampled image. Super-resolution can improve priors and appearance but cannot prove missing identity/detail. The path toward the physical limit is therefore:

1. improve optics, focal length, sensor sampling, exposure, stabilization, and atmospheric conditions;
2. retain the original pixels and avoid destructive video compression;
3. align and fuse multiple frames when motion/parallax permit;
4. detect on overlapping high-resolution tiles and multiple scales;
5. use model diversity and temporal consistency;
6. calibrate confidence against object pixel size and labeled range data;
7. report “insufficient evidence” below a measured resolution threshold.

The models are estimators operating under that information bound, not a way around it.
