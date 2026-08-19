# Quantized video tracking research

Last reviewed: 2026-08-19

## Decision

Keep `PABannier/sam3.cpp` `sam3-q8_0.ggml` as the lab's quantized,
text-prompted video backend. It is the only audited option that combines a
public checkpoint, CUDA-capable local runtime, text-based discovery, and a
memory-bank video tracker that our headless bridge already supports.

Do not describe it as SAM 3.1. It is quantized base SAM 3 and does not contain
Object Multiplex. Official SAM 3.1 remains the quality/full-feature path.

## Hugging Face audit

| Candidate | Format and target | Video status | Decision |
|---|---|---|---|
| [`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1) | Official FP32 PyTorch | Complete Object Multiplex | Keep as reference-quality backend |
| [`Sparknight/sam3.1-int8-int4-convrot`](https://huggingface.co/Sparknight/sam3.1-int8-int4-convrot) | Selective INT8/INT4 ConvRot for ComfyUI | Tracking weights exist, but the author says broad video validation is still pending | Benchmark separately; not a drop-in Meta runtime checkpoint |
| `dummy9996` / similar FP8 uploads | ComfyUI FP8 safetensors | No validated headless Meta video path | Do not integrate yet |
| [`Reza2kn/sam3.1-nvfp4-detector-no-language`](https://huggingface.co/Reza2kn/sam3.1-nvfp4-detector-no-language) | NVFP4 detector without language | Not the complete text-prompted tracker required here | Reject for this task |
| [`mlx-community/sam3.1-bf16`](https://huggingface.co/mlx-community/sam3.1-bf16) | MLX for Apple Silicon | Advertises tracking and realtime modes | Useful Mac experiment, not an NVIDIA L40S backend |
| CoreML SAM 3.1 derivatives | CoreML / Apple Neural Engine | Platform-specific and often split into detector/encoder components | Not suitable for Rocky Linux CUDA |
| [`PABannier/sam3.cpp`](https://huggingface.co/PABannier/sam3.cpp) | GGML Q8/Q4, CPU/Metal/CUDA | Base SAM 3 text and visual memory-bank tracking | Use Q8 now |

Hugging Face's model-tree metadata does not discover every community upload,
so the audit also queried the public model API by `sam3.1` and inspected model
cards and files directly.

## Why Q8 instead of Q4

Upstream publishes full SAM 3 Q8 at about 1.10 GB and Q4_0 at about 0.71 GB.
Its published Apple M4 Pro benchmark reports essentially the same tracking
latency for F16, Q8, and Q4 at 1008px. Q4 therefore saves only about 393 MB of
weight storage without demonstrated tracking speed improvement or a broad
quality evaluation. Q8 is the more conservative first benchmark.

The much smaller visual-only Q8/Q4 checkpoints remove the text encoder. They
can track a point/box-selected object, but cannot independently discover all
instances of an arbitrary text concept.

## Memory and speed controls for official SAM 3.1

Quantizing weights alone does not solve every out-of-memory error. Video frames,
vision-encoder activations, masks, temporal state, and other GPU processes can
consume more memory than the checkpoint itself. This lab therefore applies:

- decoded-frame CPU offload by default in simple and advanced official video runs;
- configurable grounding batch size: `1` minimum VRAM, `4` balanced, `16` maximum throughput;
- CUDA expandable segments in the dashboard launcher to reduce allocator fragmentation;
- one queued dashboard inference at a time;
- explicit model cleanup after a completed request.

Batch size controls a real tradeoff: smaller batches lower peak VRAM but reduce
vision-encoder throughput. CPU frame offload saves substantial VRAM on long
videos but adds host-to-device transfers.

## Best future low-memory architecture

For maximum speed with acceptable quality, benchmark a cascade rather than a
single model:

1. Run text discovery only on the first frame and sparse keyframes.
2. Initialize a tiny visual tracker from the discovered boxes/masks.
3. Propagate between keyframes with EdgeTAM Q8 or SAM 2.1 Tiny/Small Q8.
4. Re-run full SAM 3 or SAM 3.1 only after low confidence, occlusion, scene
   change, or detector/tracker disagreement.

Upstream `sam3.cpp` publishes EdgeTAM Q8 at about 20 MB and reports much lower
tracking latency than full SAM 3, but EdgeTAM has no text discovery. The hybrid
must therefore be evaluated for small/distant-object recall and identity
stability before it replaces either current backend.
