# Dashboard guide

The dashboard is ordered from simplest to most advanced. For the first test,
use only tab **1**. You do not need to understand the other tabs yet.

## Tab 1: compare an image

1. Upload an image from the laptop.
2. Enter a useful SAM concept such as `person`, `vehicle`, `aircraft`, `boat`,
   or `animal`.
3. Leave all three models selected.
4. Leave the SAM runtime on **Official Meta CUDA**.
5. Select **Run image comparison**.

The first run may take longer because each model is loaded into GPU memory.
The result summary reports object or mask count, inference time, and the
smallest detected box side. The three annotated images show what each model
found. Raw JSON is kept inside the optional technical-details section.

By default, the target applies to all three results. SAM 3 uses the text
directly. YOLO and RF-DETR first detect their known COCO classes, then the
dashboard keeps matching classes. For example, `person` keeps only `person`;
`vehicle` maps to bicycle, car, motorcycle, airplane, bus, train, truck, and
boat. Clear **Only show YOLO/RF-DETR boxes matching this target** when you
deliberately want to inspect every known class.

## Tab 2: track an object in video

1. Upload an MP4 video.
2. Describe the object type, for example `vehicle`.
3. Choose a tracking engine/profile:
   - **Official balanced** uses CPU frame storage and a four-frame grounding batch.
   - **Official minimum VRAM** uses one grounding frame at a time.
   - **Quantized Q8** uses base SAM 3's memory-bank tracker with low weight
     memory, but the pinned Linux runtime is CPU-only and can be very slow.
   - **Official maximum speed** requires a mostly free GPU.
4. The safe default is **Quick test — first 60 frames**. Use it to check the
   prompt and memory profile before selecting 300 frames or the whole video.
5. Select **Track through video**.

The output is a browser-compatible H.264 annotated MP4. The ZIP contains every
mask plus the manifest. A 60-frame result is only two seconds at 30 FPS; this
was the cause of earlier short outputs. Choose **Whole uploaded video** to
process the entire input. In the advanced tab, a positive frame limit
is the exact maximum number of unique output frames and `0` means the full
propagation range.
The quantized option is not SAM 3.1 and cannot be used to claim Object
Multiplex quality or speed. On the Linux server, use a short Q8 test first;
use an official minimum-VRAM profile when you need NVIDIA GPU acceleration.

Version 0.3.3 keeps finite official runs bounded inside Meta's detector as
well as at the dashboard output. Earlier version 0.2.5 avoided a final-frame
bug by leaving Meta's internal range unbounded; that could make a 60-frame
request prepare the whole uploaded video. Whole-video tracking can still
exceed VRAM because temporal state grows with video length and object count.

## Tab 3: advanced SAM image prompts

Use this tab for SAM-only experiments.

- **Text prompt:** describe a concept and optionally add positive or negative
  example boxes.
- **Visual prompt:** select a click action and click the uploaded image. A
  positive point marks the target; a negative point excludes an area. Select
  box mode and click two opposite corners to build a box.
- **Previous logits:** feed the `.npy` result from an earlier official run back
  into SAM for iterative refinement.

Only fill the section that matches the selected prompt method.

## Tab 4: advanced SAM video

Text mode is still the easiest. Visual mode supports explicit objects and later
corrections. The syntax examples are displayed next to the fields. CPU offload
reduces GPU memory use but can make processing slower.
The grounding batch slider applies to official SAM 3.1 only: use `1` for the
lowest peak VRAM, `4` for the balanced default, and `16` only on an otherwise
free high-memory GPU.

## Tab 5: long video and RTSP surveillance

Use this tab instead of whole-video mode for a very long recording or camera
feed. Both paths process fixed overlapping chunks, save results incrementally,
and run every chunk in an isolated CUDA worker. Worker exit is the hard VRAM
cleanup boundary.

Start with 60 frames, 8 overlap frames, grounding batch 1, and 16 active
objects. For a long file, use one maximum chunk as a smoke test before changing
the value to `0`. For RTSP, use a short maximum duration first; `0` runs until
**Stop safely** is selected.

The latest annotated segment replaces the previous preview after every chunk.
`index.json` contains current totals, `frames.jsonl` is written one unique
frame at a time, and `chunks.jsonl` inside the output directory contains peak
CUDA measurements.

The RTSP connection originates from the server. Its queue holds at most two
pending chunks. When inference is slower than capture, old pending chunks are
dropped deliberately and the dashboard reports the count. Camera credentials
and query parameters are redacted from stored metadata.

Read [LONG_VIDEO_AND_RTSP.md](LONG_VIDEO_AND_RTSP.md) for output layout,
identity-handoff limits, reconnect behavior, and CLI examples.

## Tab 6: live video correction

This is an expert session workflow. Start a session, add a prompt, propagate,
then refine or remove objects without loading the video again. Close the
session when finished to release GPU memory.

## Tab 7: models and system

This confirms that every checkpoint and runtime exists. Do not download files
again when every row says **Ready**.

## `No module named pkg_resources`

Setuptools 82 removed the legacy module that the pinned Meta SAM source still
imports. Version 0.2 of this lab prevents that incompatible upgrade. Repair an
existing environment once with:

```bash
.venv/bin/python -m pip install --upgrade "setuptools<82" -e ".[all]"
.venv/bin/python -m pip check
.venv/bin/model-lab doctor --strict
```

## Missing `einops` or `pycocotools`

Meta's pinned SAM source imports `einops`, `pycocotools`, and `psutil` from
core image/video modules but does not list all of them as base dependencies.
Version 0.2.2 of this lab installs them directly and checks the full
image-processor import before starting the dashboard. Repair an older
environment after pulling the latest code with:

```bash
.venv/bin/python -m pip install -e ".[all]"
.venv/bin/python -c 'from sam3.model.sam3_image_processor import Sam3Processor; print("SAM image imports: OK")'
.venv/bin/model-lab doctor --strict
```

## `Got unsupported ScalarType BFloat16`

SAM 3 intentionally runs CUDA inference in BF16 on supported NVIDIA GPUs.
NumPy cannot represent BF16 directly, so version 0.2.3 converts only returned
floating-point tensors to FP32 before writing masks and JSON. GPU inference
itself remains mixed precision.

## SAM 3.1 video session option errors

The pinned SAM 3.1 Multiplex model accepts video-frame CPU offload but not
tracking-state CPU offload. Version 0.2.4 filters session arguments against
the installed model signature and disables the unsupported dashboard option.
It also accepts empty optional Gradio fields as empty lists instead of raising
`NoneType` errors.

## SAM 3.1 `expanded size of the tensor` video error

Meta's pinned SAM 3.1 source uses an inclusive propagation endpoint but an
exclusive batched-grounding endpoint when a finite internal frame window is
sent. The last requested frame can therefore contain an empty feature tensor.
Version 0.2.5 temporarily left Meta's internal window unbounded, which avoided
the empty final frame but could prepare the whole video and increase memory.
Version 0.3.3 instead aligns the propagation and detector bounds, preserving an
exact, internally bounded dashboard and CLI output limit.

## Reading the comparison correctly

A larger detection count does not automatically mean better accuracy. YOLO and
RF-DETR produce known-class boxes, while SAM produces prompt-conditioned masks.
A scientific ranking requires labeled ground truth and box, mask, and tracking
metrics measured under the same resolution and threshold policy.
