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
3. Start with 30–60 frames.
4. Select **Track through video**.

The output is an annotated MP4. The ZIP contains every mask plus the manifest.
After a short test works, increase the frame limit or use the advanced video
tab.

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

## Tab 5: live video correction

This is an expert session workflow. Start a session, add a prompt, propagate,
then refine or remove objects without loading the video again. Close the
session when finished to release GPU memory.

## Tab 6: models and system

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

## Reading the comparison correctly

A larger detection count does not automatically mean better accuracy. YOLO and
RF-DETR produce known-class boxes, while SAM produces prompt-conditioned masks.
A scientific ranking requires labeled ground truth and box, mask, and tracking
metrics measured under the same resolution and threshold policy.
