from __future__ import annotations

import gradio as gr

from ..config import LabConfig
from .root_service import RootPipelineService
from .service import PlaygroundService

APP_CSS = """
.gradio-container {
  max-width: 1480px !important;
  color: var(--body-text-color, #172033);
}
.vision-hero {
  border: 1px solid var(--border-color-primary, #cbd5e1);
  border-radius: 18px;
  padding: 24px 28px;
  margin-bottom: 14px;
  background: var(--background-fill-secondary, #f1f5f9);
  color: var(--body-text-color, #172033);
}
.vision-hero h1 {
  margin: 0 0 8px 0;
  color: var(--body-text-color, #172033) !important;
  font-size: 2rem;
}
.vision-hero p {
  margin: 0;
  color: var(--body-text-color-subdued, #475569) !important;
  font-size: 1.02rem;
}
.step-card {
  border: 1px solid var(--border-color-primary, #cbd5e1);
  border-radius: 14px;
  padding: 4px 16px 8px 16px;
  background: var(--background-fill-primary, #ffffff);
  color: var(--body-text-color, #172033);
}
.result-card {
  border: 1px solid var(--border-color-primary, #cbd5e1);
  border-left: 5px solid var(--color-accent, #2563eb);
  border-radius: 10px;
  padding: 6px 14px;
  background: var(--background-fill-secondary, #f1f5f9);
  color: var(--body-text-color, #172033);
}
.step-card h1, .step-card h2, .step-card h3,
.result-card h1, .result-card h2, .result-card h3,
.result-card p, .result-card li, .result-card td, .result-card th {
  color: var(--body-text-color, #172033) !important;
}
"""


CAPABILITIES = """
### Which model does what?

| Model | Best use in this lab | Important limitation |
|---|---|---|
| **YOLO26-L** | Fast detection of familiar classes such as people, cars, animals and aircraft | Closed class list; boxes rather than precise masks |
| **RF-DETR Large** | Transformer detector for familiar classes, often stronger on difficult objects | Closed class list; boxes rather than prompt-driven masks |
| **Official SAM 3** | Find a described concept and return precise object masks | Needs a useful prompt and the most GPU memory |
| **Official SAM 3.1** | Discover and track a described concept through video | Video processing can take time; use the 60-frame quick test before a full run |
| **SAM 3 Q8** | Smaller community fallback for memory/quality comparison | Quantized base SAM 3; CPU-only on Linux; not SAM 3.1 Object Multiplex |

The dashboard intentionally excludes SAM 3 Agent because that adds a general multimodal language model. This lab focuses on native detection, segmentation and tracking.
"""


def _record_click(mode: str, positive: str, negative: str, box: str, box_state: list, evt: gr.SelectData):
    x, y = evt.index
    point = f"{x},{y}"
    if mode == "Positive point (this is the object)":
        positive = ";".join(filter(None, [positive.strip(), point]))
        return positive, negative, box, box_state, f"Added positive point ({x}, {y})"
    if mode == "Negative point (exclude this area)":
        negative = ";".join(filter(None, [negative.strip(), point]))
        return positive, negative, box, box_state, f"Added negative point ({x}, {y})"
    corners = list(box_state or []) + [[x, y]]
    if len(corners) == 2:
        (x0, y0), (x1, y1) = corners
        box = f"{min(x0, x1)},{min(y0, y1)},{max(x0, x1)},{max(y0, y1)}"
        return positive, negative, box, [], f"Box ready: {box}"
    return positive, negative, box, corners, "Now click the opposite corner of the object"


def _clear_prompts():
    return "", "", "", [], "Visual prompts cleared"


def create_app(config: LabConfig) -> gr.Blocks:
    service = PlaygroundService(config)
    root_service = RootPipelineService(config)
    bounded_defaults = config.raw.get("bounded_video", {})
    with gr.Blocks(title="Long-range vision model lab") as app:
        gr.HTML(
            """
            <div class="vision-hero">
              <h1>Long-range vision model lab</h1>
              <p>Compare YOLO, RF-DETR and SAM 3 on images, then use SAM 3.1 to track objects through video.</p>
            </div>
            """
        )
        health_banner = gr.Markdown("Checking models and GPU…", elem_classes="result-card")
        gr.Markdown(
            "**New here?** Use the first tab from top to bottom. Advanced tabs are optional and can be ignored until the simple tests work."
        )

        with gr.Tabs():
            with gr.Tab("1 · Start here: compare an image"):
                gr.Markdown(
                    "## Your first experiment\n"
                    "Upload one image, describe the object you care about, and run the three models. "
                    "The first run can take longer while model weights are loaded into GPU memory."
                )
                with gr.Row():
                    with gr.Column(scale=3, elem_classes="step-card"):
                        gr.Markdown("### Step 1 — Upload an image")
                        quick_image = gr.Image(
                            type="filepath",
                            label="Input image",
                            sources=["upload", "clipboard"],
                        )
                    with gr.Column(scale=2, elem_classes="step-card"):
                        gr.Markdown("### Step 2 — Say what you want to find")
                        quick_target = gr.Textbox(
                            value="vehicle",
                            label="Object or concept",
                            placeholder="Examples: person, vehicle, aircraft, boat, animal",
                            info="SAM 3 uses this as text. YOLO and RF-DETR match it to their known COCO classes.",
                        )
                        quick_filter = gr.Checkbox(
                            value=True,
                            label="Only show YOLO/RF-DETR boxes matching this target",
                            info="Turn this off only when you want every class those detectors can recognize.",
                        )
                        gr.Examples(
                            examples=[
                                ["person"],
                                ["vehicle"],
                                ["aircraft"],
                                ["boat"],
                                ["animal"],
                            ],
                            inputs=[quick_target],
                            label="Example prompts",
                        )
                        gr.Markdown("### Step 3 — Choose models")
                        quick_models = gr.CheckboxGroup(
                            choices=[
                                ("YOLO26-L — fast detector", "yolo"),
                                ("RF-DETR Large — transformer detector", "rfdetr"),
                                ("SAM 3 — prompt-based masks", "sam3"),
                            ],
                            value=["yolo", "rfdetr", "sam3"],
                            label="Models to run",
                        )
                        with gr.Accordion("Optional: choose the SAM runtime", open=False):
                            quick_backend = gr.Radio(
                                choices=[
                                    ("Official Meta CUDA — recommended", "official"),
                                    ("Quantized Q8 — comparison fallback", "q8"),
                                ],
                                value="official",
                                label="SAM runtime",
                            )
                        quick_cascade = gr.Checkbox(
                            value=False,
                            label="Cascade mode: tile detectors + verify SAM on padded crops",
                            info=(
                                "Preserves source-resolution detail for small/distant objects instead of resizing "
                                "the whole frame once. Slower: runs each detector per tile and SAM per candidate."
                            ),
                        )
                        quick_fuse = gr.Checkbox(
                            value=True,
                            label="Fuse proposals across detectors before SAM verification",
                            info=(
                                "On: one ensemble cascade result. Off: each detector's tiled proposals are verified "
                                "and reported separately, so you can compare the detectors in cascade mode."
                            ),
                        )
                        quick_tile_sam = gr.Checkbox(
                            value=False,
                            label="Tile SAM 3 too (test baseline)",
                            info=(
                                "Also prompt SAM 3 directly on the tile grid, composited back to the full image. "
                                "Useful to measure how much SAM's whole-frame resize loses on small objects."
                            ),
                        )
                        gr.Markdown("### Step 4 — Run")
                        quick_run = gr.Button("Run image comparison", variant="primary", elem_id="quick-run")

                quick_summary = gr.Markdown(
                    "Results will appear here. You can inspect the annotated images below.",
                    elem_classes="result-card",
                )
                quick_gallery = gr.Gallery(
                    label="Annotated result from each model",
                    columns=3,
                    object_fit="contain",
                    height="auto",
                )
                quick_report = gr.File(label="Download full comparison report (JSON)")
                with gr.Accordion("Technical details (optional)", open=False):
                    quick_json = gr.JSON(label="Structured model output")
                quick_run.click(
                    service.quick_compare,
                    [
                        quick_image,
                        quick_target,
                        quick_models,
                        quick_backend,
                        quick_filter,
                        quick_cascade,
                        quick_fuse,
                        quick_tile_sam,
                    ],
                    [quick_summary, quick_gallery, quick_report, quick_json],
                )

            with gr.Tab("2 · Track an object in video"):
                gr.Markdown(
                    "## Simple video tracking\n"
                    "Upload a short video, describe the target, and choose a memory/speed profile. Official SAM 3.1 "
                    "has the strongest temporal feature set; quantized Q8 uses much less memory but this pinned "
                    "Linux runtime is CPU-only and can be very slow."
                )
                with gr.Row():
                    with gr.Column(scale=3, elem_classes="step-card"):
                        gr.Markdown("### Step 1 — Upload a video")
                        quick_video_input = gr.Video(label="Input video", format="mp4")
                    with gr.Column(scale=2, elem_classes="step-card"):
                        gr.Markdown("### Step 2 — Describe the target")
                        quick_video_target = gr.Textbox(
                            value="vehicle",
                            label="Object or concept to track",
                            placeholder="Examples: person, car, truck, animal",
                        )
                        quick_video_engine = gr.Dropdown(
                            choices=[
                                (
                                    "Official SAM 3.1 — balanced (CPU frames, batch 4)",
                                    "official_balanced",
                                ),
                                (
                                    "Official SAM 3.1 — minimum VRAM (CPU frames, batch 1)",
                                    "official_low_vram",
                                ),
                                (
                                    "Quantized SAM 3 Q8 — lowest memory, CPU-only on Linux",
                                    "q8",
                                ),
                                (
                                    "Official SAM 3.1 — maximum speed (batch 16; exclusive GPU)",
                                    "official_fast",
                                ),
                            ],
                            value="official_balanced",
                            label="Tracking engine and memory profile",
                            info="Use minimum VRAM on a shared GPU; maximum speed needs most of the L40S.",
                        )
                        gr.Markdown("### Step 3 — Choose how much video to process")
                        quick_frame_range = gr.Radio(
                            choices=[
                                ("Quick test — first 60 frames", "first_60"),
                                ("Longer test — first 300 frames", "first_300"),
                                ("Whole uploaded video — highest memory risk", "all"),
                            ],
                            value="first_60",
                            label="Output length",
                            info=(
                                "Start with 60 frames. Whole-video runs keep much more tracking state; use the "
                                "minimum-VRAM profile and a free GPU, and remember Q8 is CPU-only on Linux."
                            ),
                        )
                        with gr.Accordion("Optional quality setting", open=False):
                            quick_video_threshold = gr.Slider(
                                0.05,
                                0.95,
                                0.5,
                                step=0.05,
                                label="Mask confidence threshold",
                                info="Lower finds more objects but may add false positives.",
                            )
                        gr.Markdown("### Step 4 — Run")
                        quick_video_run = gr.Button("Track through video", variant="primary")
                quick_video_summary = gr.Markdown(
                    "The annotated video and downloads will appear below.", elem_classes="result-card"
                )
                quick_video_output = gr.Video(label="Annotated tracking result")
                with gr.Row():
                    quick_video_manifest = gr.File(label="Detection manifest (JSON)")
                    quick_video_archive = gr.File(label="All masks and results (ZIP)")
                with gr.Accordion("Technical details (optional)", open=False):
                    quick_video_json = gr.JSON(label="Structured tracking output")
                quick_video_run.click(
                    service.quick_video,
                    [
                        quick_video_input,
                        quick_video_target,
                        quick_video_engine,
                        quick_frame_range,
                        quick_video_threshold,
                    ],
                    [
                        quick_video_summary,
                        quick_video_output,
                        quick_video_manifest,
                        quick_video_archive,
                        quick_video_json,
                    ],
                )

            with gr.Tab("3 · Advanced SAM image prompts"):
                gr.Markdown(
                    "## Advanced image segmentation\n"
                    "Use this after the first image comparison works. Choose either a text description or visual "
                    "points/box. Only fill the controls for the prompt method you selected."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        image_input = gr.Image(
                            type="filepath",
                            label="Image — click this image when building visual prompts",
                        )
                        click_mode = gr.Radio(
                            [
                                "Positive point (this is the object)",
                                "Negative point (exclude this area)",
                                "Box corners (click twice)",
                            ],
                            value="Positive point (this is the object)",
                            label="What should an image click do?",
                        )
                        click_status = gr.Textbox(
                            value="Upload an image, select a click action, then click the image.",
                            label="Visual prompt status",
                            interactive=False,
                        )
                        clear = gr.Button("Clear points and box")
                        box_state = gr.State([])
                    with gr.Column(scale=2):
                        image_backend = gr.Dropdown(
                            choices=[
                                ("Official Meta CUDA — recommended", "official"),
                                ("Public Q8 GGML", "q8"),
                            ],
                            value="official",
                            label="SAM runtime",
                        )
                        image_mode = gr.Radio(
                            choices=[
                                ("Describe the concept with text", "text"),
                                ("Select it with points or a box", "visual"),
                            ],
                            value="text",
                            label="Prompt method",
                        )
                        with gr.Accordion("Text prompt controls", open=True):
                            image_text = gr.Textbox(
                                label="Concept to segment",
                                placeholder="Example: all distant vehicles",
                            )
                            positive_exemplars = gr.Textbox(
                                label="Optional positive example boxes",
                                placeholder="x0,y0,x1,y1; another box",
                                info="A box around an example that should match the concept.",
                            )
                            negative_exemplars = gr.Textbox(
                                label="Optional negative example boxes",
                                placeholder="x0,y0,x1,y1; another box",
                                info="A box around something that should not match.",
                            )
                        with gr.Accordion("Visual point and box controls", open=False):
                            positive = gr.Textbox(label="Positive points", placeholder="x,y; x,y")
                            negative = gr.Textbox(label="Negative points", placeholder="x,y; x,y")
                            image_box = gr.Textbox(label="Object box", placeholder="x0,y0,x1,y1")
                            multimask = gr.Checkbox(label="Return several candidate masks")
                        with gr.Accordion("Refinement and threshold", open=False):
                            mask_input = gr.File(
                                type="filepath",
                                label="Previous low-resolution logits (.npy, official only)",
                            )
                            image_threshold = gr.Slider(
                                0.01, 0.99, 0.35, step=0.01, label="Confidence threshold"
                            )
                        run_image = gr.Button("Run SAM image segmentation", variant="primary")
                image_annotated = gr.Image(label="Annotated result")
                image_gallery = gr.Gallery(label="Overlay and individual masks", columns=4)
                with gr.Row():
                    image_manifest = gr.File(label="Manifest JSON")
                    image_logits = gr.File(label="Reusable low-resolution logits")
                with gr.Accordion("Technical details", open=False):
                    image_json = gr.JSON(label="Structured result")

                image_input.select(
                    _record_click,
                    [click_mode, positive, negative, image_box, box_state],
                    [positive, negative, image_box, box_state, click_status],
                )
                clear.click(_clear_prompts, outputs=[positive, negative, image_box, box_state, click_status])
                run_image.click(
                    service.run_image,
                    [
                        image_backend,
                        image_input,
                        image_mode,
                        image_text,
                        positive,
                        negative,
                        image_box,
                        positive_exemplars,
                        negative_exemplars,
                        multimask,
                        mask_input,
                        image_threshold,
                    ],
                    [image_annotated, image_gallery, image_manifest, image_logits, image_json],
                )

            with gr.Tab("4 · Advanced SAM video controls"):
                gr.Markdown(
                    "## Advanced video run\n"
                    "Text mode discovers a concept automatically. Visual mode initializes explicit objects with "
                    "points or boxes. Use one object or correction per line."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        video_input = gr.Video(label="Input video", format="mp4")
                        video_backend = gr.Dropdown(
                            choices=[
                                ("Official SAM 3.1 Object Multiplex — recommended", "official"),
                                ("Public SAM 3 Q8", "q8"),
                            ],
                            value="official",
                            label="SAM runtime",
                        )
                        video_mode = gr.Radio(
                            choices=[
                                ("Discover and track a text-described concept", "text"),
                                ("Track explicitly selected objects", "visual"),
                            ],
                            value="text",
                            label="Prompt method",
                        )
                        video_text = gr.Textbox(label="Text concept", placeholder="Example: vehicle")
                    with gr.Column(scale=2):
                        with gr.Accordion("Visual objects and later corrections", open=False):
                            gr.Markdown(
                                "Initial example: `id:0;p:120,80;b:50,40,180,220`  \n"
                                "Correction example: `frame:25;id:0;p:130,85;n:170,90`  \n"
                                "Removal example: `25:0`"
                            )
                            video_objects = gr.Textbox(lines=4, label="Initial objects — one per line")
                            video_refinements = gr.Textbox(lines=4, label="Later corrections — one per line")
                            video_removals = gr.Textbox(lines=2, label="Object removals — official only")
                        start_frame = gr.Number(value=0, precision=0, label="Start frame")
                        max_frames = gr.Number(
                            value=int(config.raw["playground"]["max_video_frames"]),
                            precision=0,
                            label="Maximum frames (0 means all)",
                            info="A positive value is enforced as an exact output-frame limit.",
                        )
                        direction = gr.Dropdown(
                            choices=[
                                ("Forward from the start frame", "forward"),
                                ("Backward from the start frame", "backward"),
                                ("Both directions", "both"),
                            ],
                            value="forward",
                            label="Tracking direction",
                        )
                        video_threshold = gr.Slider(
                            0.01, 0.99, 0.5, step=0.01, label="Output probability threshold"
                        )
                        with gr.Accordion("GPU memory controls", open=False):
                            offload_video = gr.Checkbox(
                                value=True,
                                label="Store decoded video frames in CPU memory",
                                info="Recommended. It saves substantial VRAM on long videos with a small transfer cost.",
                            )
                            offload_state = gr.Checkbox(
                                value=False,
                                label="Store tracking state in CPU memory — unavailable in SAM 3.1 Multiplex",
                                info="Video-frame offload is supported; tracking-state offload is not.",
                                interactive=False,
                            )
                            grounding_batch_size = gr.Slider(
                                minimum=1,
                                maximum=16,
                                value=int(config.raw["sam3"].get("grounding_batch_size", 4)),
                                step=1,
                                label="SAM 3.1 grounding batch size",
                                info="1 uses the least VRAM; 4 is balanced; 16 is fastest only with a mostly free GPU.",
                            )
                        run_video = gr.Button("Run advanced video tracking", variant="primary")
                annotated_video = gr.Video(label="Annotated tracking result")
                with gr.Row():
                    video_manifest = gr.File(label="Manifest JSON")
                    video_archive = gr.File(label="All masks and manifest (ZIP)")
                with gr.Accordion("Technical details", open=False):
                    video_json = gr.JSON(label="Structured result")
                run_video.click(
                    service.run_video,
                    [
                        video_backend,
                        video_input,
                        video_mode,
                        video_text,
                        video_objects,
                        video_refinements,
                        video_removals,
                        start_frame,
                        max_frames,
                        direction,
                        offload_video,
                        offload_state,
                        video_threshold,
                        grounding_batch_size,
                    ],
                    [annotated_video, video_manifest, video_archive, video_json],
                )

            with gr.Tab("5 · Long video and RTSP surveillance"):
                gr.Markdown(
                    "## Continuous rolling-state SAM 3.1\n"
                    "Use this for very long recordings or an RTSP camera. The recommended engine keeps one "
                    "native SAM 3.1 session and ID space for the complete run while frames and obsolete state "
                    "are released continuously. No cross-window ID stitching is used."
                )
                with gr.Row():
                    with gr.Column(scale=3):  # noqa: SIM117 - Preserve Gradio layout hierarchy.
                        with gr.Tabs():
                            with gr.Tab("Long or very-long video file"):
                                bounded_file = gr.Video(label="Long input video", format="mp4")
                                bounded_file_target = gr.Textbox(
                                    value="vehicle",
                                    label="Object or concept to track",
                                )
                                bounded_max_chunks = gr.Number(
                                    value=0,
                                    precision=0,
                                    minimum=0,
                                    label="Maximum windows/chunks (0 processes the complete file)",
                                )
                                bounded_file_run = gr.Button(
                                    "Process long video safely", variant="primary"
                                )
                            with gr.Tab("RTSP surveillance feed"):
                                rtsp_url = gr.Textbox(
                                    type="password",
                                    label="RTSP URL",
                                    placeholder="rtsp://user:password@camera.example/stream",
                                    info="Credentials are used for the connection but are not written to manifests.",
                                )
                                rtsp_target = gr.Textbox(
                                    value="vehicle",
                                    label="Object or concept to track",
                                )
                                rtsp_maximum_minutes = gr.Number(
                                    value=float(bounded_defaults.get("rtsp_maximum_minutes", 10)),
                                    minimum=0,
                                    label="Maximum capture minutes (0 runs until Stop)",
                                    info="Use a finite value for unattended tests. Zero requires the Stop button.",
                                )
                                with gr.Row():
                                    rtsp_run = gr.Button("Start RTSP tracking", variant="primary")
                                    bounded_stop = gr.Button("Stop safely", variant="stop")
                    with gr.Column(scale=2, elem_classes="step-card"):
                        gr.Markdown("### Fixed VRAM limits")
                        bounded_engine = gr.Radio(
                            choices=[
                                ("Continuous native SAM 3.1 (recommended)", "continuous"),
                                ("Isolated chunks + ID stitching (fallback)", "chunked"),
                            ],
                            value="continuous",
                            label="Tracking engine",
                            info="Continuous loads SAM once and preserves its tracker state. The fallback restarts SAM for every clip.",
                        )
                        bounded_chunk_frames = gr.Slider(
                            minimum=30,
                            maximum=300,
                            value=int(bounded_defaults.get("chunk_frames", 60)),
                            step=1,
                            label="Frames per progress window",
                            info="Continuous mode does not reset at this boundary. In fallback mode this is the clip size.",
                        )
                        bounded_overlap_frames = gr.Slider(
                            minimum=0,
                            maximum=30,
                            value=int(bounded_defaults.get("overlap_frames", 8)),
                            step=1,
                            label="Fallback overlap frames",
                            info="Used only by the isolated-chunk fallback; continuous mode needs no overlap handoff.",
                        )
                        bounded_batch_size = gr.Slider(
                            minimum=1,
                            maximum=4,
                            value=int(bounded_defaults.get("grounding_batch_size", 1)),
                            step=1,
                            label="SAM 3.1 grounding batch size",
                            info="Batch 1 is the safest long-running profile on the L40S.",
                        )
                        bounded_max_objects = gr.Slider(
                            minimum=1,
                            maximum=64,
                            value=int(bounded_defaults.get("max_active_objects", 16)),
                            step=1,
                            label="Maximum active SAM objects",
                        )
                        bounded_threshold = gr.Slider(
                            minimum=0.05,
                            maximum=0.95,
                            value=0.5,
                            step=0.05,
                            label="Mask confidence threshold",
                        )
                        gr.Markdown(
                            "RTSP uses a bounded live-frame queue. If SAM is slower than the camera, old waiting "
                            "frames are dropped instead of allowing RAM and latency to grow without limit."
                        )
                bounded_summary = gr.Markdown(
                    "Choose a long file or RTSP feed to begin.", elem_classes="result-card"
                )
                bounded_live_preview = gr.Image(
                    label="Live annotated detection preview",
                    type="filepath",
                    interactive=False,
                )
                bounded_latest_video = gr.Video(
                    label="Completed annotated video/segment"
                )
                with gr.Row():
                    bounded_index = gr.File(label="Incremental run index")
                    bounded_frames = gr.File(label="Frame results (JSONL)")
                with gr.Accordion("Live bounded-memory telemetry", open=False):
                    bounded_json = gr.JSON(label="Current run state")

                bounded_file_run.click(
                    service.bounded.run_file,
                    [
                        bounded_file,
                        bounded_file_target,
                        bounded_chunk_frames,
                        bounded_overlap_frames,
                        bounded_batch_size,
                        bounded_max_objects,
                        bounded_threshold,
                        bounded_max_chunks,
                        bounded_engine,
                    ],
                    [
                        bounded_summary,
                        bounded_live_preview,
                        bounded_latest_video,
                        bounded_index,
                        bounded_frames,
                        bounded_json,
                    ],
                )
                rtsp_run.click(
                    service.bounded.run_rtsp,
                    [
                        rtsp_url,
                        rtsp_target,
                        bounded_chunk_frames,
                        bounded_overlap_frames,
                        bounded_batch_size,
                        bounded_max_objects,
                        bounded_threshold,
                        rtsp_maximum_minutes,
                        bounded_engine,
                    ],
                    [
                        bounded_summary,
                        bounded_live_preview,
                        bounded_latest_video,
                        bounded_index,
                        bounded_frames,
                        bounded_json,
                    ],
                )
                bounded_stop.click(
                    service.bounded.stop,
                    outputs=[bounded_summary, bounded_json],
                    queue=False,
                )

            with gr.Tab("6 · Expert: live video correction"):
                gr.Markdown(
                    "## Persistent SAM 3.1 Object Multiplex session\n"
                    "This is an expert workflow. Start a session, add a prompt, propagate, and then correct or "
                    "remove objects without reloading the video. Session state is lost when the dashboard stops."
                )
                with gr.Row():
                    live_video = gr.Video(label="Session video", format="mp4")
                    with gr.Column():
                        live_offload_video = gr.Checkbox(label="Offload video frames to CPU")
                        live_offload_state = gr.Checkbox(
                            value=False,
                            label="Offload tracking state to CPU — unavailable in SAM 3.1 Multiplex",
                            interactive=False,
                        )
                        live_start = gr.Button("1. Start session", variant="primary")
                        live_session_id = gr.Textbox(label="Active session ID")
                        live_status = gr.JSON(label="Latest operation")
                gr.Markdown("### 2. Add the first prompt or correct an existing object")
                with gr.Row():
                    live_frame = gr.Number(value=0, precision=0, label="Prompt frame")
                    live_object = gr.Number(value=0, precision=0, label="Object ID")
                    live_text = gr.Textbox(label="Text prompt (optional)")
                with gr.Row():
                    live_positive = gr.Textbox(label="Positive points", placeholder="x,y;x,y")
                    live_negative = gr.Textbox(label="Negative points", placeholder="x,y;x,y")
                    live_box = gr.Textbox(label="Box", placeholder="x0,y0,x1,y1")
                with gr.Row():
                    live_clear_points = gr.Checkbox(value=True, label="Replace old points")
                    live_clear_boxes = gr.Checkbox(value=True, label="Replace old boxes")
                    live_threshold = gr.Slider(0.01, 0.99, 0.5, step=0.01, label="Output threshold")
                    live_add = gr.Button("Add or refine prompt")
                    live_remove = gr.Button("Remove this object")
                gr.Markdown("### 3. Propagate the prompt through the video")
                with gr.Row():
                    live_direction = gr.Dropdown(
                        choices=[("Forward", "forward"), ("Backward", "backward"), ("Both", "both")],
                        value="both",
                        label="Direction",
                    )
                    live_prop_start = gr.Number(value=0, precision=0, label="Propagation start frame")
                    live_max_frames = gr.Number(
                        value=0,
                        precision=0,
                        label="Maximum frames (0 means all)",
                        info="A positive value is enforced as an exact output-frame limit.",
                    )
                    live_propagate = gr.Button("Propagate", variant="primary")
                    live_cancel = gr.Button("Cancel active propagation", variant="stop")
                with gr.Row():
                    live_reset = gr.Button("Reset all prompts")
                    live_close = gr.Button("Close and release session")
                live_output_video = gr.Video(label="Latest annotated propagation")
                with gr.Row():
                    live_manifest = gr.File(label="Manifest")
                    live_archive = gr.File(label="Masks and manifest")
                with gr.Accordion("Technical result", open=False):
                    live_result = gr.JSON(label="Propagation result")

                live_start.click(
                    service.sessions.start,
                    [live_video, live_offload_video, live_offload_state],
                    [live_session_id, live_status],
                )
                live_add.click(
                    service.sessions.add_prompt,
                    [
                        live_session_id,
                        live_frame,
                        live_object,
                        live_text,
                        live_positive,
                        live_negative,
                        live_box,
                        live_clear_points,
                        live_clear_boxes,
                        live_threshold,
                    ],
                    [live_status],
                )
                live_remove.click(service.sessions.remove, [live_session_id, live_frame, live_object], [live_status])
                live_reset.click(service.sessions.reset, [live_session_id], [live_status])
                live_cancel.click(service.sessions.cancel, [live_session_id], [live_status], queue=False)
                live_close.click(service.sessions.close, [live_session_id], [live_session_id, live_status])
                live_propagate.click(
                    service.sessions.propagate,
                    [live_session_id, live_direction, live_prop_start, live_max_frames, live_threshold],
                    [live_output_video, live_manifest, live_archive, live_result],
                )

            with gr.Tab("7 · Root pipeline lab"):
                gr.Markdown(
                    "## Independent root architecture lab\n"
                    "Drives the vendored `long_range_vision` pipeline untouched and separate from the lab's own "
                    "YOLO/RF-DETR/SAM adapters. Architecture: source-resolution tiles -> RF-DETR / Grounding DINO "
                    "proposals -> per-model NMS + weighted box fusion -> optional SAM 3 verify on padded ROI crops "
                    "-> (video) temporal tracker with optional appearance memory. Every knob below maps directly "
                    "to a stage of that pipeline."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### Stage A — spatial front end")
                        root_image = gr.Image(
                            type="filepath",
                            label="Input image (Stage A/B)",
                            sources=["upload", "clipboard"],
                        )
                        root_prompts = gr.Textbox(
                            value="person, vehicle",
                            label="Prompts (comma or line separated)",
                            info="Used for open-vocabulary matching and as the SAM verify text.",
                        )
                        root_models = gr.CheckboxGroup(
                            choices=[
                                ("RF-DETR Large — closed-set proposal", "rfdetr"),
                                ("Grounding DINO Tiny — open-vocabulary proposal", "grounding_dino"),
                                ("SAM 3 verify on padded ROI crops (downloads facebook/sam3)", "sam3_verify"),
                            ],
                            value=["rfdetr", "grounding_dino"],
                            label="Architecture stages to enable",
                        )
                        with gr.Accordion("Spatial front end knobs", open=True):
                            root_threshold = gr.Slider(
                                0.05, 0.95, 0.24, step=0.01, label="Detection threshold"
                            )
                            root_tile_size = gr.Slider(
                                256, 2048, 1008, step=16, label="Tile size (px)"
                            )
                            root_tile_overlap = gr.Slider(
                                0.0, 0.5, 0.2, step=0.05, label="Tile overlap"
                            )
                            root_nms_iou = gr.Slider(
                                0.1, 0.9, 0.45, step=0.05, label="Per-model NMS IoU"
                            )
                            root_ensemble_iou = gr.Slider(
                                0.1, 0.9, 0.5, step=0.05, label="Ensemble fusion IoU"
                            )
                            root_roi_padding = gr.Slider(
                                0.5, 4.0, 2.0, step=0.1, label="SAM ROI crop padding"
                            )
                            root_device = gr.Radio(
                                ["auto", "cuda", "cpu"], value="auto", label="Device"
                            )
                        gr.Markdown("### Stage B — run the image pipeline")
                        root_image_run = gr.Button("Run image through the root pipeline", variant="primary")
                        gr.Markdown("### Stage C — temporal tracker (video)")
                        root_video = gr.Video(label="Input video", format="mp4")
                        with gr.Accordion("Temporal tracker knobs", open=True):
                            root_detection_interval = gr.Slider(
                                1, 60, 5, step=1, label="Detection keyframe interval (frames)",
                                info="Detectors run on every Nth frame; SAM-style propagation is not used here.",
                            )
                            root_min_hits = gr.Slider(
                                1, 10, 2, step=1, label="Min hits to confirm a track"
                            )
                            root_max_missed = gr.Slider(
                                0, 10, 2, step=1, label="Max missed keyframes before retirement"
                            )
                            root_association_iou = gr.Slider(
                                0.05, 0.9, 0.2, step=0.05, label="Association IoU"
                            )
                            root_appearance_encoder = gr.Radio(
                                ["none", "histogram", "mobilenet_v3_small"],
                                value="none",
                                label="Appearance encoder",
                                info="histogram is portable; mobilenet_v3_small needs torchvision weights.",
                            )
                            root_appearance_weight = gr.Slider(
                                0.0, 1.0, 0.35, step=0.05, label="Appearance weight"
                            )
                            root_appearance_momentum = gr.Slider(
                                0.0, 0.99, 0.85, step=0.01, label="Appearance memory momentum"
                            )
                            root_appearance_batch = gr.Slider(
                                1, 256, 64, step=1, label="Appearance batch size"
                            )
                            root_appearance_roi_padding = gr.Slider(
                                0.0, 1.0, 0.35, step=0.05, label="Appearance ROI padding"
                            )
                            root_start_frame = gr.Number(value=0, precision=0, label="Start frame")
                            root_max_frames = gr.Number(
                                value=0, precision=0, label="Max frames (0 = whole video)"
                            )
                        root_video_run = gr.Button("Run video through the root tracker", variant="primary")
                    with gr.Column(scale=2):
                        root_summary = gr.Markdown(
                            "Stage A/B/C results appear here.", elem_classes="result-card"
                        )
                        root_gallery = gr.Gallery(
                            label="Stage-by-stage annotated outputs",
                            columns=3,
                            object_fit="contain",
                            height="auto",
                        )
                        root_image_json = gr.File(label="Download image pipeline report (JSON)")
                        with gr.Accordion("Image technical details", open=False):
                            root_image_payload = gr.JSON(label="Structured image result")
                        root_video_output = gr.Video(label="Annotated tracking result")
                        root_video_tracks = gr.File(label="Download tracks report (JSON)")
                        with gr.Accordion("Video technical details", open=False):
                            root_video_payload = gr.JSON(label="Structured video result")

                root_image_run.click(
                    root_service.run_image,
                    [
                        root_image,
                        root_prompts,
                        root_models,
                        root_threshold,
                        root_tile_size,
                        root_tile_overlap,
                        root_nms_iou,
                        root_ensemble_iou,
                        root_roi_padding,
                        root_device,
                    ],
                    [root_summary, root_gallery, root_image_json, root_image_payload],
                )
                root_video_run.click(
                    root_service.run_video_job,
                    [
                        root_video,
                        root_prompts,
                        root_models,
                        root_threshold,
                        root_tile_size,
                        root_tile_overlap,
                        root_nms_iou,
                        root_ensemble_iou,
                        root_roi_padding,
                        root_device,
                        root_detection_interval,
                        root_min_hits,
                        root_max_missed,
                        root_association_iou,
                        root_appearance_encoder,
                        root_appearance_weight,
                        root_appearance_momentum,
                        root_appearance_batch,
                        root_appearance_roi_padding,
                        root_start_frame,
                        root_max_frames,
                    ],
                    [root_summary, root_video_output, root_video_tracks, root_video_payload],
                )

            with gr.Tab("8 · Models and system status"):
                gr.Markdown(
                    "## Installation status\n"
                    "A green **Ready** value means that checkpoint or runtime exists. Downloads are resumable; "
                    "you normally do not need to use the download button again."
                )
                status_table = gr.Dataframe(
                    headers=["Component", "Ready", "Size MB", "Path"],
                    datatype=["str", "bool", "number", "str"],
                    interactive=False,
                )
                with gr.Row():
                    refresh = gr.Button("Refresh status")
                    download_choice = gr.Dropdown(
                        ["all", "yolo", "rfdetr", "sam3", "sam3-official", "sam3-q8"],
                        value="sam3",
                        label="Only download this selection",
                    )
                    download = gr.Button("Download missing files")
                with gr.Accordion("Detailed status", open=False):
                    status_json = gr.JSON(label="Status details")
                refresh.click(service.status, outputs=[status_table, status_json])
                download.click(service.download, [download_choice], [status_table, status_json])

            with gr.Tab("Help · model guide and limits"):
                gr.Markdown(CAPABILITIES)
                gr.Markdown(
                    "### How to interpret results\n"
                    "A larger object count does not prove that a model is more accurate. YOLO and RF-DETR return "
                    "known-class boxes, while SAM returns prompt-conditioned masks. A proper ranking needs labeled "
                    "ground truth, the same image resolution and task-appropriate box, mask and tracking metrics."
                )

        app.load(service.health, outputs=[health_banner])
        app.load(service.status, outputs=[status_table, status_json])
    return app
