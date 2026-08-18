from __future__ import annotations

from typing import Any

import gradio as gr

from ..config import LabConfig
from .service import PlaygroundService


CAPABILITIES = """
### What this playground exposes

| Area | Official Meta SAM 3 / 3.1 | Public Q8 fallback |
|---|---:|---:|
| Image text/concept prompt | Yes | Yes |
| Positive and negative exemplar boxes | Yes | Yes |
| Positive/negative points and box prompts | Yes | Yes |
| Multimask output | Yes | Yes |
| Mask-logit iterative refinement | Yes (`.npy` round trip) | No |
| Video text tracking | Yes | Yes |
| Video point/box multi-object tracking | Yes | Yes |
| Point refinement at arbitrary frames | Yes | Yes |
| Remove objects; forward/backward/both propagation | Yes | No |
| SAM 3.1 Object Multiplex | Yes | No (base SAM 3 memory bank) |

The official backend is the research-quality path and requires CUDA plus authenticated Meta weights. The Q8 backend is a community GGML conversion used to measure the memory/quality trade-off; it is not SAM 3.1 Object Multiplex.
"""


def _record_click(mode: str, positive: str, negative: str, box: str, box_state: list, evt: gr.SelectData):
    x, y = evt.index
    point = f"{x},{y}"
    if mode == "Positive point":
        positive = ";".join(filter(None, [positive.strip(), point]))
        return positive, negative, box, box_state, f"Added positive point ({x}, {y})"
    if mode == "Negative point":
        negative = ";".join(filter(None, [negative.strip(), point]))
        return positive, negative, box, box_state, f"Added negative point ({x}, {y})"
    corners = list(box_state or []) + [[x, y]]
    if len(corners) == 2:
        (x0, y0), (x1, y1) = corners
        box = f"{min(x0, x1)},{min(y0, y1)},{max(x0, x1)},{max(y0, y1)}"
        return positive, negative, box, [], f"Box set to {box}"
    return positive, negative, box, corners, "Click the opposite box corner"


def _clear_prompts():
    return "", "", "", [], "Prompts cleared"


def create_app(config: LabConfig) -> gr.Blocks:
    service = PlaygroundService(config)
    with gr.Blocks(title="YOLO · RF-DETR · SAM 3 Vision Lab") as app:
        gr.Markdown(
            "# Three-model long-range vision lab\n"
            "Compare YOLO26, RF-DETR Large 2026, and SAM 3; then explore the complete supported SAM prompt surface."
        )

        with gr.Tab("SAM 3 image"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_backend = gr.Dropdown(
                        choices=[("Official Meta CUDA", "official"), ("Public Q8 GGML", "q8")],
                        value="official",
                        label="Backend",
                    )
                    image_input = gr.Image(type="filepath", label="Image (click to add prompts)")
                    click_mode = gr.Radio(
                        ["Positive point", "Negative point", "Box corners"],
                        value="Positive point",
                        label="Canvas click action",
                    )
                    click_status = gr.Textbox(label="Prompt builder status", interactive=False)
                    clear = gr.Button("Clear visual prompts")
                    box_state = gr.State([])
                with gr.Column(scale=1):
                    image_mode = gr.Radio(["text", "visual"], value="text", label="Prompt path")
                    image_text = gr.Textbox(label="Text/concept", placeholder="all distant vehicles")
                    positive = gr.Textbox(label="Positive points", placeholder="x,y;x,y")
                    negative = gr.Textbox(label="Negative points", placeholder="x,y;x,y")
                    image_box = gr.Textbox(label="Visual box", placeholder="x0,y0,x1,y1")
                    positive_exemplars = gr.Textbox(
                        label="Positive concept exemplars", placeholder="x0,y0,x1,y1;..."
                    )
                    negative_exemplars = gr.Textbox(
                        label="Negative concept exemplars", placeholder="x0,y0,x1,y1;..."
                    )
                    multimask = gr.Checkbox(label="Return multiple visual masks")
                    mask_input = gr.File(
                        type="filepath", label="Previous low-resolution logits (.npy, official refinement)"
                    )
                    image_threshold = gr.Slider(0.01, 0.99, 0.35, step=0.01, label="Confidence threshold")
                    run_image = gr.Button("Run image segmentation", variant="primary")
            image_annotated = gr.Image(label="Annotated result")
            image_gallery = gr.Gallery(label="Overlay and individual masks", columns=4)
            with gr.Row():
                image_manifest = gr.File(label="Manifest JSON")
                image_logits = gr.File(label="Reusable low-resolution logits")
            image_json = gr.JSON(label="Raw structured result")

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

        with gr.Tab("SAM 3 video"):
            gr.Markdown(
                "One prompt/refinement per line. Visual object syntax: `id:0;p:120,80;n:140,90;b:50,40,180,220`. "
                "Refinement adds `frame:25`; removal syntax is `25:0`."
            )
            with gr.Row():
                with gr.Column():
                    video_backend = gr.Dropdown(
                        choices=[("Official SAM 3.1 Object Multiplex", "official"), ("Public SAM 3 Q8", "q8")],
                        value="official",
                        label="Backend",
                    )
                    video_input = gr.Video(label="Video", format="mp4")
                    video_mode = gr.Radio(["text", "visual"], value="text", label="Prompt path")
                    video_text = gr.Textbox(label="Text/concept")
                    video_objects = gr.Textbox(lines=4, label="Initial visual objects")
                    video_refinements = gr.Textbox(lines=4, label="Refinements")
                    video_removals = gr.Textbox(lines=2, label="Removals (official)")
                with gr.Column():
                    start_frame = gr.Number(value=0, precision=0, label="Start frame")
                    max_frames = gr.Number(
                        value=int(config.raw["playground"]["max_video_frames"]), precision=0, label="Max frames (0 = all)"
                    )
                    direction = gr.Dropdown(["forward", "backward", "both"], value="forward", label="Propagation")
                    video_threshold = gr.Slider(0.01, 0.99, 0.5, step=0.01, label="Output probability threshold")
                    offload_video = gr.Checkbox(label="Offload video frames to CPU")
                    offload_state = gr.Checkbox(label="Offload tracking state to CPU")
                    run_video = gr.Button("Run video tracking", variant="primary")
            annotated_video = gr.Video(label="Annotated tracking result")
            with gr.Row():
                video_manifest = gr.File(label="Manifest JSON")
                video_archive = gr.File(label="All masks + manifest (.zip)")
            video_json = gr.JSON(label="Raw structured result")
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
                ],
                [annotated_video, video_manifest, video_archive, video_json],
            )

        with gr.Tab("SAM 3.1 live session"):
            gr.Markdown(
                "Advanced official API control. Start one persistent Object Multiplex session, then add/refine "
                "prompts, propagate, remove objects, reset, cancel an active propagation, or close it. "
                "Session state lives in server memory and is lost when the playground restarts."
            )
            with gr.Row():
                live_video = gr.Video(label="Session video", format="mp4")
                with gr.Column():
                    live_offload_video = gr.Checkbox(label="Offload video frames to CPU")
                    live_offload_state = gr.Checkbox(label="Offload state to CPU")
                    live_start = gr.Button("Start session", variant="primary")
                    live_session_id = gr.Textbox(label="Session ID")
                    live_status = gr.JSON(label="Latest operation")
            with gr.Row():
                live_frame = gr.Number(value=0, precision=0, label="Prompt frame")
                live_object = gr.Number(value=0, precision=0, label="Object ID")
                live_text = gr.Textbox(label="Text prompt (optional)")
            with gr.Row():
                live_positive = gr.Textbox(label="Positive points", placeholder="x,y;x,y")
                live_negative = gr.Textbox(label="Negative points", placeholder="x,y;x,y")
                live_box = gr.Textbox(label="Box", placeholder="x0,y0,x1,y1")
            with gr.Row():
                live_clear_points = gr.Checkbox(value=True, label="Clear old points")
                live_clear_boxes = gr.Checkbox(value=True, label="Clear old boxes")
                live_threshold = gr.Slider(0.01, 0.99, 0.5, step=0.01, label="Output threshold")
                live_add = gr.Button("Add / refine prompt")
                live_remove = gr.Button("Remove object")
            with gr.Row():
                live_direction = gr.Dropdown(["forward", "backward", "both"], value="both", label="Propagation")
                live_prop_start = gr.Number(value=0, precision=0, label="Propagation start")
                live_max_frames = gr.Number(value=0, precision=0, label="Max frames (0 = all)")
                live_propagate = gr.Button("Propagate", variant="primary")
                live_cancel = gr.Button("Cancel propagation", variant="stop")
            with gr.Row():
                live_reset = gr.Button("Reset session")
                live_close = gr.Button("Close session")
            live_output_video = gr.Video(label="Latest propagation")
            with gr.Row():
                live_manifest = gr.File(label="Manifest")
                live_archive = gr.File(label="Masks + manifest")
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
            live_remove.click(
                service.sessions.remove,
                [live_session_id, live_frame, live_object],
                [live_status],
            )
            live_reset.click(service.sessions.reset, [live_session_id], [live_status])
            live_cancel.click(service.sessions.cancel, [live_session_id], [live_status], queue=False)
            live_close.click(service.sessions.close, [live_session_id], [live_session_id, live_status])
            live_propagate.click(
                service.sessions.propagate,
                [live_session_id, live_direction, live_prop_start, live_max_frames, live_threshold],
                [live_output_video, live_manifest, live_archive, live_result],
            )

        with gr.Tab("Three-model comparison"):
            compare_input = gr.Image(type="filepath", label="Image")
            compare_models = gr.CheckboxGroup(
                choices=[("YOLO26", "yolo"), ("RF-DETR", "rfdetr"), ("SAM 3", "sam3")],
                value=["yolo", "rfdetr", "sam3"],
                label="Models",
            )
            compare_prompt = gr.Textbox(value="object", label="SAM 3 concept prompt")
            compare_backend = gr.Radio(["official", "q8"], value="official", label="SAM backend")
            compare_run = gr.Button("Run comparison", variant="primary")
            compare_gallery = gr.Gallery(label="Per-model annotations", columns=3)
            compare_report = gr.File(label="Comparison JSON")
            compare_json = gr.JSON(label="Summary")
            compare_run.click(
                service.compare,
                [compare_input, compare_models, compare_prompt, compare_backend],
                [compare_gallery, compare_report, compare_json],
            )

        with gr.Tab("Models and diagnostics"):
            gr.Markdown(
                "Downloads are resumable. Official SAM downloads use the account configured by `hf auth login`; "
                "Q8 is deliberately downloaded without authentication to verify it remains non-gated."
            )
            status_table = gr.Dataframe(
                headers=["Component", "Ready", "Size MB", "Path"],
                datatype=["str", "bool", "number", "str"],
                interactive=False,
            )
            status_json = gr.JSON(label="Status details")
            with gr.Row():
                refresh = gr.Button("Refresh status")
                download_choice = gr.Dropdown(
                    ["all", "yolo", "rfdetr", "sam3", "sam3-official", "sam3-q8"],
                    value="sam3",
                    label="Download selection",
                )
                download = gr.Button("Download")
            refresh.click(service.status, outputs=[status_table, status_json])
            download.click(service.download, [download_choice], [status_table, status_json])
            app.load(service.status, outputs=[status_table, status_json])

        with gr.Tab("Feature map and limits"):
            gr.Markdown(CAPABILITIES)
            gr.Markdown(
                "This UI intentionally excludes SAM 3 Agent because it adds an MLLM—the extra reasoning stack you "
                "said you do not want in the perception pipeline. Training, fine-tuning, and benchmark evaluation "
                "remain command-line workflows rather than inference-playground controls."
            )
    return app
