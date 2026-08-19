# Long-video and RTSP operation

Release 0.4.0 adds a duration-independent SAM 3.1 path for very long files and
RTSP surveillance feeds. It is separate from the ordinary whole-video tab.

## Memory boundary

The bounded path never gives SAM 3.1 the complete long source. It:

1. decodes one fixed-size clip on CPU;
2. retains only the configured overlap frames;
3. starts one official SAM 3.1 session for that clip;
4. writes masks, telemetry, JSONL records, and an annotated MP4 segment;
5. closes the session and exits the isolated CUDA worker process; and
6. starts the next clip.

Worker-process exit is the hard CUDA cleanup boundary. Model weights are
therefore reloaded for every chunk. This is slower than keeping one predictor
alive, but it prevents stale Python or Meta-runtime tensor references from
accumulating across an unattended run.

Peak VRAM depends on the fixed chunk, grounding batch, active-object limit,
resolution, and other GPU users. It does not depend on the total recording
duration. The defaults are deliberately conservative for the L40S:

- 60 total frames per SAM session;
- 8 overlap frames;
- grounding batch 1;
- at most 16 active objects; and
- one isolated CUDA worker at a time.

Do not increase multiple limits at once. Read `cuda_peak_allocated_mb` and
`cuda_peak_reserved_mb` in `chunks.jsonl`, keep substantial headroom, and
check other GPU processes with `nvidia-smi`.

## Long or very-long file

In dashboard tab **5 · Long video and RTSP surveillance**:

1. Open **Long or very-long video file**.
2. Upload the file and enter a text concept such as `vehicle`.
3. Leave the fixed VRAM limits at their safe defaults for the first run.
4. Set **Maximum chunks** to `1` for a smoke test or `0` for the complete file.
5. Select **Process long video safely**.

The dashboard updates after every completed chunk. It shows the newest
annotated segment instead of constructing one ever-growing in-memory video.
Browser uploads copy the source into Gradio's upload storage first. For a huge
file that is already on the Rocky Linux server, prefer the CLI so it can read
that file in place.

The equivalent CLI command is:

```bash
.venv/bin/model-lab long-video \
  --input /data/very-long-video.mp4 \
  --text "vehicle" \
  --chunk-frames 60 \
  --overlap-frames 8 \
  --grounding-batch-size 1 \
  --max-active-objects 16
```

## RTSP surveillance feed

The server, not the laptop browser, connects to the RTSP endpoint. Confirm the
Rocky Linux server can route to the camera. Keep Gradio bound to localhost and
use the existing SSH tunnel.

In the RTSP section:

1. Enter the complete `rtsp://` or `rtsps://` URL.
2. Enter the concept to track.
3. Use a short maximum duration for the first test.
4. Select **Start RTSP tracking**.
5. Use **Stop safely** for a run whose maximum duration is `0`.

Credentials and URL query parameters are not written to manifests. The
dashboard field masks the URL. Avoid pasting the URL into issue reports or
terminal screenshots.

RTSP capture runs concurrently with inference but its waiting queue is fixed
at two chunks by default. If SAM is slower than the camera, the oldest pending
chunk is deleted and `dropped_rtsp_chunks` increases. This creates an explicit
coverage gap instead of unbounded RAM, disk queue, and latency growth. Each
saved frame record includes an approximate UTC capture timestamp.

The CLI form is:

```bash
.venv/bin/model-lab rtsp \
  --url "rtsp://USER:PASSWORD@CAMERA/stream" \
  --text "person" \
  --maximum-minutes 10 \
  --chunk-frames 60 \
  --overlap-frames 8 \
  --grounding-batch-size 1
```

The capture reconnects a limited number of times after a read failure. Change
queue and reconnect defaults in `configs/models.toml`.

## Identity handoff

SAM object IDs are local to one finite session. The bounded runner assigns
global IDs by comparing boxes on matching overlap frames, then retains only a
small time-limited CPU registry. The original chunk ID is preserved as
`chunk_instance_id`.

This keeps state bounded, but identity can change after a long disappearance,
severe occlusion, camera cut, or a dropped RTSP chunk. Reliable forensic
identity across those events requires an evaluated appearance/re-identification
stage and labeled tracking data; it should not be inferred from SAM IDs alone.

## Incremental outputs

Each run creates a directory under `outputs/playground/` containing:

- `index.json`: atomically updated run status and totals;
- `frames.jsonl`: one committed record per unique frame;
- `chunks.jsonl`: per-chunk paths, timings, and CUDA peaks;
- `segments/`: browser-compatible annotated MP4 clips;
- `chunks/`: per-chunk manifests and masks; and
- `worker_config.json`: non-secret local runtime configuration for isolated
workers.

An isolated worker also has a finite timeout (30 minutes by default). A timed
out process is killed and its CUDA context is reclaimed; change the timeout in
`configs/models.toml` only after measuring legitimate slow chunks.

Temporary decoded input chunks are deleted after successful processing. Output
storage still grows with retained masks and annotated segments, so continuous
production surveillance also needs a site-specific disk retention policy.
