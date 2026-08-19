# Long-video and RTSP operation

Release 0.5.0 makes continuous native SAM 3.1 tracking the default for very
long files and RTSP surveillance. The release 0.4.0 isolated-chunk engine is
still available as a fallback.

## Continuous native engine

The default `--engine continuous` path:

1. loads SAM 3.1 once;
2. decodes frames sequentially into a bounded CPU buffer;
3. advances one native Object Multiplex tracker state through every progress
   window;
4. preserves Meta's hot-start state and native object IDs across windows;
5. retains 32 frames of native state, exceeding the model's 15-prior-pointer
   and six-prior-mask-memory horizons;
6. writes masks, JSONL, track records, and video frames immediately; and
7. produces one final browser-compatible MP4 without joining video segments.

The 60-frame setting is a progress/cleanup window, not a new SAM session. It
does not require overlap frames or cross-window IoU matching. Model weights and
the tracker remain resident until the complete file/feed finishes.

Peak memory still depends on resolution, grounding batch, active objects,
model internals, and other GPU users. It should no longer grow with source
duration, but this must be confirmed on the L40S before an unattended run.
Start with grounding batch 1 and at most 16 active objects.

## Long or very-long file

Dashboard tab **5 · Long video and RTSP surveillance**:

1. Open **Long or very-long video file**.
2. Leave **Tracking engine** on **Continuous native SAM 3.1**.
3. Upload the file and enter a concept such as `vehicle`.
4. Keep the 60-frame window, batch 1, and 16-object defaults.
5. Set **Maximum chunks** to `1` for a smoke test. In continuous mode this
   means one progress window, not one independent session.
6. After checking output and VRAM, use `0` for the complete file.

For a huge file already on the Rocky Linux server, use the CLI so Gradio does
not first copy it into upload storage:

```bash
.venv/bin/model-lab long-video \
  --input /data/very-long-video.mp4 \
  --text "vehicle" \
  --engine continuous \
  --chunk-frames 60 \
  --grounding-batch-size 1 \
  --max-active-objects 16
```

Terminal progress is printed at every rolling window and at least every 25
written frames, including current and peak CUDA allocation.

## RTSP surveillance

The Rocky Linux server—not the laptop browser—connects to the RTSP endpoint.
Confirm the server can route to the camera. Keep Gradio on localhost and use
the SSH tunnel.

```bash
.venv/bin/model-lab rtsp \
  --url "rtsp://USER:PASSWORD@CAMERA/stream" \
  --text "person" \
  --engine continuous \
  --maximum-minutes 10 \
  --grounding-batch-size 1 \
  --max-active-objects 16
```

Use a finite duration first. `--maximum-minutes 0` continues until **Stop
safely** or `Ctrl-C`. Credentials and query parameters are redacted from stored
metadata.

RTSP capture runs in a producer thread with a fixed frame queue. If inference
is slower than capture, the oldest waiting frames are dropped and
`dropped_rtsp_frames` increases. This prevents unlimited latency and RAM. Each
written record includes its capture sequence and UTC timestamp. The continuous
SAM state survives across delivered frames, including after queue drops and
camera reconnects.

## Identity database and returning people

Every continuous run creates `track_identities.sqlite3` and
`identity_candidates/`. The database records:

- native SAM track ID;
- first and last observed frame;
- best mask confidence;
- best saved crop;
- an optional human-verified identity; and
- reserved embedding model/vector fields.

This is enough for a person to review and label tracks later. It is not by
itself automatic re-identification: a number such as SAM ID 27 contains no
visual information. If somebody leaves long enough for SAM to retire the
track and later receives ID 93, software needs a face embedding or person-ReID
embedding to compare ID 93's crop with the database gallery. That future
matcher must use calibrated thresholds and camera-specific validation; face
use also needs appropriate consent, access controls, and retention policy.

## Incremental outputs

The continuous run directory contains:

- `index.json`: atomic status, totals, CUDA telemetry, and rolling-state size;
- `frames.jsonl`: one committed record per delivered output frame;
- `masks/`: grayscale mask PNGs;
- `annotated.mp4`: one final H.264 video (source audio is remuxed for files);
- `track_identities.sqlite3`: track catalogue; and
- `identity_candidates/`: one best crop per observed SAM ID.

Output storage grows with retained masks/video even though RAM and VRAM are
bounded. Production surveillance needs a disk quota and retention policy.

## Isolated-chunk fallback

Use `--engine chunked` if server validation shows a leak in the pinned Meta
runtime or when a hard CUDA process-exit boundary is required:

```bash
.venv/bin/model-lab long-video \
  --input /data/very-long-video.mp4 \
  --text "vehicle" \
  --engine chunked \
  --chunk-frames 60 \
  --overlap-frames 8 \
  --grounding-batch-size 1
```

This fallback starts a fresh SAM session in each isolated process, emits
segments, and joins chunk-local IDs through overlap-box IoU. It is more robust
against runtime leaks but slower, and its identity handoff is approximate.

## Required validation

Before calling the continuous engine quality-equivalent to an ordinary
whole-video run:

1. run the same 300–1000-frame source through whole-video and continuous modes;
2. compare output frame count, native IDs, per-frame masks/boxes, and mask IoU;
3. record peak allocated/reserved VRAM after each 60-frame window;
4. verify the peak plateaus across at least ten windows; and
5. test RTSP only with a user-authorized camera and a short duration.

The implementation preserves one native forward tracker and hot-start buffer,
but exact model-weight inference cannot be validated on the development Mac.
