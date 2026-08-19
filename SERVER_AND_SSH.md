# CUDA server and SSH runbook

## 1. Server requirements

Use Linux with an NVIDIA CUDA-compatible GPU, current NVIDIA driver, Python 3.12, `git`, `cmake`, a C++ compiler, `ffmpeg`, and `ffprobe`. Meta's current official prerequisites call for PyTorch 2.7 or newer and CUDA 12.6 or newer. The bootstrap follows Meta's current example and installs PyTorch 2.10 from the CUDA 12.8 wheel index.

Example Ubuntu system packages:

```bash
sudo apt update
sudo apt install -y git cmake build-essential ffmpeg python3.12 python3.12-venv
nvidia-smi
nvcc --version
```

The official Python backend uses the NVIDIA GPU through PyTorch CUDA. The pinned Q8 C++ runtime is different: it uses CPU on Linux even when `nvcc` is installed. `SAM3_CPP_BACKEND=cpu` is therefore the only supported Q8 Linux build in this lab; Apple systems can use `metal`.

## 2. Copy the workspace

From the laptop, while in `long_range_vision`:

```bash
rsync -az --progress \
  --exclude '.venv' \
  --exclude 'models' \
  --exclude 'outputs' \
  --exclude 'runtime' \
  model_comparison_lab/ USER@SERVER_LAN_IP:~/model_comparison_lab/
```

Do not copy the Mac virtual environment or Hugging Face token files.

## 3. Authenticate and bootstrap

```bash
ssh USER@SERVER_LAN_IP
cd ~/model_comparison_lab

# If the system `hf` command is available, authenticate before bootstrap.
hf auth login
./scripts/bootstrap_server.sh
```

If `hf` is not installed globally, bootstrap without downloads, authenticate with its new environment, then download:

```bash
DOWNLOAD_MODELS=0 ./scripts/bootstrap_server.sh
.venv/bin/hf auth login
.venv/bin/model-lab models download --model all
```

The project downloads public Q8 with `token=False`, while official Meta downloads reuse the normal authenticated Hugging Face token.

## 4. Verify

```bash
.venv/bin/model-lab doctor
.venv/bin/model-lab models status
nvidia-smi
```

For strict readiness (all models, runtimes, and Python packages):

```bash
.venv/bin/model-lab doctor --strict
```

## 5. Launch

Trusted LAN:

```bash
MODEL_LAB_USER=vision \
MODEL_LAB_PASSWORD='choose-a-long-password' \
./scripts/run_playground_lan.sh
```

Find the address with `hostname -I` and open `http://SERVER_LAN_IP:7860` on the laptop. If a firewall is enabled, allow only the local subnet, not the internet. Example for a `192.168.1.0/24` LAN:

```bash
sudo ufw allow from 192.168.1.0/24 to any port 7860 proto tcp
```

Preferred SSH tunnel:

```bash
# Server terminal
MODEL_LAB_HOST=127.0.0.1 ./scripts/run_playground_lan.sh

# Laptop terminal
ssh -N -L 7860:127.0.0.1:7860 USER@SERVER_LAN_IP
```

Open `http://127.0.0.1:7860` on the laptop.

## 6. Keep it running after disconnect

```bash
tmux new -s vision-lab
cd ~/model_comparison_lab
MODEL_LAB_HOST=127.0.0.1 ./scripts/run_playground_lan.sh
```

Detach with `Ctrl-b`, then `d`. Reattach with `tmux attach -t vision-lab`.

## 7. Common failures

`401/403` for official SAM:

- Run `.venv/bin/hf auth whoami` on the server.
- Confirm the same account has accepted access for both Meta repositories.
- Do not set a new `HF_HOME` unless its token is configured too.

CUDA out of memory:

- Leave UI concurrency at one.
- Reduce SAM 3.1 `max_num_objects` in `configs/models.toml`.
- Enable video/state CPU offload in the playground.
- Start with fewer video frames.
- Stop other GPU processes after identifying them with `nvidia-smi`.
- For long recordings, use dashboard tab **Long video and RTSP surveillance**
  instead of a whole-video session. It bounds every session and exits the CUDA
  worker after each chunk.

Q8 logs `using CPU backend` on the NVIDIA server:

- This is expected for the pinned `sam3.cpp` revision; compiling GGML CUDA does
  not make that revision initialize a CUDA backend.
- Use official SAM 3.1 with the minimum-VRAM profile for GPU-accelerated video.
- Use Q8 only for a low-weight-memory comparison, and test a short frame range
  first because full-video CPU tracking can take a long time.

Browser cannot connect:

- For direct LAN, confirm the server binds `0.0.0.0`, both devices are on the same routed subnet, and the firewall allows port 7860.
- For SSH tunnel, bind the server to `127.0.0.1` and browse to the laptop's `127.0.0.1`, not the server IP.
- Never enable a public Gradio share link for private images/video.

Q8 reports `Failed to inspect video`:

- The bridge could not find `ffmpeg`/`ffprobe`. The launcher now searches the
  existing `model-lab-bootstrap` Miniforge environment automatically.
- If the tools were installed elsewhere, activate that Conda environment
  before launching and confirm `which ffmpeg` and `which ffprobe` both work.

Q8 reports `Failed to decode frame 0`:

- Version 0.3.3 replaces the upstream per-frame FFmpeg calls with one persistent
  sequential decoder, avoids the unsupported legacy `-vsync` option, and
  prints FFmpeg errors directly in the server terminal.
- Pull the update, reinstall the editable package, and rebuild the bridge; a
  Python reinstall alone does not recompile the C++ executable.

Official batch 4 now runs out of memory on a short test:

- Confirm **Quick test — first 60 frames** is selected. Version 0.3.3 restores
  a real internal Meta frame bound; earlier code could prepare the whole video
  even when the dashboard stopped after 60 returned frames.
- Restart the dashboard after an OOM, check `nvidia-smi`, and retry 60 frames
  with **Official minimum VRAM (batch 1)** before increasing the range.
- Whole-video runs may still require batch 1, fewer detected objects, an idle
  GPU. Use the bounded long-video path for the complete recording.

RTSP cannot connect or repeatedly reconnects:

- The Rocky Linux server must be able to route to the camera; the laptop's
  browser connection is irrelevant to RTSP routing.
- Confirm the URL with a short server-side FFmpeg or VLC test without posting
  credentials in logs or screenshots.
- Keep Gradio on `127.0.0.1` behind the SSH tunnel. An RTSP URL is a secret when
  it contains a username, password, or access token.
- The bounded runner uses finite OpenCV/FFmpeg open and read timeouts and the
  reconnect count configured in `configs/models.toml`.

Annotated video downloads but does not play in the dashboard:

- Version 0.3.1 encodes the result as H.264/yuv420p instead of OpenCV `mp4v`.
- Restart the dashboard after pulling the update; an already-running Python
  process continues to use the old renderer.
