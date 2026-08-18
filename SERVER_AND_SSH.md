# CUDA server and SSH runbook

## 1. Server requirements

Use Linux with an NVIDIA CUDA-compatible GPU, current NVIDIA driver, Python 3.12, `git`, `cmake`, a C++ compiler, and `ffmpeg`. Meta's current official prerequisites call for PyTorch 2.7 or newer and CUDA 12.6 or newer. The bootstrap follows Meta's current example and installs PyTorch 2.10 from the CUDA 12.8 wheel index.

Example Ubuntu system packages:

```bash
sudo apt update
sudo apt install -y git cmake build-essential ffmpeg python3.12 python3.12-venv
nvidia-smi
nvcc --version
```

If `nvcc` is unavailable, the official Python backend can still use a compatible prebuilt PyTorch CUDA wheel, but the Q8 C++ bridge will build CPU-only. Set `SAM3_CPP_BACKEND=cpu` explicitly in that case.

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

Q8 bridge is CPU-only:

- Install the CUDA toolkit so `nvcc` is present.
- Rebuild with `SAM3_CPP_BACKEND=cuda ./scripts/build_sam3_cpp.sh`.
- Read the CMake output and confirm `GGML_CUDA=ON`.

Browser cannot connect:

- For direct LAN, confirm the server binds `0.0.0.0`, both devices are on the same routed subnet, and the firewall allows port 7860.
- For SSH tunnel, bind the server to `127.0.0.1` and browse to the laptop's `127.0.0.1`, not the server IP.
- Never enable a public Gradio share link for private images/video.

