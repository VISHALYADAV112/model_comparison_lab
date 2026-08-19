# Daily start and stop commands

The installation and model downloads are already complete. Do **not** rerun
`bootstrap_server.sh` or download the models every day.

This setup uses two terminals:

- Terminal 1 is logged in to the Rocky Linux server and runs the playground.
- Terminal 2 stays on the Mac and maintains the SSH tunnel.

## 1. Mac Terminal 1: connect to the server

```bash
ssh vishal@192.168.1.216
```

## 2. Server Terminal 1: activate and start

The prompt for these commands should contain `[vishal@master ...]`.

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate model-lab-bootstrap
cd "$HOME/model_comparison_lab"
.venv/bin/model-lab doctor --strict
MODEL_LAB_HOST=127.0.0.1 MODEL_LAB_PORT=7861 ./scripts/run_playground_lan.sh
```

Leave this terminal running. The final command is a web server, so it is
supposed to keep control of the terminal until you stop it.

## 3. Mac Terminal 2: create the tunnel

Open a new terminal on the Mac. Do not first SSH into the server. Its prompt
should look like `vishalyadav@Vishals-MacBook-Air`.

```bash
ssh -N -o ExitOnForwardFailure=yes -L 7861:127.0.0.1:7861 vishal@192.168.1.216
```

Enter the server password and leave this terminal running. A successful
`ssh -N` tunnel normally displays no output.

## 4. Open the playground

Open this address in the Mac browser:

```text
http://127.0.0.1:7861
```

Images and videos can be uploaded from the Mac through the playground.

## Stop everything

1. Press `Ctrl-C` in Mac Terminal 2 to close the tunnel.
2. Press `Ctrl-C` in Server Terminal 1 to stop the playground.
3. Run `exit` in Server Terminal 1 to close the SSH connection.

The environments, source code, and model files remain installed.

## Port 7861 is occupied

Inspect it without root privileges:

```bash
ss -ltnp '( sport = :7861 )'
```

The quickest solution is to use port 7862 in both places.

Server:

```bash
MODEL_LAB_HOST=127.0.0.1 MODEL_LAB_PORT=7862 ./scripts/run_playground_lan.sh
```

Mac:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 7862:127.0.0.1:7862 vishal@192.168.1.216
```

Then open `http://127.0.0.1:7862`.

## Optional checks

```bash
.venv/bin/python -m pip check
.venv/bin/model-lab models status
nvidia-smi
df -h "$HOME"
```
