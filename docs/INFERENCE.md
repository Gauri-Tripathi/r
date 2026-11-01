# Inference with the trained OpenYap adapter

`duplex infer` accepts an export, checkpoint, or run directory containing `best.json`. It checks the saved adapter and base revision, then loads PersonaPlex with the recorded configuration and conversation prompt.

Input is a recorded WAV. The microphone helper records first; generation runs afterward. Assistant speech can overlap the input. A 12-second silence tail allows time for the response. Use one A100 80GB.

## Record a WAV (Windows)

Run locally, not in the GPU shell:

```powershell
uv pip install --python .venv/Scripts/python.exe sounddevice
.venv\Scripts\python.exe scripts/record_microphone.py --seconds 8 --out input.wav
```

An existing WAV can be used instead. Stereo inputs are downmixed, resampled to mono 24 kHz, and attenuated if necessary for peak headroom. The original input hash, applied gain and silence tail are recorded. To choose a microphone, use `--list-devices`, then `--device DEVICE_NUMBER`.

## Upload and open a GPU shell (WSL)

```bash
source .venv-linux/bin/activate
modal volume put --force duplex-data input.wav /inference/input.wav
duplex shell --config configs/openyap-full.yaml --ledger runs/budget.json --seconds 2400
```

Reconcile any stopped session whose reservation is still active before opening another shell. An existing shell can be reused if enough time remains.

## Generate inside the Modal shell

```bash
cd /opt/duplex
source .venv/bin/activate
duplex infer --assets /cache/assets \
  --adapter /runs/openyap/full-v1/export \
  --input /data/inference/input.wav \
  --out /runs/inference/demo-001 --response-seconds 12
python -m duplex.volume_commit
exit
```

For a base comparison, replace `--adapter ...` with `--base-model` and use a fresh output directory. Input is limited to 60 seconds plus the response tail. Printed model text is an internal-token diagnostic; listen to `generated.wav` for the response.

## Download and listen

```bash
modal volume get duplex-runs inference/demo-001/generated.wav response.wav
modal volume get duplex-runs inference/demo-001/inference.json response-metadata.json
```

In Windows PowerShell:

```powershell
Start-Process .\response.wav
```

Download files individually. After verifying GPU shutdown, reconcile the printed session ID locally:

```bash
duplex budget-reconcile --config configs/openyap-full.yaml \
  --ledger runs/budget.json --session SESSION_ID --confirmed-stopped
```

The adapter was trained on human conversation; correction and tool-use performance have not been established.
