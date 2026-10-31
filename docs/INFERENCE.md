# Inference with the trained OpenYap adapter

`duplex infer` loads pinned PersonaPlex and the exact saved LoRA configuration. It accepts an export directory, complete checkpoint directory, or training-run directory containing `best.json`. It validates checksums and base revision, then uses the saved conversation prompt. No teacher, ASR service, or teacher key is needed.

This is recorded-audio inference through the causal duplex runtime. The microphone helper records a WAV first; it is **not a live microphone/websocket chat server**. Generated speech may overlap the input. A default 12-second silence tail gives the model time to answer after you stop speaking. Runtime speed depends on hardware; the validated configuration remains one A100 80GB.

## Optional: record your microphone in Windows PowerShell

Run locally, not in the GPU shell:

```powershell
uv pip install --python .venv/Scripts/python.exe sounddevice
.venv\Scripts\python.exe scripts/record_microphone.py --seconds 8 --out input.wav
```

An existing WAV can be used instead. Stereo inputs are downmixed, resampled to mono 24 kHz, and attenuated if necessary for peak headroom. The original input hash, applied gain and silence tail are recorded. To choose a microphone, use `--list-devices`, then `--device DEVICE_NUMBER`.

## Upload input and open/reuse the GPU shell — WSL

```bash
source .venv-linux/bin/activate
modal volume put --force duplex-data input.wav /inference/input.wav
duplex shell --config configs/openyap-full.yaml --ledger runs/budget.json --seconds 2400
```

If an earlier reservation remains active, confirm its Modal allocation stopped and reconcile that printed session before launching another. You can reuse an already-running budgeted GPU shell if sufficient time remains. No new allocation was launched while implementing these commands.

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

For base-model comparison use `--base-model` instead of `--adapter ...`, and a fresh output directory. Use another output name for another attempt; completed inference is not overwritten. Inputs are limited to 60 seconds, with a separately bounded response tail. Only user audio enters generation—never the recorded assistant channel or expected continuation. Printed internal model text is labeled as a proxy, not an ASR transcript of the generated WAV.

## Download and listen — local WSL

```bash
modal volume get duplex-runs inference/demo-001/generated.wav response.wav
modal volume get duplex-runs inference/demo-001/inference.json response-metadata.json
```

In Windows PowerShell:

```powershell
Start-Process .\response.wav
```

Download individual files as shown, rather than passing a directory to the CLI's file download. After verifying the GPU stopped, close the printed session locally:

```bash
duplex budget-reconcile --config configs/openyap-full.yaml \
  --ledger runs/budget.json --session SESSION_ID --confirmed-stopped
```

The adapter was trained on human conversation. It is not a verified tool-use/calendar agent. These inference commands do not establish audio quality or correctness; inspect the output yourself.
