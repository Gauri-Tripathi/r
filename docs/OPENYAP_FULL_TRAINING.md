# Full OpenYap training commands

The previous run used nine minutes of training audio. This mode removes the pilot's per-conversation quota and 20-minute scan cap. It scans the complete pinned **8.922-hour** public sample, uses non-overlapping 48-second source windows, and preserves the source word timings. Partial boundary words have masked text loss; causal text-piece serialization may queue by at most 240 ms. Invalid audio/annotations are excluded and counted.

The local metadata plan found **544 training windows / about 7.15 source hours**, 76 development windows / about 1.00 hour, and 56 test windows / about 0.74 hour, before waveform-quality exclusions. These are **conversation-disjoint** splits: speakers can overlap. There is no unseen-speaker claim. For speaker-disjoint evaluation, use `--split-mode speaker` with a fresh output directory; fewer hours will be available for training.

The recipe remains rank-8 selective LoRA in layers 28–31 at LR `1e-5`. One full pass is the default; approximately **136 optimizer steps** with accumulation four, derived from actual surviving examples. The last partial accumulation uses remaining examples without wrapping to the start. Development monitoring selects the checkpoint; the same held-out speech is generated before and after. More data does not guarantee improvement.

No full-data paid run has been launched by the agent. These commands are for you to execute. Existing $18 cumulative ceiling, accounting and session limits remain; the launcher refuses allocations exceeding them.

## 1. Local preparation — WSL

Run from `/mnt/c/Users/91786/Desktop/projects/r`:

```bash
source .venv-linux/bin/activate
duplex openyap full-prepare --source data/openyap/source \
  --root data/openyap/full-v1 --assets cache/openyap-assets \
  --split-mode conversation
```

The public shards and tokenizer are already downloaded. Preparation writes roughly 6 GB of isolated WAVs across all splits. It resumes completed conversations after validating hashes. Inspect `data/openyap/full-v1/prepared.json` for actual surviving counts and hours; `plan.json` alone is not waveform validation.

## 2. Upload once, then start a budgeted A100 80GB shell

```bash
python scripts/openyap_modal_data.py upload \
  --root data/openyap/full-v1 --remote openyap/full-v1
duplex shell --config configs/openyap-full.yaml \
  --ledger runs/budget.json --seconds 3300
```

Upload only once for this dataset version. Never reset the ledger or overwrite `pilot-v1`. Encoding runs separately from base-model loading. Faster 25-frame causal Mimi chunks are verified against frame-by-frame codes and an altered-future prefix check on the actual execution device. Local CPU verification passed on real audio; the GPU command verifies again before using that mode. No teacher/TTS credentials are required.

## 3. Inside the Modal shell

```bash
cd /opt/duplex
source .venv/bin/activate
duplex openyap full-session --root /data/openyap/full-v1 \
  --assets /cache/assets --run /runs/openyap/full-v1 \
  --config configs/openyap-full.yaml --ledger /runs/budget.json
python -m duplex.volume_commit
exit
```

`full-session` checkpoints codec progress, then trains when encoding is complete and enough reservation remains. It removes the pilot's 17-step/20-minute training limits. Training still has an exposure cap, a session deadline and a checkpoint reserve. It saves initialization, optimizer/RNG/sampler, adapter checkpoints, baseline, development selection, matched generation and export.

If it reports encoding/training **partial**, exit and confirm the GPU stopped. Reconcile the printed session ID locally:

```bash
duplex budget-reconcile --config configs/openyap-full.yaml \
  --ledger runs/budget.json --session SESSION_ID --confirmed-stopped
```

Then start another budgeted shell using step 2's **shell command only**, and repeat step 3. It resumes the same encoding/checkpoint automatically. Do not rerun preparation/upload or start a new experiment to bypass exposure limits. A completed result is not trained again. The exact source/config/data identity must match across resume.

## 4. Download and inspect after completion

After shutdown/reconciliation:

```bash
python scripts/openyap_modal_data.py pull \
  --remote openyap/full-v1 --download-dir runs/openyap/full-v1
duplex openyap report --run runs/openyap/full-v1
```

In Windows PowerShell, open the listening page:

```powershell
Start-Process .\runs\openyap\full-v1\comparison.html
```

Weights: `runs/openyap/full-v1/export/adapter.safetensors`. Read `results.json` and `report.md`; listen to both speech sets. Native held-out loss and changed output do not by themselves demonstrate improved conversational quality or calendar grounding.
