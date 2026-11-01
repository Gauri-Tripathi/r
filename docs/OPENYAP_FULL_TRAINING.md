# OpenYap training

The completed run used 544 train, 76 development and 56 test windows from the pinned 8.922-hour sample. Training covered about 7.15 source hours in one pass: 136 optimizer steps at accumulation four. Speakers overlap across the conversation-disjoint splits, and all test windows come from one conversation.

Rank-8 LoRA targets temporal output projections in layers 28-31, with learning rate `1e-5`. Development loss selects the checkpoint. Native test audio loss changed from 2.66712 to 2.64902; text loss changed from 1.29411 to 1.27428. Speech-quality improvement remains unverified.

The commands below prepare or resume this workflow. Use a fresh dataset/run version for a new experiment. Completed runs are not retrained.

## 1. Local preparation — WSL

Run from `/mnt/c/Users/91786/Desktop/projects/r`:

```bash
source .venv-linux/bin/activate
duplex openyap full-prepare --source data/openyap/source \
  --root data/openyap/full-v1 --assets cache/openyap-assets \
  --split-mode conversation
```

If the source files are missing, run `python scripts/fetch_openyap.py --audio` first. Preparation writes roughly 6 GB of isolated WAVs across all splits. It resumes completed conversations after validating hashes. Inspect `data/openyap/full-v1/prepared.json` for actual surviving counts and hours; `plan.json` alone is not waveform validation.

## 2. Upload once, then start a budgeted A100 80GB shell

```bash
python scripts/openyap_modal_data.py upload \
  --root data/openyap/full-v1 --remote openyap/full-v1
duplex shell --config configs/openyap-full.yaml \
  --ledger runs/budget.json --seconds 3300
```

Upload only once for this dataset version. Never reset the ledger or overwrite `pilot-v1`. Encoding runs separately from base-model loading. Faster 25-frame causal Mimi chunks are verified against frame-by-frame codes and an altered-future prefix check on the actual execution device. Local CPU verification passed on real audio; the GPU command verifies again before using that mode.

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

`full-session` checkpoints codec progress, then trains when encoding is complete and enough reservation remains. Training still has an exposure cap, a session deadline and a checkpoint reserve. It saves initialization, optimizer/RNG/sampler, adapter checkpoints, baseline, development selection, matched generation and export.

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
