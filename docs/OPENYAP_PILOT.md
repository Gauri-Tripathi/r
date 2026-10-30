# OpenYap real-conversation feasibility pilot

This is a separate experiment from the appointment correction/live-grounding research task. It tests whether the existing PersonaPlex selective-LoRA pipeline produces any measurable held-out modeling change on real conversation. It uses no invented task labels, text teacher, TTS, voice prompt or speaker cloning.

## Completed result

The real A100 run completed **17 steps / two passes**. Mean optimizer step: **0.938 seconds**. Total measured pilot: **480.3 seconds**, including encoding, baseline, training and evaluation. All eight adapter tensors changed; frozen parameter versions stayed unchanged. Direct PCM comparison confirms all three generated speech pairs differ.

| Held-out loss | Base | Adapted | Relative change |
|---|---:|---:|---:|
| Audio | 2.83229569 | 2.83197583 | -0.01129% |
| Text | 1.52329986 | 1.52353164 | +0.01522% |

**This does not demonstrate a meaningful improvement.** The real-data training/inference pipeline works, but this tiny run produced mixed, essentially flat held-out losses. No listening or ASR quality improvement is claimed. No further paid run was started.

Modal confirms app `ap-A5zPjuNlns0TLdmoF6uudF` stopped with zero containers. Both ledgers are closed and the lease released for session `7744d959e9ac4592a7287ff8671db24d`. The conservative ledger charges 741.81 seconds including startup/confirmation; workspace metered usage increased from $6.18345250 to $6.67345261 (about **$0.49**, subject to billing finalization).

Inspect locally without GPU or network:

```powershell
.venv\Scripts\duplex.exe openyap report --run runs/openyap/pilot-v1
Start-Process .\runs\openyap\pilot-v1\comparison.html
Get-Content .\runs\openyap\pilot-v1\report.md
```

The exported adapter is `runs/openyap/pilot-v1/export/adapter.safetensors`. Existing run directories deliberately reject another paid attempt rather than overwriting completed evidence.

## Actual data

- Public dataset: `TheAgenticDataCompany/open-yap-1k`, pinned revision `a08b1c80e50159d462ddcbbabfdea5bf7ccfc626`, CC-BY-4.0. Attribution: The Agentic Data Company, Open Yap 1K (2026).
- Both sample shards downloaded locally and checked against the publisher's SHA-256 manifest. Actual sample: **8.9223 hours, 16 conversations, eight speakers**. Mixed MP3 previews are excluded.
- Real `.a.flac`/`.b.flac` tracks were verified as equal-length mono signals on a shared timeline. Selected windows are resampled to mono 24 kHz without time stretching or TTS.
- Speaker-connected components and whole conversations are assigned to train/development/test before selecting windows. No speaker crosses splits. No overlapping windows are selected within a conversation.
- First bounded selection: **34 train / six dev / seven test**, each 16 seconds. Durations: **544 / 96 / 112 session-seconds**. These are 47 actual validated windows, not all 8.9 hours of training.
- Original machine word timestamps are preserved; filler words are retained, nonlexical event tags are not fabricated as words. Windows cutting words, containing invalid timings, text/subword collisions or clipped audio reject. This strict filter biases the small subset toward slower speech; it is not representative of the whole corpus.

Artifacts: `data/openyap/source/` (downloaded sources), `data/openyap/pilot-v1/` (selected WAVs, timestamps, split manifest and exclusions). Full source shards are not uploaded to Modal; only the 144.6 MB selected subset is uploaded.

## Measured preparation and workload

A real 16-second pair cold-encoded on local CPU: **32 channel-seconds in 22.6823 wall seconds**, plus 6.2935 seconds loading. Cold throughput: 1.41079 channel-seconds/second. Full CPU encoding would take approximately 18 minutes, so the bounded GPU pilot measures actual GPU encoding before continuing.

The same rank-8, alpha-8 LoRA targets temporal output projections in layers 28–31; LR `1e-5`, accumulation four. The OpenYap branch uses a generic conversation prompt in both training and generation, instead of the appointment prompt.

There are at most **17 optimizer steps**: `floor(34 examples × 2 passes / accumulation 4)`. This is 1,088 accumulated session-seconds, at most two passes per window, rather than a blind 100-step default. Development native loss selects a checkpoint every four steps; final test outcomes do not select a checkpoint or change hyperparameters.

One A100 80GB is reserved for at most 2,400 seconds (approximately $2.72 at the conservative ledger rate including margin). Prior workspace usage is $6.1834525; the original $18 ceiling and existing cumulative ledger remain. GPU encoding must project to at most 300 seconds and leave 900 seconds for training/evaluation/shutdown, or the run stops with its partial cache retained. Training/evaluation also check the reservation deadline. No automatic retries or larger GPU substitution occur.

## Checks

The pilot records base and adapted native text/audio losses on identical held-out cached codes. It also generates three preselected test user WAVs through base and adapter with the same prompt, timing, greedy decoding and seed. Free generation receives user audio only: no assistant target speech or transcript. It saves both generated WAV sets, tensor changes, separate modality gradients, frozen-parameter version checks, actual step times, initialization, checkpoints and export.

Lower held-out loss is evidence about modeling this tiny sample. Changed output is evidence of sensitivity. Neither establishes improved conversational quality, correction handling or live grounding. Listen to both generated WAV sets; no ASR/audio-quality score is fabricated when none has been run. Results can be negative.

## Commands

```bash
source .venv-linux/bin/activate
python scripts/fetch_openyap.py --audio
duplex openyap prepare --source data/openyap/source --root data/openyap/pilot-v1 --assets cache/openyap-assets
duplex openyap encode --root data/openyap/pilot-v1 --assets cache/openyap-assets --limit 1
python scripts/openyap_modal_data.py upload
duplex shell --config configs/openyap.yaml --ledger runs/budget.json --seconds 2400 --stage openyap
```

The last command is paid and was authorized for this bounded trial. It exits automatically after encoding/training/evaluation. Preserve failures and inspect them before authorizing another attempt. After verifying actual Modal termination, reconcile the printed session ID, then download results:

```bash
duplex budget-reconcile --config configs/openyap.yaml --ledger runs/budget.json --session SESSION_ID --confirmed-stopped
python scripts/openyap_modal_data.py pull
```

Local results are under `runs/openyap/pilot-v1/`: `results.json`, `base-losses.json`, `training.jsonl`, `best.json`, base/adapted audio, checkpoints and export. `runs/openyap/pilot-plan.json` records the pre-execution bounds. Never describe a missing/incomplete result file as successful training.
