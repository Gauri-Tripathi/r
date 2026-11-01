# Correction and grounding experiment

For the OpenYap adapter, use [inference](INFERENCE.md) or [training](OPENYAP_FULL_TRAINING.md).

This guide covers the separate appointment-correction experiment. Accepted teacher responses are still missing.

Run local commands in WSL with `source .venv-linux/bin/activate`. Dataset: `research-v3`. GPU: one A100 80GB. Keep the original `pilot` artifacts.

## Scenarios and targets

Generate candidates:

```bash
duplex research candidates --out data/research-v3 --count 400 --seed 101
```

This produces 403 scheduling scenarios: 318 train, 32 development and 53 test. Groups and correction wording stay within splits. Use a new version if changing locked candidates.

Generate responses to `teacher-requests.jsonl` with your chosen text model. Save them as `data/research-v3/teacher-responses.jsonl` using the [import format](TEACHER_CONTRACT.md), then validate:

```bash
duplex targets --data data/research-v3/train-dev-candidates.jsonl \
  --import-jsonl data/research-v3/teacher-responses.jsonl --out data/research-v3/accepted.jsonl
duplex validate --data data/research-v3/accepted.jsonl
```

Review accepted responses before rendering. Research preparation rejects fixture targets.

## Audio preparation

The evaluation set has 12 development cases (394.88 seconds) and 12 test cases (421.76 seconds). Render user inputs with the cached tokenizer:

```bash
duplex research prepare-eval --data data/research-v3/dev-locked.jsonl \
  --root data/research-v3/eval-dev --assets data/research-v2/assets
duplex research prepare-eval --data data/research-v3/test-locked.jsonl \
  --root data/research-v3/eval-test --assets data/research-v2/assets
```

Generation receives user audio and facts as they arrive. Group members share an evaluation duration independent of answer length. Measure correction overlap against generated speech.

Download the cached codec for local preparation:

```bash
mkdir -p cache/research-assets
cp data/research-v2/assets/assets.json cache/research-assets/assets.json
cp data/research-v2/assets/tokenizer_spm_32k_3.model cache/research-assets/
modal volume get duplex-cache assets/tokenizer-e351c8d8-checkpoint125.safetensors \
  cache/research-assets/tokenizer-e351c8d8-checkpoint125.safetensors
duplex research prepare --data data/research-v3/accepted.jsonl \
  --root data/research-v3/measure --assets cache/research-assets \
  --cache cache/research-mimi-measure --max-audio-s 600 --measure-groups 1
```

`measure/prepared.json` records rendering, codec loading, cache hits and exclusions. The renderer uses local eSpeak. To import recordings, use `duplex audio-import` with isolated channels and checked word timings.

Set `AUDIO_CAP_S` from the measured preparation cost and available budget:

```bash
duplex research prepare --data data/research-v3/accepted.jsonl \
  --root data/research-v3/prepared --assets cache/research-assets \
  --cache cache/research-mimi --max-audio-s "$AUDIO_CAP_S" \
  --measurement data/research-v3/measure/prepared.json
```

Preparation allocates 80% of the duration cap to training and 20% to development, selecting whole groups. At least 64 training and four development examples must survive. Sessions over 750 frames, including the prompt, are excluded. Check exclusions and listen to the aligned channels.

Reported exposure includes silence and shared counterfactual prefixes. Record group counts and passes alongside duration.

## Budget and workload

Use current billing and the existing `runs/budget.json`. The plan includes loading, preparation, training, evaluation and shutdown, with a default cap of two passes.

Set `REMAINING_CREDIT` and `METERED_USD` from current billing, then record them and generate a plan:

```bash
duplex budget-account --ledger runs/budget.json \
  --remaining-credit "$REMAINING_CREDIT" --metered-usd "$METERED_USD"

duplex research plan --config configs/research.yaml \
  --data-manifest data/research-v3/prepared/prepared.json \
  --profile runs/audit-2026-11-01/remote/duplex-runs/pilot/profile.json \
  --ledger runs/budget.json --remaining-credit "$REMAINING_CREDIT" \
  --previous-metered-usd "$METERED_USD" \
  --dev-evaluation-lock data/research-v3/eval-dev/evaluation-lock.json \
  --test-evaluation-lock data/research-v3/eval-test/evaluation-lock.json \
  --extra-preparation-usd 0 --out runs/research-v3-plan.json
```

The historical profile is provisional. Replace it with a fresh measurement before training. Include any additional paid preparation in `--extra-preparation-usd`; do not execute a plan reporting `fits: false`.

## Base-model evaluation

Upload prepared inputs and open a budgeted shell:

```bash
modal volume put duplex-data data/research-v3 /research-v3
modal volume put duplex-runs runs/research-v3-plan.json /research-v3/workload-plan.json
duplex shell --config configs/research.yaml --ledger runs/budget.json --seconds 3300
```

In the GPU shell:

```bash
cd /opt/duplex
source .venv/bin/activate
duplex replay --assets /cache/assets --data /data/research-v3/eval-dev/rendered.jsonl \
  --root /data/research-v3/eval-dev --out /runs/research-v3/base-dev --limit 12
python -m duplex.volume_commit
exit
```

After confirming shutdown in Modal, reconcile the printed session ID locally:

```bash
duplex budget-reconcile --config configs/research.yaml --ledger runs/budget.json \
  --session SESSION_ID --confirmed-stopped
```

Transcribe the generated WAVs with word timestamps. Use the [ASR JSONL format](DATA.md#speech-evaluation) and check ambiguous transcripts by listening.

Download generations and score on CPU, then upload the report:

```bash
duplex research baseline --data data/research-v3/eval-dev/rendered.jsonl \
  --root data/research-v3/eval-dev --generations runs/research-v3/base-dev \
  --asr data/research-v3/base-dev-asr.jsonl --out runs/research-v3/baseline.json
modal volume put duplex-runs runs/research-v3/baseline.json /research-v3/baseline.json
```

Download directories with SDK `listdir`/`read_file`, as in `scripts/pull_smoke_audit.py`. If the base solves every development case, revise development difficulty before training. Keep test outcomes out of that decision.

## Profile and train

Refresh the budget and plan, then profile in a GPU shell:

```bash
cd /opt/duplex
source .venv/bin/activate
duplex profile --config configs/research.yaml --assets /cache/assets \
  --encoded /data/research-v3/prepared/encoded --data /data/research-v3/prepared/rendered.jsonl \
  --root /data/research-v3/prepared --run /runs/research-v3/train \
  --ledger /runs/budget.json --eval-examples 12
```

Check step time, separate audio/text gradients, memory and evaluation time against the reservation. Then train:

```bash
duplex train --config configs/research.yaml --assets /cache/assets \
  --encoded /data/research-v3/prepared/encoded --run /runs/research-v3/train \
  --profile /runs/research-v3/train/profile.json --baseline /runs/research-v3/baseline.json \
  --workload-plan /runs/research-v3/workload-plan.json \
  --ledger /runs/budget.json
python -m duplex.volume_commit
exit
```

Training starts a fresh adapter from the base and selects checkpoints by development loss. To resume, add `--resume /runs/research-v3/train/checkpoints/step-NNNNNN` with matching config, data and source. Confirm shutdown and reconcile before opening another shell.

## Test and export

Load the development-selected checkpoint from `best.json`. In the evaluation shell:

```bash
BEST=$(python -c 'import json; print(json.load(open("/runs/research-v3/train/best.json"))["checkpoint"])')
duplex replay --assets /cache/assets --data /data/research-v3/eval-test/rendered.jsonl \
  --root /data/research-v3/eval-test --out /runs/research-v3/base-test --limit 12
duplex replay --assets /cache/assets --data /data/research-v3/eval-test/rendered.jsonl \
  --root /data/research-v3/eval-test --out /runs/research-v3/adapted-test --limit 12 \
  --adapter "/runs/research-v3/train/checkpoints/$BEST"
python -m duplex.volume_commit
exit
```

Get hash-matched timestamped ASR for both sets. After downloading artifacts, set local `BEST` from the downloaded `best.json`, then score/export on CPU:

```bash
duplex evaluate --data data/research-v3/eval-test/rendered.jsonl --root data/research-v3/eval-test \
  --base runs/research-v3/base-test --adapted runs/research-v3/adapted-test \
  --base-asr data/research-v3/base-test-asr.jsonl --adapted-asr data/research-v3/adapted-test-asr.jsonl \
  --training-run runs/research-v3/train --budget-ledger runs/budget.json --out runs/research-v3/results
duplex export --checkpoint "runs/research-v3/train/checkpoints/$BEST" --out runs/research-v3/export
```

Compare correction success, preserved constraints, grounding, unsupported commitments and response delay. Include overlap, RTF, silence, truncation, repetition and ASR uncertainty. Listen for speech regressions; lexical scores cover only recognized content.

The export contains adapter tensors, configuration, pinned revisions, runtime requirements, attribution and checksums. Replay accepts either a checkpoint or export directory.
