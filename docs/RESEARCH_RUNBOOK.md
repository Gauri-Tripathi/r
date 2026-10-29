# Research continuation commands

For inference with the trained adapter, see [INFERENCE.md](INFERENCE.md).

For the current **full OpenYap conversational training** request, use [OPENYAP_FULL_TRAINING.md](OPENYAP_FULL_TRAINING.md). The commands below concern the separate appointment/teacher experiment.

Use WSL from this repository: `source .venv-linux/bin/activate`. Keep `pilot` unchanged. Active data version: `research-v3`. The validated GPU remains **one A100 80GB**, with the existing selective LoRA. Paid commands below describe a future authorized run; they have not executed in this continuation.

## Offline scenarios and text teacher

Already generated locally:

```bash
duplex research candidates --out data/research-v3 --count 400 --seed 101
```

Whole groups yield 403 candidates: 318 train / 32 dev / 53 test. Observable scripts/facts, not IDs alone, define uniqueness. Groups and correction wording stay within splits. Changed candidate locks require a new version. This is narrow synthetic scheduling data with incomplete family coverage in small subsets, not a validated research audio dataset.

**Missing step:** supply `data/research-v3/teacher-responses.jsonl` from a separately selected stronger text model using `teacher-requests.jsonl` and the [exact import contract](TEACHER_CONTRACT.md). No provider/key is selected or required elsewhere. Then validate locally:

```bash
duplex targets --data data/research-v3/train-dev-candidates.jsonl \
  --import-jsonl data/research-v3/teacher-responses.jsonl --out data/research-v3/accepted.jsonl
duplex validate --data data/research-v3/accepted.jsonl
```

Review accepted wording/provenance. Fixtures cannot pass research preparation; no OpenYap conversation labels are inferred. The optional teacher HTTP command is a separate, explicitly budgeted preparation process, never part of train/resume/export/inference.

## Separate CPU audio preparation

User-only evaluation inputs already exist: 12 development cases / 394.88 session-seconds and 12 test cases / 421.76 session-seconds. Reproduce using the locally cached tokenizer:

```bash
duplex research prepare-eval --data data/research-v3/dev-locked.jsonl \
  --root data/research-v3/eval-dev --assets data/research-v2/assets
duplex research prepare-eval --data data/research-v3/test-locked.jsonl \
  --root data/research-v3/eval-test --assets data/research-v2/assets
```

Only user WAV and arrived facts enter generation. No correction transcript, oracle user state, teacher target or continuation is passed. Duration is common within each group and independent of gold answer length. Actual overlap with generated assistant speech must be measured; reference-prefix timing alone does not establish it.

For training preparation, download the existing cached codec locally (no base load). Check any transfer charges:

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

`measure/prepared.json` reports actual TTS/encoding/loading cost, cold misses, cache hits and exclusions. It is a measurement, not the research dataset. Local eSpeak produces audio only, with separate backend provenance and no API fee. Natural audio import requires licensed isolated channels, shared lengths and checked response word timings through `duplex audio-import`; it cannot bypass the research manifest gates.

Choose `AUDIO_CAP_S` from those costs and provisional old training throughput (2.55915 seconds per four-example accumulated step, 126.32 session-seconds). For orientation, a 3600-second cap might accommodate roughly 80–100 training sessions plus development data at old lengths; actual surviving counts are unknown. Reject an unaffordable preparation cost. Set the measured cap, then:

```bash
duplex research prepare --data data/research-v3/accepted.jsonl \
  --root data/research-v3/prepared --assets cache/research-assets \
  --cache cache/research-mimi --max-audio-s "$AUDIO_CAP_S" \
  --measurement data/research-v3/measure/prepared.json
```

Selection uses whole groups, allocating 80% of the cap to train and 20% to dev. Inspect exclusions, actual counts/durations, encoded hashes and audio timelines; listen to samples. At least 64 distinct training examples and four development examples must survive. Sessions exceeding 750 frames including the native prompt are rejected, not truncated through the target. No final duration/step count is claimed before measurement.

Exposure reports sum distinct scenario-session durations; shared counterfactual prefixes and silence are not deduplicated acoustic content. Report this together with independent group counts and the per-example reuse bound.

## Accounting and complete workload plan

Append current accounting; do not reset the ledger:

```bash
duplex budget-account --ledger runs/budget.json --remaining-credit 23.87 --metered-usd 6.1834525
duplex research plan --config configs/research.yaml \
  --data-manifest data/research-v3/prepared/prepared.json \
  --profile runs/audit-2026-11-01/remote/duplex-runs/pilot/profile.json \
  --ledger runs/budget.json --remaining-credit 23.87 --previous-metered-usd 6.1834525 \
  --dev-evaluation-lock data/research-v3/eval-dev/evaluation-lock.json \
  --test-evaluation-lock data/research-v3/eval-test/evaluation-lock.json \
  --extra-preparation-usd 0 --out runs/research-v3-plan.json
```

Amounts are from 2026-11-01; refresh them after additional usage. Workspace metered cost is a conservative upper bound until project attribution is known. Replace zero with additional teacher/TTS/paid preparation estimates. The original $18 ceiling and 18,000-second cap remain. Planning includes previous usage, actual evaluation durations, profiling, loading, monitoring, training, saving and idle/shutdown margins. Old throughput is provisional; confirm it with a fresh profile. Two passes bound reuse: 100 train sessions at accumulation four permit at most 50 optimizer steps. Do not execute `fits: false` or fill four hours.

## Locked audible base baseline before training

After paid execution is authorized and affordable, upload inputs and use the budget wrapper:

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

Confirm actual GPU shutdown in Modal and reconcile locally using the printed ID:

```bash
duplex budget-reconcile --config configs/research.yaml --ledger runs/budget.json \
  --session SESSION_ID --confirmed-stopped
```

Obtain ASR from the generated WAVs, using a separately selected local/imported transcription process. **No ASR engine has been selected or run.** Never transcribe from gold answers. JSONL fields: `id`, `audio_sha256`, `text`, `words:[{text,start_s,end_s}]`, `uncertainty` (0–1), `engine`, `revision`. Nonempty speech needs timestamps; hashes must match generated WAVs. Listen for ASR/TTS/runtime failures.

Download generations and score on CPU, then upload the report:

```bash
duplex research baseline --data data/research-v3/eval-dev/rendered.jsonl \
  --root data/research-v3/eval-dev --generations runs/research-v3/base-dev \
  --asr data/research-v3/base-dev-asr.jsonl --out runs/research-v3/baseline.json
modal volume put duplex-runs runs/research-v3/baseline.json /research-v3/baseline.json
```

For directory downloads, use SDK recursive `listdir`/`read_file` as in `scripts/pull_smoke_audit.py`; installed CLI directory-get was unreliable. If the base already solves all locked cases, training stops: no capability gap was shown. Review development difficulty without inspecting final-test model outcomes. Partial failures need listening review too.

## Fresh profile and bounded supervised training

Refresh accounting/plan and open another authorized budgeted shell. Execute stages separately and inspect the fresh profile before training:

```bash
cd /opt/duplex
source .venv/bin/activate
duplex profile --config configs/research.yaml --assets /cache/assets \
  --encoded /data/research-v3/prepared/encoded --data /data/research-v3/prepared/rendered.jsonl \
  --root /data/research-v3/prepared --run /runs/research-v3/train \
  --ledger /runs/budget.json --eval-examples 12
```

Check actual step/sequence time, per-modality adapter gradients, memory, generation time and remaining reservation. If affordable:

```bash
duplex train --config configs/research.yaml --assets /cache/assets \
  --encoded /data/research-v3/prepared/encoded --run /runs/research-v3/train \
  --profile /runs/research-v3/train/profile.json --baseline /runs/research-v3/baseline.json \
  --workload-plan /runs/research-v3/workload-plan.json \
  --ledger /runs/budget.json
python -m duplex.volume_commit
exit
```

Fresh adapters start from the base, not the over-reused smoke adapter. Step-zero tensors, stream delays, bounded exposure and dev monitoring are saved. Dev loss selects a checkpoint; it does not establish speech improvement. For a genuinely interrupted run, add `--resume /runs/research-v3/train/checkpoints/step-NNNNNN` to the same command. Config/data/source mismatches reject. Do not restart to bypass exposure limits. Confirm shutdown and reconcile before another allocation.

## Matched held-out speech and export

Use the dev-selected `best.json`, not a test-selected checkpoint. In an authorized evaluation shell:

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

Report correction success, unchanged constraints, grounding, unsupported commitments, response delay, actual overlap, RTF, silence, truncation, repetition and ASR uncertainty. Listen for intelligibility/prosody regressions; lexical scores are not acoustic quality. Output changes establish sensitivity only. Report unchanged/worse outcomes and small grouped-sample uncertainty honestly.

Export includes safe adapter tensors, exact configuration/revisions, runtime source/dependency requirements, attribution and checksums. Replay accepts checkpoint or export. No teacher is used at inference. Export alone does not complete evaluation; no upload/publish happens automatically.
