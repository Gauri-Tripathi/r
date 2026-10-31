# Research status — 2026-11-01

## Current OpenYap continuation

The real OpenYap A100 pilot **completed 17 steps / two passes** on 34 unique train windows (544 seconds), with six development and seven speaker-disjoint test windows. Held-out audio loss improved 0.01129%, while text loss worsened 0.01522%: **no meaningful improvement demonstrated**. All eight adapter tensors changed, frozen parameters stayed unchanged, and three generated PCM pairs changed. Results, checkpoints, exported adapter and a listening page are downloaded under `runs/openyap/pilot-v1/`. GPU/app termination is verified, both ledgers are closed, and the lease is released. Workspace metered usage rose approximately $0.49. See [measured results and commands](docs/OPENYAP_PILOT.md).

The remainder of this status describes the preserved appointment experiment. Its historical export source hashes describe the implementation at export time; the runtime has subsequently changed for the OpenYap continuation.

The real-checkpoint run is a **smoke test**: 100 optimizer steps reused four clips 100 times. All eight LoRA tensors changed between saved steps 75 and 100. No audible improvement, adapter output sensitivity, final held-out result or stronger-teacher transfer is established. See [artifact audit](docs/SMOKE_AUDIT.md) and [machine evidence](docs/smoke-audit.json).

Original artifacts are preserved read-only under `runs/audit-2026-11-01/remote/`, with `inventory.json` checksums. The reviewed, explicitly labeled local smoke export is `runs/exports/smoke-step100-reviewed/`; its source hashes describe the implementation at that export time. An earlier export is also preserved. Historical implementation notes remain in [STATUS_HISTORY.md](docs/STATUS_HISTORY.md).

## Current implementation

- Provider-independent offline teacher contract, checked import and explicit fixture provenance. Training/resume/export do not import the optional teacher HTTP module.
- 403 candidates / 109 groups with split-specific correction wording, clean/overlap corrections, available/unavailable/unknown/stale facts and no-update/no-correction controls. Context includes original and corrected slots where applicable. Earlier versions are preserved.
- Separate audio preparation with whole-group selection, actual duration caps, exclusions, cold/cached Mimi measurements and hashed manifests. Complete sessions are rejected instead of truncating away targets.
- Locked user-only evaluation: 12 development cases / 394.88 session-seconds; 12 test cases / 421.76 session-seconds. No target continuation enters inference.
- Research training gates: checked teacher data, at least 64 distinct train examples, an audible base baseline and a baseline gap. Steps derive from measured time/exposure, capped at two effective passes by default; development loss is monitored.
- Future profiles record separate modality/adapter gradients and actual stream delays; future training saves step-zero tensors. This does not retroactively supply missing smoke-run evidence.
- Matched evaluation checks user-waveform hashes, inputs, runtime, seed, voice and decoding, with speech metrics and output sensitivity reported separately. Export includes configuration/revisions, dependency/source requirements, attribution and checksums; replay accepts exported adapters.
- Append-only billing evidence affects subsequent reservations without resetting sessions or ceilings.

## Missing execution/data

**No external teacher response JSONL has been supplied; no provider is selected.** Therefore there is no accepted stronger-teacher research audio dataset. Local eSpeak and the actual cached tokenizer rendered evaluation inputs successfully. The teacher-audio path still needs accepted data, cold Mimi measurements and listening/alignment review. No natural source audio or ordinary replay media is supplied; no OpenYap task labels are manufactured.

The locked base speech baseline, timestamped ASR/listening review, fresh real-data GPU profile, final duration/workload selection, research training, matched speech evaluation and research export remain unexecuted. [Commands and exact blockers](docs/RESEARCH_RUNBOOK.md); [teacher JSONL contract](docs/TEACHER_CONTRACT.md).

## Budget

Current user-reported balance: **$23.87**. Original ceiling: **$18**, unchanged. Closed ledger sessions charge **1,223.7463 seconds**. Modal reports **$6.1834525 workspace metered usage**; project attribution is incomplete, so it is conservatively counted. With a further $1.50 preparation/non-GPU reserve, at most **$10.3165475** remains under the ceiling before new work, subject to refreshed accounting. Teacher/TTS expenses must be budgeted separately. No four-hour fill target applies.

No paid compute, teacher/TTS API call, model upload or deployment was launched in this continuation. Final Windows suite: **54 passed, three native cases skipped**. Final Linux CPU suite: **57 passed** (63.94 seconds), including separate native audio-only/text-only gradient and context/reset tests on tiny random weights. Ruff, mypy (28 modules), secret scan and scoped security review passed. CPU tests do not establish real speech capability. [Preparation evidence](docs/research-preparation-status.json) records the current candidates, locked inputs and missing execution stages.
