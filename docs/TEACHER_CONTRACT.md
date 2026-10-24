# Offline teacher response contract

The generator is deterministic Python. A text teacher is any separately chosen stronger text LLM used **only in data preparation**. No provider is selected by this repository. Local generation and externally supplied JSONL are supported; an optional explicit HTTP preparation command remains available. TTS produces waveforms and never supplies reasoning labels. PersonaPlex plus LoRA is the student and the inference model.

`duplex research candidates --out data/research-v3` writes scenario JSONL, split/group locks and `teacher-requests.jsonl`. Each request includes conversation turns, timestamped simulator facts and expected final state. Expected state is for offline target creation/checking, never the runtime input.

Generate one response per train/development scenario externally. Return JSONL in this exact versioned schema:

```json
{"schema_version":"duplex.teacher-response.v1","id":"COPY_REQUEST_ID","scenario_sha256":"COPY_64_HEX_REQUEST_HASH","prompt_version":"scenario-verbalization-v2","teacher":{"model":"YOUR_MODEL_ID","revision":"EXACT_REVISION","access":"external"},"text":"DESIRED_ASSISTANT_RESPONSE"}
```

`access` may be `external`, `local` or `api`. IDs, hashes and prompt version must match exported requests. Import requires all IDs in the supplied scenario manifest exactly once. To import a subset, first supply a scenario manifest containing exactly that subset, keeping related variants together. Unknown/duplicate IDs, changed scenarios, extra fields, control tokens and responses failing the deterministic state/constraint checks are rejected. These lexical checks do not prove teacher strength, naturalness or complete semantic correctness; review accepted examples before rendering.

```bash
duplex targets --data data/research-v3/train-dev-candidates.jsonl \
  --import-jsonl data/research-v3/teacher-responses.jsonl \
  --out data/research-v3/accepted.jsonl
```

`accepted.jsonl` is the validated scenario dataset with response/provenance attached. `accepted.accepted.jsonl` stores accepted versioned teacher records. Model, revision, prompt version, scenario hash, audio hashes and encoded hashes bind subsequent stages. Reference fixtures are labeled `fixture_reference`; legacy `{id,text}` imports remain usable for compatibility checks but cannot pass research preparation.

The optional `targets --teacher-config ... --cache ...` command is a separate preparation process with an explicit endpoint/model/revision and bounded API budget. Its example config is deliberately unusable until selected. It is not needed for external imports. No train/resume/export command imports `duplex.teacher`, calls a teacher or needs `TEACHER_API_KEY`. Model download authentication (`HF_TOKEN`) is a separate concern. Prepared files work without teacher network access.

No API key belongs in JSONL, YAML, checkpoints, logs or W&B. Identity metadata accepts bounded model/revision identifiers, not endpoint URLs or config objects. Do not place secrets in free-form teacher response text. The current audio backend is local eSpeak, with no API credential or per-request TTS fee. A different TTS provider/backend has not been implemented or selected; do not describe the generic teacher HTTP adapter as a TTS backend.

The student objective is sequence-level distillation via supervised assistant text/audio targets and native masks. It uses no teacher logits, teacher model in GPU memory, or teacher calls at training/inference time.
