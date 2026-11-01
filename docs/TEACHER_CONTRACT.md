# Teacher responses

The teacher writes assistant responses during data preparation. Choose a text model externally, then import its outputs. Training and inference use PersonaPlex with LoRA and do not call the teacher.

## Generate requests

```bash
duplex research candidates --out data/research-v3
```

`teacher-requests.jsonl` contains conversation turns, timestamped facts and simulator state for each scenario. Return one response per train/development request:

```json
{
  "schema_version": "duplex.teacher-response.v1",
  "id": "COPY_REQUEST_ID",
  "scenario_sha256": "COPY_64_HEX_REQUEST_HASH",
  "prompt_version": "scenario-verbalization-v2",
  "teacher": {
    "model": "YOUR_MODEL_ID",
    "revision": "EXACT_REVISION",
    "access": "external"
  },
  "text": "DESIRED_ASSISTANT_RESPONSE"
}
```

`access` accepts `external`, `local` or `api`. IDs, hashes and prompt version must match the requests. Every ID in the supplied manifest must appear exactly once. For a subset, supply a matching scenario manifest and keep counterfactual groups together.

## Import and check

```bash
duplex targets --data data/research-v3/train-dev-candidates.jsonl \
  --import-jsonl data/research-v3/teacher-responses.jsonl \
  --out data/research-v3/accepted.jsonl
```

The importer rejects duplicate or unknown IDs, changed scenarios, extra fields, control tokens and responses that fail simulator checks. Review the accepted wording: validation uses lexical rules and can miss semantic errors.

Outputs:

- `accepted.jsonl`: scenarios with responses and teacher metadata.
- `accepted.accepted.jsonl`: accepted teacher records in the versioned schema.

Model/revision identifiers, scenario hashes and preparation hashes are recorded for later validation. Legacy `{id,text}` records and `fixture_reference` targets are limited to compatibility tests.

## Optional HTTP generation

`targets --teacher-config ... --cache ...` calls a separately configured endpoint with a request budget and persistent cache. Configure a model, revision and endpoint before use; no provider is selected by default.

Teacher and TTS budgets are separate. The current speech renderer uses local eSpeak. Teacher credentials are used only by the HTTP preparation command; `HF_TOKEN` is for model downloads. Keep credentials out of response files, configuration files and logs.

## Student training

Accepted text is rendered or aligned with assistant audio, then encoded with Mimi. PersonaPlex learns these text/audio sequences through supervised losses. Training reads prepared files without teacher-network access or teacher logits. Runtime inputs contain user audio and timestamped facts.
