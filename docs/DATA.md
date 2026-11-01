# Data formats

## OpenYap

The public sample contains isolated `.a.flac` and `.b.flac` tracks with machine word timestamps. Mixed previews are excluded. Preparation keeps both tracks on the same timeline and resamples to mono 24 kHz.

Full preparation assigns whole conversations to splits before selecting non-overlapping 48-second windows. Speakers may overlap across splits. Partial boundary words are masked from text loss; text-piece serialization allows at most 240 ms of queue lag. Invalid audio or timings are recorded as exclusions.

`prepared.json` records validated counts and durations. `plan.json` describes metadata coverage before waveform validation. See [training](OPENYAP_FULL_TRAINING.md) for commands.

## Correction scenarios

The deterministic simulator generates requests, spoken corrections, timestamped facts and expected final state. Counterfactual variants remain in one split. Controls cover no correction and no context update.

`Example.visible` contains only the request scope, user WAV, duration and typed events. Construction scripts, supervision and expected state remain separate. Generation accepts `ModelInput`, so those fields cannot enter inference through the dataset schema.

Teacher responses use the [versioned import contract](TEACHER_CONTRACT.md). Deterministic fixture responses are labeled and cannot pass research preparation.

User and assistant audio share a timeline but are encoded separately. Target speech begins after correction completion and accepted context delivery. Local eSpeak renders individual words; measured segment boundaries are construction timings, not independent forced alignment. Review the paired WAVs and timeline before training.

## Imported audio

`duplex audio-import` accepts an ID-keyed JSON mapping:

```json
{
  "EXAMPLE_ID": {
    "user": "relative/user.wav",
    "assistant": "relative/assistant.wav",
    "words": [{"text": "Saturday", "start_s": 18.4, "end_s": 18.9}],
    "license": "YOUR_LICENSE",
    "provenance": "source and permission record",
    "response_start_s": 18.4,
    "correction_end_s": 10.2
  }
}
```

Include every assistant word and checked timing. Both channels must have identical lengths. The importer checks target text, root-relative paths and the shared event timeline. Encoding rejects text collisions, speech during forced context, and overlong sessions rather than truncating targets.

Mimi cache keys include waveform hashes, channel assignment, sample rate, codec revision and streaming settings. Each channel starts with fresh encoder state. Text is placed on the 80 ms frame grid using word starts, PAD 3 and end-of-padding 0.

## Speech evaluation

Scoring accepts cached ASR JSONL tied to generated WAV hashes:

```json
{"id":"EXAMPLE_ID","audio_sha256":"64_HEX_CHARACTERS","text":"Recognized words.","words":[{"text":"Recognized","start_s":20.0,"end_s":20.4}],"uncertainty":0.2,"engine":"ASR_MODEL","revision":"PINNED_REVISION"}
```

Word timestamps are required for response latency. Missing ASR leaves only an internal-token diagnostic. Lexical checks do not measure prosody or intelligibility; listen to the speech too.

Correction scores exclude the previously valid prefix. Latency starts at the later of correction completion and relevant fact arrival. Confidence intervals resample whole counterfactual groups; fewer than five groups produces no interval.
