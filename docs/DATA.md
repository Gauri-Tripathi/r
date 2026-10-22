# Data, timelines, replay and scoring

`duplex scenarios` creates a deterministic candidate pool; 100 groups produce 400 short scenarios. The seed-17 pilot has 300 train / 56 dev / 44 test candidates. Related availability variants stay in one split. Template families, correction phrases, person pairs and day pairs are assigned before rendering. Time values, the initial request shape, voice and response vocabulary are shared: this is **not universal disjointness**. Reports stratify clean/overlap and family; each row also retains axes (held-out family/entity-day, single/two fields, tool-result variant).

`Example.visible` is the entire generation input type: request scope, WAV reference, duration, and typed tool events. `script`, `supervision` and `truth` are separate construction/training/evaluation fields. The inference function accepts only `ModelInput`, never `Example`. There is no inference-time teacher or request-state solver. Audio corrections have to be understood by the speech model itself. Calendar data is supplied grounding, not learned native tool calling. There are no real bookings or cancellations.

The simulator applies one-/two-field corrections without touching unchanged fields, computes the latest versioned snapshot, and derives offer/conflict/clarify/stop actions. Counterfactual variants change available/unavailable/unknown/stale facts with the same preceding dialogue. Controls include no event, corrections without updates, and updates without corrections. Ambiguous/cancel/tool-only templates have separate held-out phrasing. A trusted stronger teacher may verbalize the computed action offline; deterministic checks reject unsupported entities/dates/times, dropped fields, conflicts and commitments. Checks are conservative lexical rules, not proof of arbitrary-language correctness. Review accepted targets. Deterministic `--reference-fixture` targets are a useful test/control and **not evidence of teacher capability transfer**.

The current research contract is [versioned external teacher JSONL](TEACHER_CONTRACT.md). The legacy import below is retained only for compatibility fixtures and cannot pass research preparation.

Legacy teacher import format:

```json
{"id":"s-17-0000-available","text":"Saturday at 4 with Ananya is available. Would you like to proceed?"}
```

IDs must exactly match the selected input manifest. API teaching uses a configured schema-constrained compatible endpoint, one in-flight request, zero retries by default (maximum one), output <=256 tokens, HTTPS/timeout/response limits, and a persistent cache/API cost ledger. Missing credentials never select a paid fallback. Cache keys include input, prompt/schema version and complete nonsecret teacher configuration. Test examples are excluded from API target building. Teacher/TTS budgets are separate from Modal credit; the example teacher config has an intentionally invalid zero budget until explicitly edited.

Real rendering requires the licensed pinned SentencePiece tokenizer to measure context duration. CPU eSpeak NG renders actual English speech, with each word synthesized separately; measured WAV durations give construction boundaries, without pretending to forced-align natural speech. Tested local renderer: eSpeak NG 1.52.0 generic `en-us` voice. Prosody is robotic, and internal word onset can be uncertain within the measured segment. Resampling uses scipy polyphase to 24kHz; clipped/nonfinite/oversized/nonmono sources reject. There is no paid TTS request, downloaded neural voice or OpenYap speaker cloning. The image's installed eSpeak version is recorded per example; OS package contents can vary until a built image is pinned.

The user/assistant WAVs share one timeline, preserve silence and overlap, and are never mixed for Mimi. A pre-correction assistant prefix reflects the previously valid request. In overlap cases, the correction begins while that prefix is audible; prefix supervision ends no later than the complete correction. New target speech begins after the correction and after actual accepted context-token delivery. This intentionally exposes the native interface's forced-silence latency (potentially seconds); it is not seamless zero-latency grounding. No target needs knowledge of a future event. Construction manifest fields include prefix/correction start/end, response start, per-word timestamps, source hashes, renderer version and accepted-event frame counts. Inspect these JSON timelines beside the two WAVs; do not regard tone audio as speech.

Natural licensed WAV import uses an ID-keyed JSON mapping passed to `audio-import`:

```json
{
  "EXAMPLE_ID": {
    "user": "relative/user.wav", "assistant": "relative/assistant.wav",
    "words": [{"text":"Saturday", "start_s":18.4, "end_s":18.9}],
    "license":"YOUR_LICENSE", "provenance":"source and permission record",
    "response_start_s":18.4, "correction_end_s":10.2
  }
}
```

Provide **all** assistant words and exact checked response text, not the abbreviated documentation example. The source channels must have identical lengths. Supply checked timings from your licensed alignment tool/human review. Imported construction timings and event arrival times must already share the supplied WAV timeline. `encode` rejects speech overlapping forced context, token collisions, invalid IDs/durations and truncation. Native text pieces are placed consecutively starting from measured word starts, with PAD 3 and end-of-padding 0; words are never spread uniformly across frames. Codec timing quantizes to 80ms. Model-specific word alignment with the actual gated tokenizer/Mimi remains unverified locally.

Mimi cache keys cover WAV hash, user/assistant assignment, 24kHz, codec revision, codebooks/frame rate, causal streaming settings and preprocessing version. Every channel/example opens a fresh streaming context. The 8 codebooks per speaker are not confused with 16 modeled audio heads. No session concatenation or cross-example cache/attention reuse is allowed. Complete sessions exceeding a profiled frame limit reject rather than silently losing the response. Explicit CPU preparation prevents unbudgeted GPU codec work.

Optional ordinary replay input is JSONL:

```json
{"id":"licensed-short-1","user":"user.wav","assistant":"assistant.wav","words":[{"text":"Hello","start_s":0.8,"end_s":1.1}],"source":"licensed local subset, version and attribution","license":"YOUR_LICENSE","isolated_channels_verified":true,"duration_s":12.0}
```

`replay-encode` caps selected audio at 180s by default and 300s maximum, with each clip <=60s. It verifies isolated equal-length mono channels and checked assistant timings. The ordinary subset has no tool-grounding answers or synthetic negatives. `mix_frames` selects 10-20% ordinary conversation frames (default .15), or records that no valid replay was available. It rejects a subset whose clip granularity cannot meet that range. OpenYap is optional; metadata/card has been inspected, but no media channels were downloaded or verified. Use the pinned sample revision's isolated shard tracks only after inspecting the selected media. The full 1000h corpus is never downloaded by this repo.

Held-out scoring uses generated audible WAVs via cached ASR JSONL, not teacher-forced continuations:

```json
{"id":"EXAMPLE_ID","audio_sha256":"64_HEX_CHARACTERS","text":"Recognized spoken words.","words":[{"text":"Recognized","start_s":20.0,"end_s":20.4}],"uncertainty":0.2,"engine":"YOUR_LOCAL_ASR","revision":"IMMUTABLE_ASR_REVISION"}
```

ASR is supplied/imported from a local cached transcription tool; no transcription API is automatically called. Hashes must match generated speech. Record ASR uncertainty and listen to ambiguous examples. Without both ASR files, reports explicitly say `internal_text_proxy`; they cannot establish audible correction success or audio quality. Analysis excludes the previously valid prefix. Incorporation latency starts at `max(correction_end, latest-relevant-revision-arrival)` and ends when the first complete valid new response finishes according to ASR word times. Missing word timing yields no latency value. Report p50 if any exist, p95 only with >=20 values. Paired percentile bootstrap resamples whole counterfactual groups; fewer than five groups yields no CI. Small pilot intervals are weak evidence.

Results retain correction/joint success, unchanged constraints, latest/stale usage, commitments, hallucinated availability, clarification, repetition, truncation, silence fraction, RTF, uncertainty and listening paths. These deterministic heuristics are not a universal speech-quality judge. Ordinary regression audio and the limited benchmark replay bridge provide listening/RTF diagnostics; no aggregate public benchmark or ordinary-conversation quality claim is produced. A stronger ASR/judge can be added as a separately budgeted supplement, never the sole source of correctness.
