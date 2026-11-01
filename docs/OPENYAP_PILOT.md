# OpenYap pilot

The pilot used 34 training, six development and seven test windows, each 16 seconds long. Splits were speaker-disjoint. Training ran for 17 optimizer steps and two passes, using 544 seconds of unique training audio.

| Test loss | Base | Adapter | Relative change |
|---|---:|---:|---:|
| Audio | 2.83229569 | 2.83197583 | -0.01129% |
| Text | 1.52329986 | 1.52353164 | +0.01522% |

Losses were essentially unchanged. All eight adapter tensors changed, frozen parameters stayed unchanged, and decoded PCM differed in all three saved speech pairs. Speech-quality improvement was not established.

The run took 480.3 seconds including encoding and evaluation; mean optimizer-step time was 0.938 seconds. Workspace metered usage rose by about $0.49. Its Modal allocation stopped and its reservations were reconciled.

## Data and recipe

The source was the public OpenYap sample at revision `a08b1c80e50159d462ddcbbabfdea5bf7ccfc626`: 8.9223 hours, 16 conversations and eight speakers. The pilot selected 47 windows from it. Original isolated audio and machine word timings were used, without teacher responses or TTS.

LoRA rank 8, alpha 8, dropout 0; output projections in temporal layers 28-31. AdamW learning rate `1e-5`, accumulation four. Development loss selected the checkpoint. Strict timing filters biased the subset toward slower speech.

## Inspect saved results

```bash
duplex openyap report --run runs/openyap/pilot-v1
```

Open `runs/openyap/pilot-v1/comparison.html` to listen. Checkpoints, `results.json`, training logs and the exported adapter are in that directory.

For the larger run, use [full training](OPENYAP_FULL_TRAINING.md). Its split differs from this pilot, so compare each adapter with its matched base result rather than comparing the two test averages.
