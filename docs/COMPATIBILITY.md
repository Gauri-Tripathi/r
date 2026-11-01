# Runtime and training

Source and asset revisions are in `upstream.lock.json`. The Moshi runtime is vendored under `vendor/personaplex`.

| Component | Implementation |
|---|---|
| Model | PersonaPlex 7B; native `get_moshi_lm` and `get_mimi` loaders |
| Streams | Text, eight assistant audio codebooks, eight user audio codebooks |
| LoRA | `transformer.layers.{28,29,30,31}.self_attn.out_proj`; rank 8, alpha 8, dropout 0 |
| Trainable parameters | 262,144 across eight adapter tensors |
| Loss | Separately normalized assistant audio CE + text CE; text PAD weight 0.1 |
| Optimizer | AdamW, LR `1e-5`, betas `(0.9, 0.95)`, decay `0.01`, gradient clip 1 |
| Checkpoints | Safetensors adapters; checksummed local optimizer, RNG and sampler state |
| GPU | One A100 80GB; CUDA graphs and compilation disabled |

Stream delays are `[0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1]`. `forward_train` returns undelayed logits and validity masks. User audio, prompt frames and injected context frames are masked from supervision.

The depth transformer is frozen but remains in the autograd graph. Audio loss backpropagates through it to the temporal adapters. LoRA B starts at zero, so A receives no gradient on the first step.

Context updates are serialized one text piece per frame. The model consumes user audio and emits assistant silence during injection. Encoding and replay use the same schedule, so longer updates delay responses.

Prompt order is optional licensed voice audio, six silence frames, role text, then six silence frames. Every example starts a fresh streaming session. Checkpoints preserve the sampler and Torch RNG; GPU kernels may still be nondeterministic.
