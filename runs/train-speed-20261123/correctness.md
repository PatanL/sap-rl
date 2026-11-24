# Acceptance protocol fixed before GPU timing

Use checkpoint 25,772 with the historical AdamW moments and all four historical learning rates. Use the same saved complete CPU-generated Turtle turns for the old and new update. Compare total loss, every accumulated gradient *before clipping*, the parameter delta from one real AdamW step, and the maximum absolute difference of post-step weights. Relative L2 norms use the reference gradient or **parameter change** as denominator. A relative error normalized by the full model would hide an optimizer error.

| Precision | Absolute loss error | Global gradient relative L2 | Parameter-delta relative L2 | Largest absolute weight difference |
|---|---:|---:|---:|---:|
| FP32, TF32 disabled | ≤2e-5 | ≤3e-4 | ≤1e-3 | ≤2e-6 |
| Recorded actor BF16 autocast | ≤0.003 | ≤0.035 | ≤0.035 | ≤8e-6 |

These tolerances allow ordinary accumulation and GEMM precision differences. They do not allow a new target, changed loss weights or a different optimizer. V-trace, discount, clipping, bootstrap, auxiliary coefficients, optimizer state, sampling, reward, frozen-match fraction and segment-level maximum version lag are fixed. The reference’s 128-padded-row / 16-turn grouping still defines the opponent-forecast denominators. Physical forward/backward grouping may change.

Select the production comparison using measured end-to-end learned decisions per second, while retaining the historical 8,192 target for the main like-for-like result. Report any different decision target separately. Score the short proof against the frozen initializer and the old-path checkpoint, disclose the unequal training horizons, and compare anchor scores by bootstrapping whole paired worlds. A confidence interval spanning parity is inconclusive. It cannot certify zero loss of strength. Use independent confirmation if a screen clears parity. Do not promote or modify a running campaign.

All GPU work and all timing (including CPU substitution timing) wait for the operator’s STOPPED file. Each GPU invocation is time-bounded; the proof saves its last fully applied update if the soft wall limit intervenes.
