# Physics sanity reference, measured before revised training

The requested unclipped formula has **NMSE 0.00994671** on all 200 training cubes: per-coordinate horizontal velocity-change RMSE **0.00172360 m/s**. It just meets the predeclared near-zero threshold of 0.01; validation episodes alone are slightly above that threshold. No dynamics model was fit to obtain these values.

| Training subset | Requested formula NMSE | Requested formula RMSE (m/s) | Stop-clipped diagnostic NMSE | Moving-coast starts |
|---|---:|---:|---:|---:|
| Fitting | 0.00974751 | 0.00170626 | 0.00004532 | 70,238 |
| Validation | 0.01074347 | 0.00179131 | 0.00003667 | 17,632 |
| All training | 0.00994671 | 0.00172360 | 0.00004359 | 87,870 |

Most discrepancy is confined to the final partial sliding step. The requested formula applies a full 5 ms decrement even when the remaining speed is smaller than that decrement; the recorded cube stops instead of acquiring a reversed velocity. There are 1,887 such starts out of 87,870. Capping the decrement at the available speed reduces aggregate error by 99.56%, to NMSE 0.00004359. This capped result is a separate diagnostic; the primary reference and final Oracle comparison retain the exact requested unclipped formula.

Outside those stopping crossings, NMSE is 0.00004517. At horizontal speeds ≥0.1 m/s, observed deceleration divided by combined_friction×9.81 is 0.99998913, NMSE is 0.00004223, and mean perpendicular acceleration is 0.00030469 m/s². The high-speed deceleration ratio ranges from 0.99990057 to 1.00008074 across twelve 30° world-direction bins. These measurements agree closely with simple sliding friction over the sampled regime; they do not show a material alternative friction law. Small residuals remain in the discrete contact simulation, and this analysis does not identify their internal solver cause.

Of 87,870 moving-coast starts, 87,869 have horizontal speed >1 mm/s; 1 is included only because the preserved full-linear/angular movement rule still marks it moving. Using the original full-linear/angular thresholds, 77,936 starts are linear-only, 1 is angular-only, and 9,933 satisfy both. No samples are removed from the primary score.

![Physics sanity measurements](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/results/figures/physics_floor.png)

The normalization scalar is `S_delta_v² = 0.00029867315208091557 (m/s)²`, computed only from episodes 0–7 of cubes 0–199. A zero-change predictor has NMSE 1 on that fitting subset. All scores first average steps within an episode, then episodes within a cube, then cubes equally. Conditional speed/direction diagnostics average only represented episodes and cubes.

No retired or fresh test trajectories were opened. The analytic reference is a diagnostic baseline, not a proven irreducible error bound. It is possible for a predictor that handles the partial stopping step to outperform the unclipped reference.

[Saved measurements](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/results/physics_floor.json) · [Written revised protocol](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/REVISED_PROTOCOL.md)

Measurement SHA-256: `922b9b33ff43e20aea1e6077241160b39e46b91160bb10b051d0fd28a9511cd5`
Protocol SHA-256 recorded before scoring: `986ca6b5b06b582891e0c5145ce4c22c8965eaa3c32dcdd2fb9eac3abff59f6d`
