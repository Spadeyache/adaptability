# Step 2 data collection and integrity

The new test cubes are IDs260–289, sampled once with seed2026111519. All original cube definitions and 4,000 episode/raw traces for IDs0–199 are preserved. The20 latent-validation cubes are separate from the180 weight-training cubes in `data/split.json`.

| Fresh-test measurement | Value |
|---|---:|
| Cubes / shoves | 30 / 300 |
| Launches | 300 / 300 |
| Moving fraction of retained coast | 90.860903% |
| Maximum coast distance | 0.934926466 m |
| Retained / trimmed transitions | 21,167 / 73,587 |
| Collection wall time | 17.90 s |

All episodes reached confirmed rest. No moving transition was trimmed. The independent audit verified schema, action/state alignment, exact zero coast force, stop confirmation, the five-step tail, original training bytes, and all **1,368** sealed retired artifacts. No sampled episode was discarded or replaced.

At collection completion, fresh command-only probe scoring had not run: the written protocol defers it until the final frozen evaluation ledger. No predictive model was evaluated during collection.

[Full audit](/home/ubuntu/faliure/clean/shove_gate/push_latent/results/full_audit.json) · [Saved collection metrics](/home/ubuntu/faliure/clean/shove_gate/push_latent/results/fresh_metrics.json) · [Written protocol](/home/ubuntu/faliure/clean/shove_gate/push_latent/PROTOCOL.md)
