# PickCube five-step world model

Trained on 36,000 episodes and evaluated on 4,000 untouched episodes. With the correct context, mean held-out MSE was **1.3069**. The independently trained zero-context baseline scored **1.3964**. The relative reduction was **6.4%**.

## Step 3 · Forecast comparison

Each entry gives W / W_noz. Lower is better. The right column averages the five horizon errors equally.

| Action source | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All episodes | 0.513 / 0.539 | 0.913 / 0.971 | 1.306 / 1.395 | 1.695 / 1.816 | 2.107 / 2.260 | 6.4% |
| Noisy policy | 0.728 / 0.768 | 1.274 / 1.354 | 1.810 / 1.928 | 2.334 / 2.492 | 2.885 / 3.085 | 6.2% |
| Scripted push | 0.200 / 0.212 | 0.370 / 0.399 | 0.552 / 0.597 | 0.750 / 0.809 | 0.971 / 1.042 | 7.1% |
| Random poke | 0.394 / 0.408 | 0.734 / 0.777 | 1.054 / 1.129 | 1.364 / 1.473 | 1.689 / 1.829 | 6.8% |

The overall paired 95% bootstrap interval for the MSE reduction was [0.0843, 0.0950]. Relative gain on pooled pushes and pokes minus gain on noisy-policy episodes was +0.68 percentage points (95% interval -0.45 to +1.91). Relative gain on horizons 4–5 minus gain on horizons 1–2 was +1.13 points (+0.74 to +1.55).

![Forecast error by action source and horizon](horizon.png)

## Step 4 · Frozen context checks

| Context | Mean MSE | Increase over own z | Paired 95% interval for MSE increase |
| --- | ---: | ---: | ---: |
| Own z | 1.3069 | — | — |
| Another episode of the same action type | 1.9030 | 45.6% | [0.5689, 0.6242] |
| Training-mean z | 1.6375 | 25.3% | [0.3138, 0.3490] |

The episode's own z minimized the sampled error for **5/6 episodes**. The minimizing fractions along the line to the donor were 37 at 0.0, 39 at 0.0, 38 at 0.0, 78 at 0.1, 36 at 0.0, 77 at 0.0. Each curve uses eleven fractions and all 46 window starts in that episode.

![Replacement errors and episode interpolation curves](context_checks.png)

| True episode constant | Collection range | Held-out linear R² |
| --- | --- | ---: |
| Cube mass | 20–1000 g (log-uniform) | 0.6822 |
| Cube friction | 0.1–1.5 | 0.5513 |
| Table friction | 0.2–1.2 | 0.6904 |
| Gripper force scale | 0.3–1.5× | 0.0401 |
| Arm damping scale | 0.5–2× | 0.7000 |
| Arm stiffness scale | 0.5–1.5× | 0.5823 |
| Command delay | 0–3 steps, integer | 0.9688 |
| Cube COM x | -1 to +1 cm | 0.9713 |
| Cube COM y | -1 to +1 cm | 0.9689 |
| Cube COM z | -1 to +1 cm | 0.3098 |

Readouts below R² = 0.20 were Gripper force scale. The probe fits on training encodings and scores physical labels in their original units. It measures linear information in E(true constants). Observability from motion remains an open question.

![Held-out linear readouts of the ten constants](linear_probe.png)

## Training and evaluation details

The encoder has two 64-unit hidden layers and an eight-dimensional output. The dynamics MLP has four 512-unit hidden layers. Both models start from identical shared dynamics weights and receive the same 12,000 minibatches of 384 five-step windows. Training uses mean squared rollout error, equal weight across horizons, and AdamW with a cosine learning-rate schedule. The final update is the fixed checkpoint for both models. Normalizers use only the 36,000 training episodes. Cube mass is log-transformed before encoding.

The 39 logged state coordinates comprise joint positions and velocities, tool pose, gripper width, and cube pose and velocities. The model converts both quaternions to two rotation columns each (43 features). W receives the current state, the issued command and z. W_noz has eight zero context columns. Only the first state of each window is observed. Each horizon error compares the predicted state to the logged one in training one-step delta-SD units. All 46 five-step starts per held-out episode are averaged before resampling whole episodes. Intervals measure episode variability for this trained model pair.

[Settings](settings.json) · [Per-horizon errors and intervals](by_horizon.csv) · [Comparison](comparison.json) · [Frozen checks](checks.json) · [Interpolation measurements](interpolation.csv) · [Probe scores](probe.csv)

## Decision after the frozen checks

Scripted push episode 78 scored 0.33257 at its own z and 0.33107 at interpolation fraction 0.1.

The advantage on pushes and pokes over policy actions remains inconclusive.

Stopped after the requested checks. The proposed follow-up is a single change to a ten-step rollout, keeping z at eight dimensions and preserving the action mix. The measured advantage grew with horizon in this run. Fresh held-out episodes would test whether a longer rollout also improves recovery of the episode's own z.
