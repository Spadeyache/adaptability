# No-belief world-model baselines

All requested model seeds and adaptation trajectories are complete.

Frozen split: `results/baselines/split.json`. Full protocol: `results/baselines/protocol.json`.
Shared train-only scaling arrays: `results/baselines/normalization.npz` (provenance and formulas in `normalization.json`). Reuse these arrays for later model comparisons.
Dataset: `runs/lift_slide_belief/slip_probe/paired/fixed/data`; 200 train and 30 test cubes, 10 episodes per task. Existing data and z models are unchanged.
Test episodes 1–5 are context and 6–10 evaluation (stored IDs 0–4 / 5–9). The 51-state projection matches the latest fixed-grip z model.

Each metric uses observed cube-moving rows (>0.01 m/s). MSE pools rows and state coordinates. R² is 1−SSE/SST for normalized state deltas with each coordinate centered separately. Ten-step MSE is terminal state error scaled by the training delta SD; it uses recorded actions and predicted future states/history. The two MSE horizons use the same scale.
Mean ± sample SD is across model seeds. Every model uses the same training episode split and train-only normalizers. Only B2 reads mass/friction. B1 resets at each episode and sees only completed past transitions.
All models use a 3 × 256 ReLU MLP, Adam at 1e-3, batch 1024, and the existing early stopping rule (patience 12). B1 adds a GRU with hidden size 64 for K = 1, 5, 10, or 20 previous transitions. No model has a z table.
B2 versus B0 tests the added value of supplying true physical constants. B2 does not test learning a separate belief. To assess the accuracy cost of separating belief from dynamics, compare the later learned-belief model against B1 with matched access to past observations. B0 may already infer some physical information from the measured current state.
Weight adaptation crosses five B0 seeds with five fixed cube orders at each learning rate (50 trajectories). It runs on CPU while the base models train on GPU. Each cube gets a fresh Adam optimizer for 100 steps; all learned weights carry forward and normalization stays fixed. Adaptation summaries average orders within each model seed, then report mean ± sample SD across the five model seeds; raw order variability is also saved.
The first five literal steps contain no moving samples in any eval episode. Their moving-only metrics are N/A; the window has not been shifted.

Every moving evaluation row already has at least 20 completed past transitions. Consequently every scored B1 window is full for all requested K values. The step-index curve measures error through the episode, but does not isolate the effect of filling the history window; task phase and history content also change.

| Model / evaluation | Task | One-step normalized MSE | R² | 10-step normalized MSE |
| --- | --- | ---: | ---: | ---: |
| B0 | lift | 0.187336 ± 0.00693 | 0.782876 ± 0.00803 | 1.69788 ± 0.114 |
| B1_K1 | lift | 0.176616 ± 0.0143 | 0.7953 ± 0.0165 | 1.53318 ± 0.125 |
| B1_K5 | lift | 0.181126 ± 0.00818 | 0.790073 ± 0.00949 | 1.44779 ± 0.0938 |
| B1_K10 | lift | 0.186055 ± 0.0159 | 0.784361 ± 0.0185 | 1.37826 ± 0.1 |
| B1_K20 | lift | 0.181754 ± 0.0109 | 0.789346 ± 0.0127 | 1.3294 ± 0.148 |
| B2 | lift | 0.187075 ± 0.0142 | 0.783178 ± 0.0165 | 1.50151 ± 0.0965 |
| B0 | slow_slide | 0.0442214 ± 0.00147 | 0.8583 ± 0.0047 | 6.08263 ± 0.328 |
| B0 | shove | 0.894203 ± 0.0336 | 0.896679 ± 0.00388 | 11.1793 ± 0.897 |
| B0 | overall | 0.161267 ± 0.00387 | 0.836485 ± 0.00393 | 3.72527 ± 0.101 |
| B1_K1, steps 1–5 | lift | N/A | N/A | N/A |
| B1_K1, steps 6+ | lift | 0.176616 ± 0.0143 | 0.7953 ± 0.0165 | 1.53318 ± 0.125 |
| B1_K1 | slow_slide | 0.0405018 ± 0.0043 | 0.870218 ± 0.0138 | 5.22944 ± 0.759 |
| B1_K1, steps 1–5 | slow_slide | N/A | N/A | N/A |
| B1_K1, steps 6+ | slow_slide | 0.0405018 ± 0.0043 | 0.870218 ± 0.0138 | 5.22944 ± 0.759 |
| B1_K1 | shove | 0.845128 ± 0.0596 | 0.902349 ± 0.00688 | 9.87093 ± 0.561 |
| B1_K1, steps 1–5 | shove | N/A | N/A | N/A |
| B1_K1, steps 6+ | shove | 0.845128 ± 0.0596 | 0.902349 ± 0.00688 | 9.87093 ± 0.561 |
| B1_K1 | overall | 0.151666 ± 0.0103 | 0.84622 ± 0.0104 | 3.25832 ± 0.301 |
| B1_K1, steps 1–5 | overall | N/A | N/A | N/A |
| B1_K1, steps 6+ | overall | 0.151666 ± 0.0103 | 0.84622 ± 0.0104 | 3.25832 ± 0.301 |
| B1_K5, steps 1–5 | lift | N/A | N/A | N/A |
| B1_K5, steps 6+ | lift | 0.181126 ± 0.00818 | 0.790073 ± 0.00949 | 1.44779 ± 0.0938 |
| B1_K5 | slow_slide | 0.0414698 ± 0.00321 | 0.867117 ± 0.0103 | 5.6246 ± 0.619 |
| B1_K5, steps 1–5 | slow_slide | N/A | N/A | N/A |
| B1_K5, steps 6+ | slow_slide | 0.0414698 ± 0.00321 | 0.867117 ± 0.0103 | 5.6246 ± 0.619 |
| B1_K5 | shove | 0.828211 ± 0.023 | 0.904304 ± 0.00266 | 9.82691 ± 1.11 |
| B1_K5, steps 1–5 | shove | N/A | N/A | N/A |
| B1_K5, steps 6+ | shove | 0.828211 ± 0.023 | 0.904304 ± 0.00266 | 9.82691 ± 1.11 |
| B1_K5 | overall | 0.153916 ± 0.00481 | 0.843938 ± 0.00488 | 3.3486 ± 0.29 |
| B1_K5, steps 1–5 | overall | N/A | N/A | N/A |
| B1_K5, steps 6+ | overall | 0.153916 ± 0.00481 | 0.843938 ± 0.00488 | 3.3486 ± 0.29 |
| B1_K10, steps 1–5 | lift | N/A | N/A | N/A |
| B1_K10, steps 6+ | lift | 0.186055 ± 0.0159 | 0.784361 ± 0.0185 | 1.37826 ± 0.1 |
| B1_K10 | slow_slide | 0.0440245 ± 0.00731 | 0.85893 ± 0.0234 | 5.05356 ± 0.631 |
| B1_K10, steps 1–5 | slow_slide | N/A | N/A | N/A |
| B1_K10, steps 6+ | slow_slide | 0.0440245 ± 0.00731 | 0.85893 ± 0.0234 | 5.05356 ± 0.631 |
| B1_K10 | shove | 0.867864 ± 0.0359 | 0.899722 ± 0.00415 | 9.20475 ± 0.537 |
| B1_K10, steps 1–5 | shove | N/A | N/A | N/A |
| B1_K10, steps 6+ | shove | 0.867864 ± 0.0359 | 0.899722 ± 0.00415 | 9.20475 ± 0.537 |
| B1_K10 | overall | 0.159366 ± 0.00739 | 0.838412 ± 0.0075 | 3.07199 ± 0.2 |
| B1_K10, steps 1–5 | overall | N/A | N/A | N/A |
| B1_K10, steps 6+ | overall | 0.159366 ± 0.00739 | 0.838412 ± 0.0075 | 3.07199 ± 0.2 |
| B1_K20, steps 1–5 | lift | N/A | N/A | N/A |
| B1_K20, steps 6+ | lift | 0.181754 ± 0.0109 | 0.789346 ± 0.0127 | 1.3294 ± 0.148 |
| B1_K20 | slow_slide | 0.0415478 ± 0.00244 | 0.866867 ± 0.00781 | 4.88003 ± 0.316 |
| B1_K20, steps 1–5 | slow_slide | N/A | N/A | N/A |
| B1_K20, steps 6+ | slow_slide | 0.0415478 ± 0.00244 | 0.866867 ± 0.00781 | 4.88003 ± 0.316 |
| B1_K20 | shove | 0.871064 ± 0.0472 | 0.899353 ± 0.00546 | 9.83369 ± 0.956 |
| B1_K20, steps 1–5 | shove | N/A | N/A | N/A |
| B1_K20, steps 6+ | shove | 0.871064 ± 0.0472 | 0.899353 ± 0.00546 | 9.83369 ± 0.956 |
| B1_K20 | overall | 0.156082 ± 0.00538 | 0.841742 ± 0.00546 | 3.00966 ± 0.184 |
| B1_K20, steps 1–5 | overall | N/A | N/A | N/A |
| B1_K20, steps 6+ | overall | 0.156082 ± 0.00538 | 0.841742 ± 0.00546 | 3.00966 ± 0.184 |
| B2 | slow_slide | 0.0470988 ± 0.00256 | 0.849079 ± 0.00819 | 6.41382 ± 0.684 |
| B2 | shove | 0.80572 ± 0.0507 | 0.906903 ± 0.00586 | 9.83842 ± 0.989 |
| B2 | overall | 0.158561 ± 0.00883 | 0.839228 ± 0.00895 | 3.66626 ± 0.289 |
| Weights lr=0.0001: current cubes, pooled | lift | 0.22209 ± 0.00589 | 0.74259 ± 0.00683 | 2.3966 ± 0.225 |
| Weights lr=0.0001: first cube after 0 further cubes | lift | 0.16418 ± 0.0241 | 0.87708 ± 0.0178 | 1.7596 ± 0.516 |
| Weights lr=0.0001: first cube after 1 further cubes | lift | 0.18677 ± 0.0206 | 0.84851 ± 0.0162 | 4.5063 ± 2.76 |
| Weights lr=0.0001: first cube after 5 further cubes | lift | 0.18839 ± 0.009 | 0.84868 ± 0.00737 | 2.8254 ± 0.784 |
| Weights lr=0.0001: first cube after 20 further cubes | lift | 0.19826 ± 0.00517 | 0.8349 ± 0.00299 | 2.1777 ± 0.399 |
| Weights lr=0.0001: validation before | overall | 0.1847 ± 0.00192 | 0.83169 ± 0.00175 | 4.235 ± 0.187 |
| Weights lr=0.0001: validation before | lift | 0.21943 ± 0.00288 | 0.80586 ± 0.00255 | 2.2512 ± 0.0721 |
| Weights lr=0.0001: validation before | slow_slide | 0.05209 ± 0.00241 | 0.8404 ± 0.00737 | 6.7968 ± 0.442 |
| Weights lr=0.0001: validation before | shove | 0.92743 ± 0.0435 | 0.87494 ± 0.00587 | 9.6444 ± 0.565 |
| Weights lr=0.0001: validation after | overall | 0.32069 ± 0.0102 | 0.70777 ± 0.00928 | 13.595 ± 2.85 |
| Weights lr=0.0001: validation after | lift | 0.3613 ± 0.0143 | 0.68034 ± 0.0126 | 4.8292 ± 0.625 |
| Weights lr=0.0001: validation after | slow_slide | 0.11123 ± 0.00759 | 0.6592 ± 0.0232 | 24.158 ± 6.23 |
| Weights lr=0.0001: validation after | shove | 1.6878 ± 0.0411 | 0.7724 ± 0.00555 | 43.246 ± 10.1 |
| Weights lr=0.0001: validation change | overall | 0.13599 ± 0.0113 | -0.12392 ± 0.0103 | 9.3602 ± 2.93 |
| Weights lr=0.0001: validation change | lift | 0.14187 ± 0.0143 | -0.12552 ± 0.0126 | 2.5779 ± 0.631 |
| Weights lr=0.0001: validation change | slow_slide | 0.059135 ± 0.0064 | -0.18119 ± 0.0196 | 17.362 ± 6.37 |
| Weights lr=0.0001: validation change | shove | 0.7604 ± 0.0629 | -0.10254 ± 0.00848 | 33.601 ± 10.3 |
| Weights lr=1e-05: current cubes, pooled | lift | 0.18556 ± 0.00653 | 0.78493 ± 0.00757 | 1.6736 ± 0.107 |
| Weights lr=1e-05: first cube after 0 further cubes | lift | 0.14451 ± 0.015 | 0.89255 ± 0.0114 | 1.2156 ± 0.286 |
| Weights lr=1e-05: first cube after 1 further cubes | lift | 0.14491 ± 0.0142 | 0.89233 ± 0.0109 | 1.3345 ± 0.494 |
| Weights lr=1e-05: first cube after 5 further cubes | lift | 0.13487 ± 0.00895 | 0.89989 ± 0.00677 | 1.1693 ± 0.297 |
| Weights lr=1e-05: first cube after 20 further cubes | lift | 0.13912 ± 0.00908 | 0.89582 ± 0.00671 | 1.3627 ± 0.655 |
| Weights lr=1e-05: validation before | overall | 0.1847 ± 0.00192 | 0.83169 ± 0.00175 | 4.235 ± 0.187 |
| Weights lr=1e-05: validation before | lift | 0.21943 ± 0.00288 | 0.80586 ± 0.00255 | 2.2512 ± 0.0721 |
| Weights lr=1e-05: validation before | slow_slide | 0.05209 ± 0.00241 | 0.8404 ± 0.00737 | 6.7968 ± 0.442 |
| Weights lr=1e-05: validation before | shove | 0.92743 ± 0.0435 | 0.87494 ± 0.00587 | 9.6444 ± 0.565 |
| Weights lr=1e-05: validation after | overall | 0.19041 ± 0.00189 | 0.82648 ± 0.00173 | 4.832 ± 0.191 |
| Weights lr=1e-05: validation after | lift | 0.22382 ± 0.00335 | 0.80198 ± 0.00296 | 2.2573 ± 0.0424 |
| Weights lr=1e-05: validation after | slow_slide | 0.055601 ± 0.00177 | 0.82964 ± 0.00542 | 8.046 ± 0.519 |
| Weights lr=1e-05: validation after | shove | 0.97134 ± 0.0427 | 0.86902 ± 0.00576 | 12.695 ± 0.892 |
| Weights lr=1e-05: validation change | overall | 0.0057129 ± 0.00188 | -0.0052059 ± 0.00172 | 0.597 ± 0.272 |
| Weights lr=1e-05: validation change | lift | 0.0043895 ± 0.00102 | -0.0038836 ± 0.0009 | 0.0060392 ± 0.0887 |
| Weights lr=1e-05: validation change | slow_slide | 0.0035112 ± 0.00176 | -0.010759 ± 0.00539 | 1.2492 ± 0.64 |
| Weights lr=1e-05: validation change | shove | 0.043907 ± 0.017 | -0.0059207 ± 0.00229 | 3.0504 ± 1.19 |

All per-seed, per-cube, per-step results and sample counts are in `results/baselines/metrics.json`. Checkpoints and training logs are in `runs/baselines/`.

B0 seed 0 pilot: 43.711 seconds training; 46.634 seconds loading, fitting, saving and evaluating; 47 epochs (best 35). These are measured times.

## Exact rerun commands

```bash
cd /home/ubuntu/faliure
source scripts/activate_grip.sh
export LD_LIBRARY_PATH=/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python -m experiments.world_model_baselines freeze
python -m experiments.world_model_baselines pilot --device cuda
# Report the measured pilot runtime and review its saved metrics before the remaining commands.
python -m experiments.world_model_baselines acknowledge-pilot
python -m experiments.world_model_baselines sweep --device cuda
python -m experiments.world_model_baselines adapt --device cpu --threads 2
python -m experiments.world_model_baselines report
```

Commands reuse completed checkpoints/results. For a fresh rerun, archive `runs/baselines/` and the generated result JSON/plots first, keeping the frozen `protocol.json`, `split.json`, and `implementation.json`. Report the new pilot before acknowledging its new checksum. Do not alter the frozen split.
B1 adds a GRU and therefore parameters as well as history. B2 is the privileged-information comparator, not a guaranteed numerical optimum. Recorded-action rollouts measure forecasting, not closed-loop task success.
