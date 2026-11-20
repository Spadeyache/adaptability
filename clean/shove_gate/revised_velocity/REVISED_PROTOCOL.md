# Revised direct-force gate: horizontal velocity change

Written before the physics sanity calculation, fresh test collection, or revised model fitting on 2026-11-15. This revision is motivated by the previously reported failed gate; it is a new experiment, not a reinterpretation of that result. The old protocol, checkpoints, test cubes 200–229, and reported results remain untouched in the parent directory.

## Reasons for the revision

1. The previous primary error averaged 15 state coordinates. Friction during sliding acts directly on horizontal velocity; averaging position, orientation, vertical velocity, and angular velocity can obscure that signal. This is a motivation, not a claim that dilution caused the earlier failure. We first measure an analytic sliding reference on training cubes to check whether this proposed physical explanation matches the simulator.
2. The new primary score isolates the predicted horizontal velocity change. A training-only change scale makes its magnitude interpretable relative to the observed friction signal, rather than normalizing by the much larger spread in absolute velocity.
3. Checkpoint selection must optimize the same validation quantity used by the new gate. A predeclared validation-only extension rule avoids terminating training while this quantity is still improving. Architecture, training loss, input whitelist, and sampling otherwise remain unchanged, so the revision does not silently introduce a new objective or data weighting.
4. The old test results informed this revision. Those 30 cubes are retired from gating. Thirty newly sampled cubes, with a new fixed seed and no rejection based on properties or outcomes, provide the final held-out evaluation.
5. All previous metrics remain secondary, including recursive predictions and stopping position. A successful one-step velocity gate alone would not demonstrate useful long-horizon prediction or a learned belief model.

## Data and separation

Reuse the 200 existing training cubes (IDs 0–199), their ten direct-force shoves, and their original bytes. Episodes 0–7 fit models and normalization; episodes 8–9 select checkpoints. Draw 30 new test cubes, IDs 230–259, with seed **2026111517** and ten independently randomized shoves each. Retire IDs 200–229 entirely from fitting, selection, and this gate.

The original generator is unchanged: independent mass uniform 0.18–0.50 kg and cube friction uniform 0.4–1.0, floor friction 0.3, configured combined friction `(cube_friction + 0.3)/2`; direct COM force uniform 4–7 N, uniform direction, random position/yaw, duration uniform on 60,65,…,100 ms. Physics dt is 1 ms; model dt is 5 ms. Coast actions are exactly zero; raw coast horizon remains 1.5 s. Preserve stop confirmation and trimming (five model steps after stop onset), and retain every sampled episode including distances above 1 m. No new distance-based resampling.

True per-cube properties reside only in `data/cubes.json`. Plain input is normalized observed state15 plus applied force3. Oracle adds only true mass and combined friction. IDs, episode length, padding masks, phase, time, stop time, and termination indicators remain forbidden model inputs. Command-only probes use a fixed first100ms schedule; they cannot use trimmed length. Collection QA can inspect test trajectories for schema, launch, stop, trimming, and byte integrity, but cannot fit, score, or tune predictive models.

## Physics sanity reference, before learning

For each retained moving-coast step, let `v=(vx,vy)` and predict exactly

`delta_v_physics = -combined_friction * 9.81 * 0.005 * v / ||v||`.

At exactly zero horizontal speed define the prediction as zero. The primary reference is **unclipped**, as requested; report a stop-clipped diagnostic separately because a full-step Coulomb decrement can exceed remaining speed. Preserve the existing moving-coast mask: coast and (full linear speed >0.001 m/s or angular speed >0.05 rad/s). Report translation-only and angular-only counts/diagnostics so the retained mask is explicit.

Define scalar `S_delta_v²` from fitting episodes only: equal-weight cubes, then episodes, then eligible moving-coast steps, of `(delta_vx² + delta_vy²)/2`. All primary scores use this frozen scalar. Thus predicting zero change has NMSE 1 on fitting moving-coast steps. For any prediction, horizontal-delta NMSE is the equally aggregated mean of `((predicted_delta_vx-true_delta_vx)² + (predicted_delta_vy-true_delta_vy)²)/(2*S_delta_v²)`. Also report unnormalized MSE and per-coordinate RMSE in m/s.

Report the reference on all 200 training cubes, and separately on fitting and validation episodes. Define “near zero” prospectively as NMSE ≤0.01 (reference RMSE ≤10% of the fitting observed-change RMS). This diagnostic is not a continuation gate. If it fails, quantify speed, direction, stop-boundary, and rotational/contact deviations using training data; distinguish measured deviations from unproven simulator explanations. The analytic reference is not asserted to be an irreducible physical error floor.

## Models, selection, and extension rule

Use five paired seeds **101,211,307,401,503**, Plain and Oracle MLPs with three 256-unit ReLU hidden layers. Fit from scratch. Preserve the old full15 normalized state-change MSE training loss and its training-only state/action/context normalizers (standard deviations floored at0.001 native units). Batch size512; sample training episode uniformly, then retained step uniformly. Both families in each seed pair receive identical minibatch streams and update budgets.

Adam starts at0.001, cosine-decays to0.0001 over the first10,000 updates, then stays at0.0001 for any extension. Validate every500 updates. Select the checkpoint with the lowest equal-cube/episode **moving-coast horizontal-delta NMSE**, not the old all-state/all-step metric. Save and plot every validation check for both families, with selected checkpoints marked; retain old validation all-state error as a diagnostic.

Run at least10,000 updates per pair. At each block endpoint (10,000;20,000;…), compare each family's best validation score through `endpoint−2500` with its best score over the final2500 updates. If either family improves the previous best by at least0.1%, extend **both** models by another10,000 updates. Otherwise stop that pair. Maximum100,000 updates per pair is a declared resource bound; if this bound is reached while the extension condition still holds, report that validation convergence was not established. No test score can extend training or choose checkpoints.

## Frozen evaluation and gate

Before fitting, save the revised protocol, implementation/source hashes, data definitions, training-derived normalization, and fresh held-out byte hashes. Before any fresh predictive test scoring, finish all five pairs and freeze all ten selected checkpoints. Evaluate once on all30 fresh test cubes and ten shoves each. A resume is allowed only with identical sealed data/code/checkpoints; training is forbidden after predictive test evaluation begins.

Report primary horizontal-delta NMSE plus every previous metric for **push**, **moving coast**, and **all retained steps**. Old secondary metrics are full15 one-step NMSE, mean recursive10-step state NMSE (and tenth-step diagnostic), and stopping-position XY error. Preserve old aggregation and planned release+1.5s stopping horizon, never the observed stopping time or trimmed length. No clipping, physics correction, or ground-truth resets in learned rollouts.

The gate passes only if, on **moving-coast horizontal-delta NMSE**, `mean_Oracle <= 0.8*mean_Plain` AND `mean_Oracle + sample_SD_Oracle < mean_Plain - sample_SD_Plain` across the five paired seeds. Report all seed values, means/sampleSD, paired improvements, and the gap versus summed SD. Do not substitute another metric after seeing fresh test results.

During that same frozen evaluation, score the unchanged analytic physics reference on fresh moving-coast steps with the fitting-derived scale. Report Oracle's absolute excess error, error ratio, and RMSE relative to that reference (where the denominator is nonzero). Keep the unclipped and stop-clipped references distinct. If the gate fails, stop and report; no belief-model training is authorized in this revision.

## Artifacts

All new work lives in `clean/shove_gate/revised_velocity/`. Preserve the parent experiment as reported. Save sanity-floor results and interpretation before fitting, per-seed training/validation logs, validation curves, fresh test per-cube metrics, source/data/checkpoint seals, final gate summary, and a report with every figure derived from saved runs.
