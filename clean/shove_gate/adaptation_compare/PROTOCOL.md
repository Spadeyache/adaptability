# Step 3: frozen-model adaptation, forgetting, and return

This protocol and its reasons are written before any Step 3 simulation, adaptation, validation scoring, or predictive test evaluation. Only source inspection and protocol review preceded this file. Step 2 and its reported C2 failure remain unchanged. No dynamics checkpoint is retrained or reselected.

## Reasons and fixed interpretation

1. Step 2 fitted z only against push velocity change. That improved push prediction while degrading other state components and recursive forecasts. Step 3 fits z against the full-state one-step loss, with an explicit prior toward the training mean, and selects the procedure using a balanced three-component validation score.
2. A coast error divided by push-change energy is numerically suppressed. We therefore use separate fitting-only zero-change energy denominators for push velocity, moving-coast velocity, and full normalized-state change. All three zero-change predictors score 1 on their respective fitting populations. The frozen network input/output scalers and checkpoints are unchanged. Legacy Step 2 units are also recoverable from the saved denominators.
3. The validation cubes choose adaptation settings; fresh cubes estimate transfer. IDs 260–289 join IDs 200–259 as retired test data. No Step 3 tuning, selection, or gating uses their observations or reported scores.
4. Independent per-cube adaptation and continual adaptation answer different questions. Prefix curves restart from the frozen base for each cube and support count; the sequence experiment preserves B's weights across cubes. A stores independent z vectors. Exact A retention is a consequence of this architecture and a check of implementation, not evidence of learned resistance to interference.
5. A fixed 50 full-support updates for both A and B separates the number of observed shoves from the number of optimizer updates. Test query observations never enter adaptation. All settings are selected and reported before any fresh test properties, commands, or traces are generated.

## Data, preservation, and permissions

Reuse the complete original 200 training cubes and all their raw and model traces byte-for-byte. Keep Step 2 splits: IDs 0–179 are fitting cubes; their shoves 0–7 define metric energies only, while shoves 8–9 remain the training-cube validation set. IDs 180–199 are adaptation-validation cubes only, with support shoves 0–4 and query shoves 5–9. No weights are trained on either subset in Step 3. Adaptation of temporary model copies is explicitly allowed for B.

Fresh test IDs are **290–319**, generator seed **2026111613**, ten shoves per cube. Do not generate their properties/plans or collect them until validation selection is frozen and the chosen setting has been reported to the user. Use the exact existing independent generator and direct-COM-force simulator: mass uniform 0.18–0.50 kg, cube friction uniform 0.4–1.0, floor friction 0.3, combined friction `(cube_friction+0.3)/2`; force uniform 4–7 N with independent uniform direction, random start/yaw, duration uniform on 60,65,...,100 ms; physics dt 1 ms, model dt 5 ms, raw coast 1.5 s. Retain the existing confirmed-stop/five-step trimming rule. No rejection, replacement, or resampling.

Per-cube true physical properties occur only in cubes.json and enter only Oracle or collection integrity checks. A, B, and History do not read physical labels. No IDs, phase, lengths, masks, time, stop times, termination flags, or future observations enter a model. IDs only route/cache the explicitly requested z. History retains the exact Step 2 past-ten-transition GRU64 convention and resets each episode. Phase/moving masks are used only for reporting metrics, never adaptation loss or network inputs.

Preserve hashes of all Step 2 sources, checkpoints, normalizers, data, and results, together with its earlier retired-artifact seal. Step 3 files live only in clean/shove_gate/adaptation_compare/. Reused files may be byte-verified symlinks. Preserve all 35 frozen Step 2 candidate checkpoints; use seeds **101,211,307,401,503**. Retired IDs 200–289 must never appear in Step 3 trace loading.

## Scores and fixed scales

All errors average steps within an episode, episodes within a cube, then cubes equally. Seed summaries are mean and sample SD across the five paired seeds. No population-size or trajectory-length weighting.

Using only IDs 0–179, shoves 0–7, derive and freeze:

- `E_push`: mean of `(delta_vx^2+delta_vy^2)/2` over push steps, exactly matching Step 2's fitting push energy.
- `E_coast`: the same horizontal delta energy over moving-coast steps. Moving coast retains Step 2's mask: coast AND (full linear speed >0.001 m/s OR angular speed >0.05 rad/s).
- `E_state`: mean of all 15 squared state changes divided coordinatewise by the frozen Step 2 state standard deviation, over all retained steps.

Scores are `push_velocity_nmse = horizontal_delta_error_mse/E_push`, `coast_velocity_nmse = horizontal_delta_error_mse/E_coast`, and `full_state_nmse = mean(((predicted_delta-true_delta)/state_std)^2)/E_state`. The full-state coordinate convention is unchanged; its additional scalar energy normalization makes its zero-change baseline comparable to the two velocity terms. Save the old full-state normalized MSE and raw velocity MSE as diagnostics. Also save all-coast velocity scores for transparency; the declared coast component uses moving coast, preserving the clean pipeline's protection against stationary-step dilution.

The predeclared comparison/selection score is the unweighted arithmetic mean of these three NMSE components, called `composite`. Every table and curve reports its three components separately. P1/P2 use this composite, and parallel component-specific verdicts are reported without replacing the declared overall verdict.

## Validation-only adaptation selection

Use only cubes 180–199 to choose settings, always support shoves 0–4 and query shoves 5–9. Select one global setting by the average query composite across all 20 cubes and five seeds; no per-seed choices.

**A: two-part.** Reuse Step 2 z4, z8, and z16 checkpoints. Freeze every network weight and all original embedding rows. For each cube initialize a new z at that checkpoint's mean training embedding. Fit z with deterministic full-support Adam, learning rate **0.01**, **50 updates**. The objective is equally weighted support-episode full-state normalized one-step MSE (the original 15-coordinate Step 2 loss, before dividing by E_state) plus `lambda * mean((z-training_mean_z)^2)`. The fixed grid is dimension **{4,8,16}** and positive lambda **{0.0001,0.001,0.01,0.1,1}**. The prior remains centered at the frozen training mean. Exact selection ties prefer smaller dimension, then larger penalty. The learning rate and update count are retained from Step 2 and are not searched again.

**B: fine-tune.** Clone each seed's frozen Plain checkpoint separately for every validation cube. Update ALL dynamics weights with deterministic full-support Adam, **50 updates**, full-state support one-step loss identical to A's data term, with no z or additional weight penalty. Compare exactly two learning rates **{0.00001,0.0001}**. Select the global lower validation composite; exact ties prefer the smaller rate. Report both rates in every fresh-test comparison. Only the validation-selected rate decides the primary P1/P2; the other is a declared sensitivity comparison. No test-based rate switch.

Use all retained support steps with equal episode weighting. No support phase mask or observed length is a numeric feature. Summing independent per-cube losses is allowed for vectorized A fitting; cubes must have independent z gradients. No query or property-label gradient is allowed. Save full validation grid, adaptation losses, settings, coverage, and unchanged base-checkpoint hashes. Report the selected A dimension/penalty and B learning rate, plus both candidate-rate validation scores, before fresh collection begins.

## Fresh-cube comparison and adaptation speed

After settings are frozen/reported, collect all 30 fresh cubes without rejection. Freeze/audit new data and evaluation implementation before predictive test scoring. Open one evaluation ledger; no retuning, checkpoint changes, or new candidates after it opens.

For A and BOTH B learning rates, independently fit each fresh cube using support prefixes of **1,2,3,4,5 shoves**, always query shoves **5–9**. Each prefix starts again from the frozen training-mean z (A) or original frozen Plain checkpoint (B), with 50 updates. Save adapted vectors/weights or reproducible state hashes and all losses. A's dynamics weights remain frozen; B's original checkpoint file is never modified.

History and Oracle are evaluated without adaptation on the same query shoves. History reads only the previous ten completed transitions of that query episode. Oracle alone receives true mass and combined friction. Include unadapted Plain as a descriptive reference. No mass decoder, new world-model training, or extra hyperparameter search is part of Step 3.

## Sequential forgetting and return

Visit cubes once in fixed ascending order **290,291,...,319**, using all five support shoves at each visit. Sequence experiments are separate from the independent prefix curves.

- A keeps its frozen shared network and stores each fitted z by cube. On return, reload the stored vector without any optimization. Its first-visit vector is identical to its five-support-shove prefix fit; store all visited vectors and verify they are unchanged.
- B starts each sequence from the original Plain checkpoint and carries the adapted weights from one cube to the next. Reset Adam moments at every new cube, retaining weights only; fit exactly 50 updates using only the current cube's shoves 0–4. No replay or mixing earlier support examples. Run independent sequences for both declared learning rates and all five seeds. Save model state hashes and snapshot weights at every declared observation point.

Before the first visit, score the original training-cube validation set, IDs 0–179 shoves 8–9. For A use the selected checkpoint's original learned embedding rows for these known cubes. After total visits **5,10,15,20,25,30**, re-score cube 290 query shoves 5–9 and that same training-cube validation set. Include the post-first-visit baseline and explicit return checks after **5,15,29 other cubes**, corresponding to total visits **6,16,30**. The union of snapshots is **1,5,6,10,15,16,20,25,30**. At every union snapshot, report first-cube and training-validation scores. A's cached identical predictions may be reused only after verifying unchanged weights/z hashes; cached use must be identified.

Return means scoring only, with no additional support observation or optimization. Test query scores and training-validation scores never alter B's sequence, updates, learning rate, A's vectors, or any setting. Report each currently visited cube's query scores as an additional descriptive sequence diagnostic.

## Predeclared P1 and P2

No hypothesis is presumed to hold. All five paired seeds and all 30 test cubes/five query shoves must be represented.

**P1:** At EACH explicit return (after 5,15,29 other cubes), A's first-cube error must not rise relative to its post-first-visit error, with absolute numerical tolerance **1e-12 + 1e-10 * abs(initial_error)** per seed. Verify identical shared-weight and stored-z hashes. For the validation-selected B rate, returning composite mean minus its sample SD must be strictly greater than initial composite mean plus its sample SD. Require all three returns for overall P1 PASS. Report per-return mean change, sample SD of paired changes, initial/return seed spread, and separate component-specific P1 checks. A non-rise and B rise must both hold; equality fails B's strict spread condition. The alternate B rate receives its own explicitly secondary verdict.

**P2:** From independent per-cube adaptation curves, find the earliest predeclared prefix k in {1,2,3,4,5} whose A mean composite is <= the validation-selected B rate's five-shove mean composite. If k<5, A reaches that accuracy with fewer shoves; k=5 matches or beats it at the same number. If none qualifies, P2 FAIL. A worse mean is not accepted merely because error bars overlap. Report all prefix seed spreads and the paired A_k minus B_5 differences, plus component-specific crossing counts and both B rates. No prefix affects model/config selection. Overall Step 3 success requires both P1 and P2; otherwise report the failed criterion plainly.

## Integrity, stops, and artifacts

Before adaptation, test full-state loss/episode weighting, frozen A weights, all-weight B gradients, strict support/query separation, History past-only inputs, prior centering, per-prefix reset, sequential weight carry with optimizer reset, cached-z retention, retired-ID rejection, metric normalization, return indexing, and P1/P2 boundaries. Synthetic checks may run only after this protocol exists.

Any changed sealed source/checkpoint/data, label leakage, unexpected identity/schema/computation error, missing required coverage, nonfinite adaptation loss/output/score, or invalid simulation causes STOP and an honest partial report. Do not silently repair or rerun a scientific experiment after such a failure. Finite unfavorable predictions and failed P1/P2 are results, not reasons to resample or loosen checks. No original checkpoint, old result, or trace may be overwritten. Report preparation/audit arithmetic may be corrected without new predictions.

Save protocol/source/checkpoint/data hashes; fitting-only metric scales; every validation grid score and selected settings; a setting-announcement marker before fresh generation; collection QA; per-seed/per-cube/prefix component scores; sequence snapshot states/hashes and scores; return comparisons; all criterion arithmetic; and plots from saved records only. An independent final audit recomputes scores/criteria from saved aggregates, verifies seals and chronology, and confirms no base checkpoint changed. Stop after reporting this comparison; no subsequent experiment is implied.
