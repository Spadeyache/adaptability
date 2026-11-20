# Step 2: fresh push confirmation and object-latent adaptation

This protocol and its reasons are written before any Step 2 collection, fitting, tuning, or predictive scoring. Source inspection only preceded this file. The prior direct-force and revised-velocity experiments remain unchanged and retain their reported failures. This is a new experiment, motivated by a previously observed secondary result, not a relabeling of that result.

## Reasons and scope

1. The previously reported 56% Oracle improvement in push horizontal velocity change was secondary. We promote that exact physical quantity to the primary endpoint before collecting new test cubes, and require independent confirmation (C1).
2. A latent fitted to training objects can memorize their identities. Twenty entirely held-out development cubes therefore tune latent dimension and adaptation, while a disjoint set of 30 fresh cubes measures transfer. Neither set contributes network-weight or training-embedding gradients.
3. History measures adaptation from recent observed transitions within an episode. The two-part model measures adaptation across completed shoves by fitting only an object vector while keeping dynamics weights fixed. Mean-z and another-cube-z controls test whether improvement is object-specific.
4. All network families keep the previous full15 normalized state-change training objective, MLP width/depth, optimizer, and budget rule. Checkpoint selection switches to the preregistered push metric. Test-time z fitting uses the push velocity-change loss because that is the adaptation/query task; it never receives true physical properties.
5. Repeated optimization on test query data would invalidate confirmation. All candidate checkpoints, latent-fitting settings, decoder coefficients, and data/source hashes are frozen before a single final test-evaluation ledger opens. C1 is evaluated first; if it fails, no test z fitting occurs. History is scored alongside Plain and Oracle in this first stage so it is reported whether C1 passes or fails.

## Data and fixed splits

Use the original 200 training cubes IDs0–199 and all ten shoves with byte-identical episode and raw files. Preserve original cube-property records; split roles live in a separate split file.

- Weight-fitting cubes: IDs0–179. Shoves0–7 fit network weights, training embeddings, and normalization.
- Checkpoint-validation episodes: shoves8–9 of those same 180 cubes. These do not fit weights or normalization; their push primary score selects checkpoints and controls budget extensions. For two-part candidates they use the learned embedding of that training cube.
- Latent-validation cubes: IDs180–199. These never fit network weights or training embeddings, nor normalization, checkpoints, budget extensions, probes, or mass decoders. They are used only to choose latent size and z-fitting learning rate/update count: support shoves0–4, query shoves5–9. This separation preserves the user's restriction that the 20 validation cubes are only for tuning the z procedure, while retaining a checkpoint-validation metric.
- Fresh test cubes: IDs260–289, seed **2026111519**, ten shoves each. Retire IDs200–259 from all Step2 fitting, tuning, and gating. Preserve their files/results unchanged.

Use the same independent property/action generator without rejection or resampling: mass uniform0.18–0.50kg, cube friction uniform0.4–1.0, floor friction0.3, configured combined friction `(cube_friction+0.3)/2`; direct horizontal COM force uniform4–7N, uniform direction, random position/yaw, duration uniform on60,65,…,100ms. Physics dt1ms, model dt5ms, fixed raw coast1.5s, the existing stop-confirmation and five-step trimming rule. Retain every sampled episode, including distances above1m. Collection QA may check schema, force, rest, trimming and hashes, but not predictive performance. An invalid simulation/schema stops the run; it is never replaced with another sample.

Per-cube true mass/friction values occur only in `data/cubes.json`. No model gets IDs, step, time, phase, episode length, stop time, termination flags, padding masks, or future observations as numeric features. Cube IDs may route a training embedding or a stored adaptation vector as explicitly required, but are never supplied to the dynamics network. Actions are applied forceXYZ, exactly zero in coast. Fixed-schedule actions-only probes train/tune only within the 180 weight-fitting cubes; fresh command-only probe scores are deferred until the final evaluation ledger opens and do not select or tune dynamics models. The 20 latent-validation cubes remain reserved.

## Primary metric and aggregation

For fitting cubes0–179, shoves0–7, and PUSH transitions only, compute

`S_push² = mean_cubes(mean_episodes(mean_push_steps((delta_vx²+delta_vy²)/2)))`.

No centering is applied. Freeze this positive scalar. Horizontal-delta NMSE is the same equally weighted aggregation of `((pred_delta_vx-true_delta_vx)²+(pred_delta_vy-true_delta_vy)²)/(2*S_push²)`. Predicting zero change scores1 on the fitting push subset. Calculate predicted raw velocity change directly from network delta output times fitting state standard deviations, avoiding subtracting two rounded predicted states. Also report raw MSE and per-coordinate RMSE.

The test primary population is all PUSH steps of shoves5–9 on all30 fresh cubes, averaged steps within episode, then episodes within cube, then cubes equally. Mean and sample SD are across five paired model seeds101,211,307,401,503. Report coverage and all seed scores. Coast remains secondary. Preserve separate push, moving-coast, and all-retained-step reporting groups. Moving coast retains the previous full-linear-speed>0.001m/s OR angular-speed>0.05rad/s mask, intersected with coast, for scoring only.

## Networks and fitting

All dynamics heads are three hidden ReLU layers of256 units, predicting15 normalized state changes. State/action normalization and context normalization use only the fitting180cubes×8shoves; retain the existing0.001 native-unit standard-deviation floor.

- **Plain:** normalized current state15 + applied force3.
- **Oracle:** the same plus true mass and combined friction, normalized from the180 fitting cubes. It is a privileged ceiling only.
- **History:** current state/action plus a64-unit GRU summary of exactly ten preceding completed transitions from the same episode. Each token is normalized previous state15, action3, and observed normalized state change15. Tokens end strictly before the current transition. Missing pre-episode tokens repeat that episode's observed initial state with zero action and zero delta. There is no mask, length input, or GRU state carried between episodes. Padding depends only on episode start, never on observed stopping/trimmed length. In recursive rollouts, append predicted transitions after the forecast start, never observed future transitions.
- **Two-part:** current state/action plus z. Train four candidates with z dimensions2,4,8,16. Each has a zero-initialized `nn.Embedding(180,zdim)` learned jointly with its dynamics MLP. No true property enters this model, its loss, or its embedding. Add `1e-4 * mean(all_training_embedding_entries²)` to the full-state training loss. Candidate dimension is selected using latent-validation cubes only.

Train from scratch with Adam, batch512, uniform fitting episode then uniform retained step. Within each seed, all seven variants (Plain, Oracle, History, and four z dimensions) see identical minibatch row streams and equal update budgets. Adam lr0.001 cosine-decays to0.0001 over the first10,000 updates, then stays0.0001. Every500 updates, select each candidate's minimum checkpoint-validation PUSH horizontal-delta NMSE. Save all validation checks and chosen updates.

The previous extension rule applies jointly to the seven variants: run at least10,000 updates. At every10,000-update endpoint, compare each variant's best validation score through `endpoint−2500` with its best score over the final2500 updates. If ANY variant improves its previous best by at least0.1%, extend ALL seven by another10,000. Otherwise stop that seed. Maximum100,000 updates; report any seed still improving at this resource bound. No latent-validation or test result may alter checkpoint selection or budgets.

## Frozen-weight z fitting and validation-only choices

For an unseen cube, initialize z to the mean of the180 learned training embeddings at that candidate's selected checkpoint. Freeze every network parameter. Optimize only a new z vector, using observed PUSH transitions from the specified support shoves. The objective is their equal-episode push horizontal-delta NMSE plus `1e-4*mean((z−training_mean_z)²)`. Use full-support deterministic Adam (no minibatch selection) and no clipping/projection. No query transition, property label, ID feature, or observed termination information enters the fitting objective. Vectorizing cubes is allowed with a sum of independent per-cube losses; vectors never share gradients or observations.

Predeclare fitting learning rates **{0.01,0.03,0.1}** and update counts **{50,100,200,400}**. For each dimension, seed, and learning rate, begin at the training mean and run400 fitting updates on the20 validation cubes' support shoves0–4; save z at the four declared update counts. Score each choice only on validation shoves5–9 push steps. Choose ONE global `(dimension, learning_rate, updates)` minimizing mean validation score across all20 cubes and five seeds. Exact ties prefer smaller dimension, fewer updates, then smaller learning rate. No seed-specific choice, search expansion, or manual override.

Freeze the selected procedure before opening test query predictions. For each fresh cube and each support count k=1,2,3,4,5, independently restart from its model's training-mean z and fit on shoves0…k−1 using the SAME selected learning rate and update count. Score every prefix on the fixed query shoves5–9. This avoids confounding additional support with additional optimization passes. C2/C3 use only k=5; the prefix curve is descriptive.

For query controls, use (a) unchanged training-mean z and (b) another test cube's k=5 fitted z. The latter is a fixed cyclic derangement:260 receives261's z,…,288 receives289's z,289 receives260's z. It is never chosen by mass, similarity, or query performance.

## Mass-readout diagnostic

After candidate selection, refit z for each of the180 fitting cubes from its shoves0–4 with the frozen selected network and the same mean initialization, learning rate, updates, and objective used for new cubes. Fit an affine ordinary-least-squares map `[1,z] -> true_mass` separately per seed on those180 vectors only. Use a pseudoinverse, with no decoder hyperparameter search. Freeze coefficients before test scoring. Score it on the30 test k=5 fitted vectors and report R², MAEkg, RMSEkg and a training-mean-mass baseline. Do not substitute jointly learned embedding rows for refitted vectors. Physical labels are permitted only for this diagnostic and Oracle; decoder results do not choose models, z settings, or gates.

## Frozen evaluation and criteria

Seal data definitions, all raw/model trace hashes, implementation, fitting-only normalizers, all35 selected candidate checkpoints, validation tuning results, chosen adaptation configuration, and mass decoders before any fresh predictive test evaluation. Open one final evaluation ledger; resume only with identical frozen inputs and checkpoints. Training, retuning, new candidate selection, or fresh draws after this point are forbidden.

Stage A scores Plain, Oracle, and History on the150 fresh query shoves (5–9). Their observed past History tokens may use earlier transitions of that same query shove; the current target/future is forbidden. Compute C1 before fitting test z. If C1 fails, stop, report these baselines and mark the two-part test unperformed.

If C1 passes, Stage B performs the preregistered support-prefix z fits and query scores, then mean-z, wrong-cube-z and mass-readout diagnostics. All belong to that same frozen evaluation; query scores never feed back into fitting or selection.

- **C1:** on primary push NMSE, Oracle mean ≤0.8×Plain mean AND `Oracle mean+SD < Plain mean−SD`. Require all30 cubes and all150 query episodes. If false, stop.
- **C2:** `(Plain mean − fitted_z_5_mean)/(Plain mean − Oracle mean) >= 0.5`. The positive denominator is established by C1. This is a five-seed-mean gap-recovery criterion; also report per-seed recoveries without imposing an unrequested spread threshold.
- **C3:** `fitted_z_5_mean+SD < mean_z_mean−SD` AND `fitted_z_5_mean+SD < wrong_z_mean−SD`. Both comparisons must hold. Equality fails strict spread comparisons.

Report History next to all applicable comparisons regardless of their outcome. Report each C separately; no alternative aggregation, metric, support prefix, decoder success, or secondary result can rescue a failed criterion.

## Secondary forecasts and integrity stops

On query shoves5–9 retain full15 one-step normalized-state MSE, recursive10-step state NMSE averaged across horizons (plus tenth-step diagnostic), and stopping-position XY error in meters for push, moving coast, and all steps. Report these for each scored baseline and the k=5 fitted/mean/wrong-z models. Prefix curves require primary scores; they do not select a model. Recursive predictions use their own intermediate states/history and the preplanned applied-force sequence. The stopping horizon is planned release+1.5s, never observed stop time/trimmed length. z remains fixed throughout a query forecast. No physics correction, ground-truth reset, or output clipping.

These secondary numbers decide no criterion. Large finite secondary errors must be reported explicitly, including the offending seeds, without dropping or replacing them. They do not by themselves change the gate because the user explicitly made these outcomes secondary. Any integrity/leakage failure, changed sealed bytes, inconsistent coverage, nonfinite training/adaptation loss, nonfinite predictive output, or unexpected schema/computation failure stops execution and produces an honest incomplete/failed report. Do not silently repair, rerun with different settings, relax checks, or resample after such a failure.

## Artifacts and verification

All Step2 work lives in `clean/shove_gate/push_latent/`; preserve every prior experiment. Before fitting, verify strict past-only History, normalization/split guards, zero-initialized embedding routing, label independence outside Oracle/decoder, frozen weights during z fitting, query-independent adaptation, cyclic wrong-z assignment, criterion boundaries, and retired-test rejection. Save per-seed curves, adaptation validation grid, prefix curves, per-cube test metrics, byte/source/checkpoint/config seals, and an independent final arithmetic/integrity audit. Every plotted point comes from a saved run. No further experiment is implied by this task after reporting the frozen comparison.
