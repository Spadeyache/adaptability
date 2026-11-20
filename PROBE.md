# Dataset information probe: stopped at Step 0

**No probes were trained.** Direct mass/friction encodings were found in older saved observations. The requested stop rule therefore triggered before the one-dataset, one-seed timing run. No existing model or dataset was modified, and no new simulator data was collected.

This inventory does not establish whether physical motion reveals mass or friction. It separates suitable candidate recordings from privileged inputs, derived outputs, and recordings without the required split.

## Verified stop evidence

| Saved artifact | Actual check | Consequence |
|---|---|---|
| `runs/interaction_adaptation/initial_policy_probe/episodes/condition2_sequence0_true_attempt0.npz` | `sim__policy_observation[:,42]` = 4.90500021 N; true mass = 0.5 kg. This is mass × gravity. | Direct answer inside the observation; true-mode policy also receives it. |
| `runs/physics_conditioning/baseline_freshscene_diagnostic/traces.jsonl` | Observation columns 42 and 45 match true mass × gravity and true friction for all 15 saved lanes in the checked true-mode step. | Both answers are inside the observation; actions were generated with these inputs. |
| Legacy inspection traces | Raw `state.input_vector` includes `object_weight_n`; some policy modes override that weight with a nominal value. | Saved raw state is privileged even when the policy used a nominal weight. |

Exact values and source references: [leak_gate.json](/home/ubuntu/faliure/results/probe/leak_gate.json). The first two examples were independently checked. Wrong-parameter modes can also be invertible functions of truth; they are not automatically safe controls.

## Main lift, slide, and shove collections

These collections keep `true_mass` and `true_friction` outside observed state/action. No direct label injection was found in their documented state/action builders. Policy lifts use a constant mass guess; scripted speed, height, direction, force, and grip randomization do not read true cube parameters. The actions-only learned control has **not** been run, so this is a source audit, not a statistical control pass.

| Dataset | Tasks and total episodes per task | Cubes/configurations | Episodes per cube/task | Steps per episode (count histogram) | Mass–friction r |
|---|---|---|---|---|---|
| nov13_main | {"lift":2300,"slide":2300} | {"test":30,"train":200} | 10 | {"50":4600} | -0.0927372 |
| paired_fixed | {"lift":2300,"slow_slide":2300,"shove":2300} | {"test":30,"train":200} | 10 | {"120":4600,"180":2300} | -0.0927372 |
| paired_random | {"lift":2300,"slow_slide":2300,"shove":2300} | {"test":30,"train":200} | 10 | {"120":4600,"180":2300} | -0.0927372 |
| pilot_constant_cap | {"lift":96,"slow_slide":96} | {"diagnostic_only":4} | see per-file records; no equal-count assumption | {"120":192} | 0 |
| pilot_staged_cap | {"lift":96,"slow_slide":96} | {"diagnostic_only":4} | see per-file records; no equal-count assumption | {"120":192} | 0 |
| smoke_fixed | {"lift":20,"slow_slide":20,"shove":20} | {"test":1,"train":1} | see per-file records; no equal-count assumption | {"120":40,"180":20} | -1 |
| smoke_random | {"lift":20,"slow_slide":20,"shove":20} | {"test":1,"train":1} | see per-file records; no equal-count assumption | {"120":40,"180":20} | -1 |
| three_task_main | {"lift":2300,"slow_slide":2300,"shove":2300} | {"test":30,"train":200} | 10 | {"120":4600,"180":2300} | -0.0927372 |
| oracle_complement_cache | {"shove":412} | {"inherited_test":30} | see per-file records; no equal-count assumption | {"180":412} | 0.211823 |
| target_candidate_cache | {"shove":1538} | {"inherited_test":30} | see per-file records; no equal-count assumption | {"180":1538} | 0.211823 |

Counts include stored variants and shared recordings; they must not be added as independent cubes. The full collections share the same 230 cube identities. Smoke test cubes overlap the full training set; pilot and target caches do not provide the requested independent train/test population.

| Dataset | Exact saved roots | Generation / grip |
|---|---|---|
| nov13_main | ["runs/lift_slide_belief/nov13/data/test","runs/lift_slide_belief/nov13/data/train"] | Frozen RL-policy lifts with constant40g policy mass guess; scripted positional pushes; additive Gaussian action noise SD0.05;50commands/episode. |
| nov13_belief_banks | ["runs/lift_slide_belief/nov13/study/seed_0","runs/lift_slide_belief/nov13/study/seed_1","runs/lift_slide_belief/nov13/study/seed_2","runs/lift_slide_belief/nov13/study/seed_3","runs/lift_slide_belief/nov13/study/seed_4"] | Derived learned-belief or prediction/evaluation artifact; not a simulator transition dataset. |
| paired_fixed | ["runs/lift_slide_belief/slip_probe/paired/fixed/data/test","runs/lift_slide_belief/slip_probe/paired/fixed/data/train"] | Same three-task script and seed42 physics/motion as three_task_main; fixed30.0625N grip; additional observed sidechannels. |
| paired_random | ["runs/lift_slide_belief/slip_probe/paired/random/data/test","runs/lift_slide_belief/slip_probe/paired/random/data/train"] | Same paired scene/motion seeds; fixed30.0625N acquisition then truth-independent stratified log-force draw .25–30.0625N atstep45 for lift/slowpush; shoves shared unchanged. |
| paired_fixed_belief_banks | ["runs/lift_slide_belief/slip_probe/paired/study/fixed_grip/seed_0","runs/lift_slide_belief/slip_probe/paired/study/fixed_grip/seed_1","runs/lift_slide_belief/slip_probe/paired/study/fixed_grip/seed_2","runs/lift_slide_belief/slip_probe/paired/study/fixed_grip/seed_3","runs/lift_slide_belief/slip_probe/paired/study/fixed_grip/seed_4"] | Derived learned-belief or prediction/evaluation artifact; not a simulator transition dataset. |
| paired_random_belief_banks | ["runs/lift_slide_belief/slip_probe/paired/study/random_grip/seed_0","runs/lift_slide_belief/slip_probe/paired/study/random_grip/seed_1","runs/lift_slide_belief/slip_probe/paired/study/random_grip/seed_2","runs/lift_slide_belief/slip_probe/paired/study/random_grip/seed_3","runs/lift_slide_belief/slip_probe/paired/study/random_grip/seed_4"] | Derived learned-belief or prediction/evaluation artifact; not a simulator transition dataset. |
| pilot_constant_cap | ["runs/lift_slide_belief/slip_probe/pilot"] | Fullfactorial4mass/friction corners×3resetseeds×8gripcaps×2tasks; same original lift/slowpush; cap held throughout episode. |
| pilot_staged_cap | ["runs/lift_slide_belief/slip_probe/pilot_step45"] | Same fullfactorial pilot; fixed30.0625N untilstep45 then selectedcap independent of physics. |
| smoke_fixed | ["runs/lift_slide_belief/slip_probe/smoke_pair/fixed/data/test","runs/lift_slide_belief/slip_probe/smoke_pair/fixed/data/train"] | Two-cube smoke version of paired_fixed; IDs0/1 already occur in full training set. |
| smoke_random | ["runs/lift_slide_belief/slip_probe/smoke_pair/random/data/test","runs/lift_slide_belief/slip_probe/smoke_pair/random/data/train"] | Two-cube smoke version of paired_random; IDs0/1 already occur in full training set. |
| three_task_main | ["runs/lift_slide_belief/three_task/data/test","runs/lift_slide_belief/three_task/data/train"] | Scripted random-speed/height lift; random-speed/direction unilateral slow push; random1–4N horizontal robot-wrench shove for0.3s then withdrawal/coast. Fixed30.0625N grip. |
| force_cap_native_traces | ["runs/lift_slide_belief/three_task/diagnostics/force_cap"] | 100Hz diagnostic interventions on2heldout setups: baseline, zero-horizontal, novertical, noorientation, horizontal-only, lowfinger-cap, zeroarmvelocity, softwithdrawal. Not ordinary training trajectories. |
| oracle_complement_cache | ["runs/lift_slide_belief/three_task/diagnostics/oracle/trajectories"] | Remaining force-grid candidates absent from target cache, executed to enumerate all13candidates for150heldout setups; fixed candidate commands, not oracle-selected controls. |
| rollout_cube_one_step_metrics | ["runs/lift_slide_belief/three_task/diagnostics/rollout"] | Derived learned-belief or prediction/evaluation artifact; not a simulator transition dataset. |
| rollout_signed_distance_trials | ["runs/lift_slide_belief/three_task/diagnostics/rollout"] | Derived learned-belief or prediction/evaluation artifact; not a simulator transition dataset. |
| three_task_belief_banks | ["runs/lift_slide_belief/three_task/study/seed_0","runs/lift_slide_belief/three_task/study/seed_1","runs/lift_slide_belief/three_task/study/seed_2","runs/lift_slide_belief/three_task/study/seed_3","runs/lift_slide_belief/three_task/study/seed_4"] | Derived learned-belief or prediction/evaluation artifact; not a simulator transition dataset. |
| target_planner_choices | ["runs/lift_slide_belief/three_task/study/target_control"] | Derived learned-belief or prediction/evaluation artifact; not a simulator transition dataset. |
| target_candidate_cache | ["runs/lift_slide_belief/three_task/study/target_control/trajectories"] | Actual replayed shove candidates selected by frozen model+inferredbelief planner; candidate grid1–4N is fixed and physics-independent; true physics used only for simulator reset. |
| target_trials | ["runs/lift_slide_belief/three_task/study/target_control"] | Derived learned-belief or prediction/evaluation artifact; not a simulator transition dataset. |

For paired fixed/random and three-task data: train r = -0.141737461; test r = 0.211823308; all cubes r = -0.0927372496. These are observed sample correlations; independent sampling does not imply exactly zero sample correlation.

All per-file fields, schemas, shapes, label-constancy checks, and split identities: [inventory_lift_slide.json](/home/ubuntu/faliure/results/probe/inventory_lift_slide.json). Its groups also distinguish belief banks, planner/oracle outputs, native force traces, and raw transitions.

Raw state has joint positions/velocities, gripper width, cube position/orientation/linear and angular velocities, arm effort, and finger forces. Paired datasets additionally log relative motion, finger error, actuator/contact signals, and episode diagnostics; explicitly labeled truth, IDs, and future outcome summaries are not probe features.

## Other recorded datasets

A physical setting is not automatically a persistent cube. Where no cube IDs or existing cube-disjoint test split are saved, that requirement remains unresolved; no split was invented. Null correlation means insufficient variation or missing labels.

| Saved path | Task(s) | Episodes or images | Distinct physical configurations / objects | Steps per episode | Mass–friction r |
|---|---|---|---|---|---|
| runs/context_dynamics/nov13/confirmation | ["PickCube-v1","PushCube-v1","PullCube-v1","poke"] | 1000 | 1000 | [50] | -0.0683276 |
| runs/context_dynamics/nov13/data | ["PickCube-v1","PushCube-v1","PullCube-v1","poke"] | 40000 | 40000 | [50] | -0.00499909 |
| runs/context_dynamics/nov13/grip_data | ["PickCube-v1","PushCube-v1","PullCube-v1","poke"] | 6000 | 6000 | [50] | 0.0101477 |
| runs/context_dynamics/nov13/grip_pulses | ["PickCube-v1","PushCube-v1","PullCube-v1","poke"] | 6000 | 6000 | [50] | 0.0101477 |
| runs/context_dynamics/nov13/matched_train | ["PickCube-v1","PushCube-v1","PullCube-v1","poke"] | 8000 | 8000 | [50] | -0.0100858 |
| runs/context_dynamics/pick_gpu_2m/data | ["PickCube-v1"] | 40000 | 40000 | [50] | -0.00298785 |
| runs/context_dynamics/pick_gpu_2m/pilot | ["PickCube-v1"] | 10 | 10 | [50] | -0.127205 |
| runs/context_dynamics/pick_gpu_5m_excluded/data | ["PickCube-v1"] | 14000 | 14000 | [50] | -0.0036029 |
| runs/context_dynamics/pick_gpu_5m_excluded/pilot | ["PickCube-v1"] | 10 | 10 | [50] | -0.127205 |
| runs/mass_belief/nov17_v1/calibration | ["Pick, lift, hold"] | 40 | 40 | [100] | unknown / undefined |
| runs/mass_belief/nov17_v1/closed_loop | ["Pick, lift, hold"] | 120 | 1 | [100] | unknown / undefined |
| runs/mass_belief/nov17_v1/test | ["Pick, lift, hold"] | 60 | 1 | [100] | unknown / undefined |
| runs/mass_belief/nov17_v1/train | ["Pick, lift, hold"] | 100 | 100 | [100] | unknown / undefined |
| runs/mass_belief/nov17_v1/validation | ["Pick, lift, hold"] | 24 | 24 | [100] | unknown / undefined |
| runs/mass_belief/switch_80g_mu1/episodes | ["Pick, lift, hold"] | 60 | 1 | [100] | unknown / undefined |
| runs/rma/adapter_validation_20261107 | ["Randomized lift task; in-range validation or integration smoke"] | 216 | 9 | [50] | 2.64339e-18 |
| runs/rma/baseline_20261106_persistent/base_validation_01 | ["Randomized lift task; in-range validation or integration smoke"] | 108 | 9 | [50] | 2.64339e-18 |
| runs/rma/baseline_20261106_persistent/base_validation_02 | ["Randomized lift task; in-range validation or integration smoke"] | 108 | 9 | [50] | 2.64339e-18 |
| runs/rma/baseline_20261106_persistent/base_validation_03 | ["Randomized lift task; in-range validation or integration smoke"] | 108 | 9 | [50] | 2.64339e-18 |
| runs/rma/integration_evaluation | ["Randomized lift task; in-range validation or integration smoke"] | 2 | 1 | [50] | unknown / undefined |
| runs/interaction_estimator/contact_dataset_v1 | ["Contact snippets from independent fixed-policy lift attempts"] | 440 | 44 | 100 | -0.0350929 |
| runs/visual_mass/initial_rgb_mass/dataset | ["Initial RGB before any interaction"] | 384 | 96 | unknown / undefined | unknown / undefined |
| runs/interaction_adaptation/poststop_cli_smoke_20261027T024742/transfer/episodes | Native carry or vertical pick-and-hold as environment field specifies; no slide task | 1 | 1 | {"5":1} | unknown / undefined |
| runs/interaction_adaptation/poststop_factory_smoke_20261027T024613 | Native carry or vertical pick-and-hold as environment field specifies; no slide task | 2 | 1 | {"5":2} | unknown / undefined |
| runs/interaction_adaptation/poststop_cli_smoke_20261027T024742/retry/episodes | Native carry or vertical pick-and-hold as environment field specifies; no slide task | 3 | 1 | {"5":3} | unknown / undefined |
| runs/failure_collection/stock_short | Stock random-XYZ pick/carry goal | 10 | 1 | {"50":10} | unknown / undefined |
| runs/physics_conditioning/checkpoint_probe_smoke/six_mode_native_carry/episodes | Native training carry geometry, randomized XY and floating goal Z; CPU fresh scenes | 12 | 2 | {"50":12} | -1 |
| runs/interaction_adaptation/initial_policy_probe/episodes | Native carry or vertical pick-and-hold as environment field specifies; no slide task | 20 | 3 | {"50":20} | 0.977356 |
| runs/physics_conditioning/long_50000_guarded_20261026/diagnostics/checkpoint2001_cpu_retries/episodes | Native carry or vertical pick-and-hold as environment field specifies; no slide task | 68 | 9 | {"50":68} | -0.281601 |
| runs/physics_conditioning/long_50000_guarded_20261026/diagnostics/checkpoint501_cpu_retries/episodes | Native carry or vertical pick-and-hold as environment field specifies; no slide task | 71 | 9 | {"50":71} | -0.281601 |
| runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/transfer_vertical/episodes | Native carry or vertical pick-and-hold as environment field specifies; no slide task | 80 | 16 | {"50":80} | 0 |
| runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/transfer_native_carry/episodes | Native carry or vertical pick-and-hold as environment field specifies; no slide task | 80 | 16 | {"50":80} | 0 |
| runs/interaction_adaptation/transfer_pilot200_crossed/episodes | Native carry or vertical pick-and-hold as environment field specifies; no slide task | 80 | 16 | {"50":80} | 0 |
| runs/physics_conditioning/corrective_training_20261026/guarded100_cpu_retries/episodes | Native carry or vertical pick-and-hold as environment field specifies; no slide task | 82 | 9 | {"50":82} | -0.281601 |
| runs/interaction_adaptation/pilot200_retry_diagnostic/episodes | Native carry or vertical pick-and-hold as environment field specifies; no slide task | 82 | 9 | {"50":82} | -0.281601 |
| runs/interaction_adaptation/diagnostic_iteration2001_retries/episodes | Native carry or vertical pick-and-hold as environment field specifies; no slide task | 160 | 9 | {"50":160} | -0.281601 |
| runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/retries_vertical/episodes | Native carry or vertical pick-and-hold as environment field specifies; no slide task | 327 | 9 | {"50":327} | -0.281601 |
| runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/retries_native_carry/episodes | Native carry or vertical pick-and-hold as environment field specifies; no slide task | 350 | 9 | {"50":350} | -0.281601 |
| runs/failure_collection/varied_fixed40g | Vertical pick, lift and hold directly above spawn; no slide task | 600 | 53 | {"100":600} | -0.0526732 |
| runs/physics_conditioning/checkpoint_comparison_20261026/ckpt_2001/episodes | Native training carry geometry, randomized XY and floating goal Z; CPU fresh scenes | 1440 | 24 | {"50":1440} | -0.00286593 |
| runs/physics_conditioning/checkpoint_comparison_20261026/ckpt_5001/episodes | Native training carry geometry, randomized XY and floating goal Z; CPU fresh scenes | 1440 | 24 | {"50":1440} | -0.00286593 |
| runs/physics_conditioning/checkpoint_comparison_20261026/ckpt_8501/episodes | Native training carry geometry, randomized XY and floating goal Z; CPU fresh scenes | 1440 | 24 | {"50":1440} | -0.00286593 |
| runs/failure_collection/dense_single_pair_fixed40g | Vertical pick, lift and hold directly above spawn; no slide task | 29830 | 1775 | {"100":29830} | -0.025043 |
| runs/maniskill_pretrained/pick_cube/20261007T160411-c3dd5f0f/custom_lift | custom_lift | 20 | no persistent cube IDs | {"100":20} | friction not saved |
| runs/maniskill_pretrained/pick_cube/20261007T160411-c3dd5f0f/official_pick_cube | official_pick_cube | 20 | no persistent cube IDs | {"50":20} | friction not saved |
| runs/maniskill_pretrained/pick_cube/20261007T161108-28f3d274/custom_lift | custom_lift | 20 | no persistent cube IDs | {"100":20} | friction not saved |
| runs/maniskill_pretrained/pick_cube/20261007T161108-28f3d274/official_pick_cube | official_pick_cube | 20 | no persistent cube IDs | {"50":20} | friction not saved |
| runs/maniskill_pretrained/pick_cube/20261007T221109-3298edee/custom_lift | custom_lift | 1 | no persistent cube IDs | {"50":1} | friction not saved |
| runs/maniskill_pretrained/pick_cube/20261008T032243-6e4c9370/custom_lift | custom_lift | 1 | no persistent cube IDs | {"50":1} | friction not saved |
| runs/physics_conditioning/baseline_diagnostic_retry/traces.jsonl | physics-conditioned lift | 30 | 15 batch lanes; no cube IDs | {"50":2} | see paired physical-setting inventory |
| runs/physics_conditioning/baseline_freshscene_diagnostic/traces.jsonl | physics-conditioned lift | 30 | 15 batch lanes; no cube IDs | {"50":2} | see paired physical-setting inventory |
| runs/physics_conditioning/corrective_gpu_baselines_20261026/old2001_grid/traces.jsonl | physics-conditioned lift | 15 | 15 batch lanes; no cube IDs | {"50":1} | see paired physical-setting inventory |
| runs/physics_conditioning/corrective_gpu_baselines_20261026/old2001_offgrid/traces.jsonl | physics-conditioned lift | 9 | 9 batch lanes; no cube IDs | {"50":1} | see paired physical-setting inventory |
| runs/physics_conditioning/corrective_gpu_baselines_20261026/pilot200_grid/traces.jsonl | physics-conditioned lift | 15 | 15 batch lanes; no cube IDs | {"50":1} | see paired physical-setting inventory |
| runs/physics_conditioning/corrective_gpu_baselines_20261026/pilot200_offgrid/traces.jsonl | physics-conditioned lift | 9 | 9 batch lanes; no cube IDs | {"50":1} | see paired physical-setting inventory |
| runs/physics_conditioning/corrective_training_20261026/guarded100_grid/traces.jsonl | physics-conditioned lift | 15 | 15 batch lanes; no cube IDs | {"50":1} | see paired physical-setting inventory |
| runs/physics_conditioning/corrective_training_20261026/guarded100_offgrid/traces.jsonl | physics-conditioned lift | 9 | 9 batch lanes; no cube IDs | {"50":1} | see paired physical-setting inventory |
| runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/policy/traces.jsonl | physics-conditioned lift | 30 | 15 batch lanes; no cube IDs | {"50":2} | see paired physical-setting inventory |
| runs/physics_conditioning/pilot200_freshscene_diagnostic/traces.jsonl | physics-conditioned lift | 30 | 15 batch lanes; no cube IDs | {"50":2} | see paired physical-setting inventory |

## Legacy inspection and deployment traces

Here the unit is a saved trace file. The full manifest lists every path, row count, action-row count, field schema, and recorded step bounds. Missing physical labels are marked unknown, rather than counted as zero cubes.

| Root | Trace files | Recovered mass/friction pairs | Saved final-step/step-count histogram | r over recovered unique pairs |
|---|---|---|---|---|
| runs/article_inspired/cube_lift | 6 | unknown | {"100":6} | unknown / undefined |
| runs/checks/alignment | 9 | unknown | {"50":9} | unknown / undefined |
| runs/checks/grip_extension | 6 | unknown | {"50":6} | unknown / undefined |
| runs/deployment/20261021_234802_550312 | 1 | 1 | {"9":1} | unknown / undefined |
| runs/deployment/20261021_234906_513736 | 1 | 1 | {"9":1} | unknown / undefined |
| runs/deployment/20261021_235109_413496 | 1 | 1 | {"50":1} | unknown / undefined |
| runs/deployment/20261022_014201_067220 | 3 | 3 | {"50":3} | unknown / undefined |
| runs/diagnosis/weight-adaptation-20261008 | 85 | unknown | {"60":12,"50":13,"100":60} | unknown / undefined |
| runs/inspect/20261004T183600-9f9f3c53 | 1 | unknown | {"100":1} | unknown / undefined |
| runs/inspect/20261004T183739-111848f0 | 1 | unknown | {"100":1} | unknown / undefined |
| runs/inspect/20261004T184018-30321200 | 1 | unknown | {"100":1} | unknown / undefined |
| runs/inspect/20261004T184658-45fedb37 | 1 | unknown | {"100":1} | unknown / undefined |
| runs/inspect/20261004T184727-c1fb4e69 | 1 | unknown | {"100":1} | unknown / undefined |
| runs/inspect/20261004T190445-2812f7f4 | 1 | unknown | {"8":1} | unknown / undefined |
| runs/inspect/20261004T190612-b6c9e132 | 1 | unknown | {"100":1} | unknown / undefined |
| runs/inspect/20261004T190859-601f2196 | 1 | unknown | {"400":1} | unknown / undefined |
| runs/inspect/20261005T152308-817384cb | 1 | unknown | {"400":1} | unknown / undefined |
| runs/inspect/20261005T165416-7e88f055 | 1 | unknown | {"400":1} | unknown / undefined |
| runs/inspect/20261008T021816-05bc01b7 | 100 | unknown | {"50":100} | unknown / undefined |
| runs/inspect/20261008T024643-86f70895 | 20 | unknown | {"50":20} | unknown / undefined |
| runs/inspect/20261008T024914-21df5d38 | 20 | unknown | {"50":20} | unknown / undefined |
| runs/inspect/20261008T025442-5d43bcd3 | 60 | unknown | {"50":60} | unknown / undefined |
| runs/inspect/20261008T122327-6746d22d | 100 | unknown | {"50":100} | unknown / undefined |
| runs/inspect/20261008T130546-a6f1ea05 | 100 | unknown | {"50":100} | unknown / undefined |
| runs/inspect/20261008T190610-40c9c0c7 | 1 | unknown | {"50":1} | unknown / undefined |
| runs/inspect/20261008T192509-bc35f615 | 100 | unknown | {"50":100} | unknown / undefined |
| runs/inspect/20261008T211232-d3e19b0b | 100 | unknown | {"50":100} | unknown / undefined |
| runs/inspect/20261009T040017-2aab8e95 | 100 | 5 | {"50":100} | -1 |
| runs/inspect/20261009T105845-63d5e45c | 100 | 5 | {"50":100} | -1 |
| runs/inspect/20261009T122414-11e4a49e | 100 | 5 | {"50":100} | -1 |
| runs/maniskill_example/pick_cube | 7 | unknown | {"50":6,"49":1} | unknown / undefined |

Three saved inspection runs use `mass_correlated_linear` friction; their measured correlation across the five physical settings is −1 to floating-point precision. They cannot establish separate mass/friction identification.

## Field inventory, labels, generation, and exclusions

- [inventory_lift_slide.json](/home/ubuntu/faliure/results/probe/inventory_lift_slide.json): every NPZ file and schema in the lift/slide family, state/action column maps, per-cube label constancy, RNG provenance, source hashes, and related result artifacts.
- [inventory_historical.json](/home/ubuntu/faliure/results/probe/inventory_historical.json): context dynamics, interaction snippets, mass belief, RMA, and initial RGB. Each group lists saved fields, generation/grip, labels, splits, and correlations.
- [inventory_other.json](/home/ubuntu/faliure/results/probe/inventory_other.json): all raw failure collection, interaction adaptation, physics conditioning, pretrained policy traces, and referenced visual-review archives. Field lists and dimensions are recorded, with per-family label and policy-input checks.
- [inventory_legacy.json](/home/ubuntu/faliure/results/probe/inventory_legacy.json): every legacy JSONL/NPZ trace path and complete flattened field schemas, plus checkpoints, archived feature arrays, and aggregate logs classified separately.
- [file_inventory.json](/home/ubuntu/faliure/results/probe/file_inventory.json): candidate containers across the repo. This includes checkpoints, reports, and vendored library test fixtures; its file count is not a dataset count. RGB images are inventoried through the visual-mass manifest.

Mass/friction storage is confirmed per cube for the main lift/slide collections, and per episode or physical configuration for the historical families that save labels. Some old traces lack friction labels. The raw collection schemas save mass, weight, inertia, materials, and other truth metadata under explicit label fields; none may be blanket-concatenated with observations.

Additional limitations recorded in the inventory:

- RMA oracle and true-mode mass-belief actions were generated using true parameters even when their saved state vectors remove those parameters.
- Context-dynamics constants and true-physics embeddings are privileged side fields. The saved kinematic state is separate.
- Visual mass is an image dataset with mass determined by object size and fixed density; it has no interaction sequence and no varied-friction target.
- Fixed-friction or single-setting datasets cannot yield a meaningful friction R². Smoke datasets with one held-out cube also cannot support across-cube R².
- Existing episode splits are not automatically cube-disjoint splits. Duplicated raw/prepared/archive recordings and shared shoves cannot be treated as independent evidence.
- Correlation in feedback-generated actions can reflect observed physical responses. An actions-only probe would diagnose information in that channel; source inspection is needed to distinguish direct label use from ordinary feedback.

## Probe results

**All cells below are not run.** No training, validation, shuffled-label, actions-only, ablation, moving/still, or pooling score exists for this diagnostic. No GRU bar chart was generated. Zero scores would incorrectly imply measurements. The mean-reference R²=0 is a definition, not an executed control; a train-mean predictor evaluated on test data need not score zero.

| Dataset × task | Simple mass R² | Simple friction R² | GRU mass R² | GRU friction R² | Shuffled mass R² | Shuffled friction R² | Actions mass R² | Actions friction R² |
|---|---|---|---|---|---|---|---|---|
| nov13_main × lift | not run | not run | not run | not run | not run | not run | not run | not run |
| nov13_main × slide | not run | not run | not run | not run | not run | not run | not run | not run |
| paired_fixed × lift | not run | not run | not run | not run | not run | not run | not run | not run |
| paired_fixed × slow_slide | not run | not run | not run | not run | not run | not run | not run | not run |
| paired_fixed × shove | not run | not run | not run | not run | not run | not run | not run | not run |
| paired_random × lift | not run | not run | not run | not run | not run | not run | not run | not run |
| paired_random × slow_slide | not run | not run | not run | not run | not run | not run | not run | not run |
| paired_random × shove | not run | not run | not run | not run | not run | not run | not run | not run |
| pilot_constant_cap × lift | not run | not run | not run | not run | not run | not run | not run | not run |
| pilot_constant_cap × slow_slide | not run | not run | not run | not run | not run | not run | not run | not run |
| pilot_staged_cap × lift | not run | not run | not run | not run | not run | not run | not run | not run |
| pilot_staged_cap × slow_slide | not run | not run | not run | not run | not run | not run | not run | not run |
| smoke_fixed × lift | not run | not run | not run | not run | not run | not run | not run | not run |
| smoke_fixed × slow_slide | not run | not run | not run | not run | not run | not run | not run | not run |
| smoke_fixed × shove | not run | not run | not run | not run | not run | not run | not run | not run |
| smoke_random × lift | not run | not run | not run | not run | not run | not run | not run | not run |
| smoke_random × slow_slide | not run | not run | not run | not run | not run | not run | not run | not run |
| smoke_random × shove | not run | not run | not run | not run | not run | not run | not run | not run |
| three_task_main × lift | not run | not run | not run | not run | not run | not run | not run | not run |
| three_task_main × slow_slide | not run | not run | not run | not run | not run | not run | not run | not run |
| three_task_main × shove | not run | not run | not run | not run | not run | not run | not run | not run |
| oracle_complement_cache × shove | not run | not run | not run | not run | not run | not run | not run | not run |
| target_candidate_cache × shove | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/confirmation × PickCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/confirmation × PushCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/confirmation × PullCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/confirmation × poke | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/data × PickCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/data × PushCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/data × PullCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/data × poke | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/grip_data × PickCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/grip_data × PushCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/grip_data × PullCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/grip_data × poke | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/grip_pulses × PickCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/grip_pulses × PushCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/grip_pulses × PullCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/grip_pulses × poke | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/matched_train × PickCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/matched_train × PushCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/matched_train × PullCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/nov13/matched_train × poke | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/pick_gpu_2m/data × PickCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/pick_gpu_2m/pilot × PickCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/pick_gpu_5m_excluded/data × PickCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/context_dynamics/pick_gpu_5m_excluded/pilot × PickCube-v1 | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/mass_belief/nov17_v1/calibration × Pick, lift, hold | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/mass_belief/nov17_v1/closed_loop × Pick, lift, hold | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/mass_belief/nov17_v1/test × Pick, lift, hold | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/mass_belief/nov17_v1/train × Pick, lift, hold | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/mass_belief/nov17_v1/validation × Pick, lift, hold | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/mass_belief/switch_80g_mu1/episodes × Pick, lift, hold | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/rma/adapter_validation_20261107 × Randomized lift task; in-range validation or integration smoke | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/rma/baseline_20261106_persistent/base_validation_01 × Randomized lift task; in-range validation or integration smoke | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/rma/baseline_20261106_persistent/base_validation_02 × Randomized lift task; in-range validation or integration smoke | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/rma/baseline_20261106_persistent/base_validation_03 × Randomized lift task; in-range validation or integration smoke | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/rma/integration_evaluation × Randomized lift task; in-range validation or integration smoke | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/interaction_estimator/contact_dataset_v1 × Contact snippets from independent fixed-policy lift attempts | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/visual_mass/initial_rgb_mass/dataset × Initial RGB before any interaction | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/interaction_adaptation/poststop_cli_smoke_20261027T024742/transfer/episodes × Native carry or vertical pick-and-hold as environment field specifies; no slide task | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/interaction_adaptation/poststop_factory_smoke_20261027T024613 × Native carry or vertical pick-and-hold as environment field specifies; no slide task | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/interaction_adaptation/poststop_cli_smoke_20261027T024742/retry/episodes × Native carry or vertical pick-and-hold as environment field specifies; no slide task | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/failure_collection/stock_short × Stock random-XYZ pick/carry goal | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/checkpoint_probe_smoke/six_mode_native_carry/episodes × Native training carry geometry, randomized XY and floating goal Z; CPU fresh scenes | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/interaction_adaptation/initial_policy_probe/episodes × Native carry or vertical pick-and-hold as environment field specifies; no slide task | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/long_50000_guarded_20261026/diagnostics/checkpoint2001_cpu_retries/episodes × Native carry or vertical pick-and-hold as environment field specifies; no slide task | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/long_50000_guarded_20261026/diagnostics/checkpoint501_cpu_retries/episodes × Native carry or vertical pick-and-hold as environment field specifies; no slide task | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/transfer_vertical/episodes × Native carry or vertical pick-and-hold as environment field specifies; no slide task | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/transfer_native_carry/episodes × Native carry or vertical pick-and-hold as environment field specifies; no slide task | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/interaction_adaptation/transfer_pilot200_crossed/episodes × Native carry or vertical pick-and-hold as environment field specifies; no slide task | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/corrective_training_20261026/guarded100_cpu_retries/episodes × Native carry or vertical pick-and-hold as environment field specifies; no slide task | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/interaction_adaptation/pilot200_retry_diagnostic/episodes × Native carry or vertical pick-and-hold as environment field specifies; no slide task | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/interaction_adaptation/diagnostic_iteration2001_retries/episodes × Native carry or vertical pick-and-hold as environment field specifies; no slide task | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/retries_vertical/episodes × Native carry or vertical pick-and-hold as environment field specifies; no slide task | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/retries_native_carry/episodes × Native carry or vertical pick-and-hold as environment field specifies; no slide task | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/failure_collection/varied_fixed40g × Vertical pick, lift and hold directly above spawn; no slide task | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/checkpoint_comparison_20261026/ckpt_2001/episodes × Native training carry geometry, randomized XY and floating goal Z; CPU fresh scenes | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/checkpoint_comparison_20261026/ckpt_5001/episodes × Native training carry geometry, randomized XY and floating goal Z; CPU fresh scenes | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/checkpoint_comparison_20261026/ckpt_8501/episodes × Native training carry geometry, randomized XY and floating goal Z; CPU fresh scenes | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/failure_collection/dense_single_pair_fixed40g × Vertical pick, lift and hold directly above spawn; no slide task | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/maniskill_pretrained/pick_cube/20261007T160411-c3dd5f0f/custom_lift × custom_lift | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/maniskill_pretrained/pick_cube/20261007T160411-c3dd5f0f/official_pick_cube × official_pick_cube | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/maniskill_pretrained/pick_cube/20261007T161108-28f3d274/custom_lift × custom_lift | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/maniskill_pretrained/pick_cube/20261007T161108-28f3d274/official_pick_cube × official_pick_cube | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/maniskill_pretrained/pick_cube/20261007T221109-3298edee/custom_lift × custom_lift | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/maniskill_pretrained/pick_cube/20261008T032243-6e4c9370/custom_lift × custom_lift | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/baseline_diagnostic_retry/traces.jsonl × physics-conditioned lift | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/baseline_freshscene_diagnostic/traces.jsonl × physics-conditioned lift | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/corrective_gpu_baselines_20261026/old2001_grid/traces.jsonl × physics-conditioned lift | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/corrective_gpu_baselines_20261026/old2001_offgrid/traces.jsonl × physics-conditioned lift | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/corrective_gpu_baselines_20261026/pilot200_grid/traces.jsonl × physics-conditioned lift | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/corrective_gpu_baselines_20261026/pilot200_offgrid/traces.jsonl × physics-conditioned lift | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/corrective_training_20261026/guarded100_grid/traces.jsonl × physics-conditioned lift | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/corrective_training_20261026/guarded100_offgrid/traces.jsonl × physics-conditioned lift | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/policy/traces.jsonl × physics-conditioned lift | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/physics_conditioning/pilot200_freshscene_diagnostic/traces.jsonl × physics-conditioned lift | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/article_inspired/cube_lift × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/checks/alignment × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/checks/grip_extension × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/deployment/20261021_234802_550312 × image-estimate lift deployment | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/deployment/20261021_234906_513736 × image-estimate lift deployment | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/deployment/20261021_235109_413496 × image-estimate lift deployment | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/deployment/20261022_014201_067220 × image-estimate lift deployment | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/diagnosis/weight-adaptation-20261008 × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261004T183600-9f9f3c53 × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261004T183739-111848f0 × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261004T184018-30321200 × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261004T184658-45fedb37 × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261004T184727-c1fb4e69 × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261004T190445-2812f7f4 × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261004T190612-b6c9e132 × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261004T190859-601f2196 × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261005T152308-817384cb × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261005T165416-7e88f055 × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261008T021816-05bc01b7 × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261008T024643-86f70895 × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261008T024914-21df5d38 × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261008T025442-5d43bcd3 × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261008T122327-6746d22d × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261008T130546-a6f1ea05 × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261008T190610-40c9c0c7 × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261008T192509-bc35f615 × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261008T211232-d3e19b0b × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261009T040017-2aab8e95 × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261009T105845-63d5e45c × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/inspect/20261009T122414-11e4a49e × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |
| runs/maniskill_example/pick_cube × lift/hold/carry inspection | not run | not run | not run | not run | not run | not run | not run | not run |

## Exact commands to reproduce this inventory

These commands read existing artifacts and write only the diagnostic reports. They do not train or collect data.

```bash
cd /home/ubuntu/faliure
.venv/bin/python results/probe/index_files.py
.venv/bin/python results/probe/audit_lift_slide.py
PYTHONPATH=. .venv/bin/python results/probe/audit_historical.py
.venv/bin/python results/probe/audit_other.py
.venv/bin/python results/probe/audit_legacy.py
.venv/bin/python results/probe/audit_leak_gate.py
.venv/bin/python results/probe/build_report.py
```

There is no probe-training rerun command or first-seed runtime: implementation/training stopped at the required gate. Continuing requires settling the dataset/feature exclusions and existing-split limitations described above. Deleting explicit labels alone does not remove information already supplied to the acquisition policy.
