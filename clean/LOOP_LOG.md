# Autonomous research loop

The governing request is `clean/REQUEST.txt`. Budget: at most 20 scientific candidate iterations, five seeds each. Infrastructure repairs and the mandatory precollection pilot do not count as trained candidates.

No candidate has been trained or evaluated. No test or out-of-range observations have been opened for analysis.

## Setup, before iteration 1

- Preserved the prior scaffold report under `results/prior_scaffold/`; its pilot-only approval restriction is superseded by the current request.
- Corrected pilot membership to training cubes 0–9; previous code would have included two test cubes, but it had collected no data.
- The installed planner crashes under NumPy 2.2.6 and constructs successfully under the available NumPy 1.26.4. Isolating the compatible dependency under `clean/vendor/`; no installed dependency or old project source is modified.
- Lock metric definitions, five seeds, and training budgets in `protocol.py` and the immutable `data/protocol.json` before collection.
- Validation uses episodes 8–9 of the 200 training cubes; fresh z is fitted from episodes 0–4 with shared weights frozen. This validates held-out episodes of familiar cubes, not unseen identities.
- F5 compares learned-z versus fresh-fitted-z validation as dimension changes. The final-only rule forbids using actual held-out test errors to tune dimension.
- A supervised one-episode GRU probe is an empirical signal benchmark, not a mathematical upper bound on all possible belief estimators.

Data gate: ten-cube pilot passed all available data checks across five probe seeds. Receipt/source verification saved separately because the first running audit process predated that added check. Full dataset collected: 260 cubes, ten episodes each. No held-out motion analyzed.

## Iteration 1

Hypothesis: Initial capacity and a small object latent may explain object-specific dynamics.

Change: Initial locked reference candidate, d=2.

Base: None. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 2.577573957570958, "std": 0.0897565988033306, "values": [2.659276218547508, 2.5847576195486015, 2.473582721488002, 2.670478419161534, 2.4997748091091463]}, "history": {"mean": 2.546605714572073, "std": 0.2601643614192159, "values": [2.7376809177831922, 2.6698492905215665, 2.194811665632361, 2.782808308390389, 2.347878390532857]}, "oracle": {"mean": 2.6140823000272886, "std": 0.06903496823533682, "values": [2.6930212786106096, 2.646229085286941, 2.578251842886674, 2.638434590861181, 2.5144747024910377]}, "fitted_z": {"mean": 2.5531709141646983, "std": 0.12832693240902932, "values": [2.4155048955254146, 2.440515308179667, 2.7305849245150053, 2.582750970504012, 2.5964984720993924]}, "mean_z": {"mean": 2.559557664979124, "std": 0.13015468912429876, "values": [2.4080653053550094, 2.4500823697854726, 2.7285959810396423, 2.609179051683717, 2.6018656170317795]}, "shuffled_z": {"mean": 2.562007118785476, "std": 0.12588447459682892, "values": [2.425234376508423, 2.4490431435193165, 2.731221888188115, 2.597554771840012, 2.606981413871514]}, "another_cube_z": {"mean": 2.5630176819925246, "std": 0.12829823123069398, "values": [2.4148062972165074, 2.4536837056931264, 2.729642691594867, 2.6119464753256123, 2.6050092401325107]}, "learned_z": {"mean": 2.550994638304571, "std": 0.13370144334041578, "values": [2.394476956540459, 2.442379661736843, 2.727925421661979, 2.597835876108627, 2.5923552754749473]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": false, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": false, "S3": true, "S4": false, "plain_oracle_gap": -0.03650834245633039, "gap_recovery": null}, "mean_z_absolute_gain": 0.006386750814425746}

Iteration 1 decision: retain as the initial failed reference, not a successful model. F4/S2 fail; Oracle does not beat Plain so S1 is undefined and fails. Next test changes only the moving-step loss weight.

## Iteration 2

Hypothesis: Validation fails F4 and Oracle fails to improve on Plain; weighting moving steps more strongly should focus the shared dynamics and latent on the scored physical responses.

Change: Increase moving-step training and fitting loss weight from 1 to 10; all other settings unchanged.

Base: 1. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 2.8348129577299406, "std": 0.08933934130583746, "values": [2.833855930424469, 2.798186433867297, 2.705672544468574, 2.920170807357793, 2.9161790725315693]}, "history": {"mean": 2.6448904150686956, "std": 0.23718490654668586, "values": [2.8023718619041778, 2.5807752643011566, 2.3662419841376883, 2.962593560724081, 2.5124694042763744]}, "oracle": {"mean": 2.9132524726234568, "std": 0.09775909011550164, "values": [3.0156374005416042, 2.8414829959130206, 3.0073695320582523, 2.9054605110058525, 2.796311923598553]}, "fitted_z": {"mean": 2.7324230568616583, "std": 0.14927904992650315, "values": [2.6313183822653174, 2.679656545120206, 2.9354320151375415, 2.837682098445713, 2.578026243339514]}, "mean_z": {"mean": 2.7406246893371886, "std": 0.15022740719778288, "values": [2.640914112352582, 2.681884927884995, 2.933053624144155, 2.863768130543767, 2.583502651760443]}, "shuffled_z": {"mean": 2.741096864947781, "std": 0.14608088471421724, "values": [2.6508305003782153, 2.682013557989052, 2.935585812382558, 2.85057819820036, 2.5864762557887206]}, "another_cube_z": {"mean": 2.744242611249752, "std": 0.14938400789593903, "values": [2.6473355012524133, 2.685277647791503, 2.934551171601464, 2.8677087457875423, 2.586339989815838]}, "learned_z": {"mean": 2.7345343153635313, "std": 0.15307868713066772, "values": [2.6246695261032245, 2.6804642532933, 2.9346632747716956, 2.8543355701605106, 2.5785389524889264]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": false, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": false, "S3": true, "S4": false, "plain_oracle_gap": -0.0784395148935162, "gap_recovery": null}, "mean_z_absolute_gain": 0.008201632475530296}

Iteration 2 decision: reverted moving weight10 to1. Fitted-z validation NMSE worsened and F4 still failed.

Iteration3 preregistration: hypothesis—validation contact oscillation under high grip force masks object-specific prediction; reduce only grip-force randomization maximum from30N to8N. Base iteration1. All other model/script settings unchanged. Archive old trajectories, preserve split/cubes/protocol byte-for-byte, re-run ten-cube pilot gate, collect the same260 identities and retrain all baselines. This narrows the scientific regime; cross-version normalized errors are not directly comparable improvements of the model.

Grip8 pilot gate passed. Full collection uses four independent CPU workers for throughput, with unchanged simulator/script/sensor functions and shared immutable plans. Archive references and independent process provenance are saved in receipts and full manifest. No held-out observations enter diagnostics.

## Iteration 3

Hypothesis: Validation F4 fails while high-force contact oscillations dominate the locked error; lowering only the grip-force upper bound should remove that unstable contact regime and reveal object-dependent dynamics.

Change: Change only the grip-force randomization upper bound from 30 N to 8 N; restore the retained iteration-1 model settings and preserve the immutable cube split and properties.

Base: 1. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 0.35567075420991145, "std": 0.01076059665740359, "values": [0.36652109067503874, 0.36823330509023067, 0.34661223646396555, 0.3473680542807489, 0.34961908453957347]}, "history": {"mean": 0.3741893682740813, "std": 0.01939707774458302, "values": [0.3788907130913467, 0.40571205528693427, 0.36095671923164063, 0.35770372604925704, 0.36768362771122765]}, "oracle": {"mean": 0.3618347317928204, "std": 0.0447526829830977, "values": [0.3441277408758489, 0.44107821134396963, 0.3324232274933246, 0.34146526483890277, 0.3500792144120561]}, "fitted_z": {"mean": 0.34897522150014126, "std": 0.019329719033827936, "values": [0.38033285776771364, 0.3450299942443816, 0.3488325256166669, 0.32747121656553624, 0.34320951330640803]}, "mean_z": {"mean": 0.3536029962001557, "std": 0.02008451229224202, "values": [0.3867456262490738, 0.3482478275880243, 0.353596361398515, 0.33262204093014835, 0.34680312483501696]}, "shuffled_z": {"mean": 0.36001784262989933, "std": 0.018205188195428035, "values": [0.38991197142735573, 0.3571868802220861, 0.3583645333036134, 0.34029585009352703, 0.3543299781029141]}, "another_cube_z": {"mean": 0.35748460864840037, "std": 0.02034465707843712, "values": [0.390743146006558, 0.3513936274717856, 0.35779327189705157, 0.3355360379971009, 0.3519569598695061]}, "learned_z": {"mean": 0.35011369350592936, "std": 0.020120519625665814, "values": [0.3828975599953651, 0.3441539788466706, 0.3510113587060879, 0.3283792290384932, 0.34412634094302985]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": false, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": false, "S3": true, "S4": false, "plain_oracle_gap": -0.0061639775829089305, "gap_recovery": null}, "mean_z_absolute_gain": 0.004627774700014409}

Iteration3 decision: keep as the failed reference for the8N regime. F4/S2 remain failed and the aggregate Plain→Oracle gap is nonpositive. Lower scores versus30N describe a different data regime, not a model improvement on the same observations. Continue within8N using the unchanged locked metric.

<!-- search-reference-start-04 -->

Search starts at iteration 4 from iteration 3. Lowest fitted-z validation NMSE among finite candidates passing F1, F2, F3, F5, F6, F7, F8, and F9. F4 remains a required success check; a failing model may only serve as a reference for further corrections. F4=False; F5=True. This does not assert success.

<!-- search-command-04 -->

Exact candidate command (from repository root):

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.run candidate --iteration 4 --config /home/ubuntu/faliure/clean/configs/search_iteration_04.json --base-iteration 3 --hypothesis 'The retained reference fails validation F4. Oracle has not established a positive Plain-to-Oracle error gap. A larger latent may represent distinct object effects; F5 will test whether learned-z gains transfer to freshly fitted z.' --change 'Change only z_dim from 2 to 4; every other configuration setting is retained.' --device cuda
```

## Iteration 4

Hypothesis: The retained reference fails validation F4. Oracle has not established a positive Plain-to-Oracle error gap. A larger latent may represent distinct object effects; F5 will test whether learned-z gains transfer to freshly fitted z.

Change: Change only z_dim from 2 to 4; every other configuration setting is retained.

Base: 3. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 0.35567075420991145, "std": 0.010760596657403573, "values": [0.36652109067503874, 0.3682333050902306, 0.34661223646396555, 0.3473680542807489, 0.34961908453957347]}, "history": {"mean": 0.3741893682740812, "std": 0.019397077744583043, "values": [0.3788907130913467, 0.40571205528693427, 0.3609567192316405, 0.35770372604925704, 0.3676836277112276]}, "oracle": {"mean": 0.3618347317928204, "std": 0.04475268298309771, "values": [0.3441277408758489, 0.44107821134396963, 0.3324232274933246, 0.3414652648389027, 0.3500792144120561]}, "fitted_z": {"mean": 0.3318177302164186, "std": 0.014850263678878744, "values": [0.31563927139663983, 0.3521732322755225, 0.32233095479211293, 0.32733338082056124, 0.3416118117972563]}, "mean_z": {"mean": 0.3362214343952755, "std": 0.01680654746901343, "values": [0.31480408879725524, 0.3577917131235974, 0.3282827587173187, 0.33261333675164617, 0.34761527458655983]}, "shuffled_z": {"mean": 0.3453706604265525, "std": 0.015438992550582281, "values": [0.32504801134323175, 0.36456947126170347, 0.3372401410717895, 0.3443889780084673, 0.35560670044757037]}, "another_cube_z": {"mean": 0.34251144605321704, "std": 0.0165899717624283, "values": [0.3212949419214322, 0.36251303224006676, 0.3333443513502485, 0.34019324349257785, 0.35521166126175996]}, "learned_z": {"mean": 0.33339730644421406, "std": 0.016723087500718405, "values": [0.3142359908506258, 0.35647412438404485, 0.32363865242366785, 0.3290165353406773, 0.3436212292220546]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": false, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": false, "S3": true, "S4": false, "plain_oracle_gap": -0.0061639775829089305, "gap_recovery": null}, "mean_z_absolute_gain": 0.0044037041788569375}

<!-- search-decision-04 -->

Iteration 4 decision: **kept**. Fitted-z validation NMSE decreased relative to the retained eligible reference. F4 still fails; kept only as a failed reference, not a successful model. Retained reference: iteration 4.

<!-- search-command-05 -->

Exact candidate command (from repository root):

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.run candidate --iteration 5 --config /home/ubuntu/faliure/clean/configs/search_iteration_05.json --base-iteration 4 --hypothesis 'The retained reference fails validation F4. Oracle has not established a positive Plain-to-Oracle error gap. Additional latent capacity may recover object effects missed by the current embedding; F5 guards against identity memorization.' --change 'Change only z_dim from 4 to 8; every other configuration setting is retained.' --device cuda
```

## Iteration 5

Hypothesis: The retained reference fails validation F4. Oracle has not established a positive Plain-to-Oracle error gap. Additional latent capacity may recover object effects missed by the current embedding; F5 guards against identity memorization.

Change: Change only z_dim from 4 to 8; every other configuration setting is retained.

Base: 4. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 0.35567075420991145, "std": 0.010760596657403582, "values": [0.36652109067503874, 0.3682333050902306, 0.34661223646396555, 0.3473680542807489, 0.3496190845395734]}, "history": {"mean": 0.3741893682740813, "std": 0.01939707774458302, "values": [0.3788907130913467, 0.40571205528693427, 0.36095671923164063, 0.35770372604925704, 0.36768362771122765]}, "oracle": {"mean": 0.3618347317928204, "std": 0.04475268298309771, "values": [0.3441277408758489, 0.44107821134396963, 0.3324232274933246, 0.3414652648389027, 0.3500792144120561]}, "fitted_z": {"mean": 0.35288038095805, "std": 0.01511226042184803, "values": [0.34977766539585353, 0.37825219316061287, 0.3487091132914263, 0.35011640009059625, 0.33754653285176134]}, "mean_z": {"mean": 0.36029899000665705, "std": 0.014330801900354791, "values": [0.3605124164655876, 0.38437389299626284, 0.35717749156450174, 0.3517255381431657, 0.34770561086376744]}, "shuffled_z": {"mean": 0.37761223038923164, "std": 0.016827739670343667, "values": [0.37597106354824383, 0.4058407873761016, 0.37485148256413403, 0.37026346072901023, 0.36113435772866836]}, "another_cube_z": {"mean": 0.3706657301481023, "std": 0.015586846741209888, "values": [0.3697211597789881, 0.3970907201382765, 0.36823127805228284, 0.35989838835717863, 0.3583871044137852]}, "learned_z": {"mean": 0.3524142289500059, "std": 0.016385248009261763, "values": [0.34995084133195964, 0.3805512719510507, 0.3479989984777249, 0.34570224554941326, 0.3378677874398809]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": false, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": false, "S3": true, "S4": false, "plain_oracle_gap": -0.0061639775829089305, "gap_recovery": null}, "mean_z_absolute_gain": 0.007418609048607028}

<!-- search-decision-05 -->

Iteration 5 decision: **reverted**. Fitted-z validation NMSE did not decrease; retained the lower-error eligible reference. Retained reference: iteration 4.

<!-- search-command-06 -->

Exact candidate command (from repository root):

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.run candidate --iteration 6 --config /home/ubuntu/faliure/clean/configs/search_iteration_06.json --base-iteration 4 --hypothesis 'The retained reference fails validation F4. Oracle has not established a positive Plain-to-Oracle error gap. The largest allowed latent may expose a capacity limitation; F5 must still pass on fresh-z validation.' --change 'Change only z_dim from 4 to 16; every other configuration setting is retained.' --device cuda
```

## Iteration 6

Hypothesis: The retained reference fails validation F4. Oracle has not established a positive Plain-to-Oracle error gap. The largest allowed latent may expose a capacity limitation; F5 must still pass on fresh-z validation.

Change: Change only z_dim from 4 to 16; every other configuration setting is retained.

Base: 4. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 0.35567075420991145, "std": 0.01076059665740359, "values": [0.36652109067503874, 0.36823330509023067, 0.34661223646396555, 0.3473680542807489, 0.34961908453957347]}, "history": {"mean": 0.3741893682740813, "std": 0.01939707774458302, "values": [0.3788907130913467, 0.40571205528693427, 0.36095671923164063, 0.35770372604925704, 0.36768362771122765]}, "oracle": {"mean": 0.3618347317928204, "std": 0.04475268298309771, "values": [0.3441277408758489, 0.44107821134396963, 0.3324232274933246, 0.3414652648389027, 0.3500792144120561]}, "fitted_z": {"mean": 0.3456669743368003, "std": 0.011162592299080983, "values": [0.3343890852438473, 0.3606737220655711, 0.3350445972645989, 0.3469428564656938, 0.3512846106442906]}, "mean_z": {"mean": 0.3531995671626591, "std": 0.009680610549062477, "values": [0.34521541941982314, 0.36084108078269655, 0.3407297628157508, 0.36235735200691044, 0.3568542207881147]}, "shuffled_z": {"mean": 0.377068033738451, "std": 0.01134533919018831, "values": [0.3636592010507759, 0.38934841764576883, 0.36872654273075384, 0.3877123991336414, 0.3758936081313149]}, "another_cube_z": {"mean": 0.3711979498227869, "std": 0.009026992487377664, "values": [0.36191773646313463, 0.3770363068595002, 0.36102827535552057, 0.38020353754973385, 0.37580389288604515]}, "learned_z": {"mean": 0.34770917823188463, "std": 0.009765312663522381, "values": [0.3383988707253426, 0.3580178332217014, 0.336271713478016, 0.35419205539577236, 0.35166541833859083]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": true, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": true, "S3": true, "S4": true, "plain_oracle_gap": -0.0061639775829089305, "gap_recovery": null}, "mean_z_absolute_gain": 0.007532592825858775}

<!-- search-decision-06 -->

Iteration 6 decision: **reverted**. Fitted-z validation NMSE did not decrease; retained the lower-error eligible reference. Retained reference: iteration 4.

<!-- search-command-07 -->

Exact candidate command (from repository root):

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.run candidate --iteration 7 --config /home/ubuntu/faliure/clean/configs/search_iteration_07.json --base-iteration 4 --hypothesis 'The retained reference fails validation F4. Oracle has not established a positive Plain-to-Oracle error gap. Removing latent shrinkage may let prediction gradients distinguish cubes more strongly.' --change 'Change only z_l2 from 0.0001 to 0.0; every other configuration setting is retained.' --device cuda
```

## Iteration 7

Hypothesis: The retained reference fails validation F4. Oracle has not established a positive Plain-to-Oracle error gap. Removing latent shrinkage may let prediction gradients distinguish cubes more strongly.

Change: Change only z_l2 from 0.0001 to 0.0; every other configuration setting is retained.

Base: 4. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 0.35567075420991145, "std": 0.01076059665740359, "values": [0.36652109067503874, 0.36823330509023067, 0.34661223646396555, 0.3473680542807489, 0.34961908453957347]}, "history": {"mean": 0.3741893682740813, "std": 0.019397077744583022, "values": [0.3788907130913467, 0.40571205528693427, 0.36095671923164063, 0.35770372604925704, 0.3676836277112276]}, "oracle": {"mean": 0.3618347317928204, "std": 0.04475268298309771, "values": [0.3441277408758489, 0.44107821134396963, 0.3324232274933246, 0.3414652648389027, 0.3500792144120561]}, "fitted_z": {"mean": 0.3318162029324086, "std": 0.014846506713144676, "values": [0.31563925174196883, 0.3521672232166471, 0.3223355505909033, 0.3273331583743149, 0.3416058307382089]}, "mean_z": {"mean": 0.3362224266922288, "std": 0.016805756336652485, "values": [0.31480351982212695, 0.3577920619283722, 0.32828979769858607, 0.3326130617787322, 0.34761369223332667]}, "shuffled_z": {"mean": 0.34538128270670343, "std": 0.015437390433208058, "values": [0.325058737280028, 0.36457681195985103, 0.33725496403541494, 0.34439863913759905, 0.35561726112062403]}, "another_cube_z": {"mean": 0.3425223092118105, "std": 0.01659001818990551, "values": [0.3213030633417145, 0.36252139985308957, 0.3333581639131647, 0.3402045864867732, 0.3552243324643106]}, "learned_z": {"mean": 0.33339703169891155, "std": 0.016720963379246563, "values": [0.31423511075431315, 0.3564704520278451, 0.32364450293168134, 0.3290158378274212, 0.343619254953297]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": false, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": false, "S3": true, "S4": false, "plain_oracle_gap": -0.0061639775829089305, "gap_recovery": null}, "mean_z_absolute_gain": 0.004406223759820194}

<!-- search-decision-07 -->

Iteration 7 decision: **kept**. Fitted-z validation NMSE decreased relative to the retained eligible reference. F4 still fails; kept only as a failed reference, not a successful model. Retained reference: iteration 7.

<!-- search-command-08 -->

Exact candidate command (from repository root):

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.run candidate --iteration 8 --config /home/ubuntu/faliure/clean/configs/search_iteration_08.json --base-iteration 7 --hypothesis 'The retained reference fails validation F4. Oracle has not established a positive Plain-to-Oracle error gap. More frozen-weight latent updates may resolve incomplete adaptation convergence.' --change 'Change only fit_steps from 200 to 1000; every other configuration setting is retained.' --device cuda
```

## Iteration 8

Hypothesis: The retained reference fails validation F4. Oracle has not established a positive Plain-to-Oracle error gap. More frozen-weight latent updates may resolve incomplete adaptation convergence.

Change: Change only fit_steps from 200 to 1000; every other configuration setting is retained.

Base: 7. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 0.35567075420991145, "std": 0.010760596657403582, "values": [0.36652109067503874, 0.3682333050902306, 0.34661223646396555, 0.3473680542807489, 0.3496190845395734]}, "history": {"mean": 0.3741893682740813, "std": 0.019397077744583026, "values": [0.3788907130913468, 0.40571205528693427, 0.36095671923164063, 0.35770372604925704, 0.36768362771122765]}, "oracle": {"mean": 0.3618347317928204, "std": 0.04475268298309771, "values": [0.3441277408758489, 0.44107821134396963, 0.3324232274933246, 0.3414652648389027, 0.3500792144120561]}, "fitted_z": {"mean": 0.33131654313549813, "std": 0.014961859579981156, "values": [0.3149903157020938, 0.3512378225974032, 0.32135274863007723, 0.32700197280054893, 0.3419998559473674]}, "mean_z": {"mean": 0.3362224266922288, "std": 0.016805756336652485, "values": [0.31480351982212695, 0.3577920619283722, 0.32828979769858607, 0.3326130617787322, 0.34761369223332667]}, "shuffled_z": {"mean": 0.34564457196528486, "std": 0.01574923506829732, "values": [0.32436732103182675, 0.3634212905135512, 0.33790664660843406, 0.34417883307779495, 0.3583487685948175]}, "another_cube_z": {"mean": 0.3425223092118105, "std": 0.01659001818990551, "values": [0.3213030633417145, 0.36252139985308957, 0.3333581639131647, 0.3402045864867732, 0.3552243324643106]}, "learned_z": {"mean": 0.33339703169891155, "std": 0.016720963379246567, "values": [0.3142351107543132, 0.35647045202784516, 0.32364450293168134, 0.32901583782742116, 0.343619254953297]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": false, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": false, "S3": true, "S4": false, "plain_oracle_gap": -0.0061639775829089305, "gap_recovery": null}, "mean_z_absolute_gain": 0.004905883556730661}

<!-- search-decision-08 -->

Iteration 8 decision: **kept**. Fitted-z validation NMSE decreased relative to the retained eligible reference. F4 still fails; kept only as a failed reference, not a successful model. Retained reference: iteration 8.

<!-- search-command-09 -->

Exact candidate command (from repository root):

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.run candidate --iteration 9 --config /home/ubuntu/faliure/clean/configs/search_iteration_09.json --base-iteration 8 --hypothesis 'The retained reference fails validation F4. Oracle has not established a positive Plain-to-Oracle error gap. A smaller fitting learning rate may reduce overshoot and noisy object-latent estimates.' --change 'Change only fit_lr from 0.02 to 0.005; every other configuration setting is retained.' --device cuda
```

## Iteration 9

Hypothesis: The retained reference fails validation F4. Oracle has not established a positive Plain-to-Oracle error gap. A smaller fitting learning rate may reduce overshoot and noisy object-latent estimates.

Change: Change only fit_lr from 0.02 to 0.005; every other configuration setting is retained.

Base: 8. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 0.35567075420991145, "std": 0.010760596657403598, "values": [0.36652109067503874, 0.36823330509023067, 0.34661223646396555, 0.3473680542807489, 0.3496190845395734]}, "history": {"mean": 0.3741893682740813, "std": 0.019397077744583005, "values": [0.3788907130913468, 0.4057120552869342, 0.36095671923164063, 0.35770372604925704, 0.36768362771122765]}, "oracle": {"mean": 0.3618347317928204, "std": 0.04475268298309771, "values": [0.3441277408758489, 0.44107821134396963, 0.3324232274933246, 0.3414652648389027, 0.3500792144120561]}, "fitted_z": {"mean": 0.3310284023524471, "std": 0.015031272709208388, "values": [0.3145874780519345, 0.3511505862132379, 0.3215983506801176, 0.32605741180438763, 0.341748185012558]}, "mean_z": {"mean": 0.3362224266922288, "std": 0.016805756336652485, "values": [0.31480351982212695, 0.3577920619283722, 0.32828979769858607, 0.3326130617787322, 0.34761369223332667]}, "shuffled_z": {"mean": 0.3449436146260203, "std": 0.01548475603668496, "values": [0.3246506869452759, 0.36318528732227157, 0.33702648430094884, 0.3427799462012425, 0.3570756683603629]}, "another_cube_z": {"mean": 0.3425223092118105, "std": 0.01659001818990551, "values": [0.3213030633417145, 0.36252139985308957, 0.3333581639131647, 0.3402045864867732, 0.3552243324643106]}, "learned_z": {"mean": 0.33339703169891155, "std": 0.016720963379246567, "values": [0.3142351107543132, 0.35647045202784516, 0.32364450293168134, 0.32901583782742116, 0.343619254953297]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": false, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": false, "S3": true, "S4": false, "plain_oracle_gap": -0.0061639775829089305, "gap_recovery": null}, "mean_z_absolute_gain": 0.0051940243397816666}

<!-- search-decision-09 -->

Iteration 9 decision: **kept**. Fitted-z validation NMSE decreased relative to the retained eligible reference. F4 still fails; kept only as a failed reference, not a successful model. Retained reference: iteration 9.

<!-- search-command-10 -->

Exact candidate command (from repository root):

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.run candidate --iteration 10 --config /home/ubuntu/faliure/clean/configs/search_iteration_10.json --base-iteration 9 --hypothesis 'The retained reference fails validation F4. Oracle has not established a positive Plain-to-Oracle error gap. A smaller shared network may reduce memorization of shared trajectories and increase the useful contribution of z.' --change 'Change only width from 128 to 64; every other configuration setting is retained.' --device cuda
```

## Iteration 10

Hypothesis: The retained reference fails validation F4. Oracle has not established a positive Plain-to-Oracle error gap. A smaller shared network may reduce memorization of shared trajectories and increase the useful contribution of z.

Change: Change only width from 128 to 64; every other configuration setting is retained.

Base: 9. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 0.35606451855258076, "std": 0.010005642864971213, "values": [0.3559721391515644, 0.36796357170577365, 0.34136044812465416, 0.36189775881472824, 0.3531286749661832]}, "history": {"mean": 0.41789551209882514, "std": 0.019556575837951548, "values": [0.43627729403876486, 0.43295272850021216, 0.41446710916662405, 0.4187514422141044, 0.3870289865744202]}, "oracle": {"mean": 0.347282342243284, "std": 0.003774296673052255, "values": [0.35322908876247866, 0.3467648727017616, 0.34392266244586955, 0.3482375974249758, 0.34425748988133437]}, "fitted_z": {"mean": 0.36695774369846423, "std": 0.013074492495991507, "values": [0.3799427790063622, 0.3583752434596852, 0.3636039618220297, 0.35177558470998177, 0.38109114949426215]}, "mean_z": {"mean": 0.3727323329397553, "std": 0.01139167931591074, "values": [0.38523103075661747, 0.36333431758917867, 0.3715724228031172, 0.3601147883069522, 0.38340910524291094]}, "shuffled_z": {"mean": 0.3800661250632851, "std": 0.010793823776220979, "values": [0.38904692873547847, 0.37258459763141333, 0.377807874262173, 0.3677174796907056, 0.39317374499665475]}, "another_cube_z": {"mean": 0.37829462976449657, "std": 0.012020117656026003, "values": [0.3911258400383725, 0.3689289590775499, 0.3771070171802629, 0.3644755189851192, 0.3898358135411786]}, "learned_z": {"mean": 0.3679842563968493, "std": 0.011505217644802754, "values": [0.3820670653940347, 0.3607641760216731, 0.3653926082768646, 0.3543772873325481, 0.37732014495912564]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": false, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": false, "S3": true, "S4": false, "plain_oracle_gap": 0.00878217630929673, "gap_recovery": -1.2403787810946145}, "mean_z_absolute_gain": 0.005774589241291039}

<!-- search-decision-10 -->

Iteration 10 decision: **reverted**. Fitted-z validation NMSE did not decrease; retained the lower-error eligible reference. Retained reference: iteration 9.

<!-- search-command-11 -->

Exact candidate command (from repository root):

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.run candidate --iteration 11 --config /home/ubuntu/faliure/clean/configs/search_iteration_11.json --base-iteration 9 --hypothesis 'The retained reference fails validation F4. Oracle has not established a positive Plain-to-Oracle error gap. Further reducing shared-network width may test whether shared trajectory fitting suppresses object-specific latent use.' --change 'Change only width from 128 to 32; every other configuration setting is retained.' --device cuda
```

Selection correction after iteration10: required F4 feasibility now takes priority whenever an eligible current-data candidate passes it; minimize the same locked fitted-z NMSE within that set. Iteration6 already passedF4, so a lower-error F4-failing model should not displace it as the reference for further corrections. No metric, threshold, split, or model configuration changed. The original iteration11 proposal was interrupted in the audit guard before training or evaluation and is archived under results/aborted_candidate_plans/. Ten scientific iterations remain completed.

<!-- search-reference-start-11 -->

Search starts at iteration 11 from iteration 6. F4 is a fixed required constraint, not an alternate metric. A reference passing F4 and the other required checks exists, so select the lowest fitted-z validation NMSE within that feasible set; F4-failing candidates cannot displace it. F4=True; F5=True. This does not assert success.

<!-- search-command-11-base-06-4193c8ae4ab0 -->

Exact candidate command (from repository root; any earlier command for the same iteration with a different retained base is superseded):

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.run candidate --iteration 11 --config /home/ubuntu/faliure/clean/configs/search_iteration_11.json --base-iteration 6 --hypothesis 'Oracle has not established a positive Plain-to-Oracle error gap. Further reducing shared-network width may test whether shared trajectory fitting suppresses object-specific latent use.' --change 'Change only width from 128 to 32; every other configuration setting is retained.' --device cuda
```

## Iteration 11

Hypothesis: Oracle has not established a positive Plain-to-Oracle error gap. Further reducing shared-network width may test whether shared trajectory fitting suppresses object-specific latent use.

Change: Change only width from 128 to 32; every other configuration setting is retained.

Base: 6. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 0.3841612516135237, "std": 0.00774641962266958, "values": [0.38330435757059694, 0.3895789057111816, 0.3803336668586077, 0.3936694714678933, 0.3739198564593391]}, "history": {"mean": 0.44805126213203134, "std": 0.01944140091639523, "values": [0.43560835814002546, 0.4722993038558127, 0.4346500312497671, 0.4316997876448196, 0.46599882976973217]}, "oracle": {"mean": 0.3886713959341309, "std": 0.020043113471260393, "values": [0.40245570186462587, 0.39221625514081493, 0.3538470706824388, 0.3930281599650251, 0.4018097920177499]}, "fitted_z": {"mean": 0.3799163319925218, "std": 0.01131254228382255, "values": [0.3877639458114183, 0.3837176091038091, 0.36024150018687623, 0.38680617845444226, 0.3810524264060631]}, "mean_z": {"mean": 0.3882202886534266, "std": 0.01120703140824124, "values": [0.3915062828591354, 0.3955915388348245, 0.36850538265384863, 0.39469321169889626, 0.39080502722042804]}, "shuffled_z": {"mean": 0.3992660616570948, "std": 0.011268984105969116, "values": [0.39953984898938105, 0.4100422561018868, 0.38073619148644133, 0.4062019588553835, 0.39981005285238097]}, "another_cube_z": {"mean": 0.39873983946425484, "std": 0.011636678144841555, "values": [0.4062221208431528, 0.4054998919987536, 0.3784108103201385, 0.4038046856235398, 0.3997616885356894]}, "learned_z": {"mean": 0.38062861461163683, "std": 0.010907434220088288, "values": [0.3904314006916982, 0.38196400434389133, 0.3622907499101937, 0.3870405368054112, 0.38141638130698996]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": false, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": false, "S3": true, "S4": false, "plain_oracle_gap": -0.004510144320607201, "gap_recovery": null}, "mean_z_absolute_gain": 0.008303956660904832}

<!-- search-decision-11-base-06-4193c8ae4ab0 -->

Iteration 11 decision: **reverted**. This candidate fails required F4 and cannot displace a reference that passes F4 and the other required checks, even with lower NMSE. Preserve the fixed feasibility constraint; the locked metric is unchanged. Retained reference: iteration 6.

<!-- search-command-12-base-06-bcdfb498b15e -->

Exact candidate command (from repository root; any earlier command for the same iteration with a different retained base is superseded):

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.run candidate --iteration 12 --config /home/ubuntu/faliure/clean/configs/search_iteration_12.json --base-iteration 6 --hypothesis 'Oracle has not established a positive Plain-to-Oracle error gap. More shared-network capacity may reduce underfitting of contact dynamics and strengthen the Oracle reference.' --change 'Change only width from 128 to 256; every other configuration setting is retained.' --device cuda
```

## Iteration 12

Hypothesis: Oracle has not established a positive Plain-to-Oracle error gap. More shared-network capacity may reduce underfitting of contact dynamics and strengthen the Oracle reference.

Change: Change only width from 128 to 256; every other configuration setting is retained.

Base: 6. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 0.3419465188656493, "std": 0.019808578459911963, "values": [0.35023990530559723, 0.3399345204219253, 0.33469318985839114, 0.3155660827480423, 0.36929889599429055]}, "history": {"mean": 0.35636888564683067, "std": 0.026753728404781484, "values": [0.36241397380016466, 0.3771242755974984, 0.38392318708784573, 0.3381150062218646, 0.32026798552678015]}, "oracle": {"mean": 0.34116351286538693, "std": 0.020478351231324333, "values": [0.36364877737095663, 0.3245102808416682, 0.3225259795666953, 0.332315260899409, 0.3628172656482054]}, "fitted_z": {"mean": 0.3534556342044894, "std": 0.01851783579603122, "values": [0.35694311468741274, 0.36061124416962104, 0.33421378909654886, 0.3365293454415291, 0.3789806776273349]}, "mean_z": {"mean": 0.3527340550362702, "std": 0.019878072530163125, "values": [0.3524514422533659, 0.370694690720234, 0.33204805253270253, 0.33392238467499935, 0.3745537050000493]}, "shuffled_z": {"mean": 0.3925024741556038, "std": 0.008687970384390415, "values": [0.39376972831992746, 0.39665549785875426, 0.3935563946625155, 0.3778416844991729, 0.4006890654376487]}, "another_cube_z": {"mean": 0.3792947863782824, "std": 0.016240742797492536, "values": [0.38066205470430814, 0.394512470741892, 0.3676440744001206, 0.3584743678026895, 0.395180964242402]}, "learned_z": {"mean": 0.34917305634958706, "std": 0.01964559894254288, "values": [0.34885047201396024, 0.3686806940736411, 0.327968738021423, 0.33132155934892926, 0.36904381828998156]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": false, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": true, "S3": true, "S4": false, "plain_oracle_gap": 0.0007830060002623762, "gap_recovery": -14.698629812521885}, "mean_z_absolute_gain": -0.0007215791682191885}

<!-- search-decision-12-base-06-bcdfb498b15e -->

Iteration 12 decision: **reverted**. This candidate fails required F4 and cannot displace a reference that passes F4 and the other required checks, even with lower NMSE. Preserve the fixed feasibility constraint; the locked metric is unchanged. Retained reference: iteration 6.

<!-- search-command-13-base-06-579abbd14aab -->

Exact candidate command (from repository root; any earlier command for the same iteration with a different retained base is superseded):

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.run candidate --iteration 13 --config /home/ubuntu/faliure/clean/configs/search_iteration_13.json --base-iteration 6 --hypothesis 'Oracle has not established a positive Plain-to-Oracle error gap. An additional hidden layer may capture nonlinear contact responses that a shallow shared model misses.' --change 'Change only depth from 2 to 3; every other configuration setting is retained.' --device cuda
```

## Iteration 13

Hypothesis: Oracle has not established a positive Plain-to-Oracle error gap. An additional hidden layer may capture nonlinear contact responses that a shallow shared model misses.

Change: Change only depth from 2 to 3; every other configuration setting is retained.

Base: 6. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 0.35427510131667517, "std": 0.036213121227545095, "values": [0.4186508417307005, 0.3313101928815722, 0.3393015043271002, 0.3414207084296826, 0.3406922592143205]}, "history": {"mean": 0.39353544122381273, "std": 0.022582565992183282, "values": [0.40022655781190397, 0.41717883921025256, 0.40974058362482674, 0.3635376352655521, 0.37699359020652823]}, "oracle": {"mean": 0.34422946438716295, "std": 0.014140957483789324, "values": [0.32507431559083677, 0.3588899123617667, 0.3364686397020471, 0.34392726207495355, 0.3567871922062105]}, "fitted_z": {"mean": 0.3563601976662305, "std": 0.020672578755920126, "values": [0.3437568199810933, 0.3464483352683392, 0.36169606139973665, 0.3902114671132331, 0.3396883045687501]}, "mean_z": {"mean": 0.35642013745535156, "std": 0.023902496023806355, "values": [0.3403257078696275, 0.3634098976906355, 0.35656415285890164, 0.3920233053404269, 0.32977762351716605]}, "shuffled_z": {"mean": 0.4069992455360728, "std": 0.025514502529963175, "values": [0.39678530590454636, 0.40649767487855576, 0.4101691119103104, 0.4458496015973005, 0.37569453338965075]}, "another_cube_z": {"mean": 0.39190568864796543, "std": 0.028651918323993216, "values": [0.36923581116740467, 0.4083745100880888, 0.3872446293906413, 0.43210646028903765, 0.3625670323046547]}, "learned_z": {"mean": 0.34963386652740425, "std": 0.026784950005288244, "values": [0.3330547383002991, 0.3531593522834556, 0.3537876446186943, 0.38961955521082003, 0.31854804222375227]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": true, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": true, "S3": true, "S4": true, "plain_oracle_gap": 0.010045636929512214, "gap_recovery": -0.2075623839668854}, "mean_z_absolute_gain": 5.993978912105513e-05}

<!-- search-decision-13-base-06-579abbd14aab -->

Iteration 13 decision: **reverted**. Fitted-z validation NMSE did not decrease; retained the lower-error eligible reference. Retained reference: iteration 6.

<!-- search-command-14-base-06-7d37b36a5dd1 -->

Exact candidate command (from repository root; any earlier command for the same iteration with a different retained base is superseded):

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.run candidate --iteration 14 --config /home/ubuntu/faliure/clean/configs/search_iteration_14.json --base-iteration 6 --hypothesis 'Oracle has not established a positive Plain-to-Oracle error gap. A smaller training learning rate may reduce optimization noise in the shared laws and object embeddings.' --change 'Change only lr from 0.001 to 0.0003; every other configuration setting is retained.' --device cuda
```

## Iteration 14

Hypothesis: Oracle has not established a positive Plain-to-Oracle error gap. A smaller training learning rate may reduce optimization noise in the shared laws and object embeddings.

Change: Change only lr from 0.001 to 0.0003; every other configuration setting is retained.

Base: 6. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 0.35589088589032786, "std": 0.006427518130777838, "values": [0.3482774633731695, 0.3504032248985203, 0.3575275080543078, 0.36371809420317447, 0.3595281389224671]}, "history": {"mean": 0.4250655175120964, "std": 0.04004356772455203, "values": [0.4537190447158087, 0.4459946847750048, 0.3606487665306391, 0.4113241883479107, 0.4536409031911188]}, "oracle": {"mean": 0.35640183005441933, "std": 0.018811694568203804, "values": [0.35226488132914424, 0.38406066420502194, 0.3421733409787042, 0.33783104394628305, 0.3656792198129433]}, "fitted_z": {"mean": 0.35135347368984543, "std": 0.013390376880156226, "values": [0.33987273111383937, 0.3521439968573269, 0.341570448364471, 0.349768676186245, 0.3734115159273449]}, "mean_z": {"mean": 0.35657379986126403, "std": 0.013751771121894798, "values": [0.3439259986275804, 0.35944254707108925, 0.34556875225347367, 0.35575987601648473, 0.3781718253376921]}, "shuffled_z": {"mean": 0.36353919158474207, "std": 0.014096373704818563, "values": [0.349707855932416, 0.3680693858316502, 0.35174848134456327, 0.3635601815461259, 0.38461005326895514]}, "another_cube_z": {"mean": 0.3607187411818513, "std": 0.013553092406293913, "values": [0.34825132496213984, 0.36338943893479836, 0.35035278014490856, 0.3593664421689235, 0.3822337196984863]}, "learned_z": {"mean": 0.35149511955259677, "std": 0.013501462386044091, "values": [0.33839843068123426, 0.3529176968619123, 0.3418688915814272, 0.35128033502604955, 0.3730102436123607]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": false, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": false, "S3": true, "S4": false, "plain_oracle_gap": -0.0005109441640914758, "gap_recovery": null}, "mean_z_absolute_gain": 0.005220326171418599}

<!-- search-decision-14-base-06-7d37b36a5dd1 -->

Iteration 14 decision: **reverted**. This candidate fails required F4 and cannot displace a reference that passes F4 and the other required checks, even with lower NMSE. Preserve the fixed feasibility constraint; the locked metric is unchanged. Retained reference: iteration 6.

<!-- search-command-15-base-06-0343a7843f16 -->

Exact candidate command (from repository root; any earlier command for the same iteration with a different retained base is superseded):

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.run candidate --iteration 15 --config /home/ubuntu/faliure/clean/configs/search_iteration_15.json --base-iteration 6 --hypothesis 'Oracle has not established a positive Plain-to-Oracle error gap. A larger training learning rate may improve convergence within the unchanged update budget.' --change 'Change only lr from 0.001 to 0.003; every other configuration setting is retained.' --device cuda
```

## Iteration 15

Hypothesis: Oracle has not established a positive Plain-to-Oracle error gap. A larger training learning rate may improve convergence within the unchanged update budget.

Change: Change only lr from 0.001 to 0.003; every other configuration setting is retained.

Base: 6. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 0.37959607652667726, "std": 0.0155445810148876, "values": [0.3720767440637964, 0.3991626212972255, 0.39328764685885503, 0.36467188634593556, 0.3687814840675739]}, "history": {"mean": 0.39316363141473676, "std": 0.034299882540389494, "values": [0.42625600643350653, 0.413307731925247, 0.34433254425270254, 0.37082565845705273, 0.411096216005175]}, "oracle": {"mean": 0.3917844940852115, "std": 0.05144521250411094, "values": [0.470992355721438, 0.3866364807924978, 0.40626723237565954, 0.3563564931478014, 0.3386699083886608]}, "fitted_z": {"mean": 0.3987110579215215, "std": 0.021330094353861937, "values": [0.4179384389564545, 0.3733754013864827, 0.4211374184129971, 0.3998316283829664, 0.3812724024687071]}, "mean_z": {"mean": 0.3889329029963352, "std": 0.026465923684622562, "values": [0.3733815686055394, 0.4164096721373777, 0.4161035481574865, 0.3816825850139159, 0.3570871410673566]}, "shuffled_z": {"mean": 0.44519139982201245, "std": 0.023326008281307875, "values": [0.4511723655638166, 0.4396750193884285, 0.4789532128154272, 0.441945879967179, 0.414210521375211]}, "another_cube_z": {"mean": 0.4438040407207218, "std": 0.026306045951121855, "values": [0.43627332611836794, 0.4760381978971073, 0.4620837736157579, 0.43670610335049614, 0.4079188026218796]}, "learned_z": {"mean": 0.39547245712249435, "std": 0.02265224248916222, "values": [0.39625463566431335, 0.4098477067277944, 0.41943359200227964, 0.39181867439551615, 0.3600076768225681]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": false, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": true, "S3": true, "S4": false, "plain_oracle_gap": -0.012188417558534226, "gap_recovery": null}, "mean_z_absolute_gain": -0.0097781549251863}

<!-- search-decision-15-base-06-0343a7843f16 -->

Iteration 15 decision: **reverted**. This candidate fails required F4 and cannot displace a reference that passes F4 and the other required checks, even with lower NMSE. Preserve the fixed feasibility constraint; the locked metric is unchanged. Retained reference: iteration 6.

<!-- search-command-16-base-06-fcee68dd5691 -->

Exact candidate command (from repository root; any earlier command for the same iteration with a different retained base is superseded):

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.run candidate --iteration 16 --config /home/ubuntu/faliure/clean/configs/search_iteration_16.json --base-iteration 6 --hypothesis 'Oracle has not established a positive Plain-to-Oracle error gap. Giving moving transitions more loss weight may focus fitting on the transitions used by the locked metric.' --change 'Change only moving_weight from 1.0 to 10.0; every other configuration setting is retained.' --device cuda
```

## Iteration 16

Hypothesis: Oracle has not established a positive Plain-to-Oracle error gap. Giving moving transitions more loss weight may focus fitting on the transitions used by the locked metric.

Change: Change only moving_weight from 1.0 to 10.0; every other configuration setting is retained.

Base: 6. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 0.3999053747481719, "std": 0.01844590082826335, "values": [0.410066131081445, 0.4233095088150449, 0.37949865346479067, 0.3831526087521314, 0.4034999716274472]}, "history": {"mean": 0.40906464557884636, "std": 0.023532353721530883, "values": [0.4399465490679764, 0.42441744539641546, 0.3888732676357772, 0.4078511817459855, 0.38423478404807704]}, "oracle": {"mean": 0.4137251598203756, "std": 0.03357919393250082, "values": [0.4091044871322888, 0.4557344981797991, 0.420320094860984, 0.36246441516199196, 0.4210023037668141]}, "fitted_z": {"mean": 0.3990003831697164, "std": 0.006915854590290907, "values": [0.38855081163315414, 0.4028110175462433, 0.39800475993448303, 0.3984894317789391, 0.40714589495576253]}, "mean_z": {"mean": 0.4076260217548944, "std": 0.011644478343270214, "values": [0.3872813135247874, 0.41677201814914666, 0.4112994895593369, 0.4101618418388955, 0.4126154457023054]}, "shuffled_z": {"mean": 0.44551634718914856, "std": 0.0137680207103, "values": [0.42529565908942873, 0.44924545939960053, 0.4558201699792079, 0.45886054479995325, 0.4383599026775524]}, "another_cube_z": {"mean": 0.4328346431040432, "std": 0.010335145548219182, "values": [0.41447162850964886, 0.4363050031140985, 0.43634207569600564, 0.4378551279501818, 0.43919938025028116]}, "learned_z": {"mean": 0.39682283537546803, "std": 0.010573882197322977, "values": [0.380017016231219, 0.40693776948519583, 0.3985879968073003, 0.39442599974010706, 0.4041453946135178]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": true, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": true, "S3": true, "S4": true, "plain_oracle_gap": -0.013819785072203716, "gap_recovery": null}, "mean_z_absolute_gain": 0.008625638585177953}

<!-- search-decision-16-base-06-fcee68dd5691 -->

Iteration 16 decision: **reverted**. Fitted-z validation NMSE did not decrease; retained the lower-error eligible reference. Retained reference: iteration 6.

<!-- search-command-17-base-06-5fe38735534b -->

Exact candidate command (from repository root; any earlier command for the same iteration with a different retained base is superseded):

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.run candidate --iteration 17 --config /home/ubuntu/faliure/clean/configs/search_iteration_17.json --base-iteration 6 --hypothesis 'Oracle has not established a positive Plain-to-Oracle error gap. A short recursive training loss may favor shared dynamics that remain useful beyond one step.' --change 'Change only prediction_horizon from 1 to 3; every other configuration setting is retained.' --device cuda
```

## Iteration 17

Hypothesis: Oracle has not established a positive Plain-to-Oracle error gap. A short recursive training loss may favor shared dynamics that remain useful beyond one step.

Change: Change only prediction_horizon from 1 to 3; every other configuration setting is retained.

Base: 6. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 0.34024918088429934, "std": 0.008334959574069389, "values": [0.34687021733595136, 0.342024407725077, 0.3355885002447846, 0.32832660419282206, 0.3484361749228618]}, "history": {"mean": 0.3772350509469867, "std": 0.06771331465611877, "values": [0.4776331402780133, 0.3020107574667933, 0.3577757360256492, 0.3411595273191632, 0.4075960936453147]}, "oracle": {"mean": 0.3474961177050424, "std": 0.026937672020590938, "values": [0.38008351498563075, 0.3563423478416581, 0.3251692427651569, 0.3146904823642357, 0.3611950005685307]}, "fitted_z": {"mean": 0.3832855897524142, "std": 0.02160288805296215, "values": [0.3940775602075617, 0.4013996067501161, 0.3460410705503679, 0.38711201137689755, 0.3877976998771277]}, "mean_z": {"mean": 0.3985014894896547, "std": 0.02365092539092592, "values": [0.41739118910954276, 0.41550740101244904, 0.35871229584466435, 0.39986373576042183, 0.4010328257211953]}, "shuffled_z": {"mean": 0.41693372728097683, "std": 0.025675173807873534, "values": [0.4377646668719652, 0.437738683649995, 0.37497602846326644, 0.41928365564897263, 0.41490560177068486]}, "another_cube_z": {"mean": 0.41706354804950185, "std": 0.0279356064274556, "values": [0.4468573427883783, 0.4297559446082439, 0.3716200825437733, 0.41724912715199425, 0.41983524315511955]}, "learned_z": {"mean": 0.38500989677930464, "std": 0.022635849361954934, "values": [0.400482181309305, 0.4017899808920557, 0.34640630732244104, 0.3851723881983938, 0.3911986261743279]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": false, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": false, "S3": true, "S4": false, "plain_oracle_gap": -0.007246936820743066, "gap_recovery": null}, "mean_z_absolute_gain": 0.015215899737240468}

<!-- search-decision-17-base-06-5fe38735534b -->

Iteration 17 decision: **reverted**. This candidate fails required F4 and cannot displace a reference that passes F4 and the other required checks, even with lower NMSE. Preserve the fixed feasibility constraint; the locked metric is unchanged. Retained reference: iteration 6.

<!-- search-command-18-base-06-4e7b46b15173 -->

Exact candidate command (from repository root; any earlier command for the same iteration with a different retained base is superseded):

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.run candidate --iteration 18 --config /home/ubuntu/faliure/clean/configs/search_iteration_18.json --base-iteration 6 --hypothesis 'Oracle has not established a positive Plain-to-Oracle error gap. The longest allowed recursive training loss may reduce unstable shared dynamics while preserving the locked evaluation metric.' --change 'Change only prediction_horizon from 1 to 5; every other configuration setting is retained.' --device cuda
```

## Iteration 18

Hypothesis: Oracle has not established a positive Plain-to-Oracle error gap. The longest allowed recursive training loss may reduce unstable shared dynamics while preserving the locked evaluation metric.

Change: Change only prediction_horizon from 1 to 5; every other configuration setting is retained.

Base: 6. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 0.3499950465809303, "std": 0.022440076132338255, "values": [0.33981209735444246, 0.3689612069635972, 0.3777241248308224, 0.32414466217713184, 0.3393331415786575]}, "history": {"mean": 0.37073368021339864, "std": 0.026979488773345082, "values": [0.35234084922590975, 0.39265004282148736, 0.40666551288831676, 0.3475790029564861, 0.3544329931747933]}, "oracle": {"mean": 0.3430917103735664, "std": 0.021814070443122266, "values": [0.3620861793205474, 0.36528045280406723, 0.3418954498386084, 0.31193187077553763, 0.3342645991290714]}, "fitted_z": {"mean": 0.3494525671243072, "std": 0.018215106317515092, "values": [0.35626073066904895, 0.3569259979595292, 0.34469574262654107, 0.3206384760755769, 0.3687418882908399]}, "mean_z": {"mean": 0.35727273232193124, "std": 0.01899451572322866, "values": [0.3675342650859463, 0.36309330060523687, 0.3525558626062388, 0.3269207496303215, 0.3762594836819126]}, "shuffled_z": {"mean": 0.37231449321983934, "std": 0.018895574834118876, "values": [0.37848945636562503, 0.3809291918798871, 0.3684686063696814, 0.3419312244114988, 0.3917539870725046]}, "another_cube_z": {"mean": 0.36930119274873324, "std": 0.019908145492826275, "values": [0.3803104422001434, 0.37474178101513594, 0.36266984871598373, 0.33848482988681644, 0.3902990619255868]}, "learned_z": {"mean": 0.34681625503548497, "std": 0.018460481002935472, "values": [0.35515825077182145, 0.35457884166516224, 0.3421513788084736, 0.3171577691426168, 0.36503503478935073]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": false, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": false, "S3": true, "S4": false, "plain_oracle_gap": 0.006903336207363886, "gap_recovery": 0.07858221595007742}, "mean_z_absolute_gain": 0.007820165197624018}

<!-- search-decision-18-base-06-4e7b46b15173 -->

Iteration 18 decision: **reverted**. This candidate fails required F4 and cannot displace a reference that passes F4 and the other required checks, even with lower NMSE. Preserve the fixed feasibility constraint; the locked metric is unchanged. Retained reference: iteration 6.

<!-- search-command-19-base-06-672d71c73e77 -->

Exact candidate command (from repository root; any earlier command for the same iteration with a different retained base is superseded):

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.run candidate --iteration 19 --config /home/ubuntu/faliure/clean/configs/search_iteration_19.json --base-iteration 6 --hypothesis 'Oracle has not established a positive Plain-to-Oracle error gap. Stronger latent regularization may limit identity-specific encoding and improve fresh-z fitting.' --change 'Change only z_l2 from 0.0001 to 0.01; every other configuration setting is retained.' --device cuda
```

## Iteration 19

Hypothesis: Oracle has not established a positive Plain-to-Oracle error gap. Stronger latent regularization may limit identity-specific encoding and improve fresh-z fitting.

Change: Change only z_l2 from 0.0001 to 0.01; every other configuration setting is retained.

Base: 6. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 0.35567075420991145, "std": 0.01076059665740359, "values": [0.36652109067503874, 0.36823330509023067, 0.34661223646396555, 0.3473680542807489, 0.34961908453957347]}, "history": {"mean": 0.3741893682740813, "std": 0.01939707774458302, "values": [0.3788907130913467, 0.40571205528693427, 0.36095671923164063, 0.35770372604925704, 0.36768362771122765]}, "oracle": {"mean": 0.36183473179282044, "std": 0.044752682983097695, "values": [0.344127740875849, 0.44107821134396963, 0.3324232274933246, 0.34146526483890277, 0.3500792144120561]}, "fitted_z": {"mean": 0.34553908469896644, "std": 0.01136816597878108, "values": [0.33442440256375777, 0.3608050241713967, 0.33438964978584296, 0.3466341580576902, 0.3514421889161447]}, "mean_z": {"mean": 0.3531439831022458, "std": 0.009792481511046244, "values": [0.34516575788804277, 0.360983943836213, 0.3404137699687695, 0.36218557020066144, 0.35697087361754254]}, "shuffled_z": {"mean": 0.3764195742743408, "std": 0.011354273975744782, "values": [0.36319589242403283, 0.38876035790855135, 0.36767621060778377, 0.38696587975331054, 0.3754995306780255]}, "another_cube_z": {"mean": 0.3704773869998607, "std": 0.009166231005701514, "values": [0.3611963848090236, 0.37647961386896334, 0.3599784629167746, 0.37946093560638944, 0.3752715377981528]}, "learned_z": {"mean": 0.3475662328002347, "std": 0.00997225133510082, "values": [0.33830178708750513, 0.358028182914531, 0.3356445430120472, 0.3541262546847881, 0.3517303963023021]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": true, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": true, "S3": true, "S4": true, "plain_oracle_gap": -0.006163977582908986, "gap_recovery": null}, "mean_z_absolute_gain": 0.007604898403279381}

<!-- search-decision-19-base-06-672d71c73e77 -->

Iteration 19 decision: **kept**. Fitted-z validation NMSE decreased relative to the retained eligible reference. At least one locked success criterion still fails. Retained reference: iteration 19.

<!-- search-command-20-base-19-759fafb77695 -->

Exact candidate command (from repository root; any earlier command for the same iteration with a different retained base is superseded):

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.run candidate --iteration 20 --config /home/ubuntu/faliure/clean/configs/search_iteration_20.json --base-iteration 19 --hypothesis 'Oracle has not established a positive Plain-to-Oracle error gap. A larger fitting learning rate may overcome slow adaptation within the fixed fitting-step count.' --change 'Change only fit_lr from 0.02 to 0.1; every other configuration setting is retained.' --device cuda
```

## Iteration 20

Hypothesis: Oracle has not established a positive Plain-to-Oracle error gap. A larger fitting learning rate may overcome slow adaptation within the fixed fitting-step count.

Change: Change only fit_lr from 0.02 to 0.1; every other configuration setting is retained.

Base: 19. Status: running; validation only.

Results: {"summary": {"plain": {"mean": 0.35567075420991145, "std": 0.01076059665740359, "values": [0.36652109067503874, 0.36823330509023067, 0.34661223646396555, 0.3473680542807489, 0.34961908453957347]}, "history": {"mean": 0.3741893682740813, "std": 0.01939707774458302, "values": [0.3788907130913467, 0.40571205528693427, 0.36095671923164063, 0.35770372604925704, 0.36768362771122765]}, "oracle": {"mean": 0.3618347317928204, "std": 0.04475268298309771, "values": [0.3441277408758489, 0.44107821134396963, 0.3324232274933246, 0.3414652648389027, 0.3500792144120561]}, "fitted_z": {"mean": 0.34944655823094084, "std": 0.012516392314933749, "values": [0.3387251199012825, 0.36628409124675604, 0.33659718473267575, 0.3483179986797028, 0.3573083965942875]}, "mean_z": {"mean": 0.3531439831022458, "std": 0.009792481511046244, "values": [0.34516575788804277, 0.360983943836213, 0.3404137699687695, 0.36218557020066144, 0.35697087361754254]}, "shuffled_z": {"mean": 0.37723961643829457, "std": 0.00967493166451181, "values": [0.3648605628322663, 0.38707013513960997, 0.3695800798793873, 0.38496722634798064, 0.3797200779922285]}, "another_cube_z": {"mean": 0.37047738699986077, "std": 0.009166231005701526, "values": [0.36119638480902366, 0.3764796138689634, 0.35997846291677454, 0.37946093560638944, 0.3752715377981528]}, "learned_z": {"mean": 0.3475662328002347, "std": 0.00997225133510082, "values": [0.33830178708750513, 0.358028182914531, 0.3356445430120472, 0.3541262546847881, 0.3517303963023021]}}, "checks": {"F1": true, "F2": true, "F3": true, "F4": true, "F5": true, "F6": true, "F7": true, "F8": true, "F9": true}, "success": {"S1": false, "S2": true, "S3": true, "S4": true, "plain_oracle_gap": -0.0061639775829089305, "gap_recovery": null}, "mean_z_absolute_gain": 0.0036974248713049773}

<!-- search-decision-20-base-19-759fafb77695 -->

Iteration 20 decision: **reverted**. Fitted-z validation NMSE did not decrease; retained the lower-error eligible reference. Retained reference: iteration 19.

<!-- search-budget-exhausted -->

All 20 scientific candidate iterations are complete without validation success. Selected iteration 19 for the single final evaluation. F4 is a fixed required constraint, not an alternate metric. A reference passing F4 and the other required checks exists, so select the lowest fitted-z validation NMSE within that feasible set; F4-failing candidates cannot displace it. No threshold was changed.

## Final completion

All20 candidate iterations completed without validation success. Candidate19 was selected from candidates passing all failure checks. Exactly one final evaluation covered both held-out groups and allfive seeds/ablations. In-range S1–S4 passed; out-of-range S2/S3 failed. The independent saved-result integrity audit passed. The final pass is limited by the small/noisy Oracle gap, familiar script mixture in test scoring, weak property decoding of z, and no rollout gain. No model, metric, or data changes followed final evaluation.
