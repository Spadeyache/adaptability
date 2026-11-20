# Object-specific belief adaptation in simulated shoves

Saved-results brief · Kent · 20 November 2026

**Setup.** We tested a shared next-state predictor with one learned belief row per cube, using simulated shoves. The proposed separation is common dynamics in the network and persistent object information in the row; interpretable physical constants are not established. Step 2 used 180 weight-training cubes, 20 cubes for adaptation settings, 30 unseen test cubes and five independent training runs. Each test cube supplied five fitting shoves and five separate scoring shoves. [S1]

**Leak check.** Applied forces were generated independently of mass and friction. Two actions-only prediction checks returned negative R²: mass −0.113/−0.117 and combined friction −0.022/−0.025, meaning worse than predicting the test-set average. These checks found no useful property signal in commands; they do not prove every possible leak absent. [S1, S2]

**Known-physics check.** Compared with Plain (state and force only), giving the predictor true mass AND combined cube–floor friction reduced push velocity-change error from 0.209563 ± 0.00515 to 0.080712 ± 0.00937 (61.49%). The metric is squared prediction error divided by a fixed training-data scale (NMSE; lower is better); ± denotes standard deviation across training seeds. This comparison does not isolate mass from friction. [S1]

**Belief test.** With the network frozen, fitting only the new row reduced query push NMSE to 0.161539 ± 0.0307 (22.92% below Plain), beating both the average row and another cube’s fitted row. Mean error fell from 0.207674 to 0.161539 with one to five fitting shoves. A separate linear mass decoder achieved test R² 0.876836 ± 0.0553; the beliefs used no mass labels, but this decoder used training-cube mass labels. Belief fitting recovered 37.27% of the Plain-to-Oracle gap, below the predeclared 50% target, and did not beat the recent-history model beyond seed spread. [S1]

**Drift comparison.** Step 3 reused frozen checkpoints and visited 30 new cubes in one fixed order. At learning rate 1e−4, independently resetting and adapting all weights per cube gave mean push NMSE 0.107807 versus Plain’s 0.127834. In the sequential run, the mean over cubes immediately after their own adaptation was 0.170115; training-cube validation composite error rose from 0.366298 to 0.637299. At the validation-selected rate 1e−5, the corresponding sequential means were 0.134966 and 0.378233. Composite means the average of push-velocity, moving-coast velocity and full-state normalized errors. These are not final-model scores on all visited cubes; the predeclared forgetting and speed criteria both failed. [S3, S4]

**Scope.** Oracle did not improve prediction during coast (sliding after force release), although the explicit friction formula predicted coast changes accurately. Push-focused belief fitting worsened coast and full-state prediction, and longer forecasts were unreliable. This is idealized direct-force simulation, not demonstrated robot contact or learning that accumulates across objects. [S1, S3, S5]

## Sources

- **S1:** [clean/shove_gate/push_latent/RESULTS.md](/home/ubuntu/faliure/clean/shove_gate/push_latent/RESULTS.md:7), lines 7–84; 140–177; 283–290.
- **S2:** [clean/shove_gate/push_latent/collection.py](/home/ubuntu/faliure/clean/shove_gate/push_latent/collection.py:37), lines 37–45; 84–90.
- **S3:** [clean/shove_gate/adaptation_compare/RESULTS.md](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/RESULTS.md:3), lines 3–18; 58–89; 150–208; 224.
- **S4:** [clean/shove_gate/adaptation_compare/results/gain_damage_partial.csv](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/results/gain_damage_partial.csv:2), rows 2–3.
- **S5:** [clean/shove_gate/revised_velocity/RESULTS.md](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/RESULTS.md:3), lines 3–29; 44.
