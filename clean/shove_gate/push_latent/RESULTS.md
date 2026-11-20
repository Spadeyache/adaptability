# Step 2: push confirmation and object-latent adaptation

C1 **PASS**, C2 **FAIL**, C3 **PASS**.

**The overall preregistered success criteria are not met: C2 failed.**

Primary population: push steps of shoves 5–9 on the 30 fresh cubes 260–289. Mean ± sample SD is across five paired seeds. Lower error is better.

| Model / control | Push horizontal-delta NMSE | Improvement vs Plain | Beyond seed spread? |
|---|---:|---:|---|
| Plain | 0.209563 ± 0.00515 | reference | — |
| History (GRU 64) | 0.190823 ± 0.012 | +8.94% | yes |
| Oracle | 0.0807122 ± 0.00937 | +61.49% | yes |
| Two-part: fitted z (5 shoves) | 0.161539 ± 0.0307 | +22.92% | yes |
| Two-part: mean z | 0.213803 ± 0.0105 | -2.02% | no |
| Two-part: another cube’s z | 0.276197 ± 0.0128 | -31.80% | no |

| Model / control | Raw horizontal Δv MSE ((m/s)²) | Pooled-coordinate Δv RMSE (m/s) |
|---|---:|---:|
| Plain | 0.00051874844 | 0.02277605 |
| History (GRU 64) | 0.00047235835 | 0.021733807 |
| Oracle | 0.00019979323 | 0.014134823 |
| Two-part: fitted z (5 shoves) | 0.00039986951 | 0.019996737 |
| Two-part: mean z | 0.00052924258 | 0.023005273 |
| Two-part: another cube’s z | 0.00068369313 | 0.026147526 |

Raw MSE equals the saved five-seed mean NMSE times the frozen fitting push-energy scale. The pooled-coordinate RMSE is the square root of that MSE, pooling x/y coordinates and all five seeds after the stated cube/episode averaging.


![Primary comparison](/home/ubuntu/faliure/clean/shove_gate/push_latent/results/figures/primary_comparison.png)

C1 requires ≥20% Oracle improvement and `Oracle mean+SD < Plain mean−SD`. Measured improvement is **+61.49%**, with mean gap **0.128851** versus summed seed SD **0.0145246**.

C2 requires fitted z to recover at least 50% of the Plain-to-Oracle gap. It recovers **37.27%**.

History recovers **14.54%** of the same gap as a descriptive comparison; History does not decide C2 or C3.

| C3 comparison | Mean gap favoring fitted z | Sum of seed SDs | Beyond spread? |
|---|---:|---:|---|
| Mean z → fitted z | 0.0522639 | 0.0411537 | yes |
| Another cube’s z → fitted z | 0.114659 | 0.0434684 | yes |

For fitted z versus History, the mean gap favoring fitted z is **0.0292839**, compared with combined seed SD **0.0427121**. The gap does not exceed that spread.

The mean-z and wrong-z spread comparisons must both pass C3. No other metric or support prefix can rescue a failed criterion.

| Support shoves | Query push NMSE |
|---:|---:|
| 1 | 0.207674 ± 0.0209 |
| 2 | 0.191387 ± 0.0268 |
| 3 | 0.177477 ± 0.0276 |
| 4 | 0.174575 ± 0.0281 |
| 5 | 0.161539 ± 0.0307 |

![Support prefix curve](/home/ubuntu/faliure/clean/shove_gate/push_latent/results/figures/support_prefix.png)

Each prefix independently restarts at training-mean z and uses identical selected fitting settings. All query scores use shoves 5–9. The wrong-z control uses the fixed cyclic next-cube mapping, never a similarity or performance match.

## Affine mass readout

An affine least-squares map from support-refitted z to true mass was fitted on 180 weight-training cubes only, then frozen. The 20 latent-validation cubes did not fit or select it. These scores do not decide any criterion.

| Diagnostic | R² | MAE (kg) | RMSE (kg) |
|---|---:|---:|---:|
| Fitted-z affine decoder | 0.876836 ± 0.0553 | 0.0267676 ± 0.00696 | 0.0339173 ± 0.00755 |
| Training-mean mass | -0.120587 ± 0 | 0.0968598 ± 0 | 0.104316 ± 0 |

## Protocol, data, and separation

[The written protocol and reasons](/home/ubuntu/faliure/clean/shove_gate/push_latent/PROTOCOL.md) preceded all Step 2 collection, fitting, tuning, and scoring. The prior 56% push benefit was secondary; fresh cubes independently confirm or reject it as a predeclared primary endpoint. The previous test sets/results remain untouched.

IDs 0–179 fit weights on shoves 0–7; shoves 8–9 select checkpoints and budgets. IDs 180–199 are reserved solely for latent dimension and fitting choices. Fresh IDs 260–289 use seed 2026111519; IDs 200–259 are retired. Original training traces are byte-identical.

Fresh collection: **300/300 launches**, **90.86%** moving coast after trimming, maximum coast **0.934926 m**; **21,167** retained and **73,587** stationary transitions trimmed. All episodes reached rest, no moving transition was trimmed, and no sample was rejected or replaced. The integrity audit preserved **1,368** retired artifacts.

Primary normalization is fitting-only push change energy **0.00247537921214 (m/s)²**, averaged equally by cube, episode, and push step. A zero-change predictor scores 1 on that fitting subset. All primary/query scores use this same scalar.

State/action inputs exclude IDs, phase, lengths, masks, stop times, termination indicators, and future observations. True properties enter only Oracle and the separate mass-readout diagnostic. History uses ten strictly preceding transitions from the current episode, with fixed initial-state/zero-action/zero-delta padding and no mask. Recursive History uses predicted transitions after the forecast start.

## Training and validation-only choices

All seven variants retain the full 15 normalized state-change training loss and three 256-unit ReLU hidden layers. History adds GRU 64. The four two-part candidates use zero-initialized per-training-cube embeddings of dimension 2,4,8,16 with 1e-4 L2. No mass/friction label enters a two-part model.

Every seed shares minibatches and update budgets across all seven variants. Checkpoints minimize fitting-cube held-out-shove push NMSE. The predeclared final 2,500-update improvement rule extends every variant by 10,000 updates if any variant improves its previous best by ≥0.1%, with a 100,000-update resource cap.

| Seed | Updates per variant | Still improving at cap? |
|---:|---:|---|
| 101 | 70,000 | no |
| 211 | 30,000 | no |
| 307 | 30,000 | no |
| 401 | 60,000 | no |
| 503 | 50,000 | no |

![All checkpoint-validation curves](/home/ubuntu/faliure/clean/shove_gate/push_latent/results/figures/validation_curves.png)

| Seed | Variant | Selected update | Selected validation push NMSE |
|---:|---|---:|---:|
| 101 | plain | 15,500 | 0.145007 |
| 101 | oracle | 43,000 | 0.0513697 |
| 101 | history | 69,500 | 0.145157 |
| 101 | z2 | 33,000 | 0.156322 |
| 101 | z4 | 33,500 | 0.147621 |
| 101 | z8 | 66,000 | 0.143199 |
| 101 | z16 | 51,500 | 0.144265 |
| 211 | plain | 17,000 | 0.145437 |
| 211 | oracle | 10,500 | 0.0672417 |
| 211 | history | 22,500 | 0.139538 |
| 211 | z2 | 14,500 | 0.159513 |
| 211 | z4 | 26,000 | 0.153438 |
| 211 | z8 | 22,500 | 0.154311 |
| 211 | z16 | 29,500 | 0.157344 |
| 307 | plain | 10,500 | 0.15251 |
| 307 | oracle | 26,500 | 0.0693448 |
| 307 | history | 26,500 | 0.13706 |
| 307 | z2 | 18,500 | 0.157587 |
| 307 | z4 | 27,500 | 0.163378 |
| 307 | z8 | 17,000 | 0.150178 |
| 307 | z16 | 14,000 | 0.157171 |
| 401 | plain | 16,500 | 0.152263 |
| 401 | oracle | 19,000 | 0.0529552 |
| 401 | history | 50,000 | 0.136235 |
| 401 | z2 | 47,000 | 0.147745 |
| 401 | z4 | 18,000 | 0.157872 |
| 401 | z8 | 12,000 | 0.147231 |
| 401 | z16 | 11,500 | 0.161933 |
| 503 | plain | 28,500 | 0.14635 |
| 503 | oracle | 41,000 | 0.0492566 |
| 503 | history | 50,000 | 0.139421 |
| 503 | z2 | 26,500 | 0.154622 |
| 503 | z4 | 22,500 | 0.167213 |
| 503 | z8 | 28,500 | 0.154299 |
| 503 | z16 | 40,000 | 0.148589 |

The 20 latent-validation cubes selected one global configuration: **z dimension 16, learning rate 0.01, 50 updates**. Its validation query mean NMSE is **0.114005**. Selection used support shoves 0–4 and query shoves 5–9 across all five seeds; no test result or true physical property selected it.

![Latent validation grid](/home/ubuntu/faliure/clean/shove_gate/push_latent/results/figures/latent_validation_grid.png)

## Secondary prediction metrics

These scores decide no criterion. All groups use query shoves 5–9. Old state metrics retain full 15-coordinate normalization. Ten-step forecasts are recursive; stopping forecasts continue to planned release + 1.5 s, never an observed stop time or trimmed length. No clipping, physics correction, or observed-state reset is used.

| Group | Metric | Model | Mean ± seed SD |
|---|---|---|---:|
| Push | Horizontal velocity-change NMSE | Plain | 0.209563 ± 0.00515 |
| Push | Horizontal velocity-change NMSE | History (GRU 64) | 0.190823 ± 0.012 |
| Push | Horizontal velocity-change NMSE | Oracle | 0.0807122 ± 0.00937 |
| Push | Horizontal velocity-change NMSE | Two-part: fitted z (5 shoves) | 0.161539 ± 0.0307 |
| Push | Horizontal velocity-change NMSE | Two-part: mean z | 0.213803 ± 0.0105 |
| Push | Horizontal velocity-change NMSE | Two-part: another cube’s z | 0.276197 ± 0.0128 |
| Push | Full-state one-step NMSE | Plain | 0.0901272 ± 0.006 |
| Push | Full-state one-step NMSE | History (GRU 64) | 0.077845 ± 0.00893 |
| Push | Full-state one-step NMSE | Oracle | 0.0946271 ± 0.0126 |
| Push | Full-state one-step NMSE | Two-part: fitted z (5 shoves) | 0.12269 ± 0.0119 |
| Push | Full-state one-step NMSE | Two-part: mean z | 0.10526 ± 0.0108 |
| Push | Full-state one-step NMSE | Two-part: another cube’s z | 0.138293 ± 0.0182 |
| Push | Mean ten-step rollout NMSE | Plain | 0.180999 ± 0.0164 |
| Push | Mean ten-step rollout NMSE | History (GRU 64) | 0.137274 ± 0.0142 |
| Push | Mean ten-step rollout NMSE | Oracle | 0.15861 ± 0.032 |
| Push | Mean ten-step rollout NMSE | Two-part: fitted z (5 shoves) | 0.252805 ± 0.0418 |
| Push | Mean ten-step rollout NMSE | Two-part: mean z | 0.152745 ± 0.0128 |
| Push | Mean ten-step rollout NMSE | Two-part: another cube’s z | 0.26745 ± 0.0335 |
| Push | Tenth-step-only NMSE | Plain | 0.260436 ± 0.0428 |
| Push | Tenth-step-only NMSE | History (GRU 64) | 0.175576 ± 0.021 |
| Push | Tenth-step-only NMSE | Oracle | 0.199872 ± 0.0384 |
| Push | Tenth-step-only NMSE | Two-part: fitted z (5 shoves) | 0.334005 ± 0.0516 |
| Push | Tenth-step-only NMSE | Two-part: mean z | 0.193364 ± 0.0163 |
| Push | Tenth-step-only NMSE | Two-part: another cube’s z | 0.361056 ± 0.0429 |
| Push | Stopping-position error (m) | Plain | 0.382707 ± 0.12 |
| Push | Stopping-position error (m) | History (GRU 64) | 0.236824 ± 0.0578 |
| Push | Stopping-position error (m) | Oracle | 7.23305 ± 15.1 |
| Push | Stopping-position error (m) | Two-part: fitted z (5 shoves) | 4.36607 ± 4.18 |
| Push | Stopping-position error (m) | Two-part: mean z | 0.931747 ± 0.862 |
| Push | Stopping-position error (m) | Two-part: another cube’s z | 5.05492 ± 6.58 |
| Moving coast | Horizontal velocity-change NMSE | Plain | 0.0268817 ± 0.00257 |
| Moving coast | Horizontal velocity-change NMSE | History (GRU 64) | 0.0253209 ± 0.00126 |
| Moving coast | Horizontal velocity-change NMSE | Oracle | 0.0310052 ± 0.00284 |
| Moving coast | Horizontal velocity-change NMSE | Two-part: fitted z (5 shoves) | 0.0624679 ± 0.00611 |
| Moving coast | Horizontal velocity-change NMSE | Two-part: mean z | 0.030067 ± 0.00306 |
| Moving coast | Horizontal velocity-change NMSE | Two-part: another cube’s z | 0.0677617 ± 0.00624 |
| Moving coast | Full-state one-step NMSE | Plain | 0.0855499 ± 0.00396 |
| Moving coast | Full-state one-step NMSE | History (GRU 64) | 0.0642742 ± 0.0202 |
| Moving coast | Full-state one-step NMSE | Oracle | 0.0854529 ± 0.0104 |
| Moving coast | Full-state one-step NMSE | Two-part: fitted z (5 shoves) | 0.117837 ± 0.0074 |
| Moving coast | Full-state one-step NMSE | Two-part: mean z | 0.109473 ± 0.0135 |
| Moving coast | Full-state one-step NMSE | Two-part: another cube’s z | 0.125929 ± 0.0141 |
| Moving coast | Mean ten-step rollout NMSE | Plain | 0.12383 ± 0.00971 |
| Moving coast | Mean ten-step rollout NMSE | History (GRU 64) | 0.100222 ± 0.00816 |
| Moving coast | Mean ten-step rollout NMSE | Oracle | 0.15741 ± 0.0217 |
| Moving coast | Mean ten-step rollout NMSE | Two-part: fitted z (5 shoves) | 0.196799 ± 0.0231 |
| Moving coast | Mean ten-step rollout NMSE | Two-part: mean z | 0.11832 ± 0.00374 |
| Moving coast | Mean ten-step rollout NMSE | Two-part: another cube’s z | 0.204993 ± 0.0251 |
| Moving coast | Tenth-step-only NMSE | Plain | 0.166077 ± 0.0196 |
| Moving coast | Tenth-step-only NMSE | History (GRU 64) | 0.121902 ± 0.0135 |
| Moving coast | Tenth-step-only NMSE | Oracle | 0.223595 ± 0.0343 |
| Moving coast | Tenth-step-only NMSE | Two-part: fitted z (5 shoves) | 0.26263 ± 0.0359 |
| Moving coast | Tenth-step-only NMSE | Two-part: mean z | 0.130131 ± 0.00817 |
| Moving coast | Tenth-step-only NMSE | Two-part: another cube’s z | 0.267061 ± 0.0241 |
| Moving coast | Stopping-position error (m) | Plain | 0.341983 ± 0.101 |
| Moving coast | Stopping-position error (m) | History (GRU 64) | 0.216887 ± 0.0661 |
| Moving coast | Stopping-position error (m) | Oracle | 0.925026 ± 1.2 |
| Moving coast | Stopping-position error (m) | Two-part: fitted z (5 shoves) | 1.34906 ± 0.978 |
| Moving coast | Stopping-position error (m) | Two-part: mean z | 0.352439 ± 0.174 |
| Moving coast | Stopping-position error (m) | Two-part: another cube’s z | 1.38728 ± 1.29 |
| All retained steps | Horizontal velocity-change NMSE | Plain | 0.0600679 ± 0.00274 |
| All retained steps | Horizontal velocity-change NMSE | History (GRU 64) | 0.0561239 ± 0.00247 |
| All retained steps | Horizontal velocity-change NMSE | Oracle | 0.0388443 ± 0.00281 |
| All retained steps | Horizontal velocity-change NMSE | Two-part: fitted z (5 shoves) | 0.0756547 ± 0.00501 |
| All retained steps | Horizontal velocity-change NMSE | Two-part: mean z | 0.0627219 ± 0.00309 |
| All retained steps | Horizontal velocity-change NMSE | Two-part: another cube’s z | 0.106391 ± 0.00776 |
| All retained steps | Full-state one-step NMSE | Plain | 0.0762245 ± 0.00337 |
| All retained steps | Full-state one-step NMSE | History (GRU 64) | 0.0593705 ± 0.0132 |
| All retained steps | Full-state one-step NMSE | Oracle | 0.078023 ± 0.0103 |
| All retained steps | Full-state one-step NMSE | Two-part: fitted z (5 shoves) | 0.107482 ± 0.00788 |
| All retained steps | Full-state one-step NMSE | Two-part: mean z | 0.0964508 ± 0.0101 |
| All retained steps | Full-state one-step NMSE | Two-part: another cube’s z | 0.117612 ± 0.0142 |
| All retained steps | Mean ten-step rollout NMSE | Plain | 0.117104 ± 0.00605 |
| All retained steps | Mean ten-step rollout NMSE | History (GRU 64) | 0.090614 ± 0.00917 |
| All retained steps | Mean ten-step rollout NMSE | Oracle | 0.142412 ± 0.0203 |
| All retained steps | Mean ten-step rollout NMSE | Two-part: fitted z (5 shoves) | 0.189757 ± 0.0222 |
| All retained steps | Mean ten-step rollout NMSE | Two-part: mean z | 0.110227 ± 0.00442 |
| All retained steps | Mean ten-step rollout NMSE | Two-part: another cube’s z | 0.198859 ± 0.0261 |
| All retained steps | Tenth-step-only NMSE | Plain | 0.165759 ± 0.012 |
| All retained steps | Tenth-step-only NMSE | History (GRU 64) | 0.115444 ± 0.0147 |
| All retained steps | Tenth-step-only NMSE | Oracle | 0.20656 ± 0.0309 |
| All retained steps | Tenth-step-only NMSE | Two-part: fitted z (5 shoves) | 0.26255 ± 0.0334 |
| All retained steps | Tenth-step-only NMSE | Two-part: mean z | 0.131961 ± 0.00272 |
| All retained steps | Tenth-step-only NMSE | Two-part: another cube’s z | 0.270192 ± 0.0269 |
| All retained steps | Stopping-position error (m) | Plain | 0.345863 ± 0.103 |
| All retained steps | Stopping-position error (m) | History (GRU 64) | 0.219356 ± 0.0632 |
| All retained steps | Stopping-position error (m) | Oracle | 1.97287 ± 3.5 |
| All retained steps | Stopping-position error (m) | Two-part: fitted z (5 shoves) | 1.86965 ± 1.53 |
| All retained steps | Stopping-position error (m) | Two-part: mean z | 0.449983 ± 0.283 |
| All retained steps | Stopping-position error (m) | Two-part: another cube’s z | 2.46563 ± 3.19 |

Individual stopping-position errors are retained below, including any large finite recursive failures; these are prediction errors, not distances traveled by real simulated cubes.

| Seed | Model | Push stopping error (m) | Moving-coast stopping error (m) | All-step stopping error (m) |
|---:|---|---:|---:|---:|
| 101 | Plain | 0.328285861 | 0.276449692 | 0.281426614 |
| 101 | History (GRU 64) | 0.173146569 | 0.155292842 | 0.158848873 |
| 101 | Oracle | 0.503818114 | 0.38105311 | 0.397377202 |
| 101 | Two-part: fitted z (5 shoves) | 0.649405589 | 0.474059461 | 0.511219649 |
| 101 | Two-part: mean z | 0.211018776 | 0.189548239 | 0.193685859 |
| 101 | Two-part: another cube’s z | 0.524442263 | 0.408219656 | 0.434415384 |
| 211 | Plain | 0.583906872 | 0.508284242 | 0.5181324 |
| 211 | History (GRU 64) | 0.294009958 | 0.254738118 | 0.259240898 |
| 211 | Oracle | 0.608003803 | 0.520695206 | 0.528646141 |
| 211 | Two-part: fitted z (5 shoves) | 4.30875266 | 1.87707572 | 2.28409066 |
| 211 | Two-part: mean z | 1.66962006 | 0.61441128 | 0.809339884 |
| 211 | Two-part: another cube’s z | 3.37483421 | 1.43012078 | 1.77897963 |
| 307 | Plain | 0.401540632 | 0.36052296 | 0.359826511 |
| 307 | History (GRU 64) | 0.301626125 | 0.314402709 | 0.310152054 |
| 307 | Oracle | 0.339762731 | 0.306091425 | 0.313386182 |
| 307 | Two-part: fitted z (5 shoves) | 0.876158183 | 0.607016097 | 0.646890674 |
| 307 | Two-part: mean z | 0.330068971 | 0.240181453 | 0.254817434 |
| 307 | Two-part: another cube’s z | 0.867420467 | 0.710840701 | 0.732875497 |
| 401 | Plain | 0.304539006 | 0.307877321 | 0.306774649 |
| 401 | History (GRU 64) | 0.216151593 | 0.177243104 | 0.183172397 |
| 401 | Oracle | 0.556400195 | 0.355674558 | 0.38666393 |
| 401 | Two-part: fitted z (5 shoves) | 5.04868318 | 0.98791772 | 1.63109876 |
| 401 | Two-part: mean z | 2.05233144 | 0.442475377 | 0.699267242 |
| 401 | Two-part: another cube’s z | 3.9927745 | 0.792524207 | 1.29045203 |
| 503 | Plain | 0.29526449 | 0.256778723 | 0.263157076 |
| 503 | History (GRU 64) | 0.199185285 | 0.182758918 | 0.185366155 |
| 503 | Oracle | 34.1572892 | 3.06161767 | 8.23828642 |
| 503 | Two-part: fitted z (5 shoves) | 10.947331 | 2.79921374 | 4.27496269 |
| 503 | Two-part: mean z | 0.395694762 | 0.27557897 | 0.292807042 |
| 503 | Two-part: another cube’s z | 16.5151451 | 3.59468912 | 8.09143576 |

**Large finite stopping-prediction errors remain included.** The largest seed/group means are Oracle, seed 503, push: 34.157289 m; Two-part: another cube’s z, seed 503, push: 16.515145 m; Two-part: fitted z (5 shoves), seed 503, push: 10.947331 m. These recursive failures do not change C1–C3, and no affected seed was removed or replaced.


## Seed-level primary scores

| Seed | Plain | History (GRU 64) | Oracle | Two-part: fitted z (5 shoves) | Two-part: mean z | Two-part: another cube’s z |
|---:|---:|---:|---:|---:|---:|---:|
| 101 | 0.203072944 | 0.203460151 | 0.0829646857 | 0.130087602 | 0.213171301 | 0.28728914 |
| 211 | 0.215847091 | 0.180150055 | 0.0846893406 | 0.200228598 | 0.223455675 | 0.254087613 |
| 307 | 0.213612752 | 0.186855599 | 0.0929122221 | 0.13336357 | 0.196742724 | 0.278608843 |
| 401 | 0.208255102 | 0.179818432 | 0.0739334789 | 0.183509896 | 0.221162474 | 0.280557905 |
| 503 | 0.207028219 | 0.203828876 | 0.069061127 | 0.160503765 | 0.214480958 | 0.280443152 |

## Actions-only independence diagnostic

Fresh commands were scored only after the frozen final-evaluation ledger opened. Fitting/tuning used only the 180 weight-training cubes, with an internal 162/18 split; the 20 latent-validation cubes were excluded.

| Probe | Mass R² ± seed SD | Combined-friction R² ± seed SD |
|---|---:|---:|
| extra_trees | -0.112920 ± 0.000246 | -0.021767 ± 0.000743 |
| polynomial_ridge | -0.116540 ± 0.00119 | -0.024622 ± 0.00409 |

## Saved evidence

- [Written protocol and reasons](/home/ubuntu/faliure/clean/shove_gate/push_latent/PROTOCOL.md)
- [Collection report](/home/ubuntu/faliure/clean/shove_gate/push_latent/DATA_REPORT.md)
- [Frozen final comparisons](/home/ubuntu/faliure/clean/shove_gate/push_latent/results/summary.json)
- [Final evaluation ledger](/home/ubuntu/faliure/clean/shove_gate/push_latent/results/final_evaluation.json)
- [Data integrity audit](/home/ubuntu/faliure/clean/shove_gate/push_latent/results/full_audit.json)
- [Independent final audit](/home/ubuntu/faliure/clean/shove_gate/push_latent/results/final_audit.json)
- [Source review and contract checks before fitting](/home/ubuntu/faliure/clean/shove_gate/push_latent/results/pretraining_review.json)
- [Frozen-input audit before fresh predictive test scoring](/home/ubuntu/faliure/clean/shove_gate/push_latent/results/pretest_audit.json)
- [All candidate checkpoint hashes](/home/ubuntu/faliure/clean/shove_gate/push_latent/results/checkpoint_seal.json)
- [Frozen adaptation and mass-readout settings](/home/ubuntu/faliure/clean/shove_gate/push_latent/results/adaptation_seal.json)
- [Complete latent validation grid](/home/ubuntu/faliure/clean/shove_gate/push_latent/results/latent_tuning.json)
- [Preserved retired artifacts](/home/ubuntu/faliure/clean/shove_gate/push_latent/results/retired_experiment_seal.json)

Every plotted point is computed from saved run records. Error bars show sample SD across optimization seeds, not confidence intervals over all possible cubes. No report operation fits a model or reruns a prediction.

Rebuild after a completed or explicitly stopped evaluation:

```bash
PYTHONPATH=clean/vendor:. MPLCONFIGDIR=/tmp/clean-mpl .venv/bin/python -m clean.shove_gate.push_latent.report
```

[Report input SHA-256 manifest](/home/ubuntu/faliure/clean/shove_gate/push_latent/results/report_inputs.json)
