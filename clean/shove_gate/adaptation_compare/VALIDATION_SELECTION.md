# Step 3: validation-only adaptation settings

A selects **z dimension 4, prior penalty 1**, Adam learning rate 0.01, 50 updates.
B selects **learning rate 1e-05**, 50 full-support Adam updates. Both declared rates remain in the test comparison.

Only cubes 180–199 selected these settings: support shoves 0–4, query shoves 5–9, mean across all 20 cubes and five seeds. No new test cube or retired test observation enters selection.

| Model / observation | Push velocity NMSE | Moving-coast velocity NMSE | Full-state NMSE | Composite |
|---|---:|---:|---:|---:|
| A selected | 0.154089 ± 0.014544 | 0.331895 ± 0.0285016 | 1.35376 ± 0.0412501 | 0.613248 ± 0.0191668 |
| B lr 1e-05 (selected) | 0.143357 ± 0.00838093 | 0.322589 ± 0.037805 | 1.26772 ± 0.0946651 | 0.577888 ± 0.0266492 |
| B lr 0.0001 | 0.129797 ± 0.0138257 | 0.473068 ± 0.0476942 | 1.35895 ± 0.0921798 | 0.653939 ± 0.0270122 |

All three components have their own fitting-only zero-change energy scale. The composite is their unweighted average; lower is better. Scales use IDs 0–179, shoves 0–7, while frozen network input/output scalers are unchanged.

| Scale | Value |
|---|---:|
| Push horizontal Δv energy ((m/s)²) | 0.00247537921214 |
| Moving-coast horizontal Δv energy ((m/s)²) | 0.000294527390245 |
| Full normalized-state Δ energy | 0.162185466745 |

A minimizes full-state support loss plus the declared penalty toward training-mean z. B minimizes the identical full-state data term while updating all Plain weights. Every fit uses all retained support steps, with equal episode weighting, and exactly 50 updates.

A exact ties prefer smaller dimension, then larger penalty. B exact ties prefer smaller learning rate. No hyperparameter grid is extended after these choices.

| z dimension | Prior penalty | Validation composite ± seed SD |
|---:|---:|---:|
| 4 | 0.0001 | 0.643694 ± 0.0415621 |
| 4 | 0.001 | 0.643487 ± 0.0415708 |
| 4 | 0.01 | 0.641326 ± 0.0409097 |
| 4 | 0.1 | 0.628129 ± 0.0405787 |
| 4 | 1 | 0.613248 ± 0.0191668 |
| 8 | 0.0001 | 0.68323 ± 0.0348448 |
| 8 | 0.001 | 0.682577 ± 0.0354892 |
| 8 | 0.01 | 0.680129 ± 0.0350521 |
| 8 | 0.1 | 0.674301 ± 0.031707 |
| 8 | 1 | 0.64471 ± 0.02894 |
| 16 | 0.0001 | 0.705117 ± 0.0451141 |
| 16 | 0.001 | 0.70465 ± 0.0451982 |
| 16 | 0.01 | 0.702957 ± 0.0455484 |
| 16 | 0.1 | 0.695027 ± 0.0482903 |
| 16 | 1 | 0.681988 ± 0.0406997 |

[Protocol and reasons](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/PROTOCOL.md)

![Validation penalty grid](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/results/figures/validation_penalty.png)
