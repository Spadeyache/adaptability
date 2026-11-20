# Figure captions

A training seed changes a run's random choices; seed standard deviation describes the variation across separately trained models. Coast means sliding after the applied force is removed, and normalized error means squared prediction error divided by a fixed training-data scale.

**setup**

During training, many cubes share one prediction network while each has a learned row in a belief table; for a new cube, the network is frozen and its row starts at the training average, is fitted on five shoves, tested on five separate shoves, and saved. This separates shared prediction from object-specific adaptation, while “laws” and “constants” remain the proposed interpretation of that split rather than a demonstrated physical identification. [Training and fitting procedure](/home/ubuntu/faliure/clean/shove_gate/push_latent/RESULTS.md:72), [support/query separation](/home/ubuntu/faliure/clean/shove_gate/push_latent/RESULTS.md:57), [stored rows](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/RESULTS.md:58).

**fig_leak**

Two saved predictors using only commanded actions fail to predict mass or combined cube–floor friction on unseen cubes, with negative R² meaning worse than predicting the test-set average; error bars show standard deviation across training seeds. Together with independently generated commands, this supports the absence of an action-label shortcut in these checks, but does not prove that every possible shortcut is absent. [Saved checks](/home/ubuntu/faliure/clean/shove_gate/push_latent/RESULTS.md:283), [generator](/home/ubuntu/faliure/clean/shove_gate/push_latent/collection.py:37).

**fig_physics**

Giving the network true mass and combined cube–floor friction reduces push velocity-change error by 61.49% relative to Plain, which sees only state and force; the dimensionless normalized error uses a fixed training-data scale and error bars show seed standard deviation. This establishes useful predictive information in the supplied properties on fresh cubes, while their joint input prevents attributing the gain to mass alone. [Saved comparison](/home/ubuntu/faliure/clean/shove_gate/push_latent/RESULTS.md:11), [inputs](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/RESULTS.md:18), [normalization](/home/ubuntu/faliure/clean/shove_gate/push_latent/RESULTS.md:76).

**fig_belief**

With the network frozen, fitting the cube's belief from five shoves lowers error on its other five shoves by 22.92% versus Plain and beats average and wrong-cube beliefs; the companion curve shows independently restarted fits using progressively more shoves, with error bars and bands indicating seed standard deviation. The gain supports useful object-specific adaptation, but recovers only 37.27% of the Plain-to-Oracle gap and therefore fails the predeclared 50% target. [Comparison](/home/ubuntu/faliure/clean/shove_gate/push_latent/RESULTS.md:14), [specificity](/home/ubuntu/faliure/clean/shove_gate/push_latent/RESULTS.md:38), [curve procedure](/home/ubuntu/faliure/clean/shove_gate/push_latent/RESULTS.md:47), [criterion](/home/ubuntu/faliure/clean/shove_gate/push_latent/RESULTS.md:34).

**fig_mass**

Each panel compares true and predicted mass on unseen cubes for one training seed, using a frozen straight-line map from fitted belief to kilograms; mean R² is 0.876836 ± 0.0553 across seeds, where R² measures how well mass variation is explained. Mass labels were absent from belief learning and fitting but were used to train the readout, so this demonstrates mass information in the belief rather than label-free calibration of its coordinates. [Readout procedure and scores](/home/ubuntu/faliure/clean/shove_gate/push_latent/RESULTS.md:61).

**fig_drift**

For both learning rates, the left panel tracks cumulative new-cube push error after each cube's own adaptation, with markers for independently reset per-cube adaptation, while the right tracks held-out training-cube error averaged across push velocity, moving-coast velocity, and full state; Plain is the no-adaptation reference and bands show seed standard deviation. This single fixed visiting order shows deterioration relative to Plain, especially at the higher rate, but it is not a final-network rescore of all visited cubes and the predeclared first-cube forgetting test failed. [Sequence definitions and endpoints](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/results/gain_damage_partial.csv:2), [independent adaptation](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/RESULTS.md:9), [failed return check](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/RESULTS.md:89), [scope](/home/ubuntu/faliure/clean/shove_gate/adaptation_compare/RESULTS.md:224).
