# Physical parameter sweeps

[Experiment index](../experiments.md)

**Status:** Collection complete; interpretation exploratory.

## Question

How do success and failure change as physical conditions vary under a fixed policy
and supplied mass estimate?

## Method

Collect 29,830 attempts across 2,983 settings and ten starting seeds, with a fixed
40 g estimate supplied to the frozen collection policy. Vary single factors and
pairs in the same or opposite directions. The design includes mass, friction,
size and other declared factors. Numeric pair sweeps follow diagonals, not a
complete factorial grid.

Plot actual parameter values per recorded attempt, with green successes and red
failures. The dense plot plan defines 307 comparisons.

## Result

The recordings and plots show where tested conditions succeed or fail. Some
intermediate settings mix success and failure across starts. For example, the
nominal 100 g / 0.4 object-friction setting succeeded on four of ten starts.

These plots preserve individual outcomes rather than assigning a causal
failure label from the parameter values.

## Limitation

Observed success frequencies are conditional on the tested policy, start
distribution and physics. Seed dependence does not by itself establish intrinsic
randomness for identical complete states. More samples could improve probability
estimates; a probability map alone would not explain the cause of each failure.

Actual mass and friction are known to the simulator, but are not automatically
available at real deployment. They would need measurement or estimation. Changing
physical parameters is also different from changing the estimate supplied to
the policy.

## Next step

Use the sweeps and manual replay review to identify candidate explanations.
Test which robot-observable signals or controlled interactions distinguish them;
connect that evidence to the proposed [estimate updates](interaction_estimate_updates.md).

## Code and evidence

- [Collection guide](../../robot_adaptation/collection/README.md)
- [Collection settings](../../configs/failure_collection_varied.json)
- [Dataset and collection plan](../../runs/failure_collection/dense_single_pair_fixed40g/)
- [Dense plot report](../../data_check/plots/dense_single_pair_all/index.html)
- Examples: [mass only](../../data_check/plots/dense_single_pair_all/mass_only.png), [mass/friction together](../../data_check/plots/dense_single_pair_all/mass_object_friction_together.png), [mass/friction opposite](../../data_check/plots/dense_single_pair_all/mass_object_friction_opposite.png)
- [Plot guide](../../robot_adaptation/analysis/plots/README.md) and [editable plot plan](../../robot_adaptation/analysis/plots/plots_dense.md)
- [Earlier 600-attempt dataset](../../runs/failure_collection/varied_fixed40g/) and [historical plot gallery](../../data_check/plots/index.html)
