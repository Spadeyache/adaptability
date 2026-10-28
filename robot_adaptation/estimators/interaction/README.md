# Interaction-based mass and friction estimator

This supervised baseline predicts object mass (kilograms) and object material
friction (static and dynamic coefficients equal) from completed contact bouts.
The 48-unit GRU encodes each bout. Completed bouts are averaged within an attempt,
and attempt embeddings are accumulated with equal weight across attempts.
The prediction head produces positive values by predicting standardized log
parameters. Training supervises every evidence prefix from one to five attempts.

## Prepare and train

Use the shared environment from the repository root:

```bash
source scripts/activate_grip.sh
python -m robot_adaptation.estimators.interaction.data \
  --source runs/failure_collection/dense_single_pair_fixed40g \
  --output runs/interaction_estimator/contact_dataset_v1
python -m robot_adaptation.estimators.interaction.train \
  --dataset runs/interaction_estimator/contact_dataset_v1 \
  --output runs/interaction_estimator/gru_v1 --epochs 300 --threads 2
python -m robot_adaptation.estimators.interaction.evaluate \
  --dataset runs/interaction_estimator/contact_dataset_v1 \
  --checkpoint runs/interaction_estimator/gru_v1/best.pt \
  --output runs/interaction_estimator/gru_v1/test_metrics.json
```

Outputs refuse to replace existing run directories. Source files, configuration,
dataset identity, training history and checkpoint hashes are saved with the run.
Checkpoint selection uses validation errors only. Condition-bootstrap confidence
intervals in the final evaluation resample physical conditions, not overlapping
snippets or repeated random orderings.

## Inputs and boundaries

`data.FEATURE_SPECS` defines the ordered inputs: encoder joint position/velocity,
finger opening, object pose relative to the hand, applied actions, motor limits,
measured finger forces and measured object/hand motion responses. The estimator
never receives supplied mass, actual mass/friction, condition IDs or simulator
policy observations. Poses and forces in the existing data are ideal simulated
sensor proxies; a real sensing stack is not implemented.

The default contact detector uses measured finger compression above 0.05 N.
A bout starts with the action producing the first detected contact and ends
after three consecutive absent-contact samples, 0.25 seconds of observed goal
holding, or the recording horizon. Success is an observable proxy using object
goal distance, joint speed and bilateral finger compression. Weak terminal
contact, object descent and duration provide additional completed-bout outcome
features. These outcomes are available only after their bout finishes.
Simulator contact alignment can be requested as an explicit offline diagnostic;
the trained default and deployment predictor use the same observable detector.

State arrays contain T+1 entries and commands/responses T entries. A snippet's
`start:end` slice contains action[start:end] and responses, with state[end] as its
final observation. No-contact attempts are recorded separately and leave the
predictor's physical estimate unchanged.

## Persistent inference

```python
from robot_adaptation.estimators.interaction.predict import InteractionPredictor

predictor = InteractionPredictor("runs/interaction_estimator/gru_v1/best.pt")
initial = predictor.predict()
estimate = predictor.update_episode(completed_episode_inputs)
# estimate has mass_kg, friction, attempts, informative_attempts and snippets.
predictor.reset()  # when a different object is introduced
```

The mapping needs only the input fields above plus
`input__object_goal_distance_m`, `input__object_pose_world_m_wxyz` and
`meta__state_time_s`. Label and simulator fields can be omitted. One completed
attempt is one evidence update; repeated calls with the same attempt would
double-count evidence and must be avoided by the caller.

## Scope of the first experiment

The clean historical subset contains 440 independently reset attempts at 44
mass/friction configurations, holding all other declared factors nominal.
Static and dynamic object friction are equal. The original design consists of
single-factor sweeps and two diagonals; it is not a full crossed grid.
Whole conditions are assigned to 26 training, nine validation and nine test
groups before snippets are extracted. All normalization uses training data only.

Pooling attempts with the same known condition during offline training and
evaluation is supervised evidence aggregation. It does not demonstrate adaptive
retry behavior, estimator transfer to a new policy, real-robot sensor robustness,
or causal attribution of a failure. Those require separate closed-loop tests.
