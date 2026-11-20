"""Freeze splits, estimands, provenance, and adaptation orders before training."""
import json
from datetime import datetime, timezone

import numpy as np

from .data import DATA, RESULTS, ROOT, TASKS, digest, episode_bounds, load_rows, write_json


def freeze():
    output = RESULTS / 'protocol.json'
    if output.exists():
        saved = json.loads(output.read_text())
        for name, expected in saved['input_sha256'].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f'Frozen input changed: {name}')
        if digest(RESULTS / 'split.json') != saved['split_sha256']:
            raise ValueError('Frozen split changed')
        print(f'Existing frozen protocol verified: {output}', flush=True)
        return saved
    rows = {split: load_rows(split) for split in ('train', 'test')}
    rng = np.random.default_rng(721)
    held = []
    for cube in range(200):
        for task in range(3):
            held.extend([[cube, task, int(ep)] for ep in sorted(rng.choice(10, 2, replace=False))])
    entries, counts = [], {}
    for split, data in rows.items():
        start, _ = episode_bounds(data)
        keys = np.column_stack([data[key][np.unique(start)] for key in ('cube_id', 'task', 'episode_id')])
        expected = np.array([(cube, task, ep) for cube in (range(200) if split == 'train' else range(200, 230))
                             for task in range(3) for ep in range(10)])
        np.testing.assert_array_equal(keys, expected)
        if data['state'].shape[1] != 51 or data['action'].shape[1] != 8:
            raise ValueError('Unexpected state/action dimensions')
        counts[split] = dict(rows=len(start), episodes=len(keys), cubes=len(np.unique(data['cube_id'])))
        if split == 'test':
            for cube, task, episode in keys:
                entries.append(dict(cube_id=int(cube), task=TASKS[task], task_code=int(task),
                                    episode_number=int(episode + 1), episode_id=int(episode),
                                    role='context' if episode < 5 else 'eval'))
    write_json(RESULTS / 'split.json', dict(schema_version=1, episode_numbering='Human 1..10; stored episode_id 0..9',
               test_episodes=entries, train_validation_episodes=held, training_split_seed=721))
    input_files = [DATA / 'dataset.json', *sorted(DATA.glob('*/*.npz')),
                   ROOT / 'experiments/lift_slide_belief/model.py',
                   ROOT / 'experiments/lift_slide_belief/slip_data.py',
                   ROOT / 'experiments/lift_slide_belief/slip_study.py']
    protocol = dict(schema_version=1, frozen_utc=datetime.now(timezone.utc).isoformat(),
        data_directory=str(DATA.relative_to(ROOT)), data_counts=counts,
        input_sha256={str(path.relative_to(ROOT)): digest(path) for path in input_files},
        split_file=str((RESULTS / 'split.json').relative_to(ROOT)), split_sha256=digest(RESULTS / 'split.json'),
        headline_task='lift', tasks=list(TASKS), seeds=list(range(5)),
        state=dict(dimensions=51, raw_state=[0,43], relative_position=[43,46],
                   relative_velocity=[46,49], finger_position_error=[49,51],
                   note='Same observed state projection as the latest fixed-grip z model; no task/cube/episode identifiers as inputs.'),
        metrics=dict(mask='Stored cube_moving: observed pre-action linear speed > 0.01 m/s.',
            one_step_mse='Mean squared normalized delta error, pooled over all moving rows and 51 coordinates.',
            r2='1 - total SSE / total SST for the same normalized deltas; center each target coordinate over selected rows. Variance-weighted multioutput R2. Zero SST -> null.',
            rollout_10_mse='Terminal error mean(((predicted state t+10 - observed state t+10) / train delta_sd)^2), all 51 coordinates. Moving starts with 10 remaining same-episode transitions only. Recorded actions, predicted future states, no clipping or observed future history.',
            aggregation='Within each seed pool eligible rows by task and overall; across seeds report arithmetic mean and sample SD (ddof=1). Overall pools rows, not task means.',
            unavailable='An empty moving subset is null with count 0. Never substitute nonmoving samples.',
            nonfinite='Stop on nonfinite predictions; never discard divergent rows.'),
        normalization=dict(fit='Training episodes only; shared split and same statistics across models/seeds.',
                           sd='Population SD, minimum 1e-4, float64 accumulation.',
                           oracle='Mass/friction mean and SD fitted on training rows only; only B2 loads these labels.'),
        training=dict(mlp_hidden=[256,256,256], activation='ReLU', optimizer='Adam', lr=1e-3,
                      batch_size=1024, max_epochs=1000, patience=12, min_relative_improvement=0.001,
                      early_stopping='All rows in 2 held-out episodes per train cube/task, same split as z model.',
                      loss='Normalized delta MSE, all training rows; no z, no latent table, no weight decay.'),
        history=dict(k=[1,5,10,20], gru_hidden=64, transition='Normalized (state, action, next_state), dimension 110.',
                     sequence='Last K completed same-episode transitions, oldest first, right pad and mask; no current next_state.',
                     reset='Empty at every episode start, including eval episodes; never carries context episodes.',
                     rollout='Append predicted transitions and retain last K after each prediction.',
                     early_late='Literal recorded step 0..4 versus step >=5; moving subset only.'),
        adaptation=dict(base_seeds=list(range(5)), order_seeds=list(range(5)),
                        orders=[np.random.default_rng(seed).permutation(np.arange(200,230)).tolist() for seed in range(5)],
                        design='Cross all 5 B0 seeds with all 5 orders, independently for each learning rate (50 trajectories).',
                        learning_rates=[1e-4,1e-5], steps_per_cube=100, batch_size=1024,
                        optimizer='Fresh Adam at each cube; carry every learned weight between cubes. Fixed normalization buffers.',
                        fit_rows='All rows of the current test cube context lifts, episodes 0..4. Sample batches with replacement.',
                        retained_k=[0,1,5,20], scoring='Current cube eval lifts after its adaptation; first cube at each k; full train validation before and after final cube.'),
        run_gate='Run and report measured B0 seed 0 runtime and metrics before any other model or adaptation run; stop and report failed or suspicious runs.')
    write_json(output, protocol)
    print(json.dumps(dict(protocol=str(output), split=str(RESULTS / 'split.json'), counts=counts)), flush=True)
    return protocol


if __name__ == '__main__':
    freeze()
