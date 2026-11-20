"""Actions-only generator diagnostic; does not read dynamics or train gate models.

Uses all 230 preplanned command schedules. This is intentionally separate from
the ten-cube physics pilot: it tests the randomized command generator, not
recoverability from measured motion. Neither identifiers nor episode length
enter a feature. Ground-truth labels are read only from data/cubes.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent
SEEDS = (101, 211, 307, 401, 503)
TARGETS = ('mass_kg', 'friction', 'combined_friction')
FEATURES = (
    'force_cap_n', 'cos_push_direction', 'sin_push_direction',
    'contact_offset_m', 'cube_start_x_m', 'cube_start_y_m',
    'cos_start_yaw', 'sin_start_yaw',
    'commanded_push_force_x_n', 'commanded_push_force_y_n',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features(plans):
    """Inputs are the independently randomized settings, never body responses."""
    output = []
    for p in plans:
        direction = float(p['direction_rad'])
        yaw = float(p['yaw_rad'])
        cap = float(p['cap_n'])
        cosine, sine = np.cos(direction), np.sin(direction)
        output.append([
            cap, cosine, sine, float(p['contact_offset_m']),
            *map(float, p['start_xy']), np.cos(yaw), np.sin(yaw),
            .95 * cap * cosine, .95 * cap * sine,
        ])
    result = np.asarray(output, dtype=np.float64)
    if result.shape != (len(plans), len(FEATURES)) or not np.isfinite(result).all():
        raise ValueError('Invalid action-only feature matrix')
    return result


def make_model(family, parameter, seed):
    if family == 'polynomial_ridge':
        return make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(), Ridge(alpha=float(parameter)),
        )
    if family == 'extra_trees':
        return ExtraTreesRegressor(
            n_estimators=192, min_samples_leaf=int(parameter),
            max_features=1.0, random_state=seed, n_jobs=2,
        )
    raise ValueError(family)


def cube_bootstrap(y, prediction, cube_ids, seed, repetitions=1000):
    """Resample held-out cube clusters, retaining their ten action episodes."""
    groups = [np.flatnonzero(cube_ids == cid) for cid in np.unique(cube_ids)]
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repetitions):
        index = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        scores.append(r2_score(y[index], prediction[index], multioutput='raw_values'))
    quantile = np.quantile(np.asarray(scores), [.025, .975], axis=0)
    return {target: quantile[:, j].tolist() for j, target in enumerate(TARGETS)}


def run_generator(root=ROOT):
    root = Path(root)
    cube_path, plan_path = root / 'data/cubes.json', root / 'data/plans.json'
    source_hashes = {str(p.relative_to(root)): digest(p) for p in (cube_path, plan_path)}
    started = time.perf_counter()
    cubes = {int(c['cube_id']): c for c in json.loads(cube_path.read_text())['cubes']}
    plans = json.loads(plan_path.read_text())
    if not isinstance(plans, list):
        raise ValueError('Expected plans.json to contain a list')
    identities = [(int(p['cube_id']), int(p['episode_id'])) for p in plans]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate cube/episode in command plans')
    for cid in cubes:
        if sorted(ep for cube, ep in identities if cube == cid) != list(range(10)):
            raise ValueError(f'Expected exactly episodes0–9 for cube{cid}')

    x = features(plans)
    cube_ids = np.asarray([int(p['cube_id']) for p in plans])
    labels = np.asarray([[float(cubes[cid][target]) for target in TARGETS] for cid in cube_ids])
    train_cubes = np.asarray(sorted(cid for cid, cube in cubes.items() if cube['split'] == 'train'))
    test_cubes = np.asarray(sorted(cid for cid, cube in cubes.items() if cube['split'] == 'test'))
    if len(train_cubes) != 200 or len(test_cubes) != 30:
        raise ValueError('Expected200train and30test cube definitions')
    train = np.isin(cube_ids, train_cubes)
    test = np.isin(cube_ids, test_cubes)
    if not np.all(train | test) or np.any(train & test):
        raise ValueError('Non-disjoint cube split')

    # The same test predictions cannot influence hyperparameter selection.
    parameter_grids = {
        'polynomial_ridge': (1., 10., 100., 1000., 10000.),
        'extra_trees': (8, 32, 128),
    }
    records = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(train_cubes)
        fit = np.isin(cube_ids, shuffled[:180])
        validation = np.isin(cube_ids, shuffled[180:])
        center = labels[fit].mean(axis=0)
        scale = labels[fit].std(axis=0)
        if np.any(scale <= 0):
            raise ValueError('Degenerate training labels')
        # Fit the two independent targets only. Cube friction is recovered
        # from the train-only affine material relationship for reporting, so
        # ExtraTrees does not inadvertently weight friction twice.
        fit_target = ((labels[fit] - center) / scale)[:, [0, 2]]
        val_target = ((labels[validation] - center) / scale)[:, [0, 2]]
        for family, grid in parameter_grids.items():
            candidates = []
            for parameter in grid:
                model = make_model(family, parameter, seed)
                model.fit(x[fit], fit_target)
                prediction = model.predict(x[validation])
                # Friction and combined friction are redundant targets; avoid
                # counting friction twice in model-selection weighting.
                error = float(np.mean((prediction - val_target)**2))
                candidates.append((error, parameter))
            selected_error, selected = min(candidates, key=lambda item: item[0])
            final_center = labels[train].mean(axis=0)
            final_scale = labels[train].std(axis=0)
            model = make_model(family, selected, seed)
            model.fit(x[train], ((labels[train] - final_center) / final_scale)[:, [0, 2]])
            predicted_independent = model.predict(x[test]) * final_scale[[0, 2]] + final_center[[0, 2]]
            transform_design = np.column_stack((labels[train, 2], np.ones(train.sum())))
            transform = np.linalg.lstsq(transform_design, labels[train, 1], rcond=None)[0]
            if not np.allclose(transform_design @ transform, labels[train, 1], rtol=0, atol=1e-10):
                raise ValueError('Cube and combined friction are not affinely related; revise probe targets before fitting')
            prediction = np.empty((test.sum(), len(TARGETS)))
            prediction[:, [0, 2]] = predicted_independent
            prediction[:, 1] = prediction[:, 2] * transform[0] + transform[1]
            scores = r2_score(labels[test], prediction, multioutput='raw_values')
            record = {
                'family': family, 'seed': seed,
                'validation_fit_cubes': 180, 'validation_cubes': 20,
                'selected_parameter': selected,
                'validation_standardized_mse': selected_error,
                'validation_candidates': [{'parameter': p, 'mse': error} for error, p in candidates],
                'test_r2': dict(zip(TARGETS, map(float, scores))),
                'test_cube_bootstrap_95_percent_interval': cube_bootstrap(
                    labels[test], prediction, cube_ids[test], seed + 9000),
                'all_test_r2_below_0_1': bool(np.all(scores < .1)),
            }
            records.append(record)
            print(json.dumps({'family': family, 'seed': seed, 'test_r2': record['test_r2']}), flush=True)

    mean_prediction = np.broadcast_to(labels[train].mean(axis=0), labels[test].shape)
    baseline = r2_score(labels[test], mean_prediction, multioutput='raw_values')
    aggregate = {}
    for family in parameter_grids:
        values = np.asarray([[record['test_r2'][target] for target in TARGETS]
                             for record in records if record['family'] == family])
        aggregate[family] = {
            target: {'mean_r2': float(values[:, j].mean()),
                     'seed_std_r2': float(values[:, j].std(ddof=1)),
                     'min_r2': float(values[:, j].min()), 'max_r2': float(values[:, j].max()),
                     'mean_r2_minus_constant_baseline': float(values[:, j].mean() - baseline[j])}
            for j, target in enumerate(TARGETS)
        }
    result = {
        'scope': 'Generator-only action probe using all preplanned commands; no additional physical cubes collected and no dynamics inputs.',
        'input_sha256': source_hashes, 'source_sha256': digest(__file__),
        'feature_names': FEATURES, 'targets': TARGETS,
        'no_numeric_identifier_features': True,
        'no_episode_duration_or_termination_features': True,
        'no_measured_contact_force_features': True,
        'train_cubes': 200, 'test_cubes': 30,
        'train_episodes': int(train.sum()), 'test_episodes': int(test.sum()),
        'selection': 'Per seed,180 training cubes for fit and20 training cubes for validation; choose hyperparameter by validation standardized MSE; refit all200 training cubes. No test-label tuning.',
        'confidence_interval': 'Percentile95% over1000 held-out-cube cluster bootstrap resamples; ten episodes retained together. Diagnostic, not proof of independence.',
        'constant_train_mean_test_r2': dict(zip(TARGETS, map(float, baseline))),
        'seeds': SEEDS, 'models': records, 'aggregate': aggregate,
        'all_fitted_probe_r2_below_0_1': all(record['all_test_r2_below_0_1'] for record in records),
        'interpretation': 'R2<=0 means no advantage over the held-out target mean. Negative scores are reported literally, not described as mathematically equal to zero. The empirical criterion flags positive predictability at R2>=0.1; it does not prove generator independence.',
        'elapsed_seconds': time.perf_counter() - started,
    }
    return result


def run_pilot(root=ROOT):
    """Actual saved-command audit, with separate outcome-length diagnostics."""
    root = Path(root)
    cubes = {int(c['cube_id']): c for c in json.loads((root/'data/cubes.json').read_text())['cubes']}
    paths = sorted((root/'data/episodes').glob('*.npz'))
    if len(paths) != 100:
        raise ValueError('Pilot probe requires exactly 100 newly collected episodes')
    x_rows, lengths, ids, targets, identities = [], [], [], [], []
    hashes = {}
    for path in paths:
        hashes[path.name] = digest(path)
        with np.load(path, allow_pickle=False) as ep:
            cid, episode = int(ep['cube_id'][0]), int(ep['episode_id'][0])
            identities.append((cid, episode)); ids.append(cid)
            if cid not in range(10):
                raise ValueError('Probe read a cube beyond the authorized ten')
            if not np.all(ep['cube_id'] == cid) or not np.all(ep['episode_id'] == episode):
                raise ValueError('Identity changes inside episode')
            push = ep['action'][ep['phase'] == 'push']
            if push.shape != (16, 8) or not np.array_equal(push, np.broadcast_to(push[0], push.shape)):
                raise ValueError('Expected 16 identical commanded push rows per episode')
            # This is a sufficient representation of the whole fixed prefix:
            # all sixteen rows are verified identical, not averaged responses.
            x_rows.append(push[0].astype(np.float64))
            lengths.append(len(ep['step']))
            targets.append([float(cubes[cid][key]) for key in TARGETS])
    if set(identities) != {(i,e) for i in range(10) for e in range(10)}:
        raise ValueError('Expected ten distinct shoves for each of the same ten cubes')
    x = np.asarray(x_rows); length = np.asarray(lengths, dtype=np.float64)[:,None]
    ids, y = np.asarray(ids), np.asarray(targets)
    feature_sets = {
        'commands_only': x,
        'retained_length_only': length,
        'commands_plus_retained_length': np.column_stack((x,length)),
    }
    baseline = np.empty_like(y)
    for cid in np.unique(ids):
        train, heldout = ids != cid, ids == cid
        baseline[heldout] = y[train].mean(axis=0)
    baseline_r2 = r2_score(y, baseline, multioutput='raw_values')
    baseline_sse = np.square(y-baseline).sum(axis=0)
    records=[]
    # These hyperparameters are fixed before reading probe scores. No choice
    # uses held-out-cube labels; all ten folds are evaluated exactly once.
    parameters = {'polynomial_ridge': 100., 'extra_trees': 5}
    for name, matrix in feature_sets.items():
        for family, parameter in parameters.items():
            for seed in SEEDS:
                prediction = np.empty_like(y)
                for cid in np.unique(ids):
                    train, heldout = ids != cid, ids == cid
                    center = y[train].mean(axis=0)
                    scale = y[train].std(axis=0)
                    model = make_model(family, parameter, seed)
                    model.fit(matrix[train], ((y[train]-center)/scale)[:,[0,2]])
                    independent = model.predict(matrix[heldout])*scale[[0,2]]+center[[0,2]]
                    prediction[heldout,0] = independent[:,0]
                    prediction[heldout,2] = independent[:,1]
                    affine = np.linalg.lstsq(np.column_stack((y[train,2],np.ones(train.sum()))),
                                             y[train,1],rcond=None)[0]
                    prediction[heldout,1] = independent[:,1]*affine[0]+affine[1]
                raw_r2 = r2_score(y,prediction,multioutput='raw_values')
                skill = 1.-np.square(y-prediction).sum(axis=0)/baseline_sse
                row=dict(feature_set=name,family=family,seed=seed,
                         pooled_out_of_fold_r2=dict(zip(TARGETS,map(float,raw_r2))),
                         skill_vs_fold_training_mean=dict(zip(TARGETS,map(float,skill))))
                records.append(row)
                print(json.dumps({'pilot_feature_set':name,'family':family,'seed':seed,
                                  'r2':row['pooled_out_of_fold_r2']}),flush=True)
    aggregate={}
    for name in feature_sets:
        aggregate[name]={}
        for family in parameters:
            subset=[row for row in records if row['feature_set']==name and row['family']==family]
            aggregate[name][family]={}
            for target in TARGETS:
                raw=np.array([row['pooled_out_of_fold_r2'][target] for row in subset])
                skill=np.array([row['skill_vs_fold_training_mean'][target] for row in subset])
                aggregate[name][family][target]=dict(mean_r2=float(raw.mean()),seed_std_r2=float(raw.std(ddof=1)),
                    min_r2=float(raw.min()),max_r2=float(raw.max()),
                    mean_skill_vs_fold_training_mean=float(skill.mean()))
    return dict(
        scope='Actual newly collected 100-episode pilot, ten held-out-cube folds; no additional simulation.',
        episode_sha256=hashes,
        folds=10,cubes=10,episodes=100,seeds=SEEDS,
        fitting='Leave one whole cube out; fit on nine cubes and predict all ten shoves of the held-out cube. Pool all100 out-of-fold predictions before computing R2. No per-constant-cube R2.',
        fixed_hyperparameters=parameters,
        feature_sets={
            'commands_only':'Eight actual action channels from the verified identical16-step push prefix. No numeric IDs, motion, stop time, length, mask or outcome.',
            'retained_length_only':'One deliberately exposed outcome feature: number of retained transitions after stop-based trimming. Leakage diagnostic only, not permitted model input.',
            'commands_plus_retained_length':'Eight commanded-action channels plus retained length, a deliberate length-leakage diagnostic.',
        },
        constant_fold_training_mean_r2=dict(zip(TARGETS,map(float,baseline_r2))),
        baseline_note='With ten equal-size cube groups, the leave-one-cube-out mean baseline has pooled R2=1-(10/9)^2=-0.234568. Baseline-relative skill is separately named and is not raw R2.',
        models=records,aggregate=aggregate,
        limitation='Only ten independent cube identities. Seed SD describes fitting randomness, not dataset uncertainty. Negative R2 is reported literally and is not relabeled near zero.',
        length_policy='Stop-based trimming makes length an observed outcome. Do not feed length, termination mask, final step index, padding pattern or future episode endpoint into Plain or Oracle. Use cube/episode-balanced losses so longer episodes do not silently get more weight.',
    )


def run(root=ROOT):
    root=Path(root).resolve()
    path=root/'results/action_probe.json'
    input_hashes={str(p.relative_to(root)):digest(p) for p in [root/'data/cubes.json',root/'data/plans.json']}
    if path.exists():
        old=json.loads(path.read_text())
        if old.get('input_sha256') != input_hashes or old.get('source_sha256') != digest(__file__):
            raise RuntimeError('Existing action probe differs; preserve/archive it before recomputing')
        for name,sha in old['pilot']['episode_sha256'].items():
            if digest(root/'data/episodes'/name) != sha:
                raise RuntimeError('Pilot episode changed after action-probe evaluation')
        return old
    started=time.perf_counter()
    generator=run_generator(root)
    pilot=run_pilot(root)
    result=dict(scope='Two distinct audits: 200/30 preplanned-command generator check and actual ten-cube pilot leave-one-cube-out command/length checks.',
                input_sha256=input_hashes,source_sha256=digest(__file__),
                generator=generator,pilot=pilot,elapsed_seconds=time.perf_counter()-started)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    run(args.root.resolve())
