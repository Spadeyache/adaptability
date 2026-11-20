"""Read-only historical dataset inventory. Writes only inventory_historical.json."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name('inventory_historical.json')
FAMILIES = ['context_dynamics', 'interaction_estimator', 'mass_belief', 'visual_mass', 'rma']


def read(path):
    return json.loads(Path(path).read_text())


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def schema(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: dict(shape=list(archive[key].shape), dtype=str(archive[key].dtype))
                for key in archive.files}


def correlation(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.any(np.ptp(values, axis=0) == 0):
        return None
    return float(np.corrcoef(values.T)[0, 1])


def physics_summary(constants):
    c = np.asarray(constants, dtype=float)
    unique, counts = np.unique(c, axis=0, return_counts=True)
    pairs = np.unique(c[:, :2], axis=0)
    return dict(unique_full_physics_configurations=len(unique), unique_mass_friction_pairs=len(pairs),
                episodes_per_full_configuration=dict(min=int(counts.min()), max=int(counts.max()),
                                                    histogram={str(k): int(v) for k, v in Counter(counts).items()}),
                mass_kg_range=[float(c[:, 0].min()), float(c[:, 0].max())],
                friction_range=[float(c[:, 1].min()), float(c[:, 1].max())],
                mass_friction_pearson_unique_full_configurations=correlation(unique[:, :2]),
                mass_friction_pearson_unique_mass_friction_pairs=correlation(pairs),
                mass_friction_pearson_episode_weighted=correlation(c[:, :2]),
                null_correlation_reason='Fewer than two configurations or a constant variable.'
                if correlation(pairs) is None else None)


def disjoint(values, held):
    values = np.asarray(values)
    train = {tuple(x) for x in values[~held]}
    validation = {tuple(x) for x in values[held]}
    return len(train & validation)


def context_group(folder):
    meta = read(folder / 'dataset.json')
    paths = sorted(folder.glob('shard_*.npz'))
    arrays = defaultdict(list)
    total_steps, state_samples = 0, 0
    for path in paths:
        with np.load(path, allow_pickle=False) as a:
            for key in ('episode_id', 'seed', 'constants', 'policy_guess_kg', 'task_id', 'source_id', 'is_validation', 'pair_id'):
                if key in a:
                    arrays[key].append(a[key])
            total_steps += int(np.prod(a['command'].shape[:2]))
            state_samples += int(np.prod(a['state'].shape[:2]))
            assert a['state'].shape[1] == a['command'].shape[1] + 1
            assert a['state'].shape[2] == 39 and a['command'].shape[2] == 5
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    n = len(arrays['episode_id'])
    assert len(np.unique(arrays['episode_id'])) == n
    held = arrays.get('is_validation', (arrays['episode_id'] // 4) % 10 == 9)
    cfg = meta['config']
    result = dict(family='context_dynamics', path=relative(folder), npz_files=len(paths),
        task_names=meta['tasks'], task_id_counts={str(k): int(v) for k, v in Counter(arrays['task_id']).items()},
        episodes=n, action_steps=total_steps, observed_state_samples=state_samples,
        steps_per_episode=sorted(set([total_steps // n])),
        cube_ids_present=False, identity='episode_id identifies a reset, not a persistent cube. Physics configurations below are not claimed to be cube IDs.',
        **physics_summary(arrays['constants']),
        fields_example_file=relative(paths[0]), all_saved_fields=schema(paths[0]), state_columns=meta['state'],
        action_columns=meta['command'], constant_columns=meta['constants'],
        labels=dict(storage='constants[episode,0]=mass_kg; constants[episode,1]=cube friction; other columns describe nuisance physics. GPU also saves requested_constants.',
                    constancy='Stored once per episode. Collector sets physics before rollout and does not change mass or friction during it; no repeated per-step mass/friction readback is saved.'),
        split=dict(rule=meta['split'], train_episodes=int((~held).sum()), validation_episodes=int(held.sum()),
                   shared_full_physics_configurations=disjoint(arrays['constants'], held),
                   shared_mass_friction_pairs=disjoint(arrays['constants'][:, :2], held),
                   cube_disjoint='No persistent cube IDs. Whole-episode split; exact sampled-configuration overlap checked above.'),
        collection_config=cfg, manifest_sha256=digest(folder / 'dataset.json'),
        completion=read(folder / 'complete.json') if (folder / 'complete.json').exists() else None,
        direct_mass_friction_in_saved_state_action=False,
        generator='GPU policy+noise/scripted push/random poke in PickCube' if 'source_id' in arrays else 'CPU Pick policy+noise, Push/Pull scripts, poke/grip scripts, occasional random command segments',
        grip='Opening/strength commands; effective motor force is scaled by independently randomized grip_scale. Finger material friction is fixed in config.',
        physics_conditioned_commands='Policy mass guess sampled independently from true mass; motion RNG separated from physics. Applied delay and effective grip caps do depend on randomized delay/grip-scale nuisance constants. Feedback actions respond to observed physical motion.',
        source_evidence=['experiments/context_dynamics/collect.py:100-154', 'experiments/context_dynamics/collect_gpu.py:22-40,74-116',
                         'robot_adaptation/simulation/context_dynamics.py:11-14,64-99,118-124'],
        limitations=['Raw state omits joint effort and force sensing.',
                     'Historical z encoders explicitly consume constants; their z is privileged, not an inferred belief.'],
    )
    if 'source_id' in arrays:
        result['source_counts'] = {meta['sources'][k]: int(v) for k, v in Counter(arrays['source_id']).items()}
    if 'pair_id' in arrays:
        result['paired_replay_groups'] = len(np.unique(arrays['pair_id']))
    result['true_mass_vs_independent_policy_guess_pearson'] = correlation(np.column_stack((arrays['constants'][:, 0], arrays['policy_guess_kg'])))
    if '5m_excluded' in str(folder):
        result['excluded_warning'] = 'Incomplete excluded 5m scene-spacing dataset; do not pool with corrected 64m collection. Count reflects actual saved shards, not requested episodes.'
    return result


def mass_group(folder):
    paths = sorted(folder.glob('*.npz'))
    base = ROOT / 'runs/mass_belief/nov17_v1'
    cfg = read(base / 'config.json')
    switch = 'switch_80g_mu1' in str(folder)
    switch_cfg = read(folder.parent / 'config.json') if switch else None
    constants, steps, initial_weights, supplied, modes = [], [], [], [], Counter()
    for path in paths:
        with np.load(path, allow_pickle=False) as a:
            mu = float(a['static_friction']) if 'static_friction' in a else cfg['friction']
            constants.append([float(a['mass_g']) / 1000, mu])
            steps.append(len(a['x']))
            initial_weights.append(float(a['initial'][42]))
            supplied.extend(a['supplied_g'].tolist())
            modes['oracle_true_mass' if path.stem.endswith('_true') else 'belief' if path.stem.endswith('_belief') else 'probe'] += 1
            assert a['x'].shape[1] == 25
    return dict(family='mass_belief', path=relative(folder), npz_files=len(paths), episodes=len(paths),
        task='Pick, lift, hold', action_steps=sum(steps), steps_per_episode=sorted(set(steps)),
        cube_ids_present=False, identity='No cube ID. train/validation/calibration have independently sampled scalar masses; fixed test/closed-loop/switch groups repeat one physical configuration with fresh reset seeds.',
        **physics_summary(constants), all_saved_fields=schema(paths[0]), fields_example_file=relative(paths[0]),
        feature_columns={'x[0:9]': 'post-action qpos including fingers', 'x[9:18]': 'post-action qvel',
                         'x[18:20]': 'post-action measured finger compression forces', 'x[20:25]': 'preceding applied five-dimensional command'},
        labels=dict(storage='mass_g scalar in every NPZ; friction is scalar in switch NPZ, otherwise constant config.json friction only.',
                    constancy='Physics set once per episode. Each file stores one mass; no per-step true physics series.'),
        existing_split=folder.name, cube_disjoint_split='Explicit episode directories; no cube-ID split. Test and closed-loop use the same 400g condition; switch uses 80g/friction1.',
        generator='Frozen v1 policy, five command channels including opening and strength',
        grip='Fixed trained force range; policy selects strength from observations and supplied mass estimate.',
        controller_modes=dict(modes), supplied_mass_g_range=[min(supplied), max(supplied)],
        initial_observation_weight_n_range=[min(initial_weights), max(initial_weights)],
        physics_conditioned_commands='closed_loop *_true passes actual mass to policy at every step; *_belief passes inferred mass; other recordings use fixed40g or independent random training guesses. x excludes the supplied-weight observation, but oracle commands can carry privileged-controller information.',
        direct_mass_friction_in_saved_x=False,
        initial_observation_warning='initial[42] is the configured40g reported weight at reset, not actual object mass. Do not mistake the45-vector for the25 estimator inputs.',
        configuration=switch_cfg or cfg,
        source_evidence=['experiments/mass_belief/collect.py:16-18,29-54',
                         'experiments/mass_belief/evaluate.py:74-97', 'experiments/mass_belief/switch.py:44-57'])


def rma_group(folder):
    records = list(csv.DictReader((folder / 'episodes.csv').open()))
    constants = np.array([[float(x['mass_kg']), float(x['friction'])] for x in records])
    paths = [folder / x['trace'] for x in records]
    assert set(paths) == set(folder.glob('episode_*.npz'))
    steps, constant_z = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as a:
            steps.append(len(a['action']))
            assert a['state'].shape == (len(a['action']), 44)
            constant_z.append(bool(np.all(a['z'] == a['z'][:1])))
    return dict(family='rma', path=relative(folder), npz_files=len(paths), episodes=len(paths),
        task='Randomized lift task; in-range validation or integration smoke',
        action_steps=sum(steps), observed_state_samples=sum(steps), reconstructable_next_state_transitions=sum(x - 1 for x in steps),
        steps_per_episode=sorted(set(steps)), cube_ids_present=False,
        identity='Physical grid cell (mass,friction) plus reset seed and oracle/adapted controller mode; no persistent cube ID.',
        **physics_summary(constants), generator='RMA policy with true-physics encoder in oracle mode, history adapter in adapted mode',
        grip='Policy opening/strength commands; configured finger friction and force range',
        controller_modes=dict(Counter(x['mode'] for x in records)),
        all_saved_fields=schema(paths[0]), fields_example_file=relative(paths[0]),
        labels=dict(storage='episodes.csv mass_kg and friction joined by trace filename; NPZ z is the eight-dimensional true-physics encoding and z_hat is the controller latent.',
                    constancy='One grid mass/friction per episode; z is exactly constant across all stored steps in every file.',
                    constant_z_verified_all_files=all(constant_z)),
        state_columns='state44 selects original46 observation coordinates except42(weight) and45(friction). Last two retained force coordinates divided by40. qpos/qvel/pose/goal/grasp/contact-force observations remain.',
        direct_mass_friction_in_saved_state_action=False,
        privileged_sidefields=['z (derived from true mass/friction)', 'z_hat equals privileged z in oracle mode'],
        physics_conditioned_commands='Oracle mode action=policy(state,encode(true_mass,true_friction)); adapted mode uses observed history. These must not be pooled as label-free acquisition.',
        existing_split='Development validation grid or integration smoke. These files do not define cube-disjoint train/test datasets.',
        manifest=read(folder / 'manifest.json'),
        source_evidence=['robot_adaptation/policy/rma/model.py:10-19,40-51', 'experiments/rma/evaluate.py:64-98'])


def interaction_group():
    folder = ROOT / 'runs/interaction_estimator/contact_dataset_v1'
    meta, episodes, snippets = [read(folder / x) for x in ('manifest.json', 'episodes.json', 'snippets.json')]
    with np.load(folder / 'snippets.npz', allow_pickle=False) as a:
        x, offsets = a['x'], a['offsets']
    conditions = {}
    for ep in episodes:
        if ep['condition_id'] in conditions:
            assert conditions[ep['condition_id']] == ep['target']
        conditions[ep['condition_id']] = ep['target']
    # Raw acquisition roots have a separate audit; do not duplicate that scan.
    from robot_adaptation.estimators.interaction.data import FEATURE_NAMES
    assert list(FEATURE_NAMES) == meta['feature_names']
    target = np.array([ep['target'] for ep in episodes])
    return dict(family='interaction_estimator', path=relative(folder), task='Contact snippets from independent fixed-policy lift attempts',
        npz_files=1, source_episodes=len(episodes), snippets=len(snippets), recorded_feature_rows=len(x),
        source_steps_per_episode=100, source_action_steps=100 * len(episodes),
        snippet_lengths=dict(min=int(np.diff(offsets).min()), median=float(np.median(np.diff(offsets))), max=int(np.diff(offsets).max())),
        cube_ids_present=False, identity='condition_id identifies repeated physical configuration, not persistent object identity.',
        **physics_summary(target), generator='Fixed40g-report frozen policy, independently reset attempts; contact bouts aligned by measured compression.',
        grip='Policy strength/opening; measured actuator cap and force included in x',
        all_saved_fields=schema(folder / 'snippets.npz'), feature_columns=meta['feature_names'],
        labels=dict(storage='target=[mass_kg,object_friction] in episodes.json/snippets.json, constant within condition.',
                    constancy_verified=True),
        split=meta['splits'], split_rule='Condition-disjoint train/validation/test, seed73; all snippets from a condition share split.',
        source_root=meta['source'], direct_label_injection_found=False,
        verification='Stored51-column manifest exactly matches current FEATURE_NAMES selecting measured input__ signals. No label__ column selected by the feature builder. Raw source files are covered by the separate acquisition audit.',
        nuisance_physics_audit=read(folder / 'physical_readback_audit.json'),
        scope=meta['scope'], physics_conditioned_commands='Supplied mass fixed40g independently of actual physics; feedback naturally depends on observed dynamics.',
        source_evidence=['robot_adaptation/estimators/interaction/data.py:18-33,57-71,140-182'])


def visual_group():
    folder = ROOT / 'runs/visual_mass/initial_rgb_mass/dataset'
    meta = read(folder / 'dataset.json')
    objects, image_shapes, missing = {}, Counter(), []
    for record in meta['records']:
        objects.setdefault(record['object_id'], []).append(record)
        path = folder / record['image']
        if not path.exists():
            missing.append(str(path))
        else:
            with Image.open(path) as image:
                image_shapes[str((image.size, image.mode))] += 1
    for views in objects.values():
        assert len({x['split'] for x in views}) == len({x['mass_kg'] for x in views}) == 1
    constants = [[v[0]['mass_kg'], meta['config']['static_friction']] for v in objects.values()]
    return dict(family='visual_mass', path=relative(folder), task='Initial RGB before any interaction',
        objects=len(objects), images=len(meta['records']), missing_images=missing, views_per_object=dict(Counter(len(v) for v in objects.values())),
        image_shapes=dict(image_shapes), action_steps=0, per_step_fields=None,
        fields='PNG RGB uint8 pixels only as model input. dataset.json records image path, object_id, split, mass_kg, size_m, seed, initial_pose_world.',
        labels=dict(storage='Per-image manifest mass_kg; fixed friction in configuration', constancy='Mass and split verified identical across each object\'s views.'),
        **physics_summary(constants), cube_ids_present=True,
        split_objects=dict(Counter(v[0]['split'] for v in objects.values())), split_images=dict(Counter(v['split'] for v in meta['records'])),
        generator='Rendering only; no policy or grip action', direct_label_injection_found=False,
        physical_confound='Mass is defined as density1000kg/m3 × side_length^3. Visible size predicts mass by construction; this is not independently varied hidden mass. No text label is rendered by generator.',
        config=meta['config'], manifest_sha256=digest(folder / 'dataset.json'),
        source_evidence=['robot_adaptation/estimators/visual_mass/generate.py:76-88,132-168'])


def main():
    groups = []
    for meta in sorted((ROOT / 'runs/context_dynamics').rglob('dataset.json')):
        print('context', relative(meta.parent), flush=True)
        groups.append(context_group(meta.parent))
    for folder in sorted((ROOT / 'runs/mass_belief').rglob('*')):
        if folder.is_dir() and any(folder.glob('*.npz')):
            first = next(folder.glob('*.npz'))
            with np.load(first, allow_pickle=False) as a:
                if 'x' in a and 'mass_g' in a:
                    groups.append(mass_group(folder))
    for path in sorted((ROOT / 'runs/rma').rglob('episodes.csv')):
        if any(path.parent.glob('episode_*.npz')):
            groups.append(rma_group(path.parent))
    groups.append(interaction_group())
    groups.append(visual_group())
    known = {x['path'] for x in groups}
    derived = []
    for family in FAMILIES:
        parents = sorted({p.parent for p in (ROOT / 'runs' / family).rglob('*.npz')})
        for folder in parents:
            if relative(folder) not in known:
                derived.append(dict(path=relative(folder), role='Derived diagnostic/model-output arrays, not raw acquisition episodes',
                                    files={p.name:schema(p) for p in sorted(folder.glob('*.npz'))}))
    csv_only = []
    for family in FAMILIES:
        for path in sorted((ROOT / 'runs' / family).rglob('*.csv')):
            if relative(path.parent) in known:
                continue
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            csv_only.append(dict(path=relative(path), rows=len(rows), columns=list(rows[0]) if rows else [],
                                 role='Derived summary/prediction table; no additional raw trajectory assumed'))
    no_raw = []
    for family in FAMILIES:
        for manifest in sorted((ROOT / 'runs' / family).rglob('manifest.json')):
            if relative(manifest.parent) in known or list(manifest.parent.glob('*.npz')):
                continue
            no_raw.append(dict(path=relative(manifest.parent), manifest=relative(manifest),
                               note='Manifest/checkpoints/logs present; no raw NPZ directly in this directory. Nested datasets are listed separately.'))
    dup_a = ROOT / 'runs/context_dynamics/nov13/grip_data'
    dup_b = ROOT / 'runs/context_dynamics/nov13/grip_pulses'
    duplicates = sum(digest(p) == digest(dup_b / p.name) for p in dup_a.glob('shard_*.npz'))
    equal_arrays = Counter()
    for p in dup_a.glob('shard_*.npz'):
        with np.load(p, allow_pickle=False) as a, np.load(dup_b / p.name, allow_pickle=False) as b:
            for key in ('constants', 'episode_id', 'seed', 'command', 'state'):
                equal_arrays[key] += int(np.array_equal(a[key], b[key]))
    artifact = dict(schema_version=1, scope='Historical families outside lift_slide; inventory only, no collection/training/model evaluation.',
        groups=groups, derived_npz_groups=derived, derived_csv_tables=csv_only, manifests_without_direct_raw_npz=no_raw,
        associated_raw_roots=[dict(path='runs/failure_collection/dense_single_pair_fixed40g', relationship='Raw source of440 selected interaction episodes', inventory='results/probe/inventory_other.json'),
                              dict(path='runs/failure_collection/varied_fixed40g', relationship='Policy/checkpoint provenance for context and mass_belief; not their recorded trajectories', inventory='results/probe/inventory_other.json')],
        duplicate_audit=dict(paths=[relative(dup_a), relative(dup_b)], same_named_shards=60, byte_identical_shards=duplicates,
                             shards_with_identical_arrays=dict(equal_arrays),
                             warning='Matching seeds/physics do not represent independently sampled objects. Different actions/states are separate acquisition variants; original source revision explaining their difference is not reconstructed here.'),
        stop_rule=dict(global_probe_training_status='STOPPED: parent verified true-mode label injection in separate interaction_adaptation retries and physics_conditioning data; see results/probe/inventory_other.json. The two fixed40g raw source roots themselves do not supply actual mass.',
                       proven_true_mass_friction_copy_in_these_audited_saved_state_or_action=False,
                       privileged_controller_groups=['rma oracle trajectories', 'mass_belief closed_loop *_true'],
                       privileged_sidefields=['context constants and encoded physics z', 'rma z and oracle z_hat', 'all explicitly named physical labels'],
                       distinction='Measured kinematics/contact forces are dynamic physical observations. Direct true-physics encoder inputs and truth-conditioned controllers are privileged, even when raw saved state columns exclude the labels.'),
        limitations=['Current source paths/line references document the inspected builders; equality with every historical source revision is not asserted.',
                     'All trajectory file counts and physical labels were read from actual archives or their explicit per-trace CSV joins; field schemas are shown from a representative file per group.',
                     'No persistent cube IDs are invented from mass/friction values, reset seeds, or filenames.',
                     'Raw failure_collection roots are inventoried separately by the acquisition auditor; no duplicate scan or probe training was performed.'],
        audit_source_sha256=digest(Path(__file__)))
    OUT.write_text(json.dumps(artifact, indent=2, allow_nan=False) + '\n')
    print('saved', OUT, 'raw groups', len(groups), 'derived groups', len(derived), flush=True)


if __name__ == '__main__':
    main()
