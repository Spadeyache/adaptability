"""Sequential weight adaptation of B0, using only observed context lifts."""
from datetime import datetime, timezone
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from .data import ROOT, RESULTS, TASKS, digest, load_rows, masks, write_json
from .evaluate import evaluate
from .model import load_checkpoint, save_checkpoint


METRICS = ('one_step_mse', 'one_step_r2', 'rollout_10_mse')
SEED_NAMESPACE = 271828


def _sync(model):
    if model.device.type == 'cuda':
        torch.cuda.synchronize(model.device)


def _seed(base_seed, order_id, position, cube):
    # The same minibatch draws are used across learning rates for a seed/order.
    sequence = np.random.SeedSequence([SEED_NAMESPACE, base_seed, order_id, position, cube])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _buffers(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()}


def _unchanged_buffers(model, reference):
    current = dict(model.named_buffers())
    if current.keys() != reference.keys() or any(
            not torch.equal(current[name], value) for name, value in reference.items()):
        raise RuntimeError('Adaptation changed a fixed normalization buffer')


def _check_evaluation(value, path='evaluation'):
    """Empty subsets may be null; nonfinite predictions must stop the run."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ('one_step_nonfinite_rows', 'rollout_nonfinite_starts') and child:
                raise FloatingPointError(f'{path}.{key} reports {child} nonfinite predictions')
            _check_evaluation(child, path+'.'+key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_evaluation(child, f'{path}[{index}]')
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise FloatingPointError(f'Nonfinite value at {path}')


def _failure_safe(value):
    """Retain failed numeric values explicitly without emitting invalid JSON."""
    if isinstance(value, dict):
        return {key: _failure_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_failure_safe(child) for child in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return repr(float(value))
    return value


def _timed_evaluation(model, rows, selection):
    _sync(model)
    started = monotonic()
    result = evaluate(model, rows, selection=selection)
    _sync(model)
    return result, monotonic()-started


def _adapt_cube(model, rows, context, *, learning_rate, steps, batch_size, torch_seed):
    """Fresh Adam, replacement minibatches, and every B0 weight trainable."""
    if model.kind != 'B0':
        raise ValueError('Sequential weight adaptation requires a trained B0 checkpoint')
    indices = np.flatnonzero(context)
    if not len(indices) or steps < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError('Context and the fixed optimization schedule must be nonempty')
    if any(key in rows for key in ('true_mass', 'true_friction')):
        raise ValueError('Adaptation must not load physical labels')
    reference = _buffers(model)
    _sync(model)
    started = monotonic()
    state, action, following = [torch.as_tensor(rows[key][indices], dtype=torch.float32,
                                               device=model.device)
                                for key in ('state', 'action', 'next_state')]
    inputs, targets = model.normalize(torch.cat((state, action), 1), following-state)
    if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
        raise FloatingPointError('Nonfinite normalized context observations')
    model.train().requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device=model.device).manual_seed(torch_seed)
    with torch.no_grad():
        before = (model(inputs)-targets).square().mean().item()
    losses = []
    for update in range(steps):
        block = torch.randint(len(indices), (batch_size,), device=model.device, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs[block])-targets[block]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite context loss at update {update}')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        after = (model(inputs)-targets).square().mean().item()
    if not all(torch.isfinite(value).all() for value in model.parameters()) or not np.isfinite([before, after]).all():
        raise FloatingPointError('Nonfinite weights or context loss after adaptation')
    _unchanged_buffers(model, reference)
    _sync(model)
    return dict(context_rows=len(indices), updates=steps, batch_size=batch_size, replacement=True,
                fresh_adam=True, learning_rate=float(learning_rate), torch_seed=torch_seed,
                context_mse_before=before, context_mse_after=after, batch_losses=losses,
                seconds=monotonic()-started, normalization_buffers_unchanged=True)


def _prepare_cubes(test_rows, roles):
    """Keep complete lift episodes while avoiding repeated full-dataset transfers."""
    prepared = {}
    for cube in np.unique(test_rows['cube_id']):
        selected = (test_rows['cube_id'] == cube) & (test_rows['task'] == 0)
        rows = {name: value[selected] for name, value in test_rows.items()}
        context, scoring = roles['context'][selected], roles['eval'][selected]
        if len(rows['state']) != 1200 or int(context.sum()) != 600 or int(scoring.sum()) != 600:
            raise ValueError(f'Cube {cube} must have 600 context and 600 evaluation lift rows')
        if (not np.array_equal(np.unique(rows['episode_id'][context]), np.arange(5))
                or not np.array_equal(np.unique(rows['episode_id'][scoring]), np.arange(5, 10))):
            raise ValueError(f'Context/evaluation lift episode IDs differ for cube {cube}')
        if np.any(context & scoring) or not np.all(context | scoring):
            raise ValueError('Context and evaluation must partition complete lift episodes')
        prepared[int(cube)] = dict(rows=rows, context=context, evaluation=scoring)
    return prepared


@torch.no_grad()
def _evaluation_target_sst(model, cube_data):
    targets = []
    for block in cube_data.values():
        rows = block['rows']
        selected = block['evaluation'] & rows['cube_moving']
        state, following = [torch.as_tensor(rows[key][selected], dtype=torch.float32, device=model.device)
                            for key in ('state', 'next_state')]
        values = (following-state-model.delta_mean)/model.delta_sd
        targets.append(values.double().cpu().numpy())
    targets = np.concatenate(targets)
    return float(np.square(targets-targets.mean(0)).sum())


def _pool_current(records, target_sst):
    """Pool differently adapted cubes using fixed common target normalization."""
    values = [record['evaluation']['per_task']['lift'] for record in records]
    if any((value['moving_rows'] and value['one_step_sse'] is None)
           or (value['rollout_starts'] and value['rollout_10_mse'] is None) for value in values):
        raise ValueError('A current-cube metric is unavailable despite eligible rows')
    moving = sum(value['moving_rows'] for value in values)
    starts = sum(value['rollout_starts'] for value in values)
    sse = sum(value['one_step_sse'] for value in values if value['one_step_sse'] is not None)
    return dict(selected_rows=sum(value['selected_rows'] for value in values), moving_rows=moving,
                one_step_sse=sse, one_step_sst=target_sst,
                one_step_mse=sse/(moving*51) if moving else None,
                one_step_r2=1-sse/target_sst if moving and target_sst > 0 else None,
                rollout_starts=starts,
                rollout_10_mse=sum(value['rollout_10_mse']*value['rollout_starts']
                    for value in values if value['rollout_starts'])/starts if starts else None)


def _adapt_sequence(model, cube_data, order, configuration, *, base_seed, order_id,
                    learning_rate, report, progress=None):
    fixed = _buffers(model)
    first = int(order[0])
    for position, cube in enumerate(order):
        cube = int(cube)
        block = cube_data[cube]
        report['stage'] = dict(name='context_adaptation', position=position, cube_id=cube)
        fit = _adapt_cube(model, block['rows'], block['context'], learning_rate=learning_rate,
            steps=configuration['steps_per_cube'], batch_size=configuration['batch_size'],
            torch_seed=_seed(base_seed, order_id, position, cube))
        _unchanged_buffers(model, fixed)
        record = dict(position=position, cube_id=cube, adaptation=fit)
        report['current_cubes'].append(record)
        report['stage']['name'] = 'current_cube_evaluation'
        result, elapsed = _timed_evaluation(model, block['rows'], block['evaluation'])
        record.update(evaluation=result, evaluation_seconds=elapsed)
        _check_evaluation(result)
        if position in configuration['retained_k']:
            report['stage']['name'] = 'first_cube_retention'
            if position == 0:
                retained, elapsed = result, 0.
            else:
                retained, elapsed = _timed_evaluation(
                    model, cube_data[first]['rows'], cube_data[first]['evaluation'])
            report['retention'][str(position)] = dict(cube_id=first, further_cubes=position,
                evaluation=retained, evaluation_seconds=elapsed, reused_current_evaluation=position == 0)
            _check_evaluation(retained)
        if progress is not None:
            progress(report)


def _difference(after, before):
    return {key: after[key]-before[key] if after[key] is not None and before[key] is not None else None
            for key in METRICS}


def _summary_record(report, path):
    before = report['train_validation_before']
    after = report['train_validation_after']
    reference = report['retention']['0']['evaluation']['per_task']['lift']
    return dict(base_seed=report['identity']['base_seed'], order_id=report['identity']['order_id'],
        learning_rate=report['identity']['learning_rate'], cube_order=report['identity']['cube_order'],
        first_cube_id=report['identity']['cube_order'][0], report=str(path.resolve()), report_sha256=digest(path),
        final_checkpoint=report['final_checkpoint'], final_checkpoint_sha256=report['final_checkpoint_sha256'],
        elapsed_seconds=report['elapsed_seconds'], current_cube_pooled=report['current_cube_pooled'],
        current_cubes=[dict(position=row['position'], cube_id=row['cube_id'],
            metrics=row['evaluation']['per_task']['lift'], torch_seed=row['adaptation']['torch_seed'])
            for row in report['current_cubes']],
        retention={key: value['evaluation']['per_task']['lift'] for key, value in report['retention'].items()},
        retention_change_from_k0={key: _difference(value['evaluation']['per_task']['lift'], reference)
                                  for key, value in report['retention'].items()},
        train_validation_before=dict(overall=before['overall'], **before['per_task']),
        train_validation_after=dict(overall=after['overall'], **after['per_task']),
        train_validation_change={task: _difference(
            after['overall'] if task == 'overall' else after['per_task'][task],
            before['overall'] if task == 'overall' else before['per_task'][task]) for task in ('overall', *TASKS)})


def _hierarchical(values):
    """Average orders within each independent trained seed, then summarize seeds."""
    result = {}
    for metric in METRICS:
        observed = [(seed, order, row[metric]) for seed, order, row in values]
        seeds = sorted({seed for seed, _, _ in observed})
        if any(value is not None and not np.isfinite(value) for _, _, value in observed):
            raise FloatingPointError(f'Nonfinite adaptation summary metric: {metric}')
        per_seed = {str(seed): [value for source, _, value in observed if source == seed] for seed in seeds}
        means = {seed: float(np.mean(group)) if all(value is not None for value in group) else None
                 for seed, group in per_seed.items()}
        within = {seed: float(np.std(group, ddof=1)) if len(group) > 1 and means[seed] is not None else None
                  for seed, group in per_seed.items()}
        complete = bool(means) and all(value is not None for value in means.values())
        available = sum(value is not None for _, _, value in observed)
        result[metric] = dict(mean=float(np.mean(list(means.values()))) if complete else None,
            sd_across_seeds=float(np.std(list(means.values()), ddof=1)) if complete and len(means) > 1 else None,
            seed_means=means, within_seed_order_sd=within, seed_count=len(means),
            trajectory_count=len(observed), available_trajectories=available, expected_trajectories=len(observed),
            all_supplied_metrics_available=complete,
            raw=[dict(base_seed=seed, order_id=order, value=float(value) if value is not None else None)
                 for seed, order, value in observed])
    return result


def aggregate_reports(reports, configuration):
    """Retain all trajectory numbers and make both seed/order levels explicit."""
    grouped = {}
    for rate in configuration['learning_rates']:
        selected = [row for row in reports if row['learning_rate'] == rate]
        if not selected:
            continue
        def summarize(getter):
            return _hierarchical([(row['base_seed'], row['order_id'], getter(row)) for row in selected])
        grouped[str(rate)] = dict(learning_rate=rate, trajectory_count=len(selected),
            current_cube_pooled=summarize(lambda row: row['current_cube_pooled']),
            retention={str(k): summarize(lambda row: row['retention'][str(k)]) for k in configuration['retained_k']},
            retention_change_from_k0={str(k): summarize(lambda row: row['retention_change_from_k0'][str(k)])
                                      for k in configuration['retained_k']},
            **{name: {task: summarize(lambda row, name=name, task=task: row[name][task])
                       for task in ('overall', *TASKS)} for name in
               ('train_validation_before', 'train_validation_after', 'train_validation_change')})
    expected = {(seed, order, rate) for seed in configuration['base_seeds']
                for order in range(len(configuration['orders'])) for rate in configuration['learning_rates']}
    actual = {(row['base_seed'], row['order_id'], row['learning_rate']) for row in reports}
    if len(actual) != len(reports) or not actual <= expected:
        raise ValueError('Duplicate or out-of-protocol adaptation trajectories')
    missing = []
    for row in reports:
        checks = [('current_cube_pooled', row['current_cube_pooled'])]
        checks.extend((f'retention/{k}', value) for k, value in row['retention'].items())
        checks.extend((f'current_cube/{value["cube_id"]}', value['metrics'])
                      for value in row.get('current_cubes', []))
        checks.extend((name+'/'+task, values) for name in ('train_validation_before', 'train_validation_after')
                      for task, values in row[name].items())
        for scope, values in checks:
            for metric in METRICS:
                if values[metric] is None:
                    missing.append(dict(base_seed=row['base_seed'], order_id=row['order_id'],
                                        learning_rate=row['learning_rate'], scope=scope, metric=metric))
    return dict(schema_version=1, complete_design=actual == expected, completed_trajectories=len(reports),
        complete_primary_metrics=bool(reports) and not missing, missing_primary_metrics=missing,
        expected_trajectories=len(expected), learning_rates=grouped, trajectories=reports,
        aggregation='Pool eligible current-cube rows within each trajectory. Average the five orders within each B0 seed, then report mean and sample SD (ddof=1) across B0 seeds. Orders are not counted as independent trained models.',
        order_variability='Each metric also retains its within-seed sample SD across orders and every seed/order value.',
        retention='k counts further cubes after adapting to the first cube. k=0 is immediately after its own adaptation.',
        differences='Signed after-minus-before values; positive MSE changes and negative R2 changes indicate degradation.',
        normalization='B0 training normalization buffers remain fixed throughout every trajectory.',
        early_window='Literal first five steps have no moving rows; unavailable early metrics remain null in the full reports.')


def run_adaptation(*, device='cpu', seeds=None, order_ids=None, learning_rates=None,
                   protocol_path=RESULTS/'protocol.json', checkpoint_root=ROOT/'runs/baselines/B0',
                   output_root=ROOT/'runs/baselines/adaptation', report_root=RESULTS/'adaptation'):
    """Run selected frozen trajectories; caller must first satisfy the B0 pilot gate."""
    from .__main__ import verify_completed_run, verify_implementation

    verify_implementation()
    protocol_path, checkpoint_root = Path(protocol_path).resolve(), Path(checkpoint_root).resolve()
    output_root, report_root = Path(output_root).resolve(), Path(report_root).resolve()
    if output_root == checkpoint_root or checkpoint_root in output_root.parents:
        raise ValueError('Adapted checkpoints must be outside the trained B0 checkpoint directory')
    protocol = json.loads(protocol_path.read_text())
    config = protocol['adaptation']
    split_path = ROOT / protocol['split_file']
    if digest(split_path) != protocol['split_sha256']:
        raise ValueError('Frozen split digest changed')
    for name, expected in protocol['input_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Frozen input changed: {name}')
    selected_seeds = list(config['base_seeds'] if seeds is None else seeds)
    selected_orders = list(range(len(config['orders']))) if order_ids is None else list(order_ids)
    selected_rates = list(config['learning_rates'] if learning_rates is None else learning_rates)
    for selected, allowed, name in ((selected_seeds, config['base_seeds'], 'base seeds'),
            (selected_orders, range(len(config['orders'])), 'order IDs'),
            (selected_rates, config['learning_rates'], 'learning rates')):
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(allowed):
            raise ValueError(f'Select unique, nonempty frozen {name}')
    sources = {str(path.relative_to(ROOT)): digest(path) for path in
               [Path(__file__).resolve(), *[Path(__file__).with_name(name).resolve()
                                           for name in ('data.py', 'model.py', 'evaluate.py')]]}
    provenance = dict(protocol_sha256=digest(protocol_path), split_sha256=digest(split_path), source_sha256=sources,
                      implementation_sha256=digest(RESULTS / 'implementation.json'))
    directory = ROOT / protocol['data_directory']
    train, test = [load_rows(split, oracle=False, directory=directory) for split in ('train', 'test')]
    if any(key in rows for rows in (train, test) for key in ('true_mass', 'true_friction')):
        raise ValueError('Physical labels were loaded into the adaptation experiment')
    train_roles, test_roles = masks(train, split_path), masks(test, split_path)
    cube_data = _prepare_cubes(test, test_roles)
    if any(sorted(order) != sorted(cube_data) for order in config['orders']):
        raise ValueError('Every frozen order must permute all held-out cubes exactly once')
    del test
    initial_validation, target_sst = {}, {}
    for base_seed in selected_seeds:
        original = checkpoint_root / f'seed_{base_seed}' / 'checkpoint.pt'
        base_evaluation = json.loads(original.with_name('evaluation.json').read_text())
        verify_completed_run(base_evaluation, original.parent, 'B0', 0, base_seed)
        original_digest = digest(original)
        for order_id in selected_orders:
            order = config['orders'][order_id]
            for rate in selected_rates:
                folder = output_root / f'seed_{base_seed}' / f'order_{order_id}' / f'lr_{rate:g}'
                report_path = folder / 'report.json'
                identity = dict(**provenance, base_seed=int(base_seed), order_id=int(order_id),
                    order_seed=int(config['order_seeds'][order_id]), learning_rate=float(rate), cube_order=order,
                    base_checkpoint=str(original), base_checkpoint_sha256=original_digest,
                    device=str(torch.device(device)), torch_version=str(torch.__version__),
                    seed_namespace=SEED_NAMESPACE, torch_seed_rule='SeedSequence[271828,base_seed,order_id,position,cube_id] uint32',
                    minibatch_draws_paired_across_learning_rates=True)
                if report_path.exists():
                    saved = json.loads(report_path.read_text())
                    if saved['identity'] != identity:
                        raise ValueError(f'Existing trajectory uses another protocol: {folder}')
                    if saved['status'] == 'complete':
                        if digest(saved['final_checkpoint']) != saved['final_checkpoint_sha256']:
                            raise ValueError(f'Completed adapted checkpoint changed: {folder}')
                        continue
                started = monotonic()
                report = dict(identity=identity, status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                              current_cubes=[], retention={}, stage=dict(name='load_B0'))
                try:
                    model, base_training = load_checkpoint(original, device=device)
                    if model.kind != 'B0' or model.history_k != 0 or model.state_dim != 51 or model.action_dim != 8:
                        raise ValueError('Expected the frozen 51-state, 8-action B0 architecture')
                    torch.manual_seed(_seed(base_seed, order_id, 0, int(order[0])))
                    fixed_buffers = _buffers(model)
                    reused = base_seed in initial_validation
                    if not reused:
                        report['stage']['name'] = 'initial_train_validation'
                        observed, elapsed = _timed_evaluation(model, train, train_roles['validation'])
                        report['train_validation_before'] = observed
                        _check_evaluation(observed)
                        initial_validation[base_seed] = observed, elapsed
                        target_sst[base_seed] = _evaluation_target_sst(model, cube_data)
                    report['train_validation_before'] = initial_validation[base_seed][0]
                    report['initial_validation_evaluation_reused'] = reused
                    report['initial_validation_seconds'] = 0. if reused else initial_validation[base_seed][1]
                    _adapt_sequence(model, cube_data, order, config, base_seed=base_seed, order_id=order_id,
                        learning_rate=rate, report=report, progress=lambda value: write_json(report_path, value))
                    report['stage']['name'] = 'final_train_validation'
                    observed, elapsed = _timed_evaluation(model, train, train_roles['validation'])
                    report.update(train_validation_after=observed, final_validation_seconds=elapsed)
                    _check_evaluation(observed)
                    _unchanged_buffers(model, fixed_buffers)
                    if digest(original) != original_digest:
                        raise RuntimeError('The trained B0 checkpoint changed during adaptation')
                    verify_implementation()
                    if (digest(protocol_path) != provenance['protocol_sha256']
                            or digest(split_path) != provenance['split_sha256']
                            or any(digest(ROOT / name) != checksum for name, checksum in sources.items())):
                        raise RuntimeError('Frozen adaptation source, protocol, or split changed during the trajectory')
                    report['current_cube_pooled'] = _pool_current(report['current_cubes'], target_sst[base_seed])
                    _check_evaluation(report['current_cube_pooled'])
                    final_path = folder / 'checkpoint.pt'
                    model.eval().requires_grad_(False)
                    save_checkpoint(final_path, model, dict(base_training, adaptation=dict(identity=identity,
                        completed_cubes=len(order), report=str(report_path), normalization_buffers_unchanged=True)))
                    report.update(status='complete', normalization_buffers_unchanged=True,
                                  original_checkpoint_unchanged=True, final_checkpoint=str(final_path),
                                  final_checkpoint_sha256=digest(final_path), elapsed_seconds=monotonic()-started,
                                  completed_utc=datetime.now(timezone.utc).isoformat(), stage=dict(name='complete'))
                    write_json(report_path, report)
                    print(f'adaptation seed{base_seed} order{order_id} lr{rate:g} complete in '
                          f'{report["elapsed_seconds"]:.1f}s', flush=True)
                    del model
                except Exception as error:
                    report.update(status='failed', elapsed_seconds=monotonic()-started,
                                  failure=dict(type=type(error).__name__, message=str(error)))
                    safe = _failure_safe(report)
                    write_json(report_path, safe)
                    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                    write_json(folder / f'failure_{stamp}.json', safe)
                    raise
    summaries = []
    for path in sorted(output_root.glob('seed_*/order_*/lr_*/report.json')):
        report = json.loads(path.read_text())
        if report['status'] == 'complete' and all(report['identity'].get(k) == v for k, v in provenance.items()):
            if digest(report['identity']['base_checkpoint']) != report['identity']['base_checkpoint_sha256']:
                raise ValueError(f'Trained source checkpoint changed for {path}')
            if digest(report['final_checkpoint']) != report['final_checkpoint_sha256']:
                raise ValueError(f'Adapted checkpoint changed for {path}')
            summaries.append(_summary_record(report, path))
    aggregate = aggregate_reports(summaries, config)
    aggregate.update(provenance, requested_selection=dict(base_seeds=selected_seeds, order_ids=selected_orders,
                                                         learning_rates=selected_rates))
    write_json(report_root / 'aggregate.json', aggregate)
    return aggregate


def render_report(aggregate, directory=RESULTS):
    """Return Markdown table rows and export the three-metric retention figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def format_metric(summary, name):
        value = summary[name]
        if value['mean'] is None:
            return 'NA'
        spread = value['sd_across_seeds']
        return f'{value["mean"]:.5g}' + (f' ± {spread:.3g}' if spread is not None else ' (SD unavailable)')

    table = []
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)
    labels = ('One-step normalized MSE', 'One-step R²', '10-step normalized MSE')
    for rate_key, group in aggregate['learning_rates'].items():
        label = f'Weights lr={group["learning_rate"]:g}'
        entries = [(label+': current cubes, pooled', 'lift', group['current_cube_pooled'])]
        for k in sorted(group['retention'], key=int):
            entries.append((label+f': first cube after {k} further cubes', 'lift', group['retention'][k]))
        for phase, title in (('train_validation_before', 'validation before'),
                             ('train_validation_after', 'validation after'),
                             ('train_validation_change', 'validation change')):
            entries.extend((label+': '+title, task, group[phase][task]) for task in ('overall', *TASKS))
        table.extend('| '+name+' | '+task+' | '+' | '.join(format_metric(values, metric) for metric in METRICS)+' |'
                     for name, task, values in entries)
        x = np.array(sorted(map(int, group['retention'])))
        for axis, metric, ylabel in zip(axes, METRICS, labels):
            means = np.array([group['retention'][str(k)][metric]['mean'] for k in x], dtype=float)
            deviations = [group['retention'][str(k)][metric]['sd_across_seeds'] for k in x]
            line, = axis.plot(x, means, marker='o', label=f'lr={group["learning_rate"]:g}')
            if all(value is not None for value in deviations):
                spread = np.asarray(deviations)
                axis.fill_between(x, means-spread, means+spread, color=line.get_color(), alpha=.15)
            axis.set(xlabel='Further cubes adapted', ylabel=ylabel, xticks=x)
            axis.grid(alpha=.2)
    axes[0].legend()
    figure.suptitle('First-cube retention: orders averaged within each B0 seed; shading ±1 SD across seeds')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        figure.savefig(directory / f'plot3_retention.{suffix}', dpi=180)
    plt.close(figure)
    return table
