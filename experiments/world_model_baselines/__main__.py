"""Frozen-protocol baseline commands. The pilot must precede the full sweep."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from .data import RESULTS, ROOT, digest, load_rows, masks, write_json
from .protocol import freeze

RUNS = ROOT / 'runs/baselines'
MODELS = [('B0', 0), *[('B1', k) for k in (1, 5, 10, 20)], ('B2', 0)]
CORE_SOURCES = tuple(f'experiments/world_model_baselines/{name}.py' for name in ('model', 'data', 'evaluate'))


def name(kind, k):
    return f'B1_K{k}' if kind == 'B1' else kind


def check_result(result):
    """Never conceal a failed run by dropping rows or averaging other seeds."""
    for task, metrics in result['per_task'].items():
        for key in ('one_step_mse', 'one_step_r2', 'rollout_10_mse'):
            value = metrics[key]
            if value is None or not np.isfinite(value):
                raise RuntimeError(f'{task}: invalid {key}; full run stopped, saved diagnostics retained')
    if result['per_task']['lift']['one_step_r2'] <= 0.01:
        raise RuntimeError('Lift R2 is near zero or negative. Stop and review the saved pilot/run before continuing.')


def verify_implementation():
    """Reject core implementation changes after the core source freeze."""
    path = RESULTS / 'implementation.json'
    if not path.exists():
        raise RuntimeError('Core implementation fingerprints must be frozen before running models')
    expected = json.loads(path.read_text())['sha256']
    if set(expected) != set(CORE_SOURCES):
        raise ValueError('Implementation manifest must identify model, data, and evaluation sources')
    for source, checksum in expected.items():
        if digest(ROOT / source) != checksum:
            raise ValueError(f'Frozen baseline implementation changed: {source}')
    return expected


def _run_identity(kind, k, seed):
    return dict(model=name(kind, k), kind=kind, history_k=k, seed=seed,
                protocol_sha256=digest(RESULTS / 'protocol.json'),
                split_sha256=digest(RESULTS / 'split.json'),
                implementation_sha256=digest(RESULTS / 'implementation.json'),
                core_source_sha256=verify_implementation())


def _verify_checkpoint(path, kind, k, seed, expected_training=None, device='cpu'):
    from .model import load_checkpoint
    model, training = load_checkpoint(path, device=device)
    expected = dict(kind=kind, history_k=k, seed=seed, state_dim=51, action_dim=8)
    if any(training.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Checkpoint training identity does not match requested run: {path}')
    if (model.kind, model.history_k, model.state_dim, model.action_dim) != (kind, k, 51, 8):
        raise ValueError(f'Checkpoint architecture does not match requested run: {path}')
    if expected_training is not None and training != expected_training:
        raise ValueError(f'Checkpoint metadata differs from saved training report: {path}')
    report_path = Path(path).parent / 'training.json'
    if not report_path.exists() or json.loads(report_path.read_text()) != training:
        raise ValueError(f'Checkpoint and training.json disagree: {path}')
    return model, training


def verify_completed_run(saved, folder, kind, k, seed):
    """Validate completed-run provenance, including the already-executed pilot."""
    folder = Path(folder)
    identity = _run_identity(kind, k, seed)
    for key in ('model', 'kind', 'history_k', 'seed', 'protocol_sha256', 'split_sha256'):
        if saved.get(key) != identity[key]:
            raise ValueError(f'Saved run has a different {key}: {folder}')
    # The first pilot predates run manifests, but already recorded these exact
    # core hashes and the checkpoint digest. Its measured times stay unchanged.
    if any(saved.get('source_sha256', {}).get(source) != checksum
           for source, checksum in identity['core_source_sha256'].items()):
        raise ValueError(f'Saved run core implementation differs from the frozen sources: {folder}')
    if ('implementation_sha256' in saved
            and saved['implementation_sha256'] != identity['implementation_sha256']):
        raise ValueError(f'Saved implementation manifest differs: {folder}')
    checkpoint = folder / 'checkpoint.pt'
    if saved.get('checkpoint_sha256') != digest(checkpoint):
        raise ValueError(f'Saved checkpoint checksum differs: {folder}')
    _verify_checkpoint(checkpoint, kind, k, seed, saved['training'])
    check_result(saved['evaluation'])


def _verified_pilot():
    path = RESULTS / 'pilot.json'
    saved = json.loads(path.read_text())
    folder = RUNS / 'B0' / 'seed_0'
    if json.loads((folder / 'evaluation.json').read_text()) != saved:
        raise ValueError('Pilot summary differs from the completed B0 seed-0 evaluation')
    verify_completed_run(saved, folder, 'B0', 0, 0)
    return saved


def run_one(kind, k, seed, *, device, rows=None):
    from .model import train_model
    from .evaluate import evaluate
    clock = monotonic()
    folder = RUNS / name(kind, k) / f'seed_{seed}'
    metrics_file = folder / 'evaluation.json'
    identity = _run_identity(kind, k, seed)
    protocol_hash = identity['protocol_sha256']
    if metrics_file.exists():
        saved = json.loads(metrics_file.read_text())
        verify_completed_run(saved, folder, kind, k, seed)
        print(f'Already complete: {folder}', flush=True)
        return saved
    manifest_path = folder / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is not None and manifest.get('identity') != identity:
        raise ValueError(f'Unfinished run provenance differs: {folder}')
    train, test = rows if rows is not None else (
        load_rows('train', oracle=kind == 'B2'), load_rows('test', oracle=kind == 'B2'))
    split = masks(train)
    if (folder / 'checkpoint.pt').exists():
        if manifest is None or manifest.get('checkpoint_sha256') != digest(folder / 'checkpoint.pt'):
            raise ValueError(f'Cannot resume a checkpoint without matching saved provenance: {folder}')
        if manifest.get('training_sha256') != digest(folder / 'training.json'):
            raise ValueError(f'Saved training report checksum differs: {folder}')
        model, training = _verify_checkpoint(folder / 'checkpoint.pt', kind, k, seed, device=device)
    else:
        write_json(manifest_path, dict(identity=identity, status='training'))
        model, training = train_model(train, split['train'], split['validation'], seed=seed,
            kind=kind, history_k=k, device=device, output_dir=folder,
            log=lambda event: print(json.dumps(dict(model=name(kind,k), seed=seed, **event)), flush=True))
        if _run_identity(kind, k, seed) != identity:
            raise ValueError('Frozen implementation, split, or protocol changed during training')
        write_json(manifest_path, dict(identity=identity, status='trained',
                   checkpoint_sha256=digest(folder / 'checkpoint.pt'),
                   training_sha256=digest(folder / 'training.json')))
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    evaluation_clock = monotonic()
    result = evaluate(model, test, selection=masks(test)['eval'])
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    if _run_identity(kind, k, seed) != identity:
        raise ValueError('Frozen implementation, split, or protocol changed during evaluation')
    saved = dict(model=name(kind,k), kind=kind, history_k=k, seed=seed,
                 protocol_sha256=protocol_hash, split_sha256=digest(RESULTS / 'split.json'),
                 implementation_sha256=identity['implementation_sha256'],
                 checkpoint_sha256=digest(folder / 'checkpoint.pt'),
                 source_sha256={str(path.relative_to(ROOT)):digest(path) for path in sorted(
                     (ROOT / 'experiments/world_model_baselines').glob('*.py'))},
                 training=training, evaluation=result,
                 evaluation_seconds=monotonic()-evaluation_clock,
                 wall_seconds=monotonic()-clock,
                 environment=dict(torch=torch.__version__, numpy=np.__version__, device=device,
                     device_name=torch.cuda.get_device_name() if device.startswith('cuda') else 'cpu',
                     matmul_precision=torch.get_float32_matmul_precision()))
    write_json(metrics_file, saved)
    check_result(result)
    print(json.dumps(dict(model=name(kind,k), seed=seed, wall_seconds=saved['wall_seconds'],
                          lift=result['per_task']['lift'])), flush=True)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze','pilot','acknowledge-pilot','sweep','adapt','report'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
        return
    if args.command == 'report':
        from .report import report
        report()
        return
    freeze()
    verify_implementation()
    if args.command == 'acknowledge-pilot':
        _verified_pilot()
        write_json(RESULTS / 'pilot_reported.json',
                   dict(pilot_sha256=digest(RESULTS / 'pilot.json'),
                        reported_utc=datetime.now(timezone.utc).isoformat(),
                        attestation='The operator has reported the measured pilot runtime and reviewed its results.'))
        print('Recorded the reported and verified B0 pilot; later runs are now eligible.', flush=True)
        return
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('high')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    if args.command == 'pilot':
        result = run_one('B0', 0, 0, device=args.device)
        write_json(RESULTS / 'pilot.json', result)
        return
    acknowledgment = RESULTS / 'pilot_reported.json'
    if not acknowledgment.exists():
        raise RuntimeError('Report the B0 pilot runtime to the user before starting the sweep or adaptation.')
    acknowledgment = json.loads(acknowledgment.read_text())
    if acknowledgment['pilot_sha256'] != digest(RESULTS / 'pilot.json'):
        raise RuntimeError('Pilot acknowledgment does not match this pilot')
    _verified_pilot()
    if args.command == 'sweep':
        for kind, k in MODELS:
            rows = (load_rows('train', oracle=kind == 'B2'), load_rows('test', oracle=kind == 'B2'))
            for seed in range(5):
                run_one(kind, k, seed, device=args.device, rows=rows)
            del rows
    else:
        from .adaptation import run_adaptation
        run_adaptation(device=args.device)


if __name__ == '__main__':
    main()
