"""Fit from whole simulation episodes; use independent splits for selection and calibration."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from robot_adaptation.collection.collect import write_json
from robot_adaptation.estimators.mass_belief.model import WindowMass
from .collect import feature


def data(folder):
    files = sorted(folder.glob('*.npz'))
    sequences, masses = [], []
    for path in files:
        with np.load(path) as ep:
            sequences.append(ep['x'])
            masses.append(float(ep['mass_g']))
    return np.stack(sequences), np.asarray(masses)


def historical(c):
    root = Path(c['source'])
    index = json.loads((root/'episodes.json').read_text())
    sequences, masses, sources = [], [], []
    for row in index:
        if set(row['changed_properties']) - {'mass_kg'}:
            continue
        with np.load(root/row['file']) as ep:
            assert np.isclose(float(ep['label__object_static_friction']), c['friction'])
            x = feature(ep['sim__policy_observation'][1:], ep['input__action_applied'])
            sequences.append(x)
            masses.append(1000*float(ep['label__mass_kg']))
            sources.append(row['file'])
    return np.stack(sequences), np.asarray(masses), sources


@torch.no_grad()
def predict(model, sequences):
    x = torch.tensor(sequences, dtype=torch.float32)
    windows = model.windows(x)
    flat = windows.flatten(0, 1)
    return torch.cat([model(batch) for batch in flat.split(512)]).reshape(windows.shape[:2]).numpy()


def load_model(run):
    c = json.loads((run/'config.json').read_text())
    model = WindowMass(c['window'], *c['mass_range_g'])
    model.load_state_dict(torch.load(run/'estimator.pt', weights_only=True, map_location='cpu'))
    return model.eval()


def train(run):
    c = json.loads((run/'config.json').read_text())
    if (run/'estimator.pt').exists():
        raise FileExistsError('Choose a new run to fit another estimator')
    torch.set_num_threads(2)
    torch.manual_seed(c['seed'])
    x, mass = data(run/'train')
    hist, hmass, sources = historical(c)
    x = torch.tensor(np.concatenate([x,hist]), dtype=torch.float32)
    labels = torch.tensor(np.concatenate([mass,hmass]), dtype=torch.float32)
    vx, vm = data(run/'validation')
    assert len(mass) == c['train_episodes'] and len(vm) == c['validation_episodes']
    model = WindowMass(c['window'], *c['mass_range_g'])
    model.center.copy_(x.mean((0,1)))
    model.scale.copy_(x.std((0,1)).clamp_min(.0001))
    windows = model.windows(x).flatten(0,1)
    targets = labels.repeat_interleave(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    best = float('inf')
    history = []
    for epoch in range(36):
        model.train()
        for batch in torch.randperm(len(targets)).split(256):
            pred = model(windows[batch])
            loss = ((pred-targets[batch])/model.span).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        val = predict(model, vx)
        score = float(np.abs(val[:, c['window']-1::c['window']]-vm[:,None]).mean())
        history.append(dict(epoch=epoch+1, validation_window_endpoint_mae_g=score))
        if score < best:
            best = score
            torch.save(model.state_dict(), run/'estimator.pt')
    write_json(run/'fit.json', dict(history=history, historical_train_files=sources,
               fresh_train=len(mass), validation=len(vm), best_validation_mae_g=best))
    model = load_model(run)
    cx, cm = data(run/'calibration')
    assert len(cm) == c['calibration_episodes']
    predictions = predict(model, cx)
    calibration = {}
    endpoints = list(range(c['window']-1, c['horizon'], c['window']))
    # Disjoint windows, at most one effective independent trial's worth of
    # precision. Repeated measurements on the same object may share bias.
    for i in endpoints:
        slope, offset = np.polyfit(cm, predictions[:, i], 1)
        residual = predictions[:,i] - slope*cm - offset
        noise = max(float(residual @ residual / (len(cm)-2)), 1.0) * len(endpoints)
        calibration[str(i)] = [float(slope), float(offset), noise]
    write_json(run/'calibration.json', calibration)
    np.savez_compressed(run/'calibration_predictions.npz', mass_g=cm, predictions_g=predictions)
    print(f'Best validation endpoint MAE {best:.2f} g', flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    train(parser.parse_args().run)
