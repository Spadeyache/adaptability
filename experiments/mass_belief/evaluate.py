"""Compare reset and carried estimates on matched five-trial sequences."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from robot_adaptation.collection.collect import write_json
from robot_adaptation.estimators.mass_belief.model import MassBelief
from .collect import episode, make_env, policy_and_limits
from .train import data, load_model, predict


def prior(c):
    low, high = c['mass_range_g']
    return MassBelief((low+high)/2, (high-low)**2/12)


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparison(run):
    c = json.loads((run/'config.json').read_text())
    calibration = json.loads((run/'calibration.json').read_text())
    torch.set_num_threads(1)
    model = load_model(run)
    paths = sorted((run/'test').glob('*.npz'))
    x, masses = data(run/'test')
    assert len(masses) == c['sequences']*c['trials'] and np.allclose(masses,c['heavy_g'])
    predictions = predict(model,x)
    rows, steps = [], []
    for i, (path, estimated, mass) in enumerate(zip(paths,predictions,masses)):
        trial = i%c['trials']+1
        if trial == 1:
            carried = prior(c)
        reset = prior(c)
        with np.load(path) as ep:
            seed = int(ep['seed'])
            success = bool(ep['success'][-1])
        assert seed==c['test_seed']+i
        for step, value in enumerate(estimated):
            if str(step) in calibration:
                for belief in (reset, carried):
                    belief.update(float(value),calibration[str(step)])
            steps.append(dict(sequence=i//c['trials'],trial=trial,seed=seed,step=step+1,
                              truth_g=mass,raw_g=float(value),reset_mean_g=reset.mean,
                              reset_variance_g2=reset.variance,carry_mean_g=carried.mean,
                              carry_variance_g2=carried.variance))
        for arm, mean, variance in [('A: estimator reset', float(estimated[-1]),None),
                                    ('B: filter carried',carried.mean,carried.variance),
                                    ('Filter reset',reset.mean,reset.variance)]:
            rows.append(dict(sequence=i//c['trials'], trial=trial, seed=seed, arm=arm,
                             truth_g=mass, estimate_g=mean, abs_error_g=abs(mean-mass),
                             variance_g2=variance, fixed_probe_success=int(success)))
    write_csv(run/'trials.csv',rows)
    write_csv(run/'steps.csv',steps)


@torch.inference_mode()
def side(run):
    c = json.loads((run/'config.json').read_text())
    calibration = json.loads((run/'calibration.json').read_text())
    torch.set_num_threads(1)
    model = load_model(run)
    policy, limits = policy_and_limits(c)
    folder = run/'closed_loop'
    folder.mkdir(exist_ok=False)
    env = make_env(c,c['heavy_g'],limits)
    rows = []
    try:
        for seq in range(c['sequences']):
            belief = prior(c)
            for t in range(c['trials']):
                seed = c['test_seed']+seq*c['trials']+t
                with np.load(run/'test'/f'{seq*c["trials"]+t:04d}.npz') as base:
                    reference = base['initial']
                for arm in ('true','belief'):
                    def update(step, observations):
                        if str(step) in calibration:
                            sequence = np.array(observations[-c['window']:])[None]
                            window = model.windows(torch.from_numpy(sequence))[:,-1]
                            belief.update(model(window).item(),calibration[str(step)])
                        return float(np.clip(belief.mean,*c['mass_range_g']))
                    initial_g = c['heavy_g'] if arm=='true' else float(np.clip(belief.mean,*c['mass_range_g']))
                    record = episode(env,policy,seed,c['horizon'],initial_g,
                                     update if arm=='belief' else None)
                    np.testing.assert_array_equal(record['initial'], reference)
                    assert np.isclose(record['mass_g'],c['heavy_g'])
                    np.savez_compressed(folder/f'{seq:02}_{t+1}_{arm}.npz',**record)
                    rows.append(dict(sequence=seq,trial=t+1,seed=seed,mode=arm,
                                     first_supplied_g=initial_g,final_supplied_g=record['supplied_g'][-1],
                                     success=int(record['success'][-1]),ever_success=int(record['success'].any())))
    finally:
        env.close()
    write_csv(run/'side_check.csv',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--stage',choices=['comparison','side'],required=True)
    a=p.parse_args()
    (comparison if a.stage=='comparison' else side)(a.run)
