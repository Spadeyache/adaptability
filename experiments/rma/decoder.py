"""Read mass/friction from a frozen RMA latent and evaluate the reconstructed physics."""
import argparse
import csv
import json
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.rma.common import config, digest, load_policy, make_env, prepare, status
from robot_adaptation.policy.rma.model import split_observation


def fit(z, labels):
    # Ridge on standardized latent coordinates, with an unpenalized intercept.
    center = z.mean(0)
    scale = z.std(0).clamp_min(1e-8)
    x = (z - center) / scale
    bias = labels.mean(0)
    weights = torch.linalg.solve(x.T @ x + 1e-4 * torch.eye(x.shape[1], dtype=x.dtype), x.T @ (labels - bias))
    return dict(center=center, scale=scale, weights=weights, bias=bias)


def decode(z, model):
    return ((z.double() - model['center']) / model['scale']) @ model['weights'] + model['bias']


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def mae(actual, predicted):
    errors = (actual - predicted).abs().mean(0)
    return dict(mass_mae_g=1000 * errors[0].item(), friction_mae=errors[1].item())


@torch.no_grad()
def probe(args):
    c = config(args.checkpoint.parent/'manifest.json')['config']
    policy = load_policy(args.checkpoint, 'cpu')
    out = prepare(args.output, c, 'physics_decoder', policy_sha256=digest(args.checkpoint), ridge=1e-4)
    train = torch.tensor(list(product(c['masses_kg'], c['frictions'])), dtype=torch.float64)
    mids = lambda v: [(lo+hi)/2 for lo, hi in zip(v[:-1], v[1:])]
    holdout = torch.tensor(list(product(mids(c['masses_kg']), mids(c['frictions']))), dtype=torch.float64)
    z = policy.encode(train.float()).double()
    model = fit(z, train)
    torch.save(model, out/'decoder.pt')
    cv = []
    for i in range(len(train)):
        keep = torch.arange(len(train)) != i
        cv.append(decode(z[i], fit(z[keep], train[keep])))
    tests = [('leave_one_pair_out', train, torch.stack(cv)),
             ('unseen_midpoints', holdout, decode(policy.encode(holdout.float()), model))]
    rows = []
    summary = {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for name, actual, predicted in tests:
        baseline = train.mean(0).expand_as(actual)
        if name == 'leave_one_pair_out':
            baseline = (train.sum(0) - actual) / (len(train)-1)
        summary[name] = dict(pairs=len(actual), decoder=mae(actual,predicted), constant=mae(actual,baseline))
        for truth, pred in zip(actual.tolist(), predicted.tolist()):
            rows.append(dict(split=name, true_mass_g=truth[0]*1000, decoded_mass_g=pred[0]*1000,
                             true_friction=truth[1], decoded_friction=pred[1]))
        for j, factor in enumerate([1000,1]):
            axes[j].scatter(actual[:,j]*factor, predicted[:,j]*factor, label=name)
    for ax, unit in zip(axes,['mass (g)','friction coefficient']):
        low,high=ax.get_xlim(); ax.plot([low,high],[low,high],'k--',linewidth=1)
        ax.set(xlabel='True '+unit, ylabel='Decoded '+unit); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'reconstruction.png', dpi=160); plt.close(fig)
    write_csv(out/'predictions.csv', rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    status(out, 'completed', decoder_sha256=digest(out/'decoder.pt'))
    print(json.dumps(summary,indent=2))


def read_reference(args):
    ref = config(args.reference/'manifest.json')
    manifest = config(args.decoder.parent/'manifest.json')
    if not (ref['policy_sha256'] == manifest['policy_sha256'] == digest(args.checkpoint)):
        raise ValueError('Reference and decoder must share the frozen teacher checkpoint')
    if digest(args.decoder) != manifest['decoder_sha256'] or ref['status'] != 'completed':
        raise ValueError('Decoder checksum mismatch or incomplete reference')
    with (args.reference/'episodes.csv').open() as file:
        rows = list(csv.DictReader(file))
    return ref, rows, torch.load(args.decoder, map_location='cpu', weights_only=True)


@torch.no_grad()
def compare(args):
    ref, rows, decoder = read_reference(args)
    c = ref['config']
    out = prepare(args.output,c,'decoder_roundtrip',policy_sha256=ref['policy_sha256'],decoder_sha256=digest(args.decoder),trajectory_reference=str(args.reference.resolve()))
    policy = load_policy(args.checkpoint,'cpu')
    env = make_env(c)
    results = []
    try:
        for row in rows:
            if row['mode'] != 'oracle':
                continue
            seed, mass, friction = int(row['seed']), float(row['mass_kg']), float(row['friction'])
            obs,_ = env.reset(seed=seed, options=dict(mass_kg=mass,friction_coefficient=friction))
            state, physical = split_observation(obs)
            with np.load(args.reference/row['trace']) as trace:
                np.testing.assert_array_equal(trace['state'][0], state[0].numpy())
                np.testing.assert_allclose(trace['z'][0], policy.encode(physical)[0].numpy(), atol=1e-6)
            prediction = decode(policy.encode(physical), decoder).float()
            reconstructed = policy.encode(prediction)
            for _ in range(c['horizon']):
                state,_ = split_observation(obs)
                action = policy.action_from_latent(state,reconstructed).clamp(-1,1)
                obs,_,_,_,info = env.step(action)
            results.append(dict(seed=seed,mass_kg=mass,friction=friction,decoded_mass_g=prediction[0,0].item()*1000,
                                decoded_friction=prediction[0,1].item(),oracle=int(row['success']),roundtrip=int(info['success'].item())))
        write_csv(out/'episodes.csv',results)
        oracle = np.array([r['oracle'] for r in results]); decoded = np.array([r['roundtrip'] for r in results]); delta = decoded-oracle
        seeds = np.array([r['seed'] for r in results]); blocks = np.array([delta[seeds==s].mean() for s in np.unique(seeds)])
        ci = np.quantile(np.random.default_rng(7).choice(blocks, (10000,len(blocks))).mean(1), [.025,.975])
        summary = dict(pairs=len(results),oracle_success=int(oracle.sum()), roundtrip_success=int(decoded.sum()),
                       gained=int((delta==1).sum()),lost=int((delta==-1).sum()), delta_pp=100*delta.mean(), paired_seed_bootstrap_95_pp=(100*ci).tolist())
        (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary))
        status(out,'completed')
    except BaseException as exc:
        status(out,'failed',error=str(exc)); raise
    finally:
        env.close()


@torch.no_grad()
def history(args):
    ref, rows, decoder = read_reference(args)
    adapted = [r for r in rows if r['mode']=='adapted']
    if not adapted:
        raise ValueError('Matched adapted trajectories are required')
    out = prepare(args.output,ref['config'],'adapter_physics_readout',trajectory_reference=str(args.reference.resolve()),
                  policy_sha256=ref['policy_sha256'],decoder_sha256=digest(args.decoder),adapter_sha256=ref['adapter_sha256'])
    records = []
    for row in adapted:
        with np.load(args.reference/row['trace']) as trace:
            est = decode(torch.from_numpy(trace['z_hat']), decoder)
            teacher = decode(torch.from_numpy(trace['z']), decoder)
        for step, (p,t) in enumerate(zip(est.tolist(),teacher.tolist())):
            records.append(dict(seed=row['seed'], step=step, true_mass_g=float(row['mass_kg'])*1000,true_friction=float(row['friction']),
                                adapted_mass_g=p[0]*1000,adapted_friction=p[1], teacher_mass_g=t[0]*1000,teacher_friction=t[1]))
    write_csv(out/'predictions.csv',records)
    summary = {}
    for step in [0,24,49]:
        subset = [r for r in records if r['step']==step]; summary[step]={}
        for mode in ['adapted','teacher']:
            summary[step][mode] = {key:float(np.mean([abs(r[mode+'_'+key]-r['true_'+key]) for r in subset])) for key in ['mass_g','friction']}
    (out/'mae_by_step.json').write_text(json.dumps(summary,indent=2)); status(out,'completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ['fit','compare','history']:
        p = commands.add_parser(command)
        p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
        if command != 'fit':
            p.add_argument('--reference',type=Path,required=True); p.add_argument('--decoder',type=Path,required=True)
    args = parser.parse_args(); {'fit':probe,'compare':compare,'history':history}[args.command](args)
