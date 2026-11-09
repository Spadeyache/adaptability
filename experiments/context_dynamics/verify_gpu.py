"""Read saved GPU rollout chunks and verify transitions, physics and episode splits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .collect import json_write
from robot_adaptation.simulation.context_dynamics import CONSTANTS


SOURCES = ('policy_noise','scripted_push','random_poke')


def inspect_shard(path, protocol, ids):
    cfg, horizon = protocol['config'], protocol['config']['steps']
    with np.load(path,allow_pickle=False) as f:
        states, commands, applied = f['state'], f['command'], f['applied']
        requested, constants = f['requested_constants'], f['constants']
        source, held = f['source_id'], f['is_validation']
        np.testing.assert_array_equal(f['episode_id'],ids)
        np.testing.assert_array_equal(f['seed'],ids+cfg['seed'])
        np.testing.assert_array_equal(f['task_id'],np.zeros(len(ids)))
        expected_split = ids==9 if protocol['stage']=='pilot' else (ids//4)%10==9
        np.testing.assert_array_equal(held,expected_split)
        if states.shape!=(len(ids),horizon+1,39) or commands.shape!=(len(ids),horizon,5):
            raise ValueError(f'Unexpected transition dimensions in {path}')
        if applied.shape!=commands.shape or constants.shape!=(len(ids),len(CONSTANTS)):
            raise ValueError('Misaligned action or physics rows')
        if not np.isfinite(states).all() or not np.isfinite(commands).all() or np.abs(commands).max()>1:
            raise ValueError(f'Invalid state or command in {path}')
        if not np.isin(source,np.arange(len(SOURCES))).all():
            raise ValueError('Unknown motion source')
        ranges = np.array([cfg[k] for k in CONSTANTS[:7]]+[cfg['com_m']]*3)
        if not ((requested>=ranges[:,0]) & (requested<=ranges[:,1])).all():
            raise ValueError('Raw physics outside configured ranges')
        guess = f['policy_guess_kg']
        if not ((guess>=cfg['mass_kg'][0]) & (guess<=cfg['mass_kg'][1])).all():
            raise ValueError('Invalid policy mass guess')
        np.testing.assert_allclose(constants,requested,rtol=2e-6,atol=1e-8)
        np.testing.assert_allclose(states[:,:,25],states[:,:,7]+states[:,:,8],atol=2e-6)
        for q in (states[:,:,21:25],states[:,:,29:33]):
            np.testing.assert_allclose(np.linalg.norm(q,axis=-1),1.,atol=.001)
        delays = constants[:,6].astype(int)
        np.testing.assert_array_equal(constants[:,6],delays)
        for delay in np.unique(delays):
            mask = delays==delay
            np.testing.assert_array_equal(applied[mask,delay:],commands[mask,:horizon-delay])
            if np.any(applied[mask,:delay]):
                raise ValueError('Nonzero command in initially empty delay queue')
        lo,hi = protocol['force_limits_n']
        expected_cap = (lo+.5*(applied[:,:,4]+1)*(hi-lo))*constants[:,None,3]
        caps = f['finger_drive_limit_n']
        if caps.shape!=(len(ids),horizon,2):
            raise ValueError('Missing finger readbacks')
        for finger in range(2):
            np.testing.assert_allclose(caps[:,:,finger],expected_cap,rtol=2e-6,atol=1e-5)
        pos = states[:,:,26:29]
        peak_mm = 1000*np.max(np.linalg.norm(pos-pos[:,0:1],axis=-1),axis=1)
        max_horizontal = np.abs(pos[:,:,:2]).max(axis=(1,2))
        if np.any(max_horizontal >= cfg['scene_spacing_m']/2 - 1):
            raise ValueError('A cube approached the boundary of its parallel environment')
        return [dict(episode_id=int(eid),seed=int(eid+cfg['seed']),source=SOURCES[int(source[i])],
                     split='validation' if held[i] else 'train',policy_guess_kg=float(guess[i]),
                     peak_cube_displacement_mm=float(peak_mm[i]),
                     max_abs_cube_xy_m=float(max_horizontal[i]),
                     **{name:float(value) for name,value in zip(CONSTANTS,constants[i])})
                for i,eid in enumerate(ids)]


def verify_dataset(folder):
    protocol = json.loads((folder/'dataset.json').read_text())
    cfg = protocol['config']
    files = sorted(folder.glob('shard_*.npz'))
    batch = cfg['shard_episodes']
    count = cfg['episodes']
    if len(files)!=(count+batch-1)//batch:
        raise ValueError('Incomplete collection')
    rows, hashes = [], {}
    for i,file in enumerate(files):
        if file.name!=f'shard_{i:04}.npz':
            raise ValueError('Missing or out-of-order chunk')
        ids = np.arange(i*batch,min((i+1)*batch,count))
        rows.extend(inspect_shard(file,protocol,ids))
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    if len(rows)!=count or len({r['seed'] for r in rows})!=count:
        raise ValueError('Episode or reset seed duplication')
    stats = {}
    for name in SOURCES:
        group = [r for r in rows if r['source']==name]
        motion = np.array([r['peak_cube_displacement_mm'] for r in group])
        if not len(motion) or not (motion>10).any():
            raise ValueError(f'No measured cube motion over 1 cm for {name}')
        stats[name] = dict(episodes=len(group),train=sum(r['split']=='train' for r in group),
                           validation=sum(r['split']=='validation' for r in group),
                           moved_over_1cm=int((motion>10).sum()),median_peak_mm=float(np.median(motion)))
    ranges = {k:[min(r[k] for r in rows),max(r[k] for r in rows)] for k in CONSTANTS}
    if not all(b>a for a,b in ranges.values()):
        raise ValueError('A hidden variable never changed')
    valid = sum(r['split']=='validation' for r in rows)
    if valid!=round(count*.1):
        raise ValueError('Unexpected holdout size')
    summary = dict(verified=True,episodes=count,shards=len(files),transition_count=count*cfg['steps'],
                   train_episodes=count-valid,validation_episodes=valid,
                   train_transitions=(count-valid)*cfg['steps'],validation_transitions=valid*cfg['steps'],
                   moved_over_1cm=sum(s['moved_over_1cm'] for s in stats.values()),sources=stats,
                   max_abs_cube_xy_m=max(r['max_abs_cube_xy_m'] for r in rows),
                   realized_ranges=ranges,gpu=protocol['gpu'],sim_backend=protocol['simulator'])
    temp = folder/'episodes.csv.tmp'
    with temp.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(folder/'episodes.csv')
    json_write(folder/'checksums.json',hashes)
    json_write(folder/'complete.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    verify_dataset(parser.parse_args().folder)
