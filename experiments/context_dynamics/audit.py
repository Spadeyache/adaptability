"""Check the real rollout arrays, settings, command queue and whole-episode split."""
import argparse
import json
from pathlib import Path

import numpy as np

from .collect import json_write
from robot_adaptation.simulation.context_dynamics import CONSTANTS, TASKS


def audit(folder):
    meta = json.loads((folder/'dataset.json').read_text())
    cfg = meta['config']
    expected = json.loads((folder/'complete.json').read_text())
    counts = np.zeros((4,2),int)
    moved = np.zeros(4,int)
    segments = np.zeros(4,int)
    limits = np.array([cfg[k] for k in CONSTANTS[:7]] + [cfg['com_m']]*3)
    low, high = np.full(10,np.inf), np.full(10,-np.inf)
    n = 0
    starts, conditions, held_out = [], [], []
    for file in sorted(folder.glob('shard_*.npz')):
        with np.load(file) as shard:
            ids, states, commands = shard['episode_id'], shard['state'], shard['command']
            params, applied = shard['constants'], shard['applied']
            np.testing.assert_array_equal(ids,np.arange(n,n+len(ids)))
            assert states.shape == (len(ids), cfg['steps']+1, 39)
            assert commands.shape == applied.shape == (len(ids), cfg['steps'],5)
            assert np.isfinite(states).all() and np.isfinite(commands).all()
            assert np.max(np.abs(commands))<=1 and (params>=limits[:,0]).all() and (params<=limits[:,1]).all()
            assert np.allclose(states[:,:,25],states[:,:,7]+states[:,:,8])
            if 'pair_id' in shard:
                np.testing.assert_array_equal(shard['pair_id'][::2],shard['pair_id'][1::2])
                np.testing.assert_allclose(states[::2,0],states[1::2,0],atol=1e-6)
                np.testing.assert_array_equal(commands[::2],commands[1::2])
                np.testing.assert_array_equal(shard['seed'][::2],shard['seed'][1::2])
                different = params[::2]!=params[1::2]
                assert (different.sum(1)==1).all()
                np.testing.assert_array_equal(different.argmax(1),shard['changed_index'][::2])
            delays = params[:,6].astype(int)
            for d in range(4):
                use = delays==d
                assert np.array_equal(applied[use,d:], commands[use,:cfg['steps']-d])
                assert not np.any(applied[use,:d])
            low = np.minimum(low, params.min(0))
            high = np.maximum(high, params.max(0))
            split = (ids//4)%10==9
            starts.append(states[:,0].astype(float))
            conditions.append(params)
            held_out.append(split)
            displacement = np.linalg.norm(states[:,:,26:29]-states[:,0:1,26:29],axis=-1).max(1)
            for task in range(4):
                use = shard['task_id']==task
                counts[task] += [np.sum(use & ~split), np.sum(use & split)]
                moved[task] += np.sum(use & (displacement>.01))
                segments[task] += np.sum(use & (shard['random_start']>=0))
            n += len(ids)
    assert n == expected['episodes'] and int(counts[:,1].sum()) == n//10
    x, y, test = np.concatenate(starts), np.concatenate(conditions), np.concatenate(held_out)
    x = (x-x[~test].mean(0))/np.maximum(x[~test].std(0),.001)
    x = np.column_stack((np.ones(n),x))
    coefficients = np.linalg.lstsq(x[~test],y[~test],rcond=1e-7)[0]
    residual = np.square(x[test]@coefficients-y[test])
    r2 = 1-residual.sum(0)/np.square(y[test]-y[test].mean(0)).sum(0)
    output = dict(episodes=n, transitions=n*cfg['steps'], train_validation=counts.sum(0).tolist(),
        tasks={task:dict(train=int(counts[i,0]),validation=int(counts[i,1]),
                         random_segments=int(segments[i]),cube_moved_over_1cm=int(moved[i]))
               for i,task in enumerate(TASKS)},
        ranges={key:[low[i],high[i]] for i,key in enumerate(CONSTANTS)},
        linear_r2_from_initial_observable_state=dict(zip(CONSTANTS,r2.tolist())))
    json_write(folder/'audit.json',output)
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    audit(p.parse_args().folder)
