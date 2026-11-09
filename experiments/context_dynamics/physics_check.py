"""Matched physical replays for two properties with weak linear readouts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from robot_adaptation.policy.frozen_v1 import load_agent
from robot_adaptation.simulation.context_dynamics import TASKS, PandaMotion, make_env
from .check import write_csv, write_json


def physical_replays(base, probes, output, count=48):
    output.mkdir(parents=True,exist_ok=False)
    rows = []
    torch.set_num_threads(1)
    for folder,task,label in [(base,0,'Policy pick'),(probes,3,'Force-pulse probe')]:
        cfg = json.loads((folder/'dataset.json').read_text())['config']
        m = json.loads(Path(cfg['policy_manifest']).read_text())
        _, history = load_agent(m['checkpoint'])
        limits = history['settings']['grip_force_limits_n']
        used = 0
        for path in sorted(folder.glob('shard_*.npz')):
            with np.load(path) as shard:
                ids = shard['episode_id']
                chosen = np.flatnonzero(((ids//4)%10==9) & (shard['task_id']==task))
                for k in chosen[:count-used]:
                    seed, actions = int(shard['seed'][k]),shard['command'][k]
                    original = shard['state'][k,0]
                    for name,column,lo,hi in [('Gripper strength',3,.3,1.5),('COM z',9,-.01,.01)]:
                        trajectories = []
                        for setting in (lo,hi,lo):
                            params = shard['constants'][k].copy()
                            params[column] = setting
                            env = make_env(TASKS[task],cfg,limits)
                            try:
                                motion = PandaMotion(env,limits,cfg['finger_friction'])
                                motion.set_episode(params,seed)
                                np.testing.assert_allclose(original,motion.state(),atol=1e-6)
                                states = []
                                for action in actions:
                                    motion.step(action)
                                    states.append(motion.state())
                                trajectories.append(np.stack(states))
                            finally:
                                env.close()
                        diff = trajectories[1]-trajectories[0]
                        repeated = trajectories[2]-trajectories[0]
                        rows.append(dict(source=label,episode=int(ids[k]),seed=seed,change=name,
                            peak_width_difference_mm=float(1000*np.max(np.abs(diff[:,25]))),
                            peak_cube_position_difference_mm=float(1000*np.linalg.norm(diff[:,26:29],axis=1).max()),
                            repeat_width_mm=float(1000*np.abs(repeated[:,25]).max()),
                            repeat_cube_mm=float(1000*np.linalg.norm(repeated[:,26:29],axis=1).max())))
                    used += 1
            if used==count:
                break
        assert used==count
    write_csv(output/'paired_physics.csv',rows)
    summary = []
    for label in ['Policy pick','Force-pulse probe']:
        for change in ['Gripper strength','COM z']:
            group=[r for r in rows if r['source']==label and r['change']==change]
            summary.append(dict(source=label,change=change,n=len(group),
                median_peak_width_mm=float(np.median([r['peak_width_difference_mm'] for r in group])),
                median_peak_cube_position_mm=float(np.median([r['peak_cube_position_difference_mm'] for r in group])),
                repeat_median_peak_cube_mm=float(np.median([r['repeat_cube_mm'] for r in group])),
                repeat_max_peak_cube_mm=float(max(r['repeat_cube_mm'] for r in group))))
    write_json(output/'summary.json',summary)
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('base',type=Path)
    p.add_argument('probes',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    physical_replays(a.base,a.probes,a.output)
