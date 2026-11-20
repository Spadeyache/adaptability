"""Authorized ten-cube pilot only. Full collection awaits the user's go."""
import json
import time
from pathlib import Path
from clean.shove_gate.protocol import ROOT,initialize,write_new,digest


def main():
    started=time.perf_counter()
    split=initialize()
    # Import and first scene timing are included in this recorded collection time.
    import numpy as np
    from clean.shove_gate.simulation import simulate,measure,STATE_COLUMNS
    from clean.shove_gate.script import ACTION_COLUMNS
    out=ROOT/'results';out.mkdir(exist_ok=True)
    if (out/'STOP.json').exists():raise RuntimeError('Pilot previously stopped: inspect STOP.json; do not resume past failure')
    original=ROOT.parent
    sources=[ROOT/n for n in ['script.py','protocol.py','simulation.py','collect.py']]+[original/'native.py']
    sources += [original/'data'/n for n in ['cubes.json','split.json','protocol.json','split_seal.json']]
    sources += [original/'shove/data'/n for n in ['cubes.json','trials.json','protocol.json','seal.json']]
    seal={str(p.relative_to(original)):digest(p) for p in sources}
    if (out/'source_seal.json').exists():
        if json.loads((out/'source_seal.json').read_text())!=seal:raise RuntimeError('Sources changed after collection began')
    else:write_new(out/'source_seal.json',seal)
    if not (ROOT/'data/schema.json').exists():write_new(ROOT/'data/schema.json',dict(state=STATE_COLUMNS,action=ACTION_COLUMNS,phase=['push','coast']))
    cubes=json.loads((ROOT/'data/cubes.json').read_text())['cubes']
    plans=json.loads((ROOT/'data/plans.json').read_text())
    pilot=[p for p in plans if p['cube_id'] in split['pilot']]
    assert len(pilot)==100 and max(p['cube_id'] for p in pilot)==9
    metrics=json.loads((out/'pilot_metrics.json').read_text()) if (out/'pilot_metrics.json').exists() else []
    done={(r['cube_id'],r['episode_id']) for r in metrics}
    for directory in ['episodes','raw']:(ROOT/'data'/directory).mkdir(exist_ok=True)
    run_times=[]
    for plan in pilot:
        key=(plan['cube_id'],plan['episode_id'])
        if key in done:continue
        t=time.perf_counter();episode,raw=simulate(cubes[plan['cube_id']],plan)
        row=dict(cube_id=plan['cube_id'],episode_id=plan['episode_id'],**measure(raw,plan))
        name=f'cube_{key[0]:03d}_episode_{key[1]:02d}.npz'
        for kind,arrays in [('episodes',episode),('raw',raw)]:
            path=ROOT/'data'/kind/name
            if path.exists():raise RuntimeError(f'Unindexed existing data: {path}')
            np.savez_compressed(path,**arrays);row[f'{kind}_sha256']=digest(path)
        row['wall_seconds']=time.perf_counter()-t;run_times.append(row['wall_seconds'])
        metrics.append(row)
        (out/'pilot_metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
        failed=[]
        if row['cap_ratio']>1.00001:failed.append('actual measured force cap exceeded')
        if not row['stopped']:failed.append('cube did not stop inside fixed coast horizon')
        if row['max_coast_contact_force_n']>1e-8:failed.append('pusher contact after release')
        if row['max_tilt_deg']>5:failed.append('cube tipped more than5degrees')
        if not .019<row['min_z_m']<=row['max_z_m']<.021:failed.append('cube left flat sliding surface')
        if failed:
            write_new(out/'STOP.json',dict(cube_id=key[0],episode_id=key[1],reasons=failed))
            raise RuntimeError(str(failed))
        print(json.dumps({k:row[k] for k in ['cube_id','episode_id','wall_seconds','stopping_distance_m','cap_ratio']}),flush=True)
    elapsed=time.perf_counter()-started
    if len(metrics)==100:
        times=[r['wall_seconds'] for r in metrics]
        timing=dict(collected_cubes=10,collected_episodes=100,transitions=21600,
                    this_invocation_wall_seconds=elapsed,episode_wall_seconds_sum=sum(times),
                    mean_episode_wall_seconds=float(np.mean(times)),median_episode_wall_seconds=float(np.median(times)),
                    total_2300_episode_projection_seconds=sum(times)*23,
                    remaining_2200_episode_projection_seconds=sum(times)*22,
                    projection='Linear extrapolation from measured serial episode collection+compression times; excludes training, audits, action probe and startup.',
                    approval_status='AWAITING_USER_GO',plain_oracle_training_started=False,belief_model_started=False)
        if not (out/'timing.json').exists():write_new(out/'timing.json',timing)
        print(json.dumps(timing,indent=2),flush=True)
    for path in sources:
        if digest(path)!=seal[str(path.relative_to(original))]:raise RuntimeError('Sealed source changed')


if __name__=='__main__':main()
