"""Pilot then full collection under the user's measured continuation condition."""
import argparse
import json
import time
from clean.shove_gate.protocol import ROOT,initialize,write_new,digest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pilot','full'],default='pilot');args=parser.parse_args()
    started=time.perf_counter();split=initialize()
    import numpy as np
    from clean.shove_gate.simulation import simulate,measure,trim_episode,STATE_COLUMNS
    from clean.shove_gate.script import ACTION_COLUMNS
    out=ROOT/'results';out.mkdir(exist_ok=True)
    if (out/'STOP.json').exists():raise RuntimeError('Existing STOP.json; investigate before resuming')
    if args.stage=='full':
        if not json.loads((out/'pilot_gate.json').read_text())['passed']:raise RuntimeError('Pilot condition failed')
        if json.loads((out/'pilot_audit.json').read_text())['status']!='passed':raise RuntimeError('Pilot audit failed')
    original=ROOT.parent
    sources=[ROOT/n for n in ['script.py','protocol.py','simulation.py','collect.py']]+[original/'native.py']
    sources += [original/'data'/n for n in ['cubes.json','split.json','protocol.json','split_seal.json']]
    sources += [original/'shove/data'/n for n in ['cubes.json','trials.json','protocol.json','seal.json']]
    seal={str(p.relative_to(original)):digest(p) for p in sources}
    if (out/'source_seal.json').exists():
        if json.loads((out/'source_seal.json').read_text())!=seal:raise RuntimeError('Sources changed after collection began')
    else:write_new(out/'source_seal.json',seal)
    if not (ROOT/'data/schema.json').exists():write_new(ROOT/'data/schema.json',dict(state=STATE_COLUMNS,action=ACTION_COLUMNS,phase=['push','coast']))
    cubes=json.loads((ROOT/'data/cubes.json').read_text())['cubes'];plans=json.loads((ROOT/'data/plans.json').read_text())
    selected=[p for p in plans if args.stage=='full' or p['cube_id'] in split['pilot']]
    ledger=out/'collection_records.jsonl'
    rows=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    done={(r['cube_id'],r['episode_id']) for r in rows}
    for kind in ['episodes','raw']:(ROOT/'data'/kind).mkdir(exist_ok=True)
    for plan in selected:
        key=(plan['cube_id'],plan['episode_id'])
        if key in done:continue
        t=time.perf_counter();episode,raw=simulate(cubes[key[0]],plan)
        row=dict(cube_id=key[0],episode_id=key[1],**measure(raw,plan))
        episode,trim=trim_episode(episode,raw,row);row.update(trim)
        name=f'cube_{key[0]:03d}_episode_{key[1]:02d}.npz'
        for kind,arrays in [('episodes',episode),('raw',raw)]:
            path=ROOT/'data'/kind/name
            if path.exists():raise RuntimeError(f'Unindexed existing data:{path}')
            np.savez_compressed(path,**arrays);row[f'{kind}_sha256']=digest(path)
        row['wall_seconds']=time.perf_counter()-t
        with ledger.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        rows.append(row)
        failed=[]
        if not row['stopped']:failed.append('Not stopped inside raw coast')
        if row['max_coast_force_n']!=0:failed.append('Nonzero coast force')
        if row['max_tilt_deg']>5:failed.append('Tilt exceeds5degrees')
        if not .019<row['min_z_m']<=row['max_z_m']<.021:failed.append('Cube left flat surface')
        if failed:
            write_new(out/'STOP.json',dict(cube_id=key[0],episode_id=key[1],reasons=failed));raise RuntimeError(str(failed))
        if key[1]==9:print(json.dumps(dict(cube_id=key[0],episodes_complete=len(rows),last_coast_distance_m=row['stopping_distance_m'])),flush=True)
    picked=[r for r in rows if args.stage=='full' or r['cube_id'] in split['pilot']]
    expected=100 if args.stage=='pilot' else 2300
    if len(picked)!=expected:raise RuntimeError('Incomplete dataset')
    metric_name='pilot_metrics.json' if args.stage=='pilot' else 'metrics.json'
    if not (out/metric_name).exists():write_new(out/metric_name,picked)
    times=[r['wall_seconds'] for r in picked]
    timing=dict(stage=args.stage,collected_cubes=len({r['cube_id'] for r in picked}),collected_episodes=len(picked),
                transitions=sum(r['retained_steps'] for r in picked),untrimmed_transitions=sum(r['untrimmed_steps'] for r in picked),
                trimmed_transitions=sum(r['trimmed_steps'] for r in picked),this_invocation_wall_seconds=time.perf_counter()-started,
                episode_wall_seconds_sum=sum(times),mean_episode_wall_seconds=float(np.mean(times)),
                plain_oracle_training_started=False,belief_model_started=False)
    tname='pilot_timing.json' if args.stage=='pilot' else 'timing.json'
    if not (out/tname).exists():write_new(out/tname,timing)
    if args.stage=='pilot':
        fraction=sum(r['launch_speed_m_s']>.02 for r in picked)/100;distance=max(r['stopping_distance_m'] for r in picked)
        gate=dict(passed=fraction>=.9 and distance<1.,launch_fraction=fraction,launch_threshold_m_s=.02,maximum_coast_distance_m=distance,
                  conditions='launch>=.9 and maxdistance<1m',user_authorization='Explicit conditional direct-force follow-up')
        if not (out/'pilot_gate.json').exists():write_new(out/'pilot_gate.json',gate)
        print(json.dumps(gate),flush=True)
    for p in sources:
        if digest(p)!=seal[str(p.relative_to(original))]:raise RuntimeError('Sealed source changed')
    print(json.dumps(timing),flush=True)

if __name__=='__main__':main()
