"""Run once, save every trace before validity checks, stop on unresolved signal."""
import argparse
import json
from pathlib import Path
import numpy as np
from clean.shove.protocol import ROOT,initialize,digest,write_json
from clean.shove.script import Settings
from clean.shove.simulation import simulate,measure


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-trials',type=int)
    args=parser.parse_args()
    trials=initialize()
    cubes=json.loads((ROOT/'data/cubes.json').read_text())['cubes']
    out=ROOT/'results';out.mkdir(exist_ok=True)
    raw=ROOT/'data/trajectories';raw.mkdir(exist_ok=True)
    # Pin originals and implementation before collection, without loading held-out motion.
    source_files=[ROOT/n for n in ['protocol.py','script.py','simulation.py','collect.py']]
    originals=[ROOT.parent/'data'/n for n in ['cubes.json','split.json','protocol.json','split_seal.json']]
    manifest={str(p.relative_to(ROOT.parent)):digest(p) for p in source_files+originals}
    if (out/'source_seal.json').exists():
        if json.loads((out/'source_seal.json').read_text())!=manifest:raise RuntimeError('Sources changed since collection')
    else:write_json(out/'source_seal.json',manifest)
    rows=json.loads((out/'metrics.json').read_text()) if (out/'metrics.json').exists() else []
    done={r['trial_id'] for r in rows}
    for trial in trials:
        if trial['trial_id'] not in done:
            if args.max_trials is not None and len(rows)>=args.max_trials:break
            settings=Settings(trial['cap_n'])
            trace=simulate(cubes[trial['cube_id']],settings)
            path=raw/f"trial_{trial['trial_id']:03d}.npz"
            if path.exists():raise RuntimeError('Existing raw trace without metrics; investigate instead of overwriting')
            np.savez_compressed(path,**trace)
            row={**trial,'sweep':cubes[trial['cube_id']]['sweep'],**measure(trace,settings),'trace_sha256':digest(path)}
            rows.append(row)
            (out/'metrics.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
            failures=[]
            if row['cap_ratio']>1.00001:failures.append('Actual contact force exceeds cap')
            if not row['stopped']:failures.append('Cube did not stop')
            if row['max_slide_contact_force_n']>1e-8:failures.append('Pusher contacts cube after release')
            if row['max_tilt_deg']>5:failures.append('Cube tipped >5 degrees')
            if not .019<row['min_cube_z_m']<=row['max_cube_z_m']<.021:failures.append('Cube not on plane')
            if failures:
                (out/'STOP.json').write_text(json.dumps({'trial':trial['trial_id'],'reasons':failures},indent=2)+'\n')
                raise RuntimeError(str(failures))
            print(json.dumps({k:row[k] for k in ['trial_id','cap_n','launch_speed_m_s','stopping_distance_m','max_contact_force_n']}),flush=True)
        subset=[r for r in rows if r['cap_n']==trial['cap_n']]
        if len(subset)==42:
            gates=[]
            for sweep in ['mass','friction']:
                group=[r for r in subset if r['sweep']==sweep]
                vals=[[r['stopping_distance_m'] for r in group if r['cube_id']==cid] for cid in sorted({r['cube_id'] for r in group})]
                effect=float(np.ptp(np.mean(vals,axis=1)));spread=float(np.max(np.ptp(vals,axis=1)))
                gates.append(dict(sweep=sweep,effect_m=effect,max_repeat_range_m=spread,passed=effect>spread))
            gate_path=out/f"gate_{trial['cap_n']:g}N.json"
            if not gate_path.exists():write_json(gate_path,gates)
            if not all(g['passed'] for g in gates):
                (out/'STOP.json').write_text(json.dumps({'strength':trial['cap_n'],'reason':'Dependence unresolved relative to repeat spread','gates':gates},indent=2)+'\n')
                break
    for p in originals:
        if digest(p)!=manifest[str(p.relative_to(ROOT.parent))]:raise RuntimeError('Original seal changed')
    print(f'Saved {len(rows)} trials',flush=True)


if __name__=='__main__':main()
