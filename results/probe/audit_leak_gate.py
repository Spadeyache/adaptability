"""Reproduce direct label leakage from saved files, without fitting a probe."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def main():
    path = ROOT/'runs/interaction_adaptation/initial_policy_probe/episodes/condition2_sequence0_true_attempt0.npz'
    with np.load(path,allow_pickle=False) as z:
        mass = float(z['label__mass_kg'])
        reported = float(z['input__reported_mass_kg'])
        weights = np.unique(z['sim__policy_observation'][:,42]).tolist()
        match = bool(np.allclose(z['sim__policy_observation'][:,42],mass*9.81,rtol=1e-6))
        evidence = [dict(path=str(path.relative_to(ROOT)),kind='true_mass_times_gravity_in_observation',
            true_mass_kg=mass,reported_mass_kg=reported,observation_field='sim__policy_observation',
            observation_shape=list(z['sim__policy_observation'].shape),zero_based_column=42,
            observed_values_n=weights,equals_true_mass_times_gravity=match,
            sources=['robot_adaptation/simulation/collection_lift.py:50-54',
                     'experiments/interaction_adaptation/evaluate.py:175-182'])]
    trace = ROOT/'runs/physics_conditioning/baseline_freshscene_diagnostic/traces.jsonl'
    with trace.open() as f:
        row = json.loads(next(f))
    episodes = json.loads(trace.with_name('episodes.json').read_text())
    paired = [e for e in episodes if e['seed']==row['seed'] and e['mode']==row['mode']]
    observed = np.asarray(row['observation'])
    masses = np.asarray([e['mass_kg'] for e in paired])
    frictions = np.asarray([e['friction'] for e in paired])
    matched_mass = bool(np.allclose(observed[:,42],masses*9.81,rtol=1e-5))
    matched_mu = bool(np.allclose(observed[:,45],frictions,rtol=1e-5))
    evidence.append(dict(path=str(trace.relative_to(ROOT)),kind='mass_and_friction_in_observation',
        label_path=str(trace.with_name('episodes.json').relative_to(ROOT)),seed=row['seed'],mode=row['mode'],
        batch_size=len(paired),observation_width=observed.shape[1],weight_column=42,friction_column=45,
        weight_equals_true_mass_times_gravity=matched_mass,friction_equals_true_friction=matched_mu,
        first_cube=dict(true_mass_kg=float(masses[0]),true_friction=float(frictions[0]),
                        input_weight_n=float(observed[0,42]),input_friction=float(observed[0,45])),
        sources=['experiments/physics_conditioning/evaluate.py:25-35',
                 'experiments/physics_conditioning/evaluate.py:113-130']))
    found = match or matched_mass or matched_mu
    report = dict(status='STOPPED_BEFORE_TRAINING' if found else 'NO_LEAK_IN_CHECKED_EXAMPLES',
        direct_leak_found=found,evidence=evidence,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        probe_runs=0,scope='These are exact examples, not an exhaustive all-file safety certification.')
    (ROOT/'results/probe/leak_gate.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
