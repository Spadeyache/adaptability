"""Separate sealed probe definitions; original clean data are read-only."""
import hashlib
import json
from pathlib import Path
import numpy as np
from clean.shove.script import Settings

ROOT=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def initialize():
    data=ROOT/'data'
    if (data/'seal.json').exists():
        return verify()
    cubes=[]
    for sweep, values in [('mass',np.linspace(.04,.50,7)),('friction',np.linspace(.4,1.,7))]:
        for i,value in enumerate(values):
            cubes.append(dict(cube_id=len(cubes),sweep=sweep,grid_index=i,
                              mass_kg=float(value) if sweep=='mass' else .27,
                              friction=float(value) if sweep=='friction' else .7))
    write_json(data/'cubes.json',{'cubes':cubes})
    trials=[]
    # Mid strength first: stop at its completed sweeps if effect < repeat spread.
    for cap in [3.,2.,4.]:
        for cube in cubes:
            for repeat in range(3):
                trials.append(dict(trial_id=len(trials),cube_id=cube['cube_id'],cap_n=cap,repeat=repeat))
    write_json(data/'trials.json',trials)
    write_json(data/'protocol.json',dict(
        settings=Settings(3.).__dict__,caps_n=[2.,3.,4.],repeats=3,
        cube_side_m=.04,pusher_radius_m=.008,pusher_mass_kg=.005,
        floor_friction=.3,pusher_friction=0.,
        scene='SAPIEN CPU PhysX, free six-DOF cube on infinite plane; no Panda arm',
        force='Conservative sum of norms of all cube-pusher contact impulses / physics dt, every physics step. Reject > cap*(1+1e-5).',
        launch='Horizontal speed at release, before first free-slide step.',
        distance='Horizontal displacement from release to onset of confirmed rest; excluding distance while pushing.',
        stop='First post-release 100 ms continuously below 1 mm/s full linear speed and .05 rad/s angular speed; time at window onset excludes confirmation tail. Zero if already stationary at release.',
        spread='Maximum minus minimum across three fresh identical deterministic scene resets; no injected noise.',
        signal_gate='After each strength: range of seven mean stopping distances must not be smaller than max within-setting repeat range, for both parameter sweeps. Equality at zero is also unresolved. Stop and report if unresolved.',
        labels='True mass/friction only in cubes.json, read by collector property setter and offline reporting. Script gets Settings only.',
        source_provenance='clean.native.set_cube_properties reused; material coefficients and cube size match clean. Infinite surface, rail pusher, 1 kHz physics are explicit probe changes.',
        no_training=True))
    names=['cubes.json','trials.json','protocol.json']
    write_json(data/'seal.json',{n:digest(data/n) for n in names})
    return verify()


def verify():
    data=ROOT/'data'
    seal=json.loads((data/'seal.json').read_text())
    for name,expected in seal.items():
        if digest(data/name)!=expected:raise RuntimeError(f'Changed sealed probe file: {name}')
    return json.loads((data/'trials.json').read_text())
