"""New independent cube identities, fixed split and independent shove plans."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SEED=2026111509


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def initialize():
    data=ROOT/'data'
    if (data/'seal.json').exists():return verify()
    cubes=[];plans=[]
    for cid in range(230):
        physical=np.random.default_rng(np.random.SeedSequence([SEED,cid,100]))
        mass=float(physical.uniform(.18,.50));mu=float(physical.uniform(.4,1.))
        cubes.append(dict(cube_id=cid,split='train' if cid<200 else 'test',
                          mass_kg=mass,friction=mu,combined_friction=(mu+.3)/2))
        for ep in range(10):
            settings=np.random.default_rng(np.random.SeedSequence([SEED,cid,ep,200]))
            plans.append(dict(cube_id=cid,episode_id=ep,cap_n=float(settings.uniform(2.,4.)),
                              direction_rad=float(settings.uniform(-np.pi,np.pi)),
                              contact_offset_m=float(settings.uniform(-.006,.006)),
                              start_xy=settings.uniform(-.10,.10,size=2).tolist(),
                              yaw_rad=float(settings.uniform(-np.pi,np.pi))))
    write_new(data/'cubes.json',{'cubes':cubes})
    write_new(data/'plans.json',plans)
    write_new(data/'split.json',dict(train=list(range(200)),test=list(range(200,230)),
                                   pilot=list(range(10)),episodes_per_cube=10,seed=SEED))
    write_new(data/'protocol.json',dict(
        stage='Ten training cubes only; await user go before remaining collection and Plain/Oracle training.',
        mass_range_kg=[.18,.50],cube_friction_range=[.4,1.],floor_material_friction=.3,
        combined_friction='(cube material friction + floor material friction)/2: verified PhysX default eAVERAGE; per-cube values only cubes.json',
        randomization='Separate independent RNG streams for property sampling and every (cube,episode) plan. No rejection based on physics labels.',
        physics_dt_s=.001,model_dt_s=.005,push_s=.08,coast_s=1.,
        fixed_lengths='16 push + 200 coast transitions; full coast recorded even after rest. No stop-time, IDs, or sequence length as model features.',
        scene='Six-DOF 4 cm cube; infinite plane; 5 g spherical pusher free in XYZ, rotations locked, gravity and damping off. No Panda controller.',
        force='Full 3D contact force = commanded force minus known pusher mass times measured delta-velocity/.001s. No pusher translational constraints/gravity/damping/other contacts. Reject norm above cap*(1+1e-5). Normal-only PhysX API retained as diagnostic.',
        pilot_revision=2,
        state='Cube position3, rotation6 (first two matrix columns), linear velocity3, angular velocity3, pusher position3, pusher velocity3. All physical observations.',
        action='Commanded XYZ force, cap, engaged flag, preplanned XYZ release target. No measured force or motion-dependent target.',
        phases=['push','coast'],stop='First 100ms continuously below full linear speed .001m/s and angular speed .05rad/s; confirmation excluded. Rest metadata for evaluation only.',
        planned_models=dict(families=['Plain','Oracle'],seeds=[101,211,307,401,503],hidden_layers=[256,256,256],
                            target='(next_state-state)/training-state std; floor .001 native units; same train-only normalization for both families',
                            inputs=dict(Plain='state+action',Oracle='state+action+true mass+combined friction'),
                            normalization='fit exclusively on training-cube training episodes; no test normalization'),
        planned_evaluation=dict(
            aggregation='Equal weight cubes; within cube equal episodes; within episode equal eligible starts.',
            one_step='MSE of normalized predicted change over state columns. Also report cube-only state columns and physical position errors.',
            ten_step='Recursive 10-step predicted states under recorded commanded actions, never observed intermediate state. Windows entirely within the named phase.',
            stopping_position='Euclidean XY error at fixed terminal time relative to recorded confirmed stopping XY; recursively predict from first push state or first coast state, respectively.',
            coast_masks='Report all 200 coast steps and moving coast (observed current linear speed>.001m/s OR angular speed>.05rad/s).',
            primary_gate='Coast one-step NMSE, all coast steps, mean Oracle <= .8*mean Plain AND Oracle mean+sampleSD < Plain mean-sampleSD across five paired seeds. Moving-coast NMSE and 10-step/stopping errors also reported, never substituted for primary.',
            seed_spread='Sample SD over five seeds; also paired seedwise improvements. No test-driven tuning.',
            no_belief_model=True),
        action_probe='Generator-only held-out-cube check may use all230 preplanned command schedules without collecting their motion; ten-cube motion pilot cannot establish generalization by itself.',
    ))
    names=['cubes.json','plans.json','split.json','protocol.json']
    write_new(data/'seal.json',{n:digest(data/n) for n in names})
    return verify()


def verify():
    data=ROOT/'data'
    for name,expected in json.loads((data/'seal.json').read_text()).items():
        if digest(data/name)!=expected:raise RuntimeError(f'Sealed file changed: {name}')
    return json.loads((data/'split.json').read_text())


if __name__=='__main__':print(initialize())
