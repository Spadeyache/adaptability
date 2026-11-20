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
            plans.append(dict(cube_id=cid,episode_id=ep,cap_n=float(settings.uniform(4.,7.)),
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
        cap_range_n=[4.,7.],launch_speed_threshold_m_s=.02,launch_fraction_target=.9,
        combined_friction='(cube material friction + floor material friction)/2: verified PhysX default eAVERAGE; per-cube values only cubes.json',
        randomization='Separate independent RNG streams for property sampling and every (cube,episode) plan. No rejection based on physics labels.',
        physics_dt_s=.001,model_dt_s=.005,push_s=.08,coast_s=1.,
        fixed_lengths='Raw diagnostic horizon:16 push+200 coast model intervals. Saved episodes retain all16 push and coast through ceil(stop onset/model dt)+5 additional steps. Trim length, masks, stop-time and IDs are excluded from model features.',
        trim_after_stop_steps=5,
        scene='Six-DOF 4 cm cube; infinite plane; 5 g spherical pusher free in XYZ, rotations locked, gravity and damping off. No Panda controller.',
        force='Full 3D contact force = commanded force minus known pusher mass times measured delta-velocity/.001s. No pusher translational constraints/gravity/damping/other contacts. Reject norm above cap*(1+1e-5). Normal-only PhysX API retained as diagnostic.',
        pilot_revision=4,
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
            score_groups=['push','moving_coast','all_steps'],
            ten_step='Recursive10-step predictions, no observed intermediate states. Group by starting transition:push,moving_coast,all_steps; push/moving-coast windows stay in their phase. Commands after a trimmed file ends are extended from the fixed preplanned coast command, never stop metadata.',
            stopping_position='Euclidean predicted terminal XY error against recorded confirmed-rest XY. Rollouts from eligible starts in each score group use the fixed physical horizon1s after release, not observed trimmed length or true stop time; zero-force coast commands are extended from the plan.',
            coast_masks='Moving coast:phase coast AND observed current full linear speed>.001m/s OR angular speed>.05rad/s, with the phase test applying to both. All steps:all retained push+coast. Report each group separately.',
            primary_gate='Moving-coast one-step NMSE:mean Oracle <= .8*mean Plain AND Oracle mean+sampleSD < Plain mean-sampleSD across five paired seeds. Push,moving-coast,all-step scores required for every comparison; no all-coast substitution.',
            seed_spread='Sample SD over five seeds; also paired seedwise improvements. No test-driven tuning.',
            no_belief_model=True),
        action_probe='Rerun on actual ten-cube pilot:pooled leave-one-cube-out R2 using fixed16-step push-command prefixes. Separately diagnose outcome-dependent trimmed length with length-only and commands+length probes. Supplement with planned200/30 generator audit; no extra motion.',
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
