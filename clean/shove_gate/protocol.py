"""Version5: direct force and a measured conditional pilot."""
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SEED=2026111509
PROTOCOL=dict(
    pilot_revision=5,
    stage='Same10cube pilot; if launch>=90% and maxcoast<1m, collect230cubes and runPlain/Oracle gate. Otherwise stop.',
    mass_range_kg=[.18,.50],cube_friction_range=[.4,1.],floor_material_friction=.3,
    force_range_n=[4.,7.],duration_range_s=[.060,.100],duration_grid_s=.005,
    launch_speed_threshold_m_s=.02,launch_fraction_target=.9,maximum_pilot_coast_distance_m=1.,
    combined_friction='(cube material friction+.3)/2; per-cube true values only cubes.json',
    randomization='Separate property/action RNGs. Force/direction/start/yaw preserve previous draws; duration uniform12..20modelsteps from separate stream. No label-based rejection.',
    physics_dt_s=.001,model_dt_s=.005,coast_s=1.5,settle_s=.1,trim_after_stop_steps=5,
    scene='Six-DOF4cm cube on infinite plane; no pusher. Direct horizontal COM force and zero external torque.',
    force='body.add_force_torque(applied_force,[0,0,0]); force mode, no mass conversion. Same vector reapplied each physicsstep during pulse;zero duringcoast.',
    state='Cube position3,rotation6(firsttwo matrixcolumns),linear velocity3,angular velocity3;15physical observations.',
    action='Applied worldforce XYZ only;zero duringcoast. No duration,time,IDs,phase,properties,stoptime or sequence-length inputs.',
    phases=['push','coast'],
    stop='First100ms continuously below full linear speed.001m/s and angular speed.05rad/s; onset excludesconfirmation.',
    trimming='Keep through ceil(stoprawindex/5)+5modelsteps. Full1.5srawcoast saved for confirmation. Length,masks,IDs,stoptime forbidden as modelinputs.',
    planned_models=dict(families=['Plain','Oracle'],seeds=[101,211,307,401,503],hidden_layers=[256,256,256],
        training_episodes=list(range(8)),validation_episodes=[8,9],updates=10000,batch_size=512,
        optimizer='Adam',learning_rate_initial=.001,learning_rate_final=.0001,schedule='cosine',
        validation_every_updates=500,selection='Minimum equalcube/episode all-step validationNMSE ontraincubes episodes8,9; freezeallmodels beforetest.',
        target='(next_state-state)/training-state std;floor.001 nativeunits',
        inputs=dict(Plain='state+action',Oracle='state+action+true mass+combined friction'),
        normalization='Onlytraincubes0..199 episodes0..7; propertynormalizer trainonly.',
        sampling='Uniformtrainingepisode then uniformtransition;pairedseed sample stream.'),
    planned_evaluation=dict(score_groups=['push','moving_coast','all_steps'],
        aggregation='Equalweight cubes,thenepisodes,theneligiblestartingtransitions. Reportemptygroupcounts.',
        one_step='MSE of normalizedstatechange over15cube coordinates.',
        ten_step='Recursive10stepforecast; mean normalizedstateMSE across10horizons. Push/movingcoastwindows staywithinphase. Fullrawstates supplytargetsaftertrim;no observedintermediate inputs.',
        stopping_position='Fromeligible starts rollout through plannedrelease+1.5s,never trimmedendpoint; Euclidean terminalXYerror againstconfirmedrestXY. Commands extendfromplan.',
        coast_masks='Movingcoast=phasecoast AND(normlinearvelocity>.001 OR normangularvelocity>.05). Allsteps=allretainedpush+coast.',
        primary_gate='Moving-coast one-stepNMSE:Oraclemean<=.8*Plainmean AND Oraclemean+sampleSD<Plainmean-sampleSD acrossfivepairedseeds.',
        seed_spread='SampleSD across5seeds; pairedimprovements reported.',
        final_test='Evaluateall10frozenmodels together once on30testcubes;no testdriven tuning.',
        no_belief_model=True),
    action_probe='Fixedfirst100ms force schedule includes independentlycommandedduration,never trimmedlength. PilotLOCO plusplanned200/30;separatelengthdiagnostic.'
)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def make_plan(cid,ep):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,cid,ep,200]))
    force=float(rng.uniform(4.,7.));direction=float(rng.uniform(-np.pi,np.pi))
    rng.uniform(-.006,.006)
    start=rng.uniform(-.10,.10,size=2).tolist();yaw=float(rng.uniform(-np.pi,np.pi))
    duration=np.random.default_rng(np.random.SeedSequence([SEED,cid,ep,300]))
    steps=int(duration.integers(12,21))
    return dict(cube_id=cid,episode_id=ep,force_n=force,direction_rad=direction,
                duration_steps=steps,duration_s=steps*.005,start_xy=start,yaw_rad=yaw)

def initialize():
    data=ROOT/'data'
    if (data/'seal.json').exists():return verify()
    if not (data/'cubes.json').exists():
        cubes=[]
        for cid in range(230):
            rng=np.random.default_rng(np.random.SeedSequence([SEED,cid,100]))
            mass=float(rng.uniform(.18,.50));mu=float(rng.uniform(.4,1.))
            cubes.append(dict(cube_id=cid,split='train' if cid<200 else 'test',mass_kg=mass,friction=mu,combined_friction=(mu+.3)/2))
        write_new(data/'cubes.json',{'cubes':cubes})
    if not (data/'split.json').exists():
        write_new(data/'split.json',dict(train=list(range(200)),test=list(range(200,230)),pilot=list(range(10)),episodes_per_cube=10,seed=SEED))
    write_new(data/'plans.json',[make_plan(c,e) for c in range(230) for e in range(10)])
    write_new(data/'protocol.json',PROTOCOL)
    write_new(data/'seal.json',{n:digest(data/n) for n in ['cubes.json','split.json','plans.json','protocol.json']})
    return verify()

def verify():
    data=ROOT/'data'
    for n,h in json.loads((data/'seal.json').read_text()).items():
        if digest(data/n)!=h:raise RuntimeError(f'Sealed file changed:{n}')
    if json.loads((data/'protocol.json').read_text())!=PROTOCOL:raise RuntimeError('Protocol code changed')
    return json.loads((data/'split.json').read_text())

if __name__=='__main__':initialize();print('Version5 protocol sealed')
