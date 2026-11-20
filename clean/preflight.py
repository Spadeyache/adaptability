"""Check installed simulator/planner compatibility before collecting episodes."""
import json,time,sys
import numpy as np
from pathlib import Path
from clean.native import make_env,execute
from clean.script import plan_episode,sample_settings
from clean.schema import observe

def main():
    out=Path(__file__).parent/'results'/('preflight_'+sys.argv[1]+'.json')
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(exist_ok=True);start=time.monotonic();env=None
    try:
        print('Creating CPU environment',flush=True);env=make_env(820000)
        print('Reading explicit state',flush=True);x=observe(env,first=True)
        print('Planning without simulator feedback',flush=True)
        actions,phases=plan_episode(env.cube.pose.raw_pose[0].cpu().numpy(),sample_settings(0))
        print('Executing nominal diagnostic episode without reading object properties',flush=True)
        states=[x]
        for action in actions:
            execute(env,action)
            states.append(observe(env))
        states=np.asarray(states)
        path=out.with_suffix('.npz')
        with path.open('xb') as handle:
            np.savez_compressed(handle,state=states[:-1],action=actions,next_state=states[1:],phase=np.asarray(phases))
        speed=np.linalg.norm(states[:-1,39:42],axis=1)
        result=dict(status='pass',state_columns=len(x),action_columns=actions.shape[1],planned_steps=len(actions),elapsed_seconds=time.monotonic()-start,
            diagnostic_only=True,properties='Environment defaults; not read, recorded, or used for experiment',
            max_cube_height_m=float(states[:,32].max()),initial_cube_height_m=float(states[0,32]),
            max_height_gain_m=float(states[:,32].max()-states[0,32]),moving_speed_threshold_m_s=.005,
            moving_step_fraction=float(np.mean(speed>.005)),trajectory_path=str(path))
    except Exception as exc:
        result=dict(status='failure',error=repr(exc),elapsed_seconds=time.monotonic()-start)
    finally:
        if env is not None:env.close()
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
    if result['status']!='pass':raise SystemExit(1)

if __name__=='__main__':main()
