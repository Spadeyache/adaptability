import hashlib,json
from pathlib import Path
import numpy as np,torch
from experiments.world_model_baselines.data import load_rows
from experiments.world_model_baselines.model import load_checkpoint
root=Path('/home/ubuntu/faliure')
folder=root/'runs/baselines/B0/seed_0'
saved=json.loads((folder/'evaluation.json').read_text())
assert saved['kind']=='B0' and saved['seed']==0
assert hashlib.sha256((folder/'checkpoint.pt').read_bytes()).hexdigest()==saved['checkpoint_sha256']
for path,expected in json.loads((root/'results/baselines/implementation.json').read_text())['sha256'].items():
 assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected
rows=load_rows('test')
assert 'true_mass' not in rows and 'true_friction' not in rows
assert set(np.unique(rows['cube_id']))==set(range(200,230))
keys=np.column_stack([rows[k] for k in ('cube_id','task','episode_id')])
_,first,count=np.unique(keys,axis=0,return_index=True,return_counts=True)
end=np.empty(len(keys),int)
for i,n in zip(first,count):
 np.testing.assert_array_equal(rows['step'][i:i+n],np.arange(n))
 np.testing.assert_array_equal(keys[i:i+n],np.tile(keys[i],(n,1)))
 end[i:i+n]=i+n
selection=np.isin(rows['episode_id'],np.arange(5,10))
assert int(selection.sum())==63000
moving=np.flatnonzero(selection&rows['cube_moving'])
starts=np.array([i for i in moving if i+9<end[i]])
model,_=load_checkpoint(folder/'checkpoint.pt','cpu')
torch.set_num_threads(1)
state,action,nextstate=[torch.as_tensor(rows[k],dtype=torch.float32) for k in ('state','action','next_state')]
one_sse=[];true=[];terminal_sse=[];nonfinite=0
with torch.no_grad():
 for chunk in np.array_split(moving,max(1,int(np.ceil(len(moving)/512)))):
  x=(torch.cat([state[chunk],action[chunk]],1)-model.input_mean)/model.input_sd
  target=(nextstate[chunk]-state[chunk]-model.delta_mean)/model.delta_sd
  predicted=model.network(x)
  assert torch.isfinite(predicted).all()
  one_sse.extend((predicted.double()-target.double()).square().sum(1).tolist())
  true.extend(target.double().tolist())
 for chunk in np.array_split(starts,max(1,int(np.ceil(len(starts)/512)))):
  rollout=state[chunk].clone()
  for step in range(10):
   normalized=(torch.cat([rollout,action[chunk+step]],1)-model.input_mean)/model.input_sd
   delta=model.network(normalized)*model.delta_sd+model.delta_mean
   rollout=rollout+delta
   assert torch.isfinite(rollout).all()
  target=nextstate[chunk+9]
  terminal_sse.extend(((rollout.double()-target.double())/model.delta_sd.double()).square().sum(1).tolist())
one_sse=np.array(one_sse);true=np.array(true);terminal_sse=np.array(terminal_sse)
report={}
for code,name in [(-1,'overall'),(0,'lift'),(1,'slow_slide'),(2,'shove')]:
 m=np.ones(len(moving),bool) if code<0 else rows['task'][moving]==code
 r=np.ones(len(starts),bool) if code<0 else rows['task'][starts]==code
 truth=true[m]
 sse=float(one_sse[m].sum());sst=float(((truth-truth.mean(0))**2).sum())
 actual=dict(moving_rows=int(m.sum()),rollout_starts=int(r.sum()),one_step_mse=sse/(51*m.sum()),one_step_r2=1-sse/sst,one_step_sst=sst,rollout_10_mse=float(terminal_sse[r].mean()/51))
 expected=saved['evaluation']['overall'] if code<0 else saved['evaluation']['per_task'][name]
 for metric in ['moving_rows','rollout_starts']: assert actual[metric]==expected[metric]
 np.testing.assert_allclose(actual['one_step_sst'],expected['one_step_sst'],rtol=1e-12)
 report[name]=dict(cpu=actual,saved={k:expected[k] for k in actual},relative_difference={k:abs(actual[k]-expected[k])/abs(expected[k]) for k in ['one_step_mse','rollout_10_mse']},r2_absolute_difference=abs(actual['one_step_r2']-expected['one_step_r2']))
print(json.dumps(report,indent=2))
result=dict(passed=all(v['relative_difference']['one_step_mse']<.005 and v['relative_difference']['rollout_10_mse']<.01 and v['r2_absolute_difference']<.001 for v in report.values()),checked_cube_ids=list(range(200,230)),evaluation_episode_ids=list(range(5,10)),same_episode_endpoint_index='next_state[start+9]',one_step_rows=len(moving),rollout_starts=len(starts),literal_first5_moving_rows=int(np.sum(selection&rows['cube_moving']&(rows['step']<5))),normalization='51 standardized delta targets; featurewise centered SST',replay_device='cpu/highest',saved_device=saved['environment'],comparisons=report,checkpoint_sha256=saved['checkpoint_sha256'],audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(root/'results/baselines/independent_pilot_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
assert result['passed'], 'Numerical drift requires inspection'
