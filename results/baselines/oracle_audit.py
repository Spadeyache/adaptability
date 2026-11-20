"""Read-only B2 label/scaling/forward audit; no fitting or model changes."""
import hashlib,json
from pathlib import Path
import numpy as np,torch
from experiments.world_model_baselines.data import load_rows
ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'results/baselines'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1)
 split=json.loads((RESULTS/'split.json').read_text());protocol=json.loads((RESULTS/'protocol.json').read_text())
 directory=ROOT/protocol['data_directory'];held={tuple(v) for v in split['train_validation_episodes']}
 test=load_rows('test',oracle=True,directory=directory)
 for path in sorted((directory/'test').glob('cube_*.npz')):
  with np.load(path,allow_pickle=False) as z:
   cube=int(z['cube_id'][0]);which=test['cube_id']==cube
   assert int(which.sum())==len(z['cube_id'])
   for key in ('task','episode_id','step','true_mass','true_friction'):
    np.testing.assert_array_equal(test[key][which],z[key])
    if key.startswith('true_'):assert np.all(z[key]==z[key][0])
 labels=[];training_cubes=[]
 for path in sorted((directory/'train').glob('cube_*.npz')):
  with np.load(path,allow_pickle=False) as z:
   cube=int(z['cube_id'][0]);training_cubes.append(cube)
   fit=np.ones(len(z['cube_id']),bool)
   for task,episode in [(task,episode) for candidate,task,episode in held if candidate==cube]:
    fit[(z['task']==task)&(z['episode_id']==episode)]=False
   labels.append(np.column_stack((z['true_mass'][fit],z['true_friction'][fit])).astype(np.float32))
 assert training_cubes==list(range(200))
 labels=np.concatenate(labels).astype(np.float64);assert len(labels)==672000
 expected_mean=labels.mean(0).astype(np.float32)
 expected_sd=np.maximum(np.sqrt(np.maximum((labels**2).mean(0)-labels.mean(0)**2,0)),1e-4).astype(np.float32)
 moving=np.flatnonzero((test['episode_id']>=5)&test['cube_moving'])
 assert len(moving)==19980
 state,action,following=[torch.as_tensor(test[key],dtype=torch.float32) for key in ('state','action','next_state')]
 raw_oracle=torch.as_tensor(np.column_stack((test['true_mass'],test['true_friction'])),dtype=torch.float32)
 checks=[]
 for path in sorted((ROOT/'runs/baselines/B2').glob('seed_*/evaluation.json')):
  saved=json.loads(path.read_text());checkpoint=path.with_name('checkpoint.pt')
  assert sha(checkpoint)==saved['checkpoint_sha256']
  weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
  assert weights['kind']=='B2'
  model=weights['model'];np.testing.assert_array_equal(model['oracle_mean'].numpy(),expected_mean)
  np.testing.assert_array_equal(model['oracle_sd'].numpy(),expected_sd)
  assert model['network.0.weight'].shape==(256,61)
  output=[];targets=[]
  with torch.no_grad():
   for chunk in np.array_split(moving,int(np.ceil(len(moving)/1024))):
    observed=(torch.cat((state[chunk],action[chunk]),1)-model['input_mean'])/model['input_sd']
    known=(raw_oracle[chunk]-model['oracle_mean'])/model['oracle_sd']
    x=torch.cat((observed,known),1)
    # Direct linear layers deliberately bypass BaselineModel.forward/normalize_oracle.
    for layer in (0,2,4,6):
     x=torch.nn.functional.linear(x,model[f'network.{layer}.weight'],model[f'network.{layer}.bias'])
     if layer!=6:x=torch.relu(x)
    assert torch.isfinite(x).all()
    target=(following[chunk]-state[chunk]-model['delta_mean'])/model['delta_sd']
    output.append((x.double()-target.double()).square().sum(1).numpy());targets.append(target.double().numpy())
  errors=np.concatenate(output);truth=np.concatenate(targets)
  scores={}
  for code,name in [(-1,'overall'),(0,'lift'),(1,'slow_slide'),(2,'shove')]:
   mask=np.ones(len(moving),bool) if code<0 else test['task'][moving]==code
   y=truth[mask];mse=float(errors[mask].mean()/51);sst=float(((y-y.mean(0))**2).sum());r2=1-float(errors[mask].sum())/sst
   previous=saved['evaluation']['overall'] if code<0 else saved['evaluation']['per_task'][name]
   assert int(mask.sum())==previous['moving_rows']
   np.testing.assert_allclose(sst,previous['one_step_sst'],rtol=1e-12)
   np.testing.assert_allclose(mse,previous['one_step_mse'],rtol=.005,atol=1e-6)
   np.testing.assert_allclose(r2,previous['one_step_r2'],rtol=0,atol=.001)
   scores[name]=dict(cpu_mse=mse,saved_mse=previous['one_step_mse'],cpu_r2=r2,saved_r2=previous['one_step_r2'],moving_rows=int(mask.sum()),mse_relative_difference=abs(mse-previous['one_step_mse'])/previous['one_step_mse'])
  checks.append(dict(seed=saved['seed'],checkpoint_sha256=sha(checkpoint),evaluation_sha256=sha(path),scores=scores))
 result=dict(passed=True,completed_seeds_checked=[v['seed'] for v in checks],expected_seeds=list(range(5)),all_five_checked=len(checks)==5,
  test_cube_ids=list(range(200,230)),source_labels_row_aligned=True,training_cubes=training_cubes,
  normalization_training_rows=len(labels),oracle_mean=expected_mean.tolist(),oracle_sd=expected_sd.tolist(),normalizers_match_training_only_rows_exactly=True,
  direct_input='59 normalized observed state/action values followed by normalized true_mass and true_friction',
  one_step_replayed_directly=True,replay_device='CPU/highest',saved_device='CUDA/high',
  interpretation='This verifies the privileged labels and their use in B2 inference. It imposes no requirement that trained B2 outperform other models.',
  checks=checks,protocol_sha256=sha(RESULTS/'protocol.json'),split_sha256=sha(RESULTS/'split.json'),source_sha256=sha(__file__))
 (RESULTS/'oracle_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(passed=True,seeds=result['completed_seeds_checked'],mean=result['oracle_mean'],sd=result['oracle_sd'],max_mse_relative_difference=max(v['mse_relative_difference'] for c in checks for v in c['scores'].values())),indent=2))
if __name__=='__main__':main()
