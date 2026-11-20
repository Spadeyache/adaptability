"""Scientific contracts for Step2 fitting and adaptation; synthetic data only."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .models import (DynamicsModel, VARIANTS, SEEDS, Z_DIMS, TRAIN_CUBES,
    build_history, equal_episode_cube_mean, fit_normalizer, fit_zs,
    fit_mass_readout, load_episodes, model_state_digest, pack_episodes,
    predict_pack, require_collection_ready, select_latent_configuration,
    extension_decision, validation_score)


def normalization():
    return dict(state_mean=[0.]*15,state_std=[1.]*15,action_mean=[0.]*3,
                action_std=[1.]*3,context_mean=[0.]*2,context_std=[1.]*2,
                horizontal_velocity_delta_mse_scale=1.)


def episode(cube=0,ep=0,n=4,delta=(.2,.3)):
    state=np.zeros((n,15),dtype=np.float32)
    state[:,0]=np.arange(n); state[:,9]=.4
    nxt=state.copy(); nxt[:,9:11]+=np.asarray(delta,dtype=np.float32)
    action=np.tile(np.array([5.,0.,0.],dtype=np.float32),(n,1))
    return dict(cube_id=cube,episode_id=ep,state=state,action=action,next_state=nxt,
                phase=np.full(n,'push'))


class LinearLatent(nn.Module):
    variant='z2'; z_dim=2
    def __init__(self):
        super().__init__(); self.scale=nn.Parameter(torch.tensor(1.))
        self.embedding=nn.Embedding(180,2); nn.init.zeros_(self.embedding.weight)
    def forward(self,state,action,context=None,history=None,z=None):
        output=torch.zeros((len(state),15),device=state.device)
        output[:,9:11]=z*self.scale
        return output


class ModelsContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_history_excludes_current_target_and_future(self):
        ep=episode(n=15); norm=normalization()
        original=build_history([ep],norm)
        changed=copy.deepcopy(ep); changed['next_state'][8:]+=500.
        altered=build_history([changed],norm)
        np.testing.assert_array_equal(original[:9],altered[:9])
        self.assertFalse(np.array_equal(original[9],altered[9]))
        self.assertEqual(original.shape,(15,10,33))

    def test_history_padding_is_physical_zero_and_resets_at_episode_boundary(self):
        norm=normalization(); norm['action_mean']=[1.,2.,3.]; norm['action_std']=[2.,4.,6.]
        a=episode(n=12); b=episode(ep=1,n=3); b['state'][0,0]=99.
        history=build_history([a,b],norm)
        np.testing.assert_array_equal(history[0,:,15:18],np.tile([-.5,-.5,-.5],(10,1)))
        np.testing.assert_array_equal(history[0,:,18:],np.zeros((10,15)))
        np.testing.assert_array_equal(history[12,:,0],np.full(10,99.))
        self.assertFalse(np.array_equal(history[11],history[12]))

    def test_push_energy_equal_cube_episode_weights_and_zero_predictor(self):
        eps=[episode(0,0,1,(1.,1.)),episode(0,1,9,(3.,3.)),episode(1,0,1,(10.,10.))]
        props={c:np.array([.3,.5],dtype=np.float32) for c in TRAIN_CUBES}
        norm=fit_normalizer(eps,props)
        self.assertAlmostEqual(norm['horizontal_velocity_delta_mse_scale'],52.5,places=5)
        self.assertNotAlmostEqual(norm['horizontal_velocity_delta_mse_scale'],182/11)
        pack=pack_episodes(eps,norm,None,'cpu',include_history=False)
        values=pack['horizontal_true_delta'].square().mean(-1).numpy()/norm['horizontal_velocity_delta_mse_scale']
        score=equal_episode_cube_mean(values,pack['row_episode'],pack['cube_ids'],pack['masks']['push'])
        self.assertAlmostEqual(score['mean'],1.,places=12)

    def test_normalizer_rejects_validation_cube_and_checkpoint_episode(self):
        for cid,eid in [(180,0),(0,8),(260,0)]:
            with self.assertRaisesRegex(RuntimeError,'non-fitting'):
                fit_normalizer([episode(cid,eid)],{})

    def test_retired_and_fresh_motion_guards_precede_file_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for cid in [200,230,259]:
                with self.assertRaisesRegex(RuntimeError,'Retired'):
                    load_episodes(root,[cid],[0],allow_test=True)
            with self.assertRaisesRegex(RuntimeError,'Test motion'):
                load_episodes(root,[260],[0])
            with self.assertRaisesRegex(RuntimeError,'ledger'):
                load_episodes(root,[260],[0],allow_test=True)

    def test_stop_artifact_prevents_new_runs_before_other_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'results').mkdir()
            (root/'results/STOP.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'STOP artifact'):
                require_collection_ready(root)

    def test_architectures_and_zero_embedding(self):
        for variant in VARIANTS:
            model=DynamicsModel(variant)
            widths=[m.out_features for m in model.net if isinstance(m,nn.Linear)]
            self.assertEqual(widths,[256,256,256,15])
            if model.z_dim:
                self.assertEqual(model.embedding.weight.shape,(180,model.z_dim))
                self.assertEqual(int(torch.count_nonzero(model.embedding.weight)),0)
            if variant=='history':
                self.assertEqual(model.gru.hidden_size,64)
                self.assertEqual(model.gru.input_size,33)

    def test_labels_cannot_enter_nonoracle_forward(self):
        state=torch.randn(3,15); action=torch.randn(3,3); history=torch.randn(3,10,33)
        for variant in ['plain','history','z2']:
            model=DynamicsModel(variant).eval(); z=torch.randn(3,2) if variant=='z2' else None
            a=model(state,action,context=torch.zeros(3,2),history=history,z=z)
            b=model(state,action,context=torch.full((3,2),999.),history=history,z=z)
            self.assertTrue(torch.equal(a,b))
        with self.assertRaisesRegex(ValueError,'Oracle requires'):
            DynamicsModel('oracle')(state,action)

    def test_heldout_cube_cannot_route_training_embedding(self):
        pack=pack_episodes([episode(180,8)],normalization(),None,'cpu',include_history=False)
        with self.assertRaisesRegex(RuntimeError,'Held-out cube'):
            predict_pack(DynamicsModel('z2'),pack,slice(None))

    def test_checkpoint_selection_is_push_velocity_not_all_state(self):
        class Fixed(nn.Module):
            variant='plain'; z_dim=0
            def __init__(self,value):super().__init__();self.value=value
            def forward(self,state,action,**kwargs):return self.value.expand(len(state),-1)
        ep=episode(delta=(.2,.3)); pack=pack_episodes([ep],normalization(),None,'cpu',include_history=False)
        good=torch.zeros((1,15)); good[:,9:11]=torch.tensor([.2,.3]); good[:,:3]=100.
        bad=torch.zeros((1,15))
        a=validation_score(Fixed(good),pack); b=validation_score(Fixed(bad),pack)
        self.assertLess(a['primary'],b['primary'])
        self.assertGreater(a['secondary_all_step'],b['secondary_all_step'])

    def test_any_variant_extension_and_exact_threshold(self):
        def logs(value):
            return {v:[dict(update=i,validation_nmse=value if v=='history' and i>7500 else 1.)
                       for i in range(500,10001,500)] for v in VARIANTS}
        a=extension_decision(logs(.999),10000)
        self.assertTrue(a['extend']); self.assertTrue(a['variants']['history']['still_improving'])
        self.assertFalse(a['variants']['plain']['still_improving'])
        self.assertFalse(extension_decision(logs(.9991),10000)['extend'])

    def test_latent_fit_updates_only_z_and_uses_push_not_other_targets(self):
        model=LinearLatent(); before=model_state_digest(model)
        ep=episode(180,0,5,(.2,.3)); ep['next_state'][:,:3]=999.
        result=fit_zs(model,[ep],normalization(),torch.zeros(2),.03,200,'cpu')
        np.testing.assert_allclose(result['snapshots'][200][0],[.2,.3],atol=2e-4)
        self.assertEqual(before,model_state_digest(model))
        self.assertTrue(result['weights_unchanged'])
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_latent_vectors_do_not_share_support_gradients(self):
        a=[episode(180,0,3,(.2,.3)),episode(181,0,7,(-.4,.1))]
        b=copy.deepcopy(a); b[0]['next_state'][:,9]+=10.
        first=fit_zs(LinearLatent(),a,normalization(),torch.zeros(2),.03,40,'cpu')
        second=fit_zs(LinearLatent(),b,normalization(),torch.zeros(2),.03,40,'cpu')
        np.testing.assert_array_equal(first['snapshots'][40][1],second['snapshots'][40][1])
        self.assertFalse(np.array_equal(first['snapshots'][40][0],second['snapshots'][40][0]))

    def test_query_cannot_enter_latent_fitting(self):
        with self.assertRaisesRegex(RuntimeError,'Query episode'):
            fit_zs(LinearLatent(),[episode(180,5)],normalization(),torch.zeros(2),.03,50,'cpu')

    @staticmethod
    def grid():
        return [dict(z_dim=d,learning_rate=lr,updates=u,seed=s,
                     query=dict(mean=1.,cubes=20,episodes=100,
                                per_cube=[dict(cube_id=c,mean=1.) for c in range(180,200)]))
                for d in Z_DIMS for lr in [.01,.03,.1] for u in [50,100,200,400] for s in SEEDS]

    def test_global_grid_tie_rule_and_validation_cube_guard(self):
        rows=self.grid(); selected,_=select_latent_configuration(rows)
        self.assertEqual((selected['z_dim'],selected['updates'],selected['learning_rate']),(2,50,.01))
        rows[0]['query']['per_cube'][0]['cube_id']=260
        with self.assertRaisesRegex(RuntimeError,'reserved validation'):
            select_latent_configuration(rows)

    def test_mass_readout_is_affine_training_only(self):
        z=np.linspace(-1.,1.,180)[:,None]; labels={c:.3+.1*z[c,0] for c in TRAIN_CUBES}
        result=fit_mass_readout(TRAIN_CUBES,z,labels)
        np.testing.assert_allclose(result['coefficients'],[.3,.1],atol=1e-12)
        self.assertAlmostEqual(result['train_r2'],1.)
        with self.assertRaisesRegex(RuntimeError,'weight-fitting'):
            fit_mass_readout(list(range(180,200)),np.zeros((20,1)),{})

    def test_nonfinite_aggregation_is_not_silently_dropped(self):
        result=equal_episode_cube_mean([1.,np.inf],[0,1],[0,1])
        self.assertIsNone(result['mean']); self.assertEqual(result['nonfinite_starts'],1)


if __name__=='__main__': unittest.main()
