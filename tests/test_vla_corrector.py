import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from lingbotvla.corrector.runtime import quantize_tokens, dequantize_tokens, build_mlp, guide_velocity, cosine_error, online_guidance
from lingbotvla.corrector.dataset import DynamicsDataset
from lingbotvla.corrector.monitor import RobustThresholdState, robust_threshold_step


class CorrectorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)

    def test_quantization_roundtrip_and_zero(self):
        z = np.random.default_rng(42).normal(size=(3,4,8)).astype(np.float32)
        q,s = quantize_tokens(z)
        np.testing.assert_allclose(dequantize_tokens(q,s),z,atol=0.02)
        q,s = quantize_tokens(np.zeros_like(z))
        self.assertTrue(np.isfinite(dequantize_tokens(q,s)).all())

    def test_episode_pairs_do_not_cross_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            paths=[]
            for e in range(2):
                p=Path(folder)/str(e);p.mkdir();paths.append(p)
                z=np.arange(14,dtype=np.float32)[:,None,None]+e*100
                q,s=quantize_tokens(z)
                for key,array in dict(z_q=q,z_scale=s,actions=np.ones((14,3)),mask=np.array([True,False,True])).items():
                    np.save(p/f'{key}.npy',array)
                (p/'complete.json').write_text(json.dumps(dict(frames=14)))
            data=DynamicsDataset(paths,10)
            self.assertEqual(len(data),8)
            for i in range(len(data)):
                np.testing.assert_allclose(data[i]['delta'],10,atol=0.1)
                self.assertEqual(float(data[i]['action'][1]),0)
            self.assertGreater(float(data[4]['z'][0,0]),99)

    def test_guidance_changes_only_valid_first_action_without_model_grads(self):
        model=build_mlp(dict(token_dim=6,action_dim=3,action_embed_dim=4,scale='custom',custom_widths=[8,8])).requires_grad_(False)
        velocity=torch.randn(1,5,3)
        x=torch.randn_like(velocity)
        z=torch.randn(1,4,6)
        mask=torch.tensor([[1,0,1]],dtype=torch.bool)
        guided,stats=guide_velocity(velocity,x,torch.tensor([0.5]),z,torch.randn_like(z),model,mask,eta=0.1)
        self.assertGreater(stats['gradient_norm'],0)
        self.assertFalse(torch.equal(guided[:,0],velocity[:,0]))
        torch.testing.assert_close(guided[:,1:],velocity[:,1:],rtol=0,atol=0)
        torch.testing.assert_close(guided[:,:,1],velocity[:,:,1],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_degenerate_motion_is_not_trigger(self):
        self.assertIsNone(cosine_error(torch.ones(1,3),torch.zeros(1,3)))
        self.assertAlmostEqual(cosine_error(torch.ones(1,3),torch.ones(1,3)),0,places=6)

    def test_threshold_bootstrap_then_sustained_change(self):
        s=RobustThresholdState()
        for _ in range(16):
            _,trigger=robust_threshold_step(0.1,s)
            self.assertFalse(trigger)
        events=[robust_threshold_step(1.5,s)[1] for _ in range(8)]
        self.assertTrue(any(events))

    def test_guidance_wrapper_restored_on_exception(self):
        class Core(torch.nn.Module):
            def predict_velocity(self,*args,**kwargs):
                return None
        core=Core();original=core.predict_velocity
        try:
            with online_guidance(core,None,None,None,None):
                raise RuntimeError('test')
        except RuntimeError:
            pass
        self.assertNotIn('predict_velocity',core.__dict__)
        self.assertEqual(core.predict_velocity,original)


if __name__=='__main__':
    unittest.main()
