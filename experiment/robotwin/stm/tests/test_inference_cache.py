import sys,types,unittest
import torch,numpy as np
from test_temporal import tm,method

class CacheInferenceTests(unittest.TestCase):
 def fixture(self):
  sys.modules['lingbotvla.models.vla.lingbot_vla.temporal_memory']=tm
  fn=method('deploy/lingbot_vla_v2_policy.py','LingbotVLAv2Server','_temporal_inputs',{'torch':torch,'np':np})
  calls=[]
  def encode(x,g):calls.append(float(x[0,0,0]));return x*2,[x+1,x+2]
  class VLA(torch.nn.Module):
   def __init__(self):
    super().__init__();self.p=torch.nn.Parameter(torch.ones(1));self.model=types.SimpleNamespace(qwenvl_with_expert=types.SimpleNamespace(qwenvl=types.SimpleNamespace(visual=torch.nn.Linear(1,1)),embed_image=encode))
  obj=types.SimpleNamespace(config=types.SimpleNamespace(temporal_memory=True),vla=VLA(),_temporal_caches={})
  obj._prepare_model_input=lambda x:x['prepared']
  def obs(t,e='e',stream='a'):
   return dict(timestamp=t,episode_id=e,stream_id=stream,prepared=dict(images=torch.full((3,5,12),t),img_masks=torch.ones(3,dtype=torch.bool),image_grid_thw=torch.tensor([1,2,10]).expand(3,3)))
  def infer(o):return fn(obj,[o['prepared']],[o])
  return obj,calls,obs,infer
 def test_repeated_explicit_history_is_encoded_once(self):
  obj,calls,obs,infer=self.fixture();a,b=obs(.5),obs(0.);o=obs(1.);o['history_observations']=[a,b]
  first=infer(o);self.assertEqual(len(calls),3)
  repeated=infer(o);self.assertEqual(len(calls),3)
  for k in ('current_image_features','history_features','history_img_masks','history_ages'):torch.testing.assert_close(first[k],repeated[k],rtol=0,atol=0)
  m=tm.PatchTemporalMemory(12,8,2);torch.nn.init.normal_(m.up.weight)
  y=m(first['current_image_features'],first['history_features'],first['history_img_masks'],first['history_ages'])
  z=m(repeated['current_image_features'],repeated['history_features'],repeated['history_img_masks'],repeated['history_ages'])
  torch.testing.assert_close(y,z,rtol=0,atol=0)
  # Current t=.5 was not observed as current earlier, but history features still hit.
  newer=obs(1.05);newer['history_observations']=[a,b];infer(newer);self.assertEqual(len(calls),4)
 def test_dense_15fps_only_one_encoder_call_per_frame(self):
  obj,calls,obs,infer=self.fixture()
  for i in range(16):out=infer(obs(i/15))
  self.assertEqual(len(calls),16);self.assertTrue(out['history_img_masks'].all())
  torch.testing.assert_close(out['history_ages'],torch.tensor([[8/15,1.]]))
  torch.testing.assert_close(out['history_features'][0,0],torch.full((3,5,12),14/15))
 def test_episode_stream_and_clock_validation(self):
  obj,calls,obs,infer=self.fixture();infer(obs(1.));infer(obs(0.,'next'))
  out=infer(obs(1.,stream='b'));self.assertFalse(out['history_img_masks'].any())
  with self.assertRaises(ValueError):infer(obs(.5,stream='b'))
  with self.assertRaises(ValueError):infer(obs(float('nan')))
  out=infer(obs(0.,'third'));self.assertFalse(out['history_img_masks'].any());self.assertEqual(len(obj._temporal_caches['a'].encoded),1)
 def test_bounded_detached_and_clear(self):
  c=tm.TemporalFeatureCache(max_entries=20);c.begin_episode('e')
  f=torch.randn(3,5,12,requires_grad=True);mask=torch.ones(3,dtype=torch.bool);g=torch.ones(3,3)
  for i in range(100):c.put_encoded(i/60,f,mask,g,[f])
  self.assertLessEqual(len(c.encoded),20);self.assertFalse(c.get_encoded(99/60)[0].requires_grad)
  c.clear();self.assertFalse(c.encoded);self.assertIsNone(c.current_encoding)
 def test_masks_and_grid_validation(self):
  obj,calls,obs,infer=self.fixture();old=obs(.5);old['prepared']['img_masks'][1]=False
  o=obs(1.);o['history_observations']=[old,None];out=infer(o)
  self.assertFalse(out['history_img_masks'][0,0,1]);self.assertFalse(out['history_img_masks'][0,1].any())
  o['prepared']['image_grid_thw']=torch.tensor([1,5,4]).expand(3,3)
  with self.assertRaises(ValueError):infer(o)

if __name__=='__main__':unittest.main(verbosity=2)
