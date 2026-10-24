import ast, importlib.util, pathlib, unittest, types, sys
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
SRC=ROOT/'src'
# Test the production module without importing the full 6B model/dependencies.
p=SRC/'lingbotvla/models/vla/lingbot_vla/temporal_memory.py'
spec=importlib.util.spec_from_file_location('temporal_memory',p);tm=importlib.util.module_from_spec(spec);spec.loader.exec_module(tm)
torch.set_num_threads(2)

def method(path,cls,name,env):
 tree=ast.parse((SRC/path).read_text());c=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
 fn=next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name==name);fn.decorator_list=[]
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),env)
 return env[name]

class TemporalTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(7);self.m=tm.PatchTemporalMemory(12,8,2);self.c=torch.randn(2,3,5,12);self.h=torch.randn(2,2,3,5,12);self.valid=torch.ones(2,2,3,dtype=torch.bool);self.ages=torch.tensor([[.5,1.],[.533333,1.]])
 def test_zero_init_and_gradient(self):
  c=self.c.requires_grad_();y=self.m(c,self.h,self.valid,self.ages);torch.testing.assert_close(y,c,rtol=0,atol=0)
  y.square().sum().backward();self.assertGreater(self.m.up.weight.grad.abs().sum(),0)
  with torch.no_grad():self.m.up.weight.add_(self.m.up.weight.grad*.001)
  self.m.zero_grad();self.m(c,self.h,self.valid,self.ages).square().sum().backward();self.assertGreater(self.m.down.weight.grad.abs().sum(),0)
 def test_missing_nan_history_is_exact_baseline(self):
  torch.nn.init.normal_(self.m.up.weight);torch.nn.init.normal_(self.m.up.bias)
  y=self.m(self.c,self.h*float('nan'),torch.zeros_like(self.valid),self.ages)
  torch.testing.assert_close(y,self.c,rtol=0,atol=0)
 def test_no_cross_patch_or_view_leakage(self):
  torch.nn.init.normal_(self.m.up.weight)
  y=self.m(self.c,self.h,self.valid,self.ages);h=self.h.clone();h[0,0,1,2,0]+=9
  delta=(self.m(self.c,h,self.valid,self.ages)-y).abs().sum(-1)
  self.assertGreater(delta[0,1,2],0);delta[0,1,2]=0;self.assertEqual(delta.sum(),0)
 def test_invalid_slot_is_ignored(self):
  torch.nn.init.normal_(self.m.up.weight);self.valid[:,0]=False
  y=self.m(self.c,self.h,self.valid,self.ages);h=self.h.clone();h[:,0]=float('nan')
  torch.testing.assert_close(y,self.m(self.c,h,self.valid,self.ages),rtol=0,atol=0)
 def test_checkpoint_roundtrip(self):
  torch.nn.init.normal_(self.m.up.weight);n=tm.PatchTemporalMemory(12,8,2);n.load_state_dict(self.m.state_dict(),strict=True)
  torch.testing.assert_close(n(self.c,self.h,self.valid,self.ages),self.m(self.c,self.h,self.valid,self.ages),rtol=0,atol=0)
 def test_causal_timestamp_cache(self):
  cache=tm.TemporalFeatureCache();ref=self.c[0];mask=torch.ones(3,dtype=torch.bool)
  for i in range(16):cache.push(i/15,'a',ref+i,mask)
  h,m,age=cache.query(1.,ref,mask);self.assertTrue(m.all());torch.testing.assert_close(h[0],ref+7);torch.testing.assert_close(h[1],ref)
  torch.testing.assert_close(age,torch.tensor([8/15,1.]))
  cache.push(0.,'b',ref,mask);h,m,_=cache.query(0.,ref,mask);self.assertFalse(m.any())
  cache.push(3.,'b',ref,mask);_,m,_=cache.query(3.,ref,mask);self.assertFalse(m.any())
  with self.assertRaises(ValueError):cache.push(2.,'b',ref,mask)
 def test_cache_is_bounded_and_detached(self):
  cache=tm.TemporalFeatureCache(max_entries=20);ref=self.c[0].requires_grad_();mask=torch.ones(3,dtype=torch.bool)
  for i in range(100):cache.push(i*.01,'a',ref,mask)
  self.assertLessEqual(len(cache.entries),20);self.assertFalse(cache.entries[-1][1].requires_grad)
 def test_fps_quantization(self):
  self.assertEqual(tm.history_frame_offsets(15),(-8,-15));self.assertEqual(tm.history_frame_offsets(30),(-15,-30))
  for fps in (15,20,24,29.97,30,60):
   for n,s in zip(tm.history_frame_offsets(fps),(.5,1.)):
    self.assertGreaterEqual(-n/fps+1e-8,s);self.assertLess(-n/fps-s,1/fps+1e-8)
 def test_future_never_used_as_history(self):
  # Execute the actual production pad_and_concat with a minimal robot fixture.
  path='lingbotvla/data/vla_data/utils.py'
  fn=method(path,'FeatureTransform','pad_and_concat',{'torch':torch,'F':torch.nn.functional})
  obj=types.SimpleNamespace(feature_config=types.SimpleNamespace(images=['cam'],joints=[]),images=['cam'],model_config=types.SimpleNamespace(temporal_memory=True),use_future_image=True,states=[],actions=[],chunk_size=2)
  # Read only the image extraction block, before unrelated robot joint packing.
  tree=ast.parse((SRC/path).read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='FeatureTransform');f=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='pad_and_concat')
  cut=next(i for i,x in enumerate(f.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Tuple) for t in x.targets))
  f.body=f.body[:cut]+[ast.Return(value=ast.Tuple(elts=[ast.Name(id=x,ctx=ast.Load()) for x in ['images','history_images','future_images']],ctx=ast.Load()))]
  env={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),'images','exec'),env)
  frames=torch.stack([torch.full((3,2,2),i,dtype=torch.uint8) for i in [10,20,30,250]])
  c,h,future=env['pad_and_concat'](obj,{'cam':frames},True)
  self.assertEqual(c['cam'][0,0,0],10);self.assertEqual(h[0]['cam'][0,0,0],20);self.assertEqual(h[1]['cam'][0,0,0],30);self.assertEqual(future['cam'][0,0,0],250)
 def test_strict_base_loading_adapter_only(self):
  path=SRC/'lingbotvla/models/module_utils.py';tree=ast.parse(path.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='load_model_weights');fn.decorator_list=[]
  # Remove annotations from the extracted function; retain actual loading control flow.
  fn.returns=None
  for a in fn.args.args:a.annotation=None
  state={}
  class Model(torch.nn.Module):
   def __init__(self):
    super().__init__();self.config=types.SimpleNamespace(temporal_memory=True,tie_word_embeddings=False);self.base=torch.nn.Linear(3,3);self.model=torch.nn.Module();self.model.temporal_memory=tm.PatchTemporalMemory(12,8,2)
  def dispatch(m,n,t,*args):m.get_parameter(n).copy_(t)
  loader=types.SimpleNamespace(get_vlm_para_fullnames=lambda m:[],map_ckpt_key=lambda k,**kw:k)
  env=dict(torch=torch,_get_weight_loader=lambda m:loader,_load_state_dict=lambda _: [list(state.items())],tqdm=lambda x,**kw:x,os=__import__('os'),empty_cache=lambda:None,_dispatch_parameter=dispatch,_dispatch_buffer=lambda *a:None,logger=types.SimpleNamespace(info_rank0=lambda *a:None))
  exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'loader','exec'),env)
  m=Model();state.update({k:v.clone() for k,v in m.state_dict().items() if k.startswith('base.')})
  with torch.no_grad():env['load_model_weights'](m,'unused',init_device='cpu',post_training=True)
  for k,v in state.items():torch.testing.assert_close(m.get_parameter(k),v,rtol=0,atol=0)
  self.assertEqual(m.model.temporal_memory.up.weight.abs().sum(),0)
  state.pop('base.bias')
  with self.assertRaises(ValueError),torch.no_grad():env['load_model_weights'](m,'unused',init_device='cpu',post_training=True)

class WiringTests(unittest.TestCase):
 def test_visual_adapter_insertion_and_cache(self):
  path=SRC/'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py'
  tree=ast.parse(path.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='FlowMatchingV2')
  fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='embed_prefix')
  # Run the real complete visual path, stopping before language embedding.
  cut=next(i for i,x in enumerate(fn.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cfg' for t in x.targets))
  fn.body=fn.body[:cut]+[ast.Return(value=ast.Tuple(elts=[ast.Name(id=x,ctx=ast.Load()) for x in ['img_emb','deepstack_embs']],ctx=ast.Load()))]
  import einops
  env={'torch':torch,'einops':einops};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'prefix','exec'),env)
  calls=[]
  def encode(images, grid):
   calls.append(images.shape)
   return images, [images+1]
  obj=types.SimpleNamespace(config=types.SimpleNamespace(temporal_memory=True),qwenvl_with_expert=types.SimpleNamespace(embed_image=encode),temporal_memory=tm.PatchTemporalMemory(12,8,2),training=False)
  current=torch.randn(2,3,5,12);hist=torch.randn(2,2,3,5,12);grid=torch.tensor([1,2,10]).expand(2,3,3);masks=torch.ones(2,3,dtype=torch.bool)
  kw=dict(image_grid_thw=grid,history_images=hist,history_grid_thw=grid[:,None].expand(2,2,3,3),history_img_masks=masks[:,None].expand(2,2,3),history_ages=torch.tensor([[.5,1.],[.5,1.]]))
  out,ds=env['embed_prefix'](obj,current,masks,None,None,**kw)
  torch.testing.assert_close(out,current,rtol=0,atol=0);self.assertEqual(len(calls),3);torch.testing.assert_close(ds[0],current+1)
  calls.clear();kw.update(current_image_features=current,current_deepstack_features=[current+1],history_features=hist)
  out,_=env['embed_prefix'](obj,current,masks,None,None,**kw);self.assertEqual(len(calls),0)
  torch.testing.assert_close(out,current,rtol=0,atol=0)
  obj.config.temporal_memory=False;calls.clear()
  out,_=env['embed_prefix'](obj,current,masks,None,None,image_grid_thw=grid)
  self.assertEqual(len(calls),1);torch.testing.assert_close(out,current,rtol=0,atol=0)

 def test_client_buffer_uses_control_time_and_copies_frames(self):
  spec=importlib.util.spec_from_file_location('temporal_client',SRC/'deploy/temporal_client.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  requests=[];client=module.TemporalObservationClient(types.SimpleNamespace(infer=lambda x:requests.append(x)),stream_id='robot1')
  client.reset('episode1')
  obs={'image':torch.zeros(1)}
  for i in range(61):
   obs['image'][0]=i;client.observe(obs,i/60)
  client.infer();payload=requests[-1]
  self.assertEqual(payload['history_observations'][0]['timestamp'],.5)
  self.assertEqual(payload['history_observations'][1]['timestamp'],0.)
  self.assertEqual(payload['history_observations'][0]['image'].item(),30)
  client.reset('episode2');client.observe(obs,0.);client.infer()
  self.assertEqual(requests[-1]['history_observations'],[None,None])

 def test_server_explicit_history_and_stream_isolation(self):
  import numpy as np
  sys.modules['lingbotvla.models.vla.lingbot_vla.temporal_memory']=tm
  fn=method('deploy/lingbot_vla_v2_policy.py','LingbotVLAv2Server','_temporal_inputs',{'torch':torch,'np':np})
  class VLA(torch.nn.Module):
   def __init__(self):
    super().__init__();self.p=torch.nn.Parameter(torch.ones(1));visual=torch.nn.Linear(1,1)
    self.model=types.SimpleNamespace(qwenvl_with_expert=types.SimpleNamespace(qwenvl=types.SimpleNamespace(visual=visual),embed_image=lambda x,g:(x,[x+1])))
  obj=types.SimpleNamespace(config=types.SimpleNamespace(temporal_memory=True),vla=VLA(),_temporal_caches={})
  obj._prepare_model_input=lambda x:x['prepared']
  def obs(t,episode='e',stream='a'):
   return dict(timestamp=t,episode_id=episode,stream_id=stream,prepared=dict(images=torch.full((3,5,12),t),img_masks=torch.ones(3,dtype=torch.bool),image_grid_thw=torch.tensor([1,2,10]).expand(3,3)))
  old0,old1,current=obs(.5),obs(0.),obs(1.)
  current['history_observations']=[old0,old1]
  out=fn(obj,[current['prepared']],[current]);self.assertTrue(out['history_img_masks'].all());self.assertEqual(out['history_features'].shape,(1,2,3,5,12))
  bad=obs(.6);current['history_observations']=[bad,old1]
  with self.assertRaises(ValueError):fn(obj,[current['prepared']],[current])
  current=obs(0.,'new');out=fn(obj,[current['prepared']],[current]);self.assertFalse(out['history_img_masks'].any())
  other=obs(1.,stream='b');out=fn(obj,[other['prepared']],[other]);self.assertFalse(out['history_img_masks'].any())

if __name__=='__main__':unittest.main(verbosity=2)
