from pathlib import Path
import json,time,os,hashlib,sys,traceback,importlib.metadata
r=Path('/workspace/poker/codex/teacher_mechanics_20260912')
assert not (r/'download-result.json').exists()
os.environ.update(HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_XET='1',DO_NOT_TRACK='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')
for k in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACEHUB_API_TOKEN']:os.environ.pop(k,None)
meta={'id': 'Qwen/Qwen3.5-9B', 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a', 'license': 'apache-2.0', 'files': [{'name': '.gitattributes', 'size': 1570, 'lfs': None}, {'name': 'LICENSE', 'size': 11544, 'lfs': None}, {'name': 'README.md', 'size': 77643, 'lfs': None}, {'name': 'chat_template.jinja', 'size': 7756, 'lfs': None}, {'name': 'config.json', 'size': 3126, 'lfs': None}, {'name': 'merges.txt', 'size': 3353259, 'lfs': None}, {'name': 'model.safetensors-00001-of-00004.safetensors', 'size': 5276436216, 'lfs': {'sha256': 'db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e', 'size': 5276436216, 'pointerSize': 135}}, {'name': 'model.safetensors-00002-of-00004.safetensors', 'size': 5335161512, 'lfs': {'sha256': '31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330', 'size': 5335161512, 'pointerSize': 135}}, {'name': 'model.safetensors-00003-of-00004.safetensors', 'size': 5368717440, 'lfs': {'sha256': '7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec', 'size': 5368717440, 'pointerSize': 135}}, {'name': 'model.safetensors-00004-of-00004.safetensors', 'size': 3325995712, 'lfs': {'sha256': 'b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6', 'size': 3325995712, 'pointerSize': 135}}, {'name': 'model.safetensors.index.json', 'size': 79657, 'lfs': None}, {'name': 'preprocessor_config.json', 'size': 390, 'lfs': None}, {'name': 'tokenizer.json', 'size': 12807982, 'lfs': {'sha256': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42', 'size': 12807982, 'pointerSize': 133}}, {'name': 'tokenizer_config.json', 'size': 16710, 'lfs': None}, {'name': 'video_preprocessor_config.json', 'size': 385, 'lfs': None}, {'name': 'vocab.json', 'size': 6722759, 'lfs': None}]}
started=time.time()
try:
 import torch,bitsandbytes as bnb
 from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM
 torch.set_num_threads(2);torch.manual_seed(73911)
 cfg=Qwen3_5TextConfig(vocab_size=512,hidden_size=64,intermediate_size=128,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=16,linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=16,linear_value_head_dim=16,layer_types=['linear_attention','linear_attention','linear_attention','full_attention'],pad_token_id=0,bos_token_id=1,eos_token_id=2)
 model=Qwen3_5ForCausalLM(cfg).to(device='cuda',dtype=torch.bfloat16).eval()
 ids=torch.tensor([[1,3,4,5,6,7,8,9]],device='cuda')
 with torch.inference_mode():
  out=model(input_ids=ids);assert torch.isfinite(out.logits).all()
  gen=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=4,do_sample=False,pad_token_id=0)
 q=bnb.nn.Linear4bit(64,64,bias=False,compute_dtype=torch.bfloat16,quant_type='nf4',compress_statistics=True)
 q.weight=bnb.nn.Params4bit(torch.randn(64,64),requires_grad=False,quant_type='nf4',compress_statistics=True);q=q.cuda()
 with torch.inference_mode(): assert torch.isfinite(q(torch.randn(2,64,device='cuda',dtype=torch.bfloat16))).all()
 pre={'tiny_hybrid_forward_finite':True,'tiny_generate_shape':list(gen.shape),'nf4_cuda_finite':True,'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),'bitsandbytes':importlib.metadata.version('bitsandbytes'),'generation_seconds':time.time()-started}
 (r/'architecture-preflight.json').write_text(json.dumps(pre,indent=2)+'\n');print('ARCHITECTURE_PREFLIGHT_PASS',flush=True)
 del model,q,out,gen;torch.cuda.empty_cache()
 import shutil
 needed=sum(x['size'] for x in meta['files']);free=shutil.disk_usage(r).free
 assert free>needed+4_000_000_000,(free,needed)
 from huggingface_hub import snapshot_download
 local=r/'official_model';local.mkdir(exist_ok=True)
 snapshot_download(meta['id'],revision=meta['revision'],allow_patterns=[x['name'] for x in meta['files'] if x['name']!='.gitattributes'],local_dir=local,token=False,max_workers=2)
 def sha(path):
  h=hashlib.sha256()
  with path.open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  return h.hexdigest()
 files={}
 for f in meta['files']:
  if f['name']=='.gitattributes':continue
  path=local/f['name'];assert path.stat().st_size==f['size']
  digest=sha(path);files[f['name']]=digest
  if f['lfs']:assert digest==f['lfs']['sha256']
 license=(local/'LICENSE').read_text();assert 'Apache License' in license and 'Version 2.0' in license
 result={'exit_code':0,'seconds':time.time()-started,'model':meta['id'],'revision':meta['revision'],'files_sha256':files,'all_lfs_hashes_match':True,'bytes':sum((local/n).stat().st_size for n in files),'no_poker_inputs':True}
except Exception as exc:
 traceback.print_exc();result={'exit_code':1,'seconds':time.time()-started,'error':repr(exc),'no_poker_inputs':True}
(r/'download-result.json').write_text(json.dumps(result,indent=2)+'\n');print('DOWNLOAD_RESULT',json.dumps({k:v for k,v in result.items() if k!='files_sha256'}),flush=True)
sys.exit(result['exit_code'])
