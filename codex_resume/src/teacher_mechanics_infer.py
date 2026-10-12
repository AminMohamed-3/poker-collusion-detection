"""Codex: fixed local Qwen teacher mechanics runner; no student or real data."""
from pathlib import Path
import argparse,hashlib,json,os,time,sys,socket,importlib.metadata
os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')
for key in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACEHUB_API_TOKEN']:os.environ.pop(key,None)
def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,d):
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,ensure_ascii=False);f.write('\n')
def block_network(*args,**kwargs):raise RuntimeError('Teacher inference is offline')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--requests',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--mode',choices=['hardware','pilot','replay'],required=True);a=ap.parse_args()
 assert not a.output.exists(), 'Each invocation has a new output directory; do not overwrite outcomes'
 a.output.mkdir(parents=True);start=time.time()
 source=Path(__file__).read_bytes();request_bytes=a.requests.read_bytes();requests=[json.loads(x) for x in request_bytes.decode().splitlines() if x.strip()]
 assert requests and len({r['request_id'] for r in requests})==len(requests)
 assert len(requests)<=24
 receipt=json.loads((a.root/'download-result.json').read_text());assert receipt['exit_code']==0 and receipt['all_lfs_hashes_match']
 assert receipt['model']=='Qwen/Qwen3.5-9B' and receipt['revision']=='c202236235762e1c871ad0ccb60c8ee5ba337b9a'
 for name,digest in receipt['files_sha256'].items():
  if not name.endswith('.safetensors'):assert sha((a.root/'official_model'/name).read_bytes())==digest
 import torch,bitsandbytes as bnb
 from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration,BitsAndBytesConfig
 socket.socket.connect=block_network;socket.create_connection=block_network
 torch.set_num_threads(2);torch.manual_seed(73911);torch.cuda.manual_seed_all(73911)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 tokenizer=AutoTokenizer.from_pretrained(a.root/'official_model',local_files_only=True,trust_remote_code=False)
 tensors=[]
 for req in requests:
  x=tokenizer.apply_chat_template(req['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt')
  assert x['input_ids'].shape[1]<=4096,(req['request_id'],x['input_ids'].shape)
  tensors.append(x)
 write(a.output/'inputs-frozen.json',{'source_sha256':sha(source),'requests_sha256':sha(request_bytes),'model_download_receipt_sha256':sha((a.root/'download-result.json').read_bytes()),'mode':a.mode,'rows':[{'request_id':r['request_id'],'input_tokens':int(x['input_ids'].shape[1]),'input_ids_sha256':sha(x['input_ids'].numpy().tobytes())} for r,x in zip(requests,tensors)],'settings':{'model_revision':receipt['revision'],'quantization':'NF4 double quantization; BF16 compute','device':'cuda:0','do_sample':False,'enable_thinking':False,'max_input_tokens':4096,'max_new_tokens':1024,'seed':73911,'use_cache':True,'attention':'sdpa','memory_guard_fraction':0.90}})
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
 torch.cuda.reset_peak_memory_stats()
 model=Qwen3_5ForConditionalGeneration.from_pretrained(a.root/'official_model',local_files_only=True,trust_remote_code=False,quantization_config=quant,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval()
 guard=int(torch.cuda.get_device_properties(0).total_memory*.9)
 assert torch.cuda.max_memory_reserved(0)<=guard, 'Loading exceeds memory guard'
 assert all(v in [0,'cuda:0',torch.device('cuda:0')] for v in model.hf_device_map.values()), 'No CPU/disk offload'
 package_names=['transformers','bitsandbytes','accelerate','torch','huggingface_hub','tokenizers','safetensors','numpy']
 write(a.output/'model-loaded.json',{'seconds':time.time()-start,'footprint_bytes':model.get_memory_footprint(),'peak_reserved':torch.cuda.max_memory_reserved(0),'guard_bytes':guard,'nf4_modules':sum(isinstance(m,bnb.nn.Linear4bit) for m in model.modules()),'packages':{n:importlib.metadata.version(n) for n in package_names},'quantization_config':model.config.quantization_config,'hf_device_map':{k:str(v) for k,v in model.hf_device_map.items()}})
 for index,(req,x) in enumerate(zip(requests,tensors)):
  x={k:v.to('cuda') for k,v in x.items()};beg=time.time()
  with torch.inference_mode():
   y=model.generate(**x,max_new_tokens=1024,do_sample=False,use_cache=True,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
  torch.cuda.synchronize();output=y[0,x['input_ids'].shape[1]:].tolist();peak=torch.cuda.max_memory_reserved(0)
  result={'request_id':req['request_id'],'request_sha256':sha(json.dumps(req,sort_keys=True,ensure_ascii=False).encode()),'input_tokens':int(x['input_ids'].shape[1]),'output_tokens':len(output),'output_ids':output,'raw_text':tokenizer.decode(output,skip_special_tokens=True),'ended_with_eos':bool(output and output[-1]==tokenizer.eos_token_id),'seconds':time.time()-beg,'peak_reserved':peak,'guard_pass':peak<=guard}
  write(a.output/f'{index:03d}.json',result)
  print(json.dumps({k:result[k] for k in ['request_id','input_tokens','output_tokens','ended_with_eos','seconds','peak_reserved']}),flush=True)
  assert peak<=guard,'Inference exceeds memory guard'
 write(a.output/'complete.json',{'requests':len(requests),'seconds':time.time()-start,'source_sha256':sha(source),'requests_sha256':sha(request_bytes),'all_outputs_frozen':True,'files_sha256':{p.name:sha(p.read_bytes()) for p in sorted(a.output.glob('*.json'))}})
if __name__=='__main__':main()
