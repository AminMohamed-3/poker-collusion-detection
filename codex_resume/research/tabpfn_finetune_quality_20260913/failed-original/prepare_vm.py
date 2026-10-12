"""Prepare the supplied VM from existing data; starts only the frozen supervisor job."""
from pathlib import Path
import hashlib,json,shutil,subprocess

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def metadata_from_existing(parent,freeze):
    copied=[]
    for m in sorted([x for x in parent['contexts'] if x['arm']=='T'],key=lambda x:(x['salt'],x['fold'],x['family'])):
        spec={'stem':m['stem'],'frame':m['frame'],'salt':m['salt'],'fold':m['fold'],'family':m['family'],'arm':'T','columns':m['columns'],'source_file':m['context_file'],'source_sha256':m['context_sha256'],'train_shape':m['train_shape'],'query_shape':m['query_shape'],'class_counts':m['class_counts'],'frames_reference_parent_unchanged':'frames/'+m['frame']+'.parquet'}
        copied.append({**spec,'context_file':'contexts/'+Path(m['context_file']).name,'context_sha256':m['context_sha256'],'byte_exact_parent_npz':True})
    return {'preflight_sha256':freeze['local_preflight_sha256'],'contexts':copied,'copied_context_count':45,'byte_exact_parent_npzs':True,'frames_reference_parent_unchanged':True,'no_new_models':True,'no_outer_query_labels_parsed':True,'no_metrics_or_inference':True}

def main():
    r=Path(__file__).resolve().parent;base=Path('/workspace/poker/codex');parent=base/'tabpfn3_relative_20260912'
    assert r==base/'tabpfn_finetune_quality_20260913'
    f=json.loads((r/'study-freeze.json').read_text())
    assert sha(parent/'context-receipt.json')==f['existing_remote_parent_context_receipt_sha256']
    ctx=metadata_from_existing(json.loads((parent/'context-receipt.json').read_text()),f)
    assert len(ctx['contexts'])==45
    (r/'contexts').mkdir(exist_ok=False)
    for m in ctx['contexts']:
        src=parent/m['context_file'];dst=r/m['context_file'];assert sha(src)==m['context_sha256']
        shutil.copy2(src,dst);assert sha(dst)==m['context_sha256']
    with (r/'context-receipt.json').open('xb') as fp:fp.write((json.dumps(ctx,indent=2)+'\n').replace('\n','\r\n').encode('utf-8'))
    shutil.copy2(parent/'run/predictions'/(f['replay_stem']+'.npy'),r/'baseline_expected.npy')
    cp=(parent/'checkpoint').resolve();assert cp.is_dir() and cp.is_relative_to(base)
    (r/'checkpoint').symlink_to(cp,target_is_directory=True)
    for n,h in f['files_sha256'].items():assert sha(r/n)==h,n
    assert sha(r/'checkpoint/tabpfn-v3-classifier-v3_default.ckpt')==f['original_checkpoint_sha256']
    script=r/'tabpfn3_finetune_vm.py';runtime=parent/'venv/bin/python'
    config=Path('/etc/supervisor/conf.d/codex-tabpfn-finetune-quality.conf');assert not config.exists() and not (r/'run').exists()
    command='/usr/bin/timeout --signal=TERM --kill-after=15s 3600 '+str(runtime)+' -u '+str(script)+' all --root '+str(r)+' --out '+str(r/'run')
    config.write_text('[program:codex_tabpfn_finetune_quality]\ncommand='+command+'\ndirectory='+str(r)+'\nautostart=false\nautorestart=false\nstartsecs=0\nstdout_logfile='+str(r/'quality.log')+'\nredirect_stderr=true\nstdout_logfile_maxbytes=0\n')
    (r/'launch-receipt.json').write_text(json.dumps({'study_freeze_sha256':sha(r/'study-freeze.json'),'supervisor_config_sha256':sha(config),'data_already_on_VM':True,'copied_contexts':45,'new_data_uploaded':False,'wallclock_limit_seconds':3600},indent=2))
    for cmd in [['supervisorctl','reread'],['supervisorctl','update','codex_tabpfn_finetune_quality'],['supervisorctl','start','codex_tabpfn_finetune_quality']]:
        q=subprocess.run(cmd,capture_output=True,text=True);assert q.returncode==0,(q.stdout,q.stderr)
    print(subprocess.run(['supervisorctl','status','codex_tabpfn_finetune_quality'],capture_output=True,text=True).stdout)
if __name__=='__main__':main()
