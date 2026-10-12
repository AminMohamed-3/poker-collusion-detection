"""Freeze local runs and reuse previously audited pool-separated pair features."""
import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from model import Config,TinyCED

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; OLD=ROOT/'reports/mil-pair-experiment'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): p.write_text(json.dumps(x,indent=2)+'\n')

def freeze():
    path=HERE/'plan.json'; assert not path.exists()
    cfg=Config(); target=TinyCED(cfg).parameter_count()
    budget=min(range(cfg.expert_hidden,100),key=lambda h: abs(TinyCED(replace(cfg,connection='residual',expert_hidden=h)).parameter_count()-target))
    variants={n:asdict(replace(cfg,connection='residual' if n.startswith('residual') else n,expert_hidden=budget if n=='residual_budget' else cfg.expert_hidden)) for n in ['residual','residual_budget','mhc','single_pass']}
    counts={n:TinyCED(Config(**c)).parameter_count() for n,c in variants.items()}
    assert abs(counts['residual_budget']/target-1)<.01
    sources={p.name:sha(p) for p in [HERE/'model.py',HERE/'prepare.py',HERE/'train.py',HERE/'evaluate.py',HERE/'test_model.py']}
    protected=[ROOT/'outputs_v13_coverage/submission_conditional_coverage.csv',*sorted((ROOT/'outputs_v14_evidence_portfolio').glob('submission_*.csv')),ROOT/'outputs_v15_bucket/submission_bucket_risk_blend.csv']
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'source_hashes':sources,'official_revision':'dba1be0a40aa45a94ad051997016db3960a90277',
        'official_references':{p.name:sha(p) for p in (HERE/'references').iterdir() if p.suffix in ['.pdf','.py','.json','.md'] or p.name=='LICENSE'},
        'variants':variants,'parameters':counts,'seeds':[7329,39117,55193],
        'training':{'epochs':20,'batch_size':64,'learning_rate':.001,'weight_decay':.01,'gradient_clip':2.,'average_last_epochs':4,'threads':2,'dtype':'float32','optimizer':'AdamW','scheduler':'One epoch warmup followed by cosine decay to 1e-5','feature_scaling':'Training-only median/IQR, IQR floor .1; clip standardized inputs to [-8,8]'},
        'data':'Previously audited 897 gameplay pair features, full/70%/40% histories with equal view count per pair. Deterministic grouping into24 feature tokens, plus one readout token. No player, pool or hand identifier as an input; no hand-order claim.',
        'architecture_scope':'4 layers, 2 causal encoder +2 decoder. Decoder global KV computed once from final encoder state and reused, layer-local SWA. Dense short-context global attention, no CSA2 indexer/compressor, no quantization, no vision, no Engram, no DSpark, no custom fused kernels or full-scale efficiency claim. Small top-k MoE computes all4 experts densely then masks; shared expert and selection-only balance bias. New random weights; no pretrained downloads.',
        'comparison':'Same backbone initialization, data/order/optimizer/steps across modes per seed. Original mHC vs Single-Pass have identical parameter counts. Extra residual baseline matches total parameter count within1% by FFN width only. All outcomes retained.',
        'primary_architecture_test':{'candidate':'single_pass','control':'residual_budget','mean_held_logloss_improvement_min':.005,'mean_pair_ap_delta_min':-.002,'rule':'Compare paired seeds over full and shortened histories; report all six cases, no best-seed selection. Exploratory reused screen, not an independent estimate.'},
        'kaggle_gate':{'recipe':'Equal mean-of-three-seeds risk blend with v13 control; retain incumbent behavior/evidence. Pure network is diagnostic only.','known_required_gain':'min(.001, .2*(.8 - mean incumbent pair_behavior))','worst_pair_behavior_delta_min':-.003,'mean_sparse_ap_delta_min':0.,'confirmation':'If screen passes, require original omitted-family floor: mean sparse AP delta>=0 and worst>=-.05. No candidate is deployable without all known folds, coherent-history and audit confirmation.'},
        'progression':'Run all twelve local fold0 fits first. Only if a fixed mHC equal blend passes the incumbent gate, run that mode on all three existing omitted-family packages. Stop on a failure; leave all production artifacts untouched. No adaptive weight, seed, depth, or learning-rate search.',
        'protected':{p.relative_to(ROOT).as_posix():sha(p) for p in protected},'no_untouched_holdout':True}
    write(path,plan); print(json.dumps({'plan_sha256':sha(path),'parameters':counts}),flush=True)

def prepare(job):
    plan=json.loads((HERE/'plan.json').read_text()); assert all(sha(HERE/p)==h for p,h in plan['source_hashes'].items())
    old=OLD/job; audit=json.loads((old/'audit.json').read_text()); package=old/'package'
    assert sha(package/'worker-plan.json')==audit['worker_plan_sha256'] and sha(old/'metadata.json')==audit['metadata_sha256']
    worker=json.loads((package/'worker-plan.json').read_text()); assert sha(package/'arrays.npz')==worker['files']['arrays.npz']
    assert audit['no_player_or_pool_overlap'] and not audit['held_labels_in_package']
    assert all(sha(ROOT/p)==h for p,h in audit['sources'].items())
    meta=json.loads((old/'metadata.json').read_text()); tr=meta['train_full']['pairs']; he=meta['full']['pairs']
    assert not {p['table_id'] for p in tr}&{p['table_id'] for p in he}
    assert not {p['pair_id'] for p in tr}&{p['pair_id'] for p in he}
    dest=HERE/job; dest.mkdir(exist_ok=False)
    oldarrays=np.load(package/'arrays.npz',allow_pickle=False)
    arrays={n:oldarrays[n] for n in ['train_pair','train_y','full_pair','shortened_pair']}
    for x in ['train_pair','full_pair','shortened_pair']: assert arrays[x].shape[1]==897 and np.isfinite(arrays[x]).all()
    np.savez_compressed(dest/'train-arrays.npz',**arrays)
    # The trainer never opens this file: held labels are evaluation-only.
    local={tag:{k:meta[tag][k] for k in ['pairs','control_risk','control_roles']} for tag in ['full','shortened']}
    write(dest/'held-metadata.json',local)
    receipt={'plan_sha256':sha(HERE/'plan.json'),'parent_audit_sha256':sha(old/'audit.json'),'original_arrays_sha256':sha(package/'arrays.npz'),
             'train_arrays_sha256':sha(dest/'train-arrays.npz'),'held_metadata_sha256':sha(dest/'held-metadata.json'),'train_pairs':len(tr),'held_pairs':len(he),'train_pools':len({p['table_id'] for p in tr}),'held_pools':len({p['table_id'] for p in he}),'no_pool_or_player_overlap':True,'train_shapes':{k:list(v.shape) for k,v in arrays.items()},'held_labels_absent_from_training_arrays':True}
    write(dest/'data-audit.json',receipt); print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--freeze',action='store_true'); p.add_argument('--job'); args=p.parse_args()
    freeze() if args.freeze else prepare(args.job)
