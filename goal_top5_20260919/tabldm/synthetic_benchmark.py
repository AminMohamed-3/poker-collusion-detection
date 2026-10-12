import gc, json, os, platform, subprocess, sys, threading, time
from pathlib import Path

import numpy as np
import psutil
import torch
from tabldm import TabLDMClassifier

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoint" / "clf_default.ckpt"
OUT = ROOT / "receipts" / "synthetic_runtime.json"
assert not OUT.exists(), "refuse overwrite"

stop = threading.Event()
gpu_samples = []
def poll_gpu():
    while not stop.is_set():
        try:
            s = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL,
            )
            for line in s.splitlines():
                pid, mib = [x.strip() for x in line.split(",")]
                if int(pid) == os.getpid(): gpu_samples.append(int(mib))
        except Exception:
            pass
        stop.wait(.25)

def rss_gib(): return psutil.Process().memory_info().rss / 2**30

def case(n_train, do_chunk_check):
    rng = np.random.default_rng(35935 + n_train)
    xtr = rng.normal(size=(n_train,128)).astype(np.float32)
    y = np.arange(n_train, dtype=np.int64) % 4
    rng.shuffle(y)
    xq = rng.normal(size=(256,128)).astype(np.float32)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0=time.perf_counter()
    clf=TabLDMClassifier(n_estimators=4,batch_size=1,random_state=35935,
                         enhance_candidates=False,
                         model_path=CKPT,allow_auto_download=False)
    clf.fit(xtr,y); tfit=time.perf_counter()-t0
    t1=time.perf_counter(); full=clf.predict_proba(xq); tfull=time.perf_counter()-t1
    rec={"n_train":n_train,"n_features":128,"n_query":256,"fit_seconds":tfit,
         "predict_full_seconds":tfull,"total_seconds":time.perf_counter()-t0,
         "output_shape":list(full.shape),"finite":bool(np.isfinite(full).all()),
         "row_sum_max_abs_error":float(np.max(np.abs(full.sum(1)-1))),
         "torch_peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
         "torch_peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,
         "rss_gib_after":rss_gib(),"parameter_count":sum(p.numel() for p in clf.model_.parameters())}
    if do_chunk_check:
        t2=time.perf_counter(); chunk=np.concatenate([clf.predict_proba(xq[i:i+64]) for i in range(0,256,64)])
        rec["predict_chunks64_seconds"]=time.perf_counter()-t2
        rec["full_vs_chunk64_max_abs_diff"]=float(np.max(np.abs(full-chunk)))
        rec["full_vs_chunk64_mean_abs_diff"]=float(np.mean(np.abs(full-chunk)))
        rec["full_vs_chunk64_allclose_1e_6"]=bool(np.allclose(full,chunk,rtol=1e-6,atol=1e-6))
    del clf,full,xtr,y,xq; gc.collect(); torch.cuda.empty_cache()
    return rec

assert torch.cuda.is_available()
thr=threading.Thread(target=poll_gpu,daemon=True);thr.start()
started=time.time()
report={"scope":"synthetic only; no poker data or quality", "seed":35935,
 "classifier":{"n_estimators":4,"batch_size":1,"enhance_candidates":False,
               "other_parameters":"official defaults; base-path fallback after default enhanced path exceeded cap"},
 "runtime":{"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,
            "cuda_device":torch.cuda.get_device_name(0)},"cases":[]}
try:
    first=case(1000,True); report["cases"].append(first)
    first_pass=(first["finite"] and first["output_shape"]==[256,4]
                and first["torch_peak_reserved_gib"]<3 and first["total_seconds"]<300)
    report["first_pass"]=bool(first_pass)
    if first_pass and time.time()-started < 300:
        report["cases"].append(case(1860,False))
finally:
    stop.set();thr.join(timeout=2)
report["wall_seconds"]=time.time()-started
report["nvidia_smi_process_peak_mib"]=max(gpu_samples) if gpu_samples else None
report["under_3gib"]=(report["nvidia_smi_process_peak_mib"] is not None and report["nvidia_smi_process_peak_mib"]<3072
                     and all(x["torch_peak_reserved_gib"]<3 for x in report["cases"]))
OUT.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
