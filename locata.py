"""LOCATA: the rigid-sphere model of the em32 against the free-field model."""
import time
import numpy as np
from scorer import sphere_replicas, freefield_replicas, PairScorer, load_task, gc_err, az_err, uvec

Z = load_task("eval", 1)
cand, freqs, mics = Z["cand"], Z["freqs"], Z["mics"]
U = uvec(cand[:, 0], cand[:, 1])
t0 = time.time(); M_sp = sphere_replicas(U, mics, freqs); print(f"sphere replicas {M_sp.shape} {time.time()-t0:.1f}s", flush=True)
S_sp = PairScorer(M_sp); del M_sp
S_ff = PairScorer(freefield_replicas(U, mics, freqs))
print(f"scorer tables built {time.time()-t0:.1f}s", flush=True)

for task in (1, 3, 5):
    Z = load_task("eval", task)
    out = {}
    for name, S in (("sphere", S_sp), ("freefield", S_ff)):
        est = []
        t1 = time.time()
        for b in range(0, len(Z["csm"]), 16):
            J = S.residuals(Z["csm"][b:b + 16])
            est.append(cand[J.argmin(1)])
        dt = (time.time() - t1) / len(Z["csm"])
        est = np.concatenate(est)
        gc = gc_err(est[:, 0], est[:, 1], Z["gt_az"], Z["gt_el"])
        az = az_err(est[:, 0], Z["gt_az"])
        out[name] = (np.median(gc), np.mean(az[az <= 30]), 100 * np.mean(az > 30), dt * 1e3)
    stored = gc_err(Z["est_az"], Z["est_el"], Z["gt_az"], Z["gt_el"])
    print(f"task{task} n={len(Z['csm'])}  stored-BEM gc-med {np.nanmedian(stored):.1f}", flush=True)
    for k, (m, ca, miss, ms) in out.items():
        print(f"   {k:>9s}: gc-med {m:5.1f}  chal-az {ca:5.2f}  miss {miss:5.1f}%   {ms:6.1f} ms/frame (batch 16)", flush=True)

# timing: single frame
Z = load_task("eval", 1)
t1 = time.time()
for n in range(10):
    S_sp.residuals(Z["csm"][n:n + 1])
print(f"single-frame scorer: {(time.time()-t1)/10*1e3:.1f} ms/frame", flush=True)
