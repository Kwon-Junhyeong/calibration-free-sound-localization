"""LOCATA Task 1, eigenmike: matching with the analytic rigid-sphere replicas of the array body and with free-field
replicas. Candidate directions every 2 degrees in azimuth and 3 degrees in elevation, band 312 to 7000 Hz."""
import numpy as np
from scorer import sphere_replicas, freefield_replicas, PairScorer, gc_err, uvec

# positions of capsules 0, 12, 14, 15, 18, 28, 29, 31 from the BEM model of the eigenmike (array frame, m)
MICS = np.array([[0, 0.039190594, 0.014969475], [-0.014969475, 0, 0.039190594],
                 [-0.03633649, -0.0009206058, -0.020967614], [-0.014969475, 0, -0.039190594],
                 [0, -0.039190594, -0.014969475], [0.014969475, 0, 0.039190594],
                 [0.03484692, 0, 0.023377791], [0.014969475, 0, -0.039190594]], np.float32).astype(float)

K = np.load("data/locata_blocks.npz")
f = K["freqs"]
band = (f >= 312) & (f <= 7000)
C = np.conj(K["csm"][:, :, :, band])           # replicas are e^{-iwt}
A, E = np.meshgrid(np.arange(0, 360, 2.0), np.arange(-90, 91, 3.0), indexing="ij")
cand = np.stack([A.ravel(), E.ravel()], 1)
U = uvec(cand[:, 0], cand[:, 1])

for name, replicas in (("rigid sphere", sphere_replicas), ("free field", freefield_replicas)):
    S = PairScorer(replicas(U, MICS, f[band]))
    est = cand[np.concatenate([S.residuals(C[b:b + 16]).argmin(1) for b in range(0, len(C), 16)])]
    e = gc_err(est[:, 0], est[:, 1], K["az"], K["el"])
    print(f"{name:>12s}: median error {np.median(e):.1f} deg over {len(e)} blocks", flush=True)
    del S
