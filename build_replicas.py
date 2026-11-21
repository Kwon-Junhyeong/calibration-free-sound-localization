"""Extract box_24x20x16 replicas at the four microphone elements from the full-surface BEM table (5 deg grid),

usage: python build_replicas.py BEM_TABLE.h5"""
import sys
import time
import numpy as np, h5py

H5 = sys.argv[1]
pos = np.array([[ 0.12,  0.06862745,  0.05803921],
                [ 0.12, -0.06862745, -0.05803921],
                [-0.12,  0.06862745, -0.05490196],
                [-0.12, -0.06862745,  0.05490196]])
print("mic pos (m):\n", pos)
t0 = time.time()
with h5py.File(H5, "r") as f:
    cen = f["centroids"][:]; grid = f["doa_grid"][:]; freqs = f["freqs"][:]
    print("p chunks:", f["p"].chunks)
    el_idx = [int(np.argmin(np.linalg.norm(cen - p, axis=1))) for p in pos]
    print("nearest elements", el_idx, "distances (mm)", [round(1e3*np.linalg.norm(cen[e]-p),1) for e, p in zip(el_idx, pos)])
    P = np.stack([f["p"][e] for e in el_idx])            # (4, 2664, 257) engineering e^{+jwt}
print(f"read {P.shape} in {time.time()-t0:.1f}s")
np.savez("data/box_replicas_5deg.npz", P=P.astype(np.complex64), grid=grid, freqs=freqs, pos=pos, elements=el_idx)
