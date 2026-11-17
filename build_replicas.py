"""BEM replicas at the four microphones of the box, from the full-surface BEM table.

usage: python build_replicas.py BEM_TABLE.h5
"""
import sys
import time
import numpy as np, h5py

H5 = sys.argv[1]
pos = np.array([[12, 6.86, 5.80], [12, -6.86, -5.80], [-12, 6.86, -5.49], [-12, -6.86, 5.49]]) / 100   # microphones (m)
t0 = time.time()
with h5py.File(H5, "r") as f:
    cen = f["centroids"][:]; grid = f["doa_grid"][:]; freqs = f["freqs"][:]
    print("p chunks:", f["p"].chunks)
    el_idx = [int(np.argmin(np.linalg.norm(cen - p, axis=1))) for p in pos]
    print("nearest elements", el_idx, "distances (mm)", [round(1e3*np.linalg.norm(cen[e]-p),1) for e, p in zip(el_idx, pos)])
    P = np.stack([f["p"][e] for e in el_idx])            # (4, 2664, 257) engineering e^{+jwt}
print(f"read {P.shape} in {time.time()-t0:.1f}s")
np.savez("data/box_replicas.npz", P=P.astype(np.complex64), grid=grid, freqs=freqs, pos=pos, elements=el_idx)

