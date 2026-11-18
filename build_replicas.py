"""Responses at the four microphones of the box, taken from the full-surface BEM table.

usage: python build_replicas.py BEM_TABLE.h5
"""
import sys
import h5py
import numpy as np

pos = np.array([[12, 6.86, 5.80], [12, -6.86, -5.80], [-12, 6.86, -5.49], [-12, -6.86, 5.49]]) / 100   # microphones, m
with h5py.File(sys.argv[1], "r") as f:
    cen = f["centroids"][:]
    elements = [int(np.argmin(np.linalg.norm(cen - p, axis=1))) for p in pos]
    P = np.stack([f["p"][e] for e in elements])                     # (mic, direction, freq)
    np.savez("data/box_replicas.npz", P=P.astype(np.complex64), grid=f["doa_grid"][:], freqs=f["freqs"][:], pos=pos)
