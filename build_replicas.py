"""Responses at the four microphones of the box, taken from the full-surface BEM table, on a 5 degree grid.

usage: python build_replicas.py BEM_TABLE.h5
"""
import sys
import h5py
import numpy as np

STEP = 5                                              # degrees; locate.py searches every direction kept here
pos = np.array([[12, 6.86, 5.80], [12, -6.86, -5.80], [-12, 6.86, -5.49], [-12, -6.86, 5.49]]) / 100   # microphones, m
with h5py.File(sys.argv[1], "r") as f:
    cen, grid = f["centroids"][:], f["doa_grid"][:]
    az = np.degrees(np.arctan2(grid[:, 1], grid[:, 0])) / STEP
    el = np.degrees(np.arcsin(np.clip(grid[:, 2], -1, 1))) / STEP
    keep = np.flatnonzero((np.abs(az - np.round(az)) < 1e-3) & (np.abs(el - np.round(el)) < 1e-3))
    elements = [int(np.argmin(np.linalg.norm(cen - p, axis=1))) for p in pos]
    P = np.stack([f["p"][e][keep] for e in elements])               # (mic, direction, freq)
    np.savez("data/box_replicas.npz", P=P.astype(np.complex64), grid=grid[keep], freqs=f["freqs"][:], pos=pos)
