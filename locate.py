"""Directions of the recorded blocks, with the channel delays estimated from the same blocks.

usage: python locate.py [REPLICAS.npz] [BLOCKS.npz]
"""
import sys
import numpy as np
from scorer import PairScorer, refine, azimuth_elevation
from delays import FS, pair_lags, estimate_delays, compensate

R = np.load(sys.argv[1] if len(sys.argv) > 1 else "data/box_replicas.npz")
K = np.load(sys.argv[2] if len(sys.argv) > 2 else "data/box_blocks.npz")
band = (R["freqs"] >= 300) & (R["freqs"] <= 7000)
f = R["freqs"][band]
grid = R["grid"].astype(float)
S = PairScorer(R["P"][:, :, band].transpose(1, 0, 2))
C = K["csm"][:, :, :, band]

_, lags = pair_lags(S, C, f)
d = estimate_delays(lags, S.pairs)
az, el = azimuth_elevation(refine(grid, S.residuals(compensate(C, d, f))))
print("channel delays (samples):", np.round(d * FS, 2))
np.savetxt("directions.csv", np.column_stack([az, el]), fmt="%.1f", delimiter=",", header="azimuth,elevation")
print(f"{len(az)} directions saved to directions.csv")
