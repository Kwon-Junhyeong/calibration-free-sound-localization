"""Channel delays estimated from unlabeled blocks of the box recordings, then the directions.
1. For each block, pair-ratio matching that also searches the delay of each pair picks the direction and keeps
   the delay of each pair at that direction.
2. The median of each pair's delay over N blocks, then least squares for one delay per microphone (microphone 1
   is the reference).
3. The delays are removed and the blocks are matched again without the delay search.
The true directions are used only for the error. The free-field responses are run as the control."""
import numpy as np
from common import (PairScorer, BEM, FF, C0, f, LAGS, FS, refine, err, channel_errors, scores_flat, delay_search)

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
A = np.zeros((6, 3))
for p, (i, j) in enumerate(PAIRS):            # delta_ij = d_i - d_j, unknowns d_1..d_3 (d_0 = 0)
    if i: A[p, i - 1] += 1
    if j: A[p, j - 1] -= 1


def stage1(S, C):
    """per block: the direction found with the delay search, and the delay of each pair at that direction."""
    lag, est = [], []
    for b in range(0, len(C), 8):
        J, num = delay_search(S, C[b:b + 8])
        d0 = J.argmin(1)
        lag.append(LAGS[num[np.arange(len(d0)), :, d0].argmax(-1)])          # (B, P) lag at the chosen direction
        est.append(d0)
    return np.concatenate(lag), np.concatenate(est)


def calibrate(lags):
    dpair = np.median(lags, 0)
    d, *_ = np.linalg.lstsq(A, dpair, rcond=None)
    return np.concatenate([[0.0], d])


def compensate(C, d):
    corr = np.exp(2j * np.pi * (d[:, None, None] - d[None, :, None]) * f[None, None, :])   # (4,4,F)
    return (C * corr[None]).astype(np.complex64)


rng = np.random.default_rng(7)
for mode, seeds in (("clean", (0,)), ("gp+d1.5", (0, 1, 2)), ("gp+d4", (0, 1, 2))):
    for seed in seeds:
        g = channel_errors(seed, mode)
        C = (g[None, :, None, :] * np.conj(g)[None, None, :, :] * C0).astype(np.complex64)
        dphi = np.angle(g[:, 1:] * np.conj(g[:, :-1])).mean(1)
        true_d = -dphi / (2 * np.pi * (f[1] - f[0])); true_d = true_d - true_d[0]
        for name, M in (("BEM", BEM), ("FF", FF)):
            if name == "FF" and mode == "gp+d1.5":
                continue
            S = PairScorer(M)
            lags, _ = stage1(S, C)
            line = []
            for N in (10, 30, 100, len(C)):
                pick = rng.choice(len(C), N, replace=False)
                d = calibrate(lags[pick])
                Cc = compensate(C, d)
                med, miss = err(refine(scores_flat(S, Cc)))
                derr = np.abs((d - true_d)[1:] * FS).max()
                line.append(f"N={N:>3d}: {med:5.1f}deg/{miss:4.1f}% (delay err {derr:4.2f} smp)")
            print(f"[{mode} s{seed}] {name:>3s} self-cal -> " + " | ".join(line), flush=True)
