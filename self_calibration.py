"""Channel delays estimated from unlabeled blocks: the delays are the same in every block, the direction is not.

1. Per block: direction and per-pair lag with the delay-marginalized scorer.
2. Over N blocks: per-pair median, then least-squares delay per microphone.
3. Compensate the delays and score again. The true directions are used only for the error.
"""
import numpy as np
from box_common import (PairScorer, BEM, FF, C0, gt, f, E, LAGS, FS, refine, err, gauge, scores_flat)

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
A = np.zeros((6, 3))
for p, (i, j) in enumerate(PAIRS):            # delta_ij = d_i - d_j, unknowns d_1..d_3 (d_0 = 0)
    if i: A[p, i - 1] += 1
    if j: A[p, j - 1] -= 1


def stage1(S, C):
    lag, est = [], []
    for b in range(0, len(C), 8):
        T, w = S.frame_terms(C[b:b + 8])
        X = (w * T)[:, :, None, :] * S.MijH[None]
        num = np.abs(X @ E) ** 2
        den = np.einsum("bpf,pdf->bpd", w, S.Mabs2, optimize=True)
        tn = (w * np.abs(T) ** 2).sum(-1)
        J = (1 - num.max(-1) / (den * tn[:, :, None] + 1e-30)).mean(1)
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
        g = gauge(seed, mode)
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
