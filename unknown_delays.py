"""Measured box with unknown channel gain, phase and delay added: body model, free-field model and SRP-PHAT."""
import numpy as np
from scorer import PairScorer, uvec

C_SND, FS = 343.0, 16000
R = np.load("data/box_replicas.npz"); K = np.load("data/box_blocks.npz")
grid, fr_all, pos = R["grid"].astype(float), R["freqs"], R["pos"]
band = (fr_all >= 300) & (fr_all <= 7000); f = fr_all[band]
k = 2 * np.pi * f / C_SND
BEM = R["P"][:, :, band].transpose(1, 0, 2)
FF = np.exp(1j * k[None, None] * (grid @ pos.T)[:, :, None])
_, first = np.unique(K["fid"], return_index=True)
sel = np.sort(np.concatenate([first, first + 1]))
C0 = K["csm"][sel][:, :, :, band]
gt = uvec(K["az"][sel].astype(float), K["el"][sel].astype(float))
LAGS = np.arange(-8, 8.01, 0.25) / FS                                   # candidate relative delays (s)
E = np.exp(2j * np.pi * f[:, None] * LAGS[None, :]).astype(np.complex64)   # (F, L)

# neighbour lists for quadratic refinement (within 8 deg)
cosg = grid @ grid.T
NB = [np.flatnonzero(cosg[d] >= np.cos(np.radians(8.0))) for d in range(len(grid))]


def refine(score, minimize=True):
    """quadratic fit of the score surface in the tangent plane of the best grid point."""
    s = score if minimize else -score
    out = np.empty((len(s), 3))
    for b in range(len(s)):
        d0 = int(s[b].argmin()); nb = NB[d0]; u = grid[d0]
        a = np.array([0, 0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0, 0])
        e1 = np.cross(u, a); e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
        x, y = grid[nb] @ e1, grid[nb] @ e2
        A = np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], 1)
        c, *_ = np.linalg.lstsq(A, s[b, nb], rcond=None)
        H = np.array([[2 * c[3], c[5]], [c[5], 2 * c[4]]])
        if np.all(np.linalg.eigvalsh(H) > 0):
            xy = np.linalg.solve(H, -c[1:3])
            if np.linalg.norm(xy) < np.radians(6):
                v = u + xy[0] * e1 + xy[1] * e2; out[b] = v / np.linalg.norm(v); continue
        out[b] = u
    return out


def err(est):
    e = np.degrees(np.arccos(np.clip((est * gt).sum(1), -1, 1)))
    return np.median(e), 100 * np.mean(e > 30)


def gauge(seed, mode):
    rng = np.random.default_rng(seed)
    a = 10 ** (rng.uniform(-6, 6, 4) / 20); ph = rng.uniform(-np.pi, np.pi, 4)
    dl = {"clean": 0, "gp": 0, "gp+d1.5": 1.5, "gp+d4": 4.0}[mode] * rng.uniform(-1, 1, 4) / FS
    if mode == "clean": a[:] = 1; ph[:] = 0
    return a[:, None] * np.exp(1j * ph[:, None]) * np.exp(-2j * np.pi * f[None, :] * dl[:, None])


def scores_flat(S, C):
    return np.concatenate([S.residuals(C[b:b + 64]) for b in range(0, len(C), 64)])


def scores_delay(S, C):
    out = []
    for b in range(0, len(C), 8):
        T, w = S.frame_terms(C[b:b + 8])
        X = (w * T)[:, :, None, :] * S.MijH[None]                             # (B,P,D,F)
        num = np.abs(X @ E) ** 2                                              # (B,P,D,L)
        den = np.einsum("bpf,pdf->bpd", w, S.Mabs2, optimize=True)
        tn = (w * np.abs(T) ** 2).sum(-1)
        out.append((1 - num.max(-1) / (den * tn[:, :, None] + 1e-30)).mean(1))
    return np.concatenate(out)


def srp(C):
    ii = np.array([0, 0, 0, 1, 1, 2]); jj = np.array([1, 2, 3, 2, 3, 3])
    st = np.conj(FF[:, ii] * np.conj(FF[:, jj])); st /= np.abs(st)
    G = C[:, ii, jj]; G = G / (np.abs(G) + 1e-20)
    return np.einsum("bpf,dpf->bd", G, st, optimize=True).real


SB, SF = PairScorer(BEM), PairScorer(FF)
rows = {}
for mode in ("clean", "gp", "gp+d1.5", "gp+d4"):
    res = {m: [] for m in ("BEM flat", "BEM +delay", "FF flat", "FF +delay", "SRP-PHAT")}
    for seed in ((0,) if mode == "clean" else (0, 1, 2)):
        g = gauge(seed, mode)
        C = (g[None, :, None, :] * np.conj(g)[None, None, :, :] * C0).astype(np.complex64)
        res["BEM flat"].append(err(refine(scores_flat(SB, C))))
        res["BEM +delay"].append(err(refine(scores_delay(SB, C))))
        res["FF flat"].append(err(refine(scores_flat(SF, C))))
        res["FF +delay"].append(err(refine(scores_delay(SF, C))))
        res["SRP-PHAT"].append(err(refine(srp(C), minimize=False)))
    print(f"\n[{mode}]  (median deg, miss%; mean over seeds)", flush=True)
    for m, v in res.items():
        v = np.array(v); print(f"   {m:>11s}: median {v[:,0].mean():6.1f}   miss {v[:,1].mean():5.1f}%", flush=True)
    rows[mode] = {m: np.array(v) for m, v in res.items()}
np.save("data/unknown_delays.npy", rows, allow_pickle=True)
