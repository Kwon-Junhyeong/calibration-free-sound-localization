"""Shared parts of the box experiments (unknown_channels.py, self_calibration.py).

On import, reads the responses of the box (data/box_replicas_5deg.npz from build_replicas.py) and the block
covariances of its recordings (data/box_blocks.npz from build_blocks.py), and keeps the first two one-second
blocks of each of the 264 recordings (528 blocks). The functions add channel errors in software, match with and
without a search over the delay of each pair, refine the best direction between grid points, measure the error
against the true directions and run the SRP-PHAT baseline.

Time convention: the BEM table and the free-field responses below are e^{+jwt}, the convention of the STFT, so
the block covariances are matched as they are (the analytic responses in scorer.py are e^{-iwt}; see there)."""
import numpy as np
from scorer import PairScorer, uvec

C_SND, FS = 343.0, 16000
R = np.load("data/box_replicas_5deg.npz"); K = np.load("data/box_blocks.npz")
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
    """median angle to the true direction (deg) and the share of blocks more than 30 deg off (%)."""
    e = np.degrees(np.arccos(np.clip((est * gt).sum(1), -1, 1)))
    return np.median(e), 100 * np.mean(e > 30)


def channel_errors(seed, mode):
    """One random set of channel errors as a (4, F) factor on the channels: gain within +-6 dB and any phase
    (modes gp, gp+d1.5, gp+d4), and a delay of up to 1.5 or 4 samples (gp+d1.5, gp+d4); clean is all ones."""
    rng = np.random.default_rng(seed)
    a = 10 ** (rng.uniform(-6, 6, 4) / 20); ph = rng.uniform(-np.pi, np.pi, 4)
    dl = {"clean": 0, "gp": 0, "gp+d1.5": 1.5, "gp+d4": 4.0}[mode] * rng.uniform(-1, 1, 4) / FS
    if mode == "clean": a[:] = 1; ph[:] = 0
    return a[:, None] * np.exp(1j * ph[:, None]) * np.exp(-2j * np.pi * f[None, :] * dl[:, None])


def scores_flat(S, C):
    """residuals of the pair-ratio matching, (B, D)."""
    return np.concatenate([S.residuals(C[b:b + 64]) for b in range(0, len(C), 64)])


def delay_search(S, C):
    """pair-ratio matching in which each pair may also have any of the candidate delays LAGS: the residual with
    the best delay of each pair, (B, D), and the squared correlation over the delays, (B, P, D, L)."""
    T, w = S.frame_terms(C)
    X = (w * T)[:, :, None, :] * S.MijH[None]                             # (B,P,D,F)
    num = np.abs(X @ E) ** 2                                              # (B,P,D,L)
    den = np.einsum("bpf,pdf->bpd", w, S.Mabs2, optimize=True)
    tn = (w * np.abs(T) ** 2).sum(-1)
    return (1 - num.max(-1) / (den * tn[:, :, None] + 1e-30)).mean(1), num


def scores_delay(S, C):
    """residuals with the delay search, (B, D), eight blocks at a time."""
    return np.concatenate([delay_search(S, C[b:b + 8])[0] for b in range(0, len(C), 8)])


def srp(C):
    ii = np.array([0, 0, 0, 1, 1, 2]); jj = np.array([1, 2, 3, 2, 3, 3])
    st = np.conj(FF[:, ii] * np.conj(FF[:, jj])); st /= np.abs(st)
    G = C[:, ii, jj]; G = G / (np.abs(G) + 1e-20)
    return np.einsum("bpf,dpf->bd", G, st, optimize=True).real
