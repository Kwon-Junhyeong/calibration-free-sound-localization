"""Pair-ratio matching (PairScorer) and the analytic responses (replicas) of a rigid sphere and of free field.

PairScorer works with any responses and block covariances as long as both use the same time convention.
sphere_replicas and freefield_replicas below are e^{-iwt}: the total pressure for a unit plane wave arriving from
direction d (d = unit vector toward the source). The STFT covariances are e^{+jwt}, so locata.py conjugates them
before matching. The box experiments use a BEM table, which is already e^{+jwt} (common.py).
"""
import itertools
import numpy as np
from scipy.special import spherical_jn, spherical_yn

C_SND = 343.0


def uvec(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], -1)


def legendre(x, nmax):
    """P_0..P_nmax at x (any shape) -> (nmax+1, *x.shape)."""
    P = np.empty((nmax + 1,) + x.shape)
    P[0] = 1.0
    if nmax:
        P[1] = x
    for n in range(1, nmax):
        P[n + 1] = ((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1)
    return P


def sphere_replicas(dirs, mics, freqs, radius=None, nmax=None):
    """Total pressure on a rigid sphere at capsule positions `mics` (M,3) for plane waves from `dirs` (D,3).
    Returns (D, M, F) complex. Uses p = sum_n (2n+1) i^n * i / ((ka)^2 h_n'(ka)) * P_n(-d . r_hat)."""
    r = np.linalg.norm(mics, axis=1)
    a = float(np.mean(r)) if radius is None else radius
    rhat = mics / r[:, None]
    ka = 2 * np.pi * np.asarray(freqs, float) * a / C_SND
    ka = np.maximum(ka, 1e-6)
    if nmax is None:
        nmax = int(np.ceil(ka.max())) + 12
    n = np.arange(nmax + 1)[:, None]
    hp = (spherical_jn(n, ka[None], derivative=True) + 1j * spherical_yn(n, ka[None], derivative=True))
    coef = (2 * n + 1) * (1j ** n) * 1j / (ka[None] ** 2 * hp)          # (N+1, F)
    x = -(dirs @ rhat.T)                                                 # (D, M)
    P = legendre(x, nmax)                                                # (N+1, D, M)
    return np.einsum("ndm,nf->dmf", P, coef, optimize=True)


def freefield_replicas(dirs, mics, freqs):
    """Point-microphone plane-wave responses in the same e^{-iwt} convention (capsules facing d hear it first)."""
    tau = -(dirs @ mics.T) / C_SND                                       # (D, M) delay relative to origin
    return np.exp(2j * np.pi * tau[:, :, None] * np.asarray(freqs)[None, None, :])


class PairScorer:
    """Pair-ratio matching, vectorized over directions and blocks.

    For pair p=(i,j): T = C_ij / C_jj, w = |C_ij| (or a given weight), M_ij = M_i / M_j.
    eps_p(theta) = 1 - |<M_ij, T>_w|^2 / (||M_ij||_w^2 ||T||_w^2);  J = mean_p eps_p.
    The normalized correlation removes the complex factor that the channels' constant gains and phases put on
    the pair. Per block the cost is two (D x F)(F) products per pair, i.e. a batched GEMV / GEMM over blocks.
    """

    def __init__(self, M, pairs=None):
        D, Mn, F = M.shape
        self.pairs = pairs or list(itertools.combinations(range(Mn), 2))
        # only conj(M_ij) and |M_ij|^2 are kept, to stay within a few hundred MB per table
        self.MijH = np.empty((len(self.pairs), D, F), np.complex64)
        for p, (i, j) in enumerate(self.pairs):
            self.MijH[p] = np.conj(M[:, i] / (M[:, j] + 1e-20))
        self.Mabs2 = (self.MijH.real ** 2 + self.MijH.imag ** 2).astype(np.float32)

    def frame_terms(self, C, weight="abs"):
        """C: (B, M, M, F) block covariances in the band, in the time convention of the responses
        -> T (B,P,F), w (B,P,F)."""
        ii = np.array([i for i, _ in self.pairs]); jj = np.array([j for _, j in self.pairs])
        Cij = C[:, ii, jj, :]
        Cjj = C[:, jj, jj, :].real + 1e-20
        T = Cij / Cjj
        if weight == "abs":
            w = np.abs(Cij)
        elif weight == "uniform":
            w = np.ones_like(Cij.real)
        elif weight == "coh":
            w = np.abs(Cij) / np.sqrt(C[:, ii, ii, :].real * Cjj)
        else:
            raise ValueError(weight)
        return T.astype(np.complex64), w.astype(np.float32)

    def residuals(self, C, weight="abs", per_pair=False):
        """Return J (B, D) (or eps (B,P,D) if per_pair)."""
        T, w = self.frame_terms(C, weight)
        wT = w * T                                                        # (B,P,F)
        num = np.einsum("bpf,pdf->bpd", wT, self.MijH, optimize=True)     # batched GEMM over pairs
        den = np.einsum("bpf,pdf->bpd", w, self.Mabs2, optimize=True)
        tn = (w * np.abs(T) ** 2).sum(-1)                                 # (B,P)
        eps = 1.0 - (np.abs(num) ** 2) / (den * tn[:, :, None] + 1e-30)
        return eps if per_pair else eps.mean(1)


def gc_err(a1, e1, a2, e2):
    a1, e1, a2, e2 = map(np.radians, (a1, e1, a2, e2))
    c = np.sin(e1) * np.sin(e2) + np.cos(e1) * np.cos(e2) * np.cos(a1 - a2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def az_err(a, b):
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)
