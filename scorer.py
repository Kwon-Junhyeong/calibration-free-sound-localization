"""Direction scoring with microphone-pair ratios. A constant gain and phase on each channel cancel in the fit."""
import itertools
import numpy as np


class PairScorer:
    """Residual of every direction after fitting one complex number per microphone pair.

    For pair (i, j): T = C_ij / C_jj from the recording and M_ij = M_i / M_j from the body model.
    eps = 1 - |<M_ij, T>_w|^2 / (||M_ij||_w^2 ||T||_w^2) with w = |C_ij|, and J is the mean over pairs.
    """

    def __init__(self, M):
        """M: body responses (direction, mic, freq)."""
        D, n_mic, F = M.shape
        self.pairs = list(itertools.combinations(range(n_mic), 2))
        self.MijH = np.empty((len(self.pairs), D, F), np.complex64)
        for p, (i, j) in enumerate(self.pairs):
            self.MijH[p] = np.conj(M[:, i] / (M[:, j] + 1e-20))
        self.Mabs2 = (self.MijH.real ** 2 + self.MijH.imag ** 2).astype(np.float32)

    def frame_terms(self, C):
        """C: covariance blocks (block, mic, mic, freq). Returns T and w, both (block, pair, freq)."""
        ii = np.array([i for i, _ in self.pairs])
        jj = np.array([j for _, j in self.pairs])
        Cij = C[:, ii, jj, :]
        T = Cij / (C[:, jj, jj, :].real + 1e-20)
        return T.astype(np.complex64), np.abs(Cij).astype(np.float32)

    def residuals(self, C, batch=64):
        """J (block, direction), lower is better."""
        out = []
        for b in range(0, len(C), batch):
            T, w = self.frame_terms(C[b:b + batch])
            num = np.einsum("bpf,pdf->bpd", w * T, self.MijH, optimize=True)
            den = np.einsum("bpf,pdf->bpd", w, self.Mabs2, optimize=True)
            tn = (w * np.abs(T) ** 2).sum(-1)
            out.append((1.0 - np.abs(num) ** 2 / (den * tn[:, :, None] + 1e-30)).mean(1))
        return np.concatenate(out)


def refine(grid, J, radius=8.0):
    """Refines each direction with a quadratic fit of J around its best grid point, in the tangent plane."""
    cos_r = np.cos(np.radians(radius))
    out = np.empty((len(J), 3))
    for b in range(len(J)):
        u = grid[int(J[b].argmin())]
        nb = np.flatnonzero(grid @ u >= cos_r)
        a = np.array([0, 0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0, 0])
        e1 = np.cross(u, a)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        x, y = grid[nb] @ e1, grid[nb] @ e2
        A = np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], 1)
        c, *_ = np.linalg.lstsq(A, J[b, nb], rcond=None)
        H = np.array([[2 * c[3], c[5]], [c[5], 2 * c[4]]])
        out[b] = u
        if np.all(np.linalg.eigvalsh(H) > 0):
            xy = np.linalg.solve(H, -c[1:3])
            if np.linalg.norm(xy) < np.radians(6):
                v = u + xy[0] * e1 + xy[1] * e2
                out[b] = v / np.linalg.norm(v)
    return out


def azimuth_elevation(u):
    return np.degrees(np.arctan2(u[:, 1], u[:, 0])), np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1)))
