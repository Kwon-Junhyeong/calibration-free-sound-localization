"""Channel delays estimated from unlabeled blocks. The delays stay fixed across blocks while the direction varies."""
import numpy as np

FS = 16000
LAGS = np.arange(-8, 8.01, 0.25) / FS          # candidate delay of each pair, s


def pair_lags(S, C, f, batch=8):
    """Scores with a fitted delay per pair. Returns J (block, direction) and the pair delays at the best direction of each block."""
    E = np.exp(2j * np.pi * f[:, None] * LAGS[None, :]).astype(np.complex64)
    J, lags = [], []
    for b in range(0, len(C), batch):
        T, w = S.frame_terms(C[b:b + batch])
        num = np.abs(((w * T)[:, :, None, :] * S.MijH[None]) @ E) ** 2           # (block, pair, direction, lag)
        den = np.einsum("bpf,pdf->bpd", w, S.Mabs2, optimize=True)
        tn = (w * np.abs(T) ** 2).sum(-1)
        j = (1 - num.max(-1) / (den * tn[:, :, None] + 1e-30)).mean(1)
        d0 = j.argmin(1)
        J.append(j)
        lags.append(LAGS[num[np.arange(len(d0)), :, d0].argmax(-1)])
    return np.concatenate(J), np.concatenate(lags)


def estimate_delays(lags, pairs, n_mic=4):
    """Median over blocks for each pair, then least squares for each microphone. Microphone 0 is the reference."""
    A = np.zeros((len(pairs), n_mic - 1))
    for p, (i, j) in enumerate(pairs):
        if i:
            A[p, i - 1] += 1
        if j:
            A[p, j - 1] -= 1
    d, *_ = np.linalg.lstsq(A, np.median(lags, 0), rcond=None)
    return np.concatenate([[0.0], d])


def compensate(C, d, f):
    corr = np.exp(2j * np.pi * (d[:, None, None] - d[None, :, None]) * f[None, None, :])
    return (C * corr[None]).astype(np.complex64)
