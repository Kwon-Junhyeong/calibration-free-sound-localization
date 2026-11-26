"""A check that needs no data: pair-ratio matching gives the same direction when every channel has its own
unknown gain and phase.

Eight capsules on a rigid sphere of radius 4.2 cm (the capsules of the LOCATA eigenmike used in locata.py), the
analytic responses of scorer.py on a 5 deg grid, and block covariances of a source in each test direction with
noise 20 dB below it. Each channel then gets a random gain within +-6 dB and a random phase. The direction is
found by pair-ratio matching and, as a comparison, by a beamformer steered with the same responses.

usage: python demo.py"""
import numpy as np
from scorer import sphere_replicas, PairScorer, gc_err, uvec

MICS = np.array([[0, 0.039190594, 0.014969475], [-0.014969475, 0, 0.039190594],
                 [-0.03633649, -0.0009206058, -0.020967614], [-0.014969475, 0, -0.039190594],
                 [0, -0.039190594, -0.014969475], [0.014969475, 0, 0.039190594],
                 [0.03484692, 0, 0.023377791], [0.014969475, 0, -0.039190594]])
f = np.fft.rfftfreq(512, 1 / 16000)
f = f[(f >= 312) & (f <= 7000)]
A, E = np.meshgrid(np.arange(0, 360, 5.0), np.arange(-90, 91, 5.0), indexing="ij")
cand = np.stack([A.ravel(), E.ravel()], 1)
M = sphere_replicas(uvec(cand[:, 0], cand[:, 1]), MICS, f)               # (D, 8, F), e^{-iwt}
S = PairScorer(M)
rng = np.random.default_rng(0)
g = 10 ** (rng.uniform(-6, 6, 8) / 20) * np.exp(1j * rng.uniform(-np.pi, np.pi, 8))   # one set of channel errors


def block(d):
    """covariance (8, 8, F) of a source at grid direction d, with uncorrelated noise 20 dB below the source."""
    m = M[d]
    C = m[:, None, :] * np.conj(m)[None, :, :]
    noise = 0.01 * np.mean(np.abs(m) ** 2)
    return C + noise * np.eye(8)[:, :, None]


def beamformer(C):
    """power of the beamformer steered with the responses, summed over frequency; larger is better."""
    num = np.einsum("dmf,mnf,dnf->d", np.conj(M), C, M, optimize=True).real
    return num / (np.abs(M) ** 2).sum(1).sum(-1)


print("true direction | pair-ratio matching: exact channels, unknown channels | beamformer: exact, unknown")
for az, el in ((40, 10), (135, -20), (220, 30), (300, 0)):
    d = int(np.argmin(gc_err(cand[:, 0], cand[:, 1], az, el)))
    C = block(d)
    Cg = g[:, None, None] * np.conj(g)[None, :, None] * C
    out = []
    for score, best in ((lambda c: S.residuals(c[None])[0], np.argmin), (beamformer, np.argmax)):
        for c in (C, Cg):
            e = cand[best(score(c))]
            out.append(f"({e[0]:3.0f}, {e[1]:3.0f}) off {gc_err(e[0], e[1], az, el):5.1f} deg")
    print(f"({az:3d}, {el:3d}) | {out[0]}, {out[1]} | {out[2]}, {out[3]}")
