"""Channel gains, phases and delays added to the measured box recordings: does the body keep the array
localizable when the channel delays are unknown? Matching with the body's replicas and with free-field replicas,
with and without the delay estimate, and SRP-PHAT. 2 one-second blocks per direction (528), 3 seeds."""
import numpy as np
from common import (PairScorer, BEM, FF, C0, refine, err, gauge, scores_flat, scores_delay, srp)

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
np.save("data/unknown_channels.npy", rows, allow_pickle=True)
