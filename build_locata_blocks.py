"""1-s spatial covariance blocks from the eigenmike recordings of LOCATA Task 1 (eval, static source), with the
true direction in the array frame. Eight of the 32 capsules, 0.5 s hop, blocks that are at least half voice-active.

usage: python build_locata_blocks.py LOCATA_DIR"""
import glob, os, sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, stft

D = sys.argv[1]
CAPS = [0, 12, 14, 15, 18, 28, 29, 31]
FS, WIN, HOP, NFFT = 16000, 16000, 8000, 512


def seconds(rows):
    return rows[:, 3] * 3600 + rows[:, 4] * 60 + rows[:, 5]


blocks, az, el, rid = [], [], [], []
for rec in sorted(glob.glob(f"{D}/eval/task1/recording*"), key=lambda p: int(p.split("recording")[-1])):
    d = f"{rec}/eigenmike"
    src = glob.glob(f"{d}/position_source_*.txt")
    if not src:
        continue
    sr, x = wavfile.read(f"{d}/audio_array_eigenmike.wav")
    x = resample_poly(x[:, CAPS].astype(np.float64), FS, sr, axis=0)
    t0 = seconds(np.loadtxt(f"{d}/audio_array_timestamps_eigenmike.txt", skiprows=1, ndmin=2))[0]
    pa = np.loadtxt(f"{d}/position_array_eigenmike.txt", skiprows=1, ndmin=2)
    ps = np.loadtxt(src[0], skiprows=1, ndmin=2)
    vad = np.loadtxt(glob.glob(f"{d}/VAD_eigenmike_*.txt")[0], skiprows=1)
    n = 0
    for s0 in range(0, len(x) - WIN + 1, HOP):
        seg = x[s0:s0 + WIN]
        v = vad[int(s0 * sr / FS):int((s0 + WIN) * sr / FS)]
        if np.std(seg) < 1e-7 or len(v) == 0 or v.mean() < 0.5:
            continue
        tc = t0 + (s0 + WIN / 2) / FS
        a = pa[np.argmin(np.abs(seconds(pa) - tc))]
        u = a[12:21].reshape(3, 3).T @ (ps[np.argmin(np.abs(seconds(ps) - tc)), 6:9] - a[6:9])
        u /= np.linalg.norm(u)
        _, _, X = stft(seg.T, fs=FS, nperseg=NFFT, noverlap=NFFT // 2, boundary=None, padded=False)
        blocks.append((np.einsum("ift,jft->ijf", X, np.conj(X)) / X.shape[2]).astype(np.complex64))
        az.append(np.degrees(np.arctan2(u[1], u[0])) % 360); el.append(np.degrees(np.arcsin(u[2])))
        rid.append(int(rec.split("recording")[-1])); n += 1
    print(os.path.basename(rec), "blocks", n, flush=True)
np.savez("data/locata_blocks.npz", csm=np.array(blocks), az=np.array(az), el=np.array(el), rec=np.array(rid),
         freqs=np.fft.rfftfreq(NFFT, 1 / FS).astype(np.float32))
print("total blocks", len(blocks))
