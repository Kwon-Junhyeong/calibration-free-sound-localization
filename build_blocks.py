"""1-s spatial covariance blocks from the 264 box recordings (ch1-4 = panels 0,3,5,6), speech-active only.

usage: python build_blocks.py RECORDINGS_DIR"""
import glob, os, re, sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

D = sys.argv[1]
files = sorted(glob.glob(f"{D}/*.wav"))
NFFT, HOP, FS = 512, 256, 16000
blocks, az, el, fid = [], [], [], []
for n, fn in enumerate(files):
    m = re.search(r"az(\d+)_el([pm])(\d+)", os.path.basename(fn))
    a = int(m.group(1)); e = int(m.group(3)) * (1 if m.group(2) == "p" else -1)
    sr, x = wavfile.read(fn)
    x = x[:, 1:5].astype(np.float64) / 32768.0
    _, _, X = stft(x.T, fs=FS, nperseg=NFFT, noverlap=NFFT - HOP, window="hann")   # (4, 257, T)
    pw = (np.abs(X[:, 10:220]) ** 2).sum((0, 1))
    floor = np.percentile(pw, 10)
    act = pw > floor * 10 ** (12 / 10)                     # 12 dB above the quiet floor
    idx = np.flatnonzero(act)
    for b in range(0, len(idx) - 31, 62):                  # ~1 s of active frames per block
        sel = idx[b:b + 62]
        Xs = X[:, :, sel]
        C = np.einsum("ift,jft->ijf", Xs, np.conj(Xs)) / len(sel)
        blocks.append(C.astype(np.complex64)); az.append(a); el.append(e); fid.append(n)
    if n % 40 == 0:
        print(n, os.path.basename(fn), "active frames", act.sum(), "blocks so far", len(blocks), flush=True)
np.savez("data/box_blocks.npz", csm=np.array(blocks), az=np.array(az), el=np.array(el), fid=np.array(fid),
         freqs=np.fft.rfftfreq(NFFT, 1 / FS).astype(np.float32))
print("total blocks", len(blocks))
