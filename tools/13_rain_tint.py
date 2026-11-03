# -*- coding: utf-8 -*-
"""给人声染上雨的音色（保留全部可懂度）。

思路：
  发声机制完全不动 —— 谐波结构、共振峰、辅音全是真人的，词一定听得清。
  只做两件事：
    1) 频谱包络轻度向雨声倾斜（雨的整体亮度曲线）
       注意只改"倾斜"，不改共振峰位置，否则元音会糊。
    2) 在人声的"气声段"（低能量、无音高帧）填一点真雨声，
       让换气和字缝里有雨的味道，而不是叠一条独立的雨轨。

用法:
    python 13_rain_tint.py <人声.wav> <输出.wav> [--amount 0.3] [--breath 0.18]
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, stft, istft

sys.path.insert(0, os.path.dirname(__file__))
import config as C

SR = C.SR
NPERSEG = 1024
NOVERLAP = 768


def load_rain():
    srcs = sorted(C.NORMALIZED.glob("*.wav"))
    x, sr = sf.read(str(srcs[0]), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x[: int(SR * 60)]


def smooth_curve(y, w):
    k = np.hanning(max(3, w))
    k /= k.sum()
    return np.convolve(y, k, mode="same")


def tilt_curve(db, w=13):
    """强平滑 -> 只保留整体倾斜，去掉共振峰峰谷。"""
    s = smooth_curve(db, w)
    h = max(1, w // 2)
    s[:h] = s[h]
    s[-h:] = s[-h - 1]
    return s


def band_db_db(spec, freqs, n_bands=40, fmin=120.0, fmax=9000.0):
    """把谱平均成对数频段曲线。"""
    centers = np.geomspace(fmin, fmax, n_bands)
    out = np.empty(n_bands)
    P = np.abs(spec) ** 2
    for i, fc in enumerate(centers):
        m = (freqs >= fc / 1.12) & (freqs < fc * 1.12)
        out[i] = 10 * np.log10(P[m].mean() + 1e-15) if m.sum() else -150.0
    return centers, out


def rain_tilt(rain, n_bands=40):
    seg = rain[: int(SR * 6)] * np.hanning(min(len(rain), int(SR * 6)))
    S = np.abs(np.fft.rfft(seg, n=1 << 16))
    f = np.fft.rfftfreq(1 << 16, 1 / SR)
    centers, db = band_db_db(S, f, n_bands)
    return centers, tilt_curve(db, 11)


def apply_tint(x, centers, rain_t, amount=0.30):
    """逐帧把频谱倾斜往雨那边拉，只动倾斜、不动共振峰。"""
    _, _, Z = stft(x, fs=SR, nperseg=NPERSEG, noverlap=NOVERLAP)
    freqs = np.fft.rfftfreq(NPERSEG, 1 / SR)
    frames = []
    for ti in range(Z.shape[1]):
        col = Z[:, ti]
        P = np.abs(col) ** 2
        if P.sum() < 1e-12:
            frames.append(np.zeros_like(freqs))
            continue
        _, db = band_db_db(np.sqrt(P), freqs, len(centers))
        cur_t = tilt_curve(db, 11)
        # 目标 = 当前倾斜 + (雨的倾斜 - 平均倾斜)*amount
        tgt = rain_t - rain_t.mean()
        cur = cur_t - cur_t.mean()
        gain_db = (tgt - cur) * amount
        # 映射回到每个 FFT bin
        g = np.interp(np.log(np.maximum(freqs, 1)), np.log(centers), gain_db)
        frames.append(g)
    if not frames:
        return x
    G = np.stack(frames, axis=1)
    Z2 = Z * (10 ** (G / 20.0))
    _, y = istft(Z2, fs=SR, nperseg=NPERSEG, noverlap=NOVERLAP)
    y = y[: len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y


def breath_fill(x, rain, amount=0.18, thresh_pct=25.0, rng=None):
    """在人声的低能量段（换气、字缝）填雨声，音色统一但不压人声。"""
    rng = rng or np.random.default_rng(3)
    n = len(x)
    w = max(16, int(SR * 0.020))
    e = np.sqrt(np.convolve(x ** 2, np.ones(w) / w, mode="same") + 1e-12)
    ref = np.percentile(e[e > 1e-6], 50) if (e > 1e-6).any() else 1e-6
    gate = e < (ref * (thresh_pct / 50.0))
    gate = smooth_curve(gate.astype(np.float64), int(SR * 0.030))
    gate = np.clip(gate, 0, 1)

    reps = int(np.ceil(n / len(rain))) + 1
    r = np.tile(rain, reps)[:n].copy()
    r = sosfilt(butter(2, [150, 6500], btype="bandpass", fs=SR, output="sos"), r)
    r -= r.mean()
    r /= (np.sqrt((r ** 2).mean()) + 1e-9)

    voice_rms = np.sqrt((x ** 2).mean()) + 1e-9
    y = x + r * gate * amount * voice_rms
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--amount", type=float, default=0.30, help="频谱倾斜染色强度")
    ap.add_argument("--breath", type=float, default=0.18, help="气声段雨声量")
    ap.add_argument("--thresh", type=float, default=25.0, help="气声门限(百分比)")
    ap.add_argument("--dry", type=float, default=0.0, help="保留多少原始信号(0-1)")
    a = ap.parse_args()

    x, sr = sf.read(a.src, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    print(f"人声 {len(x)/SR:.1f}s")

    rain = load_rain()
    centers, rt = rain_tilt(rain)
    print("雨声倾斜曲线已提取")

    y = apply_tint(x, centers, rt, amount=a.amount)
    print(f"频谱染色 amount={a.amount}")

    y = breath_fill(y, rain, amount=a.breath, thresh_pct=a.thresh)
    print(f"气声填充 breath={a.breath} 门限={a.thresh}%")

    if a.dry > 0:
        y = y * (1 - a.dry) + x * a.dry

    # 轻压缩 + 归一
    w = max(16, int(SR * 0.020))
    e = np.sqrt(np.convolve(y ** 2, np.ones(w) / w, mode="same") + 1e-12)
    ref = np.sqrt(np.convolve(y ** 2, np.ones(int(SR * 0.15)) / int(SR * 0.15),
                              mode="same") + 1e-12)
    g = np.clip((np.maximum(ref, 1e-6) / np.maximum(e, 1e-6)) ** 0.4, 0.4, 3.0)
    y = y * g
    y = y / (np.abs(y).max() + 1e-9) * 0.92
    sf.write(a.out, y.astype(np.float32), SR, subtype=C.SUBTYPE)
    print(f"[ok] {a.out}  {len(y)/SR:.1f}s")


if __name__ == "__main__":
    main()
