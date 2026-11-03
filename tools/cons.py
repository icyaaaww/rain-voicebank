# -*- coding: utf-8 -*-
"""辅音合成（Klatt 式并联共振峰）。

之前的问题：用单个带通做辅音，频谱是平的，听起来是"嘶"，尖锐刺耳。
真实擦音的频谱是**有峰的**：s 在 3.5-6kHz 有两三个峰，sh 在 2-3kHz。

做法：
    1. 白噪声先高通（不然低频压过来，峰全在 100Hz 附近）
    2. 并联多个共振峰（每个峰一个谐振器）
    3. 爆破音额外加 2ms 冲激 + 30ms 送气
"""
import numpy as np
from scipy.signal import butter, sosfilt, lfilter

SR = 44100

# 擦音：每个音一组峰（Hz, 带宽）
FRIC = {
    "s":  [(3600, 380), (4900, 600), (6400, 800)],
    "sh": [(2000, 260), (2900, 400), (4000, 560)],
    "h":  [(1300, 600), (2700, 950), (4200, 1200)],
    "f":  [(1300, 340), (3200, 900), (5000, 1300)],
    "z":  [(3600, 380), (4900, 600)],
    "j":  [(2000, 260), (2900, 400)],
    "ts": [(4200, 520), (5800, 800)],
    "ch": [(2300, 300), (3300, 460)],
    "th": [(1400, 400), (3400, 1000)],
}

# 爆破音：冲激的着色共振峰
STOP = {
    "t": [(2800, 650), (4800, 1000)],
    "k": [(2300, 550), (3900, 900)],
    "p": [(850, 380), (2000, 800)],
    "d": [(2800, 650)],
    "g": [(2300, 550)],
    "b": [(850, 380)],
}

# 鼻音 / 流音：不自造低频峰，而是借用相近元音的共振峰。
# 之前自造 (320,280) 这种谐振器，低频增益压倒一切，
# 实测 99% 能量在 400Hz 以下 —— 听起来是"咚"不是"m"。
# 真实的 m 有三个共振峰(300/1100/2500)，能量分布和元音一样宽。
SONO_VOWEL = {
    "m": "u",   # 鼻音，口型接近 u
    "n": "u",
    "r": "i",   # 日语弹音，舌位接近 i，而且要短
    "y": "i",   # 半元音 y = i 的起始
    "w": "u",   # 半元音 w = u 的起始
}
SONO_DUR = {"m": 55, "n": 55, "r": 16, "y": 45, "w": 45}


def reson(x, fc, bw, sr=SR):
    r = np.exp(-np.pi * bw / sr)
    th = 2 * np.pi * fc / sr
    return lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x)


def parallel_noise(formants, n, seed=7, sr=SR, hp=600.0):
    """白噪声 → 高通 → 并联共振峰。"""
    rr = np.random.default_rng(seed)
    nz = rr.normal(0, 1, n)
    if hp > 0:
        nz = sosfilt(butter(4, hp, btype="highpass", fs=sr, output="sos"), nz)
    y = np.zeros(n)
    for fc, bw in formants:
        if fc > sr / 2 * 0.9:
            continue
        y = y + reson(nz, fc, bw, sr)
    pk = np.abs(y).max()
    return y / pk if pk > 1e-9 else y


def make_fric(ph, dur_ms=95, seed=7, cons_gain=0.60, sr=SR):
    n = int(sr * dur_ms / 1000)
    y = parallel_noise(FRIC.get(ph, [(3000, 800)]), n, seed, sr)
    t = np.linspace(0, 1, n)
    env = np.clip(t / 0.10, 0, 1) * (0.72 + 0.28 * np.exp(
        -2.0 * np.maximum(0, t - 0.85) / 0.15))
    y = y * env * cons_gain
    fo = max(4, int(sr * 0.020))
    if fo < n:
        y[-fo:] *= np.linspace(1, 0.15, fo)
    return y


def make_stop(ph, dur_ms=78, seed=7, cons_gain=0.72, sr=SR):
    n = int(sr * dur_ms / 1000)
    rr = np.random.default_rng(seed)
    y = np.zeros(n)
    gap = int(sr * 0.014)                 # 闭塞
    ilen = int(sr * 0.0025)               # 冲激
    imp = rr.normal(0, 1, ilen) * np.hanning(ilen)
    fmts = STOP.get(ph, [(3000, 700)])
    imp = sum(reson(imp, fc, bw, sr) for fc, bw in fmts)
    imp = imp / (np.abs(imp).max() + 1e-9)
    e = min(n, gap + ilen)
    y[gap:e] += imp[: e - gap]
    al = min(n - (gap + ilen), int(sr * 0.032))
    if al > 16:
        asp = parallel_noise(fmts, al, seed + 1, sr, hp=800)
        asp *= np.linspace(0.55, 0.04, al)
        y[gap + ilen:gap + ilen + al] += asp * 0.42
    pk = np.abs(y).max()
    return y / pk * cons_gain if pk > 1e-9 else y


def make_sono(ph, f0, dur_ms=None, seed=7, cons_gain=0.55, sr=SR):
    """鼻音/流音：脉冲激励 + 相近元音的共振峰（频谱自然平衡）。"""
    import config as C
    dur_ms = dur_ms or SONO_DUR.get(ph, 55)
    n = int(sr * dur_ms / 1000)
    period = sr / max(60.0, f0)
    exc = np.zeros(n)
    i = 0
    while i < n:
        L = max(4, int(period))
        t = np.arange(L) / L
        seg = 0.5 * (1 - np.cos(np.pi * np.minimum(t / 0.35, 1)))
        e = min(n, i + L)
        exc[i:e] += seg[: e - i]
        i += L
    exc = exc / (np.abs(exc).max() + 1e-9)
    # 预加重：脉冲序列的谐波按 1/k 衰减，高频几乎没能量，
    # 结果鼻音 98% 能量挤在 400Hz 以下（听起来是"咚"）。
    # 一阶差分把高频提上来，频谱才接近真实鼻音的分布。
    exc = np.append(exc[0], np.diff(exc))
    exc = exc / (np.abs(exc).max() + 1e-9)

    vow = SONO_VOWEL.get(ph, "u")
    fmts = list(C.FORMANTS_ACTIVE.get(vow, C.FORMANTS_ACTIVE["u"]))
    # 鼻音加一个低频鼻腔共振，但仍然保留元音的中高频峰
    if ph in ("m", "n"):
        fmts = [(280, 120)] + fmts
    y = np.zeros(n)
    for fc, bw in fmts:
        y = y + reson(exc, fc, bw, sr)
    pk = np.abs(y).max()
    y = y / pk if pk > 1e-9 else y
    # 鼻音的包络：平稳，短淡出
    env = np.clip(np.linspace(0, 1, n) / 0.10, 0, 1)
    fo = max(4, int(sr * 0.012))
    if fo < n:
        env[-fo:] *= np.linspace(1, 0.5, fo)
    return y * env * cons_gain


def make_consonant(ph, f0=None, seed=7, sr=SR):
    """统一入口：按音素类型分发。"""
    if not ph:
        return np.zeros(0)
    if ph in FRIC:
        return make_fric(ph, seed=seed, sr=sr)
    if ph in STOP:
        return make_stop(ph, seed=seed, sr=sr)
    if ph in SONO_VOWEL:
        return make_sono(ph, f0 or 200.0, seed=seed, sr=sr)
    return np.zeros(0)


if __name__ == "__main__":
    from scipy.signal import find_peaks
    for tag, table in [("擦音", FRIC), ("爆破", STOP)]:
        print(f"== {tag} ==")
        for ph in table:
            y = make_consonant(ph, 200.0, seed=7)
            if len(y) == 0:
                continue
            S = np.abs(np.fft.rfft(y * np.hanning(len(y)), n=8192))
            f = np.fft.rfftfreq(8192, 1 / SR)
            pk, _ = find_peaks(S, distance=12, prominence=S.max() * 0.08)
            top = sorted(pk, key=lambda i: -S[i])[:3]
            cent = float((f * S).sum() / (S.sum() + 1e-12))
            rms = 20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-12)
            print(f"  {ph:3s} 质心 {cent:5.0f}Hz  RMS {rms:5.1f}dB  峰 "
                  + ", ".join(f"{f[i]:.0f}" for i in sorted(top)))
