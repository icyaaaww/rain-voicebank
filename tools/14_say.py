# -*- coding: utf-8 -*-
"""用 Rosenberg 声门脉冲合成一句日语（带语调曲线）。

这套参数是目前调出来最平衡的：
    激励    Rosenberg 声门脉冲（含周期/幅度微扰）
    共振峰  IIR 滤波 + 唇端微分
    辅音    短瞬态（12ms 起音 + 指数衰减），峰值 0.55 倍元音
    气声    8%（和声门同源，不是独立噪声层）
    语调    音高随音节位置起伏，句尾降调

用法:
    python 14_say.py "あめがふる、わたしはうたう"
"""
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, lfilter

sys.path.insert(0, os.path.dirname(__file__))
import config as C
import japanese as JP
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "syn", os.path.join(os.path.dirname(__file__), "02_synthesize.py"))
syn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(syn)

_spec2 = importlib.util.spec_from_file_location(
    "har", os.path.join(os.path.dirname(__file__), "11_harmonic.py"))
H = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(H)

SR = C.SR


def load_src():
    return H.load_rain()


def glottal(f0, n, sr=SR, oq=0.62, seed=7, jit=0.008, shim=0.06):
    rr = np.random.default_rng(seed)
    period = sr / float(f0)
    y = np.zeros(n)
    i = 0
    while i < n:
        p = period * (1.0 + jit * rr.normal())
        a = 1.0 * (1.0 + shim * rr.normal())
        L = max(4, int(p))
        t = np.arange(L) / L
        seg = np.zeros(L)
        T1 = oq * 0.62
        T2 = oq * 0.38
        m1 = t < T1
        seg[m1] = 0.5 * (1 - np.cos(np.pi * t[m1] / T1))
        m2 = (t >= T1) & (t < T1 + T2)
        seg[m2] = np.cos(np.pi * (t[m2] - T1) / (2 * T2))
        seg = -seg
        e = min(n, i + L)
        y[i:e] += a * seg[: e - i]
        i += L
    return y


def ffilter(x, fmts, sr=SR):
    y = x
    for fc, bw in fmts:
        r = np.exp(-np.pi * bw / sr)
        th = 2 * np.pi * fc / sr
        y = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], y)
    return y


def vowel(ph, f0, n, seed=7, asp=0.08, fmax=4800, sr=SR):
    exc = glottal(f0, n, sr, seed=seed)
    rr = np.random.default_rng(seed + 1)
    if asp > 0:
        nz = rr.normal(0, 1, n)
        nz = sosfilt(butter(2, [300, 4000], btype="bandpass", fs=sr, output="sos"), nz)
        exc = exc + nz * (np.abs(exc).mean() + 1e-9) * asp * 10
    exc = exc / (np.abs(exc).max() + 1e-9)
    y = ffilter(exc, C.FORMANTS_ACTIVE[ph], sr)
    y = y / (np.abs(y).max() + 1e-9)
    y = np.append(y[0], np.diff(y))
    y = y / (np.abs(y).max() + 1e-9)
    y = sosfilt(butter(4, fmax, btype="lowpass", fs=sr, output="sos"), y)
    return y / (np.abs(y).max() + 1e-9)


def syllable(kana, f0, dur, src, rng, cons_gain=0.55, cons_decay=7.0):
    cons, vow = JP.KANA.get(kana, ("", "a"))
    n = int(SR * dur)
    y = vowel(vow, f0, n, seed=int(rng.integers(1, 99999)))
    if cons:
        c = syn.make_consonant(cons, src, rng, f0)
        if len(c) > 0:
            nc = min(len(c), int(SR * 0.075), n // 2)
            c = c[:nc] / (np.abs(c[:nc]).max() + 1e-9)
            ce = np.exp(-np.linspace(0, cons_decay, nc))
            c = c * ce * cons_gain
            fo = min(int(SR * 0.020), nc // 3)
            mix = np.ones(nc)
            if fo > 4:
                mix[-fo:] = np.linspace(1, 0.25, fo)
            y[:nc] = c * mix + y[:nc] * (1 - mix)
    a = min(int(SR * 0.008), n // 4)
    r = min(int(SR * 0.045), n // 3)
    env = np.ones(n)
    env[:a] = np.linspace(0, 1, a) ** 0.7
    env[-r:] = np.linspace(1, 0, r) ** 1.3
    y = y * env
    y -= y.mean()
    return y / (np.abs(y).max() + 1e-9) * 0.85


def say(text, f0=392.0, dur=0.40, seed=7, cons_gain=0.55,
        phrase_pause=0.30, long_last=True):
    """带语调：前半句略升，后半句降，句尾收。"""
    src = load_src()
    rng = np.random.default_rng(seed)

    # 把文本按标点切成短语
    phrases = []
    cur = []
    for ch in text:
        if ch in "、。,.， ":
            if cur:
                phrases.append(cur)
                cur = []
        elif ch in JP.KANA:
            cur.append(ch)
    if cur:
        phrases.append(cur)

    total_kana = sum(len(p) for p in phrases)
    out = []
    idx = 0
    for pi, ph in enumerate(phrases):
        for i, k in enumerate(ph):
            # 语调：整体在 -8% ~ +6% 之间起伏，句尾下降
            pos = idx / max(1, total_kana - 1)
            # 每句内部：起略低、中抬、尾降
            local = i / max(1, len(ph) - 1) if len(ph) > 1 else 0.5
            contour = 0.02 * np.sin(np.pi * local) - 0.05 * max(0, local - 0.7) / 0.3
            # 整句末下降更多
            if pi == len(phrases) - 1 and i == len(ph) - 1:
                contour -= 0.06
            ff = f0 * (1.0 + contour)
            d = dur * (1.35 if (long_last and i == len(ph) - 1) else 1.0)
            out.append(syllable(k, ff, d, src, rng, cons_gain))
            idx += 1
        out.append(np.zeros(int(SR * phrase_pause)))

    y = np.concatenate(out)
    y = y / (np.abs(y).max() + 1e-9) * 0.92
    return y


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "あめがふる、わたしはうたう"
    name = sys.argv[2] if len(sys.argv) > 2 else "line1.wav"
    y = say(text)
    p = C.PREVIEW / name
    sf.write(str(p), y.astype(np.float32), SR, subtype=C.SUBTYPE)
    print(f"[ok] {p}  {len(y)/SR:.2f}s  「{text}」")
