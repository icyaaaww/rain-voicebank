# -*- coding: utf-8 -*-
"""用当前最优方案说一句话：雨滴脉冲 + STFT 频谱整形 + 辅音。

用法:
    python say_ja.py "わたしはあめ"
    python say_ja.py "あめがふる" --f0 392 --out xxx.wav
"""
import argparse
import sys
import os

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, stft, istft

sys.path.insert(0, os.path.dirname(__file__))
import config as C
import japanese as JP
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "syn", os.path.join(os.path.dirname(__file__), "02_synthesize.py"))
syn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(syn)

SR = C.SR


def formant_db(f, phoneme, bw_scale=1.5):
    g = np.zeros_like(f)
    for fc, bw in C.FORMANTS_ACTIVE[phoneme]:
        g += np.exp(-0.5 * ((f - fc) / (bw * bw_scale)) ** 2)
    return 20 * np.log10(np.maximum(g, 1e-3))


def vowel(phoneme, f0, dur, drops, strength=1.2, fmax=4000.0, rng=None):
    rng = rng or np.random.default_rng(0)
    n = int(SR * dur)
    exc = syn.drop_excitation(drops, f0, n, SR, rng)
    exc = sosfilt(butter(2, 90.0, btype="highpass", fs=SR, output="sos"), exc)
    exc = exc / (np.abs(exc).max() + 1e-9)

    # 频谱整形：保留脉冲的时域颗粒感，只把频谱推向共振峰形状
    _, _, Z = stft(exc, fs=SR, nperseg=1024, noverlap=768)
    tgt = formant_db(np.fft.rfftfreq(1024, 1 / SR), phoneme)
    tgt = np.clip(tgt - tgt.max(), -40, 0)
    _, y = istft(Z * (10 ** ((tgt * strength) / 20.0))[:, None],
                 fs=SR, nperseg=1024, noverlap=768)
    y = y[:n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    y = sosfilt(butter(4, fmax, btype="lowpass", fs=SR, output="sos"), y)
    return y / (np.abs(y).max() + 1e-9)


def syllable(kana, f0, dur, drops, src, rng, prev_vowel=None):
    """一个音节 = 辅音起音 + 元音主体。"""
    cons, vow = JP.KANA.get(kana, ("", "a"))
    n = int(SR * dur)
    y = vowel(vow, f0, dur, drops, rng=rng)

    # 元音包络：轻微渐入渐出
    a = min(int(SR * 0.010), n // 4)
    r = min(int(SR * 0.045), n // 3)
    env = np.ones(n)
    if a > 0:
        env[:a] = np.linspace(0, 1, a) ** 0.7
    if r > 0:
        env[-r:] = np.linspace(1, 0, r) ** 1.3
    y = y * env

    # 辅音：替换开头（清辅音期间声带本来就不振动）
    if cons:
        c = syn.make_consonant(cons, src, rng, f0)
        if len(c) > 0:
            nc = min(len(c), int(SR * 0.075), n // 2)
            c = c[:nc]
            c = c / (np.abs(c).max() + 1e-9) * 1.25
            fi = min(int(SR * 0.005), nc // 4)
            if fi > 4:
                c[:fi] *= np.linspace(0, 1, fi)
            fo = min(int(SR * 0.015), nc // 3)
            if fo > 4:
                c[-fo:] *= np.linspace(1, 0, fo)
            mix = np.ones(nc)
            if fo > 4:
                mix[-fo:] = np.linspace(1, 0, fo)
            y[:nc] = c * mix + y[:nc] * (1 - mix)
    elif prev_vowel is not None:
        # 元音开头的词，前面给一点点滑入
        fi = min(int(SR * 0.012), n // 4)
        if fi > 4:
            y[:fi] *= np.linspace(0.3, 1, fi)

    y = y - y.mean()
    return y / (np.abs(y).max() + 1e-9) * 0.88


def say(text, f0=392.0, out="say.wav", dur_per=0.42, seed=7):
    srcs = sorted(C.NORMALIZED.glob("*.wav"))
    src, sr = sf.read(str(srcs[0]), dtype="float64")
    if src.ndim > 1:
        src = src.mean(axis=1)
    if sr != SR:
        import librosa
        src = librosa.resample(src, orig_sr=sr, target_sr=SR)
    src = src[: int(SR * 60)]

    drops = syn.extract_drops(src)
    print(f"雨滴 {len(drops)} 个")
    rng = np.random.default_rng(seed)

    kana = [ch for ch in text if ch in JP.KANA]
    print(f"音节 {len(kana)} 个: {' '.join(kana)}")

    parts = []
    for i, k in enumerate(kana):
        # 轻微的音高起伏，避免机械
        fi = f0 * (1.0 + 0.012 * np.sin(i * 1.1))
        # 句尾稍微收一点
        d = dur_per * (1.25 if i == len(kana) - 1 else 1.0)
        parts.append(syllable(k, fi, d, drops, src, rng))
        parts.append(np.zeros(int(SR * 0.012)))
    y = np.concatenate(parts)
    y = y / (np.abs(y).max() + 1e-9) * 0.90
    p = C.PREVIEW / out
    sf.write(str(p), y.astype(np.float32), SR, subtype=C.SUBTYPE)
    print(f"[ok] {p}  {len(y)/SR:.2f}s")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--f0", type=float, default=392.0)
    ap.add_argument("--out", default="say.wav")
    ap.add_argument("--dur", type=float, default=0.42)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    say(a.text, a.f0, a.out, a.dur, a.seed)
