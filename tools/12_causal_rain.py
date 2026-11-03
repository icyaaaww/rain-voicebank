# -*- coding: utf-8 -*-
"""雨在唱歌（因果驱动版）。

核心思路的转变：

  之前十几版都在"合成一个既是雨又是人声的信号"——频谱融合。全部失败，
  因为人耳的听觉场景分析（ASA）判断声源数量靠的是"是否共同调制"，
  不是频谱相似度。

  这版改成参数映射：雨声是真实雨声，但它的每一个参数都由歌声驱动：
      音高   -> 雨滴密度（高音更密更亮）
      音量   -> 雨滴强度
      元音   -> 雨滴的频谱重心（a 低、i 高）
      起音   -> 音节开始时撒一阵密集雨滴

  人声保持清晰可懂，雨声保持真实——但雨在跟着歌动，
  听觉上归因为"雨在唱这首歌"。
"""
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

sys.path.insert(0, os.path.dirname(__file__))
import config as C
import japanese as JP
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "syn", os.path.join(os.path.dirname(__file__), "02_synthesize.py"))
syn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(syn)

SR = C.SR

# 元音 -> 雨滴频谱重心（相对倍率）
VOWEL_BRIGHT = {"a": 0.85, "i": 1.55, "u": 0.70, "e": 1.20, "o": 0.75}


def load_rain():
    srcs = sorted(C.NORMALIZED.glob("*.wav"))
    x, sr = sf.read(str(srcs[0]), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x[: int(SR * 60)]


def drop_cloud(drops, t0, t1, density, bright=1.0, rng=None, sr=SR):
    """在 [t0,t1] 里撒雨滴：密度决定颗数，bright 决定频谱重心。"""
    rng = rng or np.random.default_rng(0)
    n = int((t1 - t0) * sr)
    if n <= 0:
        return np.zeros(0)
    out = np.zeros(n + sr // 4)
    n_drops = max(1, int((t1 - t0) * density))
    for _ in range(n_drops):
        d = drops[int(rng.integers(0, len(drops)))]
        # 通过重采样改变亮度和长度
        ratio = bright * rng.uniform(0.85, 1.2)
        L = max(32, int(len(d) / ratio))
        g = np.interp(np.linspace(0, len(d) - 1, L), np.arange(len(d)), d)
        g = g * np.hanning(L) ** 0.5
        pos = int(rng.integers(0, max(1, n - 1)))
        amp = rng.uniform(0.35, 1.0)
        end = min(len(out), pos + L)
        out[pos:end] += g[: end - pos] * amp
    y = out[:n]
    pk = np.abs(y).max()
    return y / pk if pk > 1e-9 else y


def vocal_syllable(kana, f0, dur, drops, src, rng, strength=1.2, fmax=4000.0):
    """人声：雨滴脉冲 + 频谱整形（可懂度最好的一版）。"""
    cons, vow = JP.KANA.get(kana, ("", "a"))
    n = int(SR * dur)
    exc = syn.drop_excitation(drops, f0, n, SR, rng)
    exc = sosfilt(butter(2, 90.0, btype="highpass", fs=SR, output="sos"), exc)
    exc = exc / (np.abs(exc).max() + 1e-9)
    from scipy.signal import stft, istft
    _, _, Z = stft(exc, fs=SR, nperseg=1024, noverlap=768)
    g = np.zeros_like(np.fft.rfftfreq(1024, 1 / SR))
    ff = np.fft.rfftfreq(1024, 1 / SR)
    for fc, bw in C.FORMANTS_ACTIVE.get(vow, C.FORMANTS_ACTIVE["a"]):
        g += np.exp(-0.5 * ((ff - fc) / (bw * 1.5)) ** 2)
    db = 20 * np.log10(np.maximum(g, 1e-3))
    db = np.clip(db - db.max(), -40, 0)
    _, y = istft(Z * (10 ** ((db * strength) / 20.0))[:, None],
                 fs=SR, nperseg=1024, noverlap=768)
    y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    y = sosfilt(butter(4, fmax, btype="lowpass", fs=SR, output="sos"), y)
    y /= (np.abs(y).max() + 1e-9)

    if cons:
        c = syn.make_consonant(cons, src, rng, f0)
        if len(c) > 0:
            nc = min(len(c), int(SR * 0.07), n // 2)
            c = c[:nc] / (np.abs(c[:nc]).max() + 1e-9) * 1.2
            mix = np.ones(nc)
            fo = min(int(SR * 0.015), nc // 3)
            if fo > 4:
                mix[-fo:] = np.linspace(1, 0, fo)
            y[:nc] = c * mix + y[:nc] * (1 - mix)

    a = min(int(SR * 0.008), n // 4)
    r = min(int(SR * 0.04), n // 3)
    env = np.ones(n)
    env[:a] = np.linspace(0, 1, a) ** 0.7
    env[-r:] = np.linspace(1, 0, r) ** 1.3
    return y * env / (np.abs(y * env).max() + 1e-9) * 0.85


def sing(text, f0=392.0, dur=0.42, seed=7, rain_gain=0.55,
         density_lo=14.0, density_hi=70.0):
    src = load_rain()
    drops = syn.extract_drops(src)
    print(f"雨滴 {len(drops)} 个")
    rng = np.random.default_rng(seed)

    kana = [ch for ch in text if ch in JP.KANA]
    total = int(SR * (dur * len(kana) + 1.0))
    vocal = np.zeros(total)
    rain = np.zeros(total)

    t = 0.28
    for i, k in enumerate(kana):
        cons, vow = JP.KANA.get(k, ("", "a"))
        fi = f0 * (1.0 + 0.012 * np.sin(i * 1.1))
        d = dur * (1.3 if i == len(kana) - 1 else 1.0)

        v = vocal_syllable(k, fi, d, drops, src, rng)
        a0 = int(t * SR)
        b0 = min(total, a0 + len(v))
        vocal[a0:b0] += v[: b0 - a0]

        # ---- 雨滴被歌声驱动 ----
        pitch_n = np.clip((fi - 200) / 500.0, 0, 1)          # 音高归一
        dens = density_lo + (density_hi - density_lo) * pitch_n
        bright = VOWEL_BRIGHT.get(vow, 1.0)
        cloud = drop_cloud(drops, t, t + d * 1.05, dens, bright, rng)
        b1 = min(total, a0 + len(cloud))
        rain[a0:b1] += cloud[: b1 - a0]

        # 起音加强：音节第一个 60ms 撒一阵密集雨滴
        burst = drop_cloud(drops, t, t + 0.06, dens * 2.2, bright * 1.3, rng)
        b2 = min(total, a0 + len(burst))
        rain[a0:b2] += burst[: b2 - a0] * 0.8

        t += d + 0.012

    # 归一化并混合
    vocal = vocal / (np.abs(vocal).max() + 1e-9) * 0.85
    rain = rain / (np.abs(rain).max() + 1e-9) * 0.85
    # 雨声也做低通，避免刺耳
    rain = sosfilt(butter(4, 6000, btype="lowpass", fs=SR, output="sos"), rain)
    mix = vocal + rain * rain_gain
    mix = mix / (np.abs(mix).max() + 1e-9) * 0.92
    return mix, vocal, rain


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "わたしはあめ"
    out = sys.argv[2] if len(sys.argv) > 2 else "causal.wav"
    mix, voc, rain = sing(text)
    sf.write(str(C.PREVIEW / out), mix.astype(np.float32), SR, subtype=C.SUBTYPE)
    sf.write(str(C.PREVIEW / ("voc_" + out)), voc.astype(np.float32), SR, subtype=C.SUBTYPE)
    sf.write(str(C.PREVIEW / ("rain_" + out)), rain.astype(np.float32), SR, subtype=C.SUBTYPE)
    print(f"[ok] mix/vocal/rain -> {out}  {len(mix)/SR:.2f}s")
