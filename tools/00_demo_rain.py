# -*- coding: utf-8 -*-
"""没有雨声素材时，先用这个造一段"假雨"跑通全流程（冒烟测试用）。

生成的 raw/demo_rain.wav 只是合成信号，质感接近中雨。
真正发布声库当然要用你自己录/采的真雨。
"""
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

import config as C

rng = np.random.default_rng(20261101)
dur = 40.0
n = int(C.SR * dur)
t = np.arange(n) / C.SR


def pink_noise(n, rng):
    """1/f 噪声，频谱整形法。"""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / C.SR)
    f[0] = f[1]
    spec /= np.sqrt(f)
    out = np.fft.irfft(spec, n)
    return out / (np.abs(out).max() + 1e-9)


# --- 底层：粉红噪声雨幕 ---
bed = pink_noise(n, rng)

# 雨幕的缓慢起伏（阵风感）
lfo = np.zeros(n)
for k, (rate, amp) in enumerate([(0.07, 0.45), (0.19, 0.25), (0.53, 0.12)], start=1):
    phase = rng.uniform(0, 2 * np.pi)
    lfo += amp * np.sin(2 * np.pi * rate * t + phase)
bed *= np.clip(1.0 + 0.6 * lfo, 0.25, 1.8)

# --- 雨滴瞬态：随机短促爆点 ---
drops = np.zeros(n)
n_drops = int(dur * 90)          # 每秒约 90 滴
pos = rng.integers(0, n - C.SR // 4, size=n_drops)
for p in pos:
    L = int(C.SR * rng.uniform(0.004, 0.022))
    if p + L >= n:
        continue
    env = np.exp(-np.linspace(0, 6, L))
    burst = rng.standard_normal(L) * env * rng.uniform(0.25, 1.0)
    drops[p:p + L] += burst
drops *= 0.35

mix = bed + drops

# --- 带限，去掉过低的隆隆和过高的嘶声 ---
sos_hp = butter(4, 180.0, btype="highpass", fs=C.SR, output="sos")
sos_lp = butter(6, 9000.0, btype="lowpass", fs=C.SR, output="sos")
mix = sosfilt(sos_hp, mix)
mix = sosfilt(sos_lp, mix)

# --- 去直流 + 归一 ---
mix -= mix.mean()
mix = mix / (np.abs(mix).max() + 1e-9) * 0.89

out = C.RAW / "demo_rain.wav"
sf.write(out, mix.astype(np.float32), C.SR, subtype=C.SUBTYPE)
print(f"[ok] demo rain -> {out}  ({dur:.0f}s, {C.SR}Hz)")
