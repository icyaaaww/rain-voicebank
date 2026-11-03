# -*- coding: utf-8 -*-
"""谐波包络合成：在频域直接给每个谐波分配幅度。

为什么换这个方法：

  之前用 IIR 谐振器（apply_formants）去"滤"颗粒/脉冲激励。
  滤波器只能强调**已经存在**的谐波。基频 261Hz 时，第 1 次谐波 261、
  第 2 次 522——u 的 F1=420 和 o 的 F1=550 之间根本没有谐波，
  滤波器再准也塑造不出来，结果五个元音频谱相似度 0.95（听不出区别）。

  Source-filter 模型的标准做法是：共振峰信息体现在**谐波的相对幅度**上。
  所以这里直接对每个谐波 k·f0 计算：
      幅度 = 共振峰响应(k·f0) × 雨声频谱包络(k·f0)
  这样 F1 的位置变化会改变相邻谐波的强弱比，人耳就能分辨元音。

  第二项同时带进"雨"的音色——不是叠加噪声，而是谐波本身的包络像雨。
"""
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

sys.path.insert(0, os.path.dirname(__file__))
import config as C


# ---------------------------------------------------------------- 频谱包络
def rain_envelope(source, sr=C.SR, n_bands=48, fmin=80.0, fmax=10000.0):
    """雨声的 1/3 倍频程频谱包络（对数域，返回中心频率与 dB）。"""
    seg = source[: int(sr * 4.0)] if len(source) > sr * 4 else source
    seg = seg * np.hanning(len(seg))
    S = np.abs(np.fft.rfft(seg, n=1 << 16)) ** 2
    f = np.fft.rfftfreq(1 << 16, 1 / sr)
    centers = np.geomspace(fmin, fmax, n_bands)
    db = np.empty(n_bands)
    for i, fc in enumerate(centers):
        m = (f >= fc / 1.12) & (f < fc * 1.12)
        db[i] = 10 * np.log10(S[m].mean() + 1e-12) if m.sum() else -120.0
    # 轻度平滑，避免包络太抖导致谐波幅度乱跳
    k = np.hanning(5)
    db = np.convolve(db, k / k.sum(), mode="same")
    db -= db.max()
    return centers, db


# ---------------------------------------------------------------- 共振峰响应
def formant_response(freqs, formants):
    """每个共振峰用高斯峰近似，返回线性幅度（可多个峰叠加）。"""
    gain = np.zeros_like(freqs, dtype=np.float64)
    for fc, bw in formants:
        gain += np.exp(-0.5 * ((freqs - fc) / (bw * 1.2)) ** 2)
    return np.maximum(gain, 1e-4)


# ---------------------------------------------------------------- 合成
def synth_vowel(phoneme, f0, dur, env_f, env_db, rng=None, sr=C.SR,
                air=0.08, tilt=1.2, buzz=0.18):
    """按谐波包络合成一个元音。

    air  : 雨幕气声底的比例
    tilt : 谐波自然衰减（1.4 ≈ -8.5dB/oct，接近女声）
    buzz : 额外叠一点雨滴瞬态，保住"雨落下来"的颗粒感
    """
    rng = rng or np.random.default_rng(0)
    n = int(sr * dur)
    t = np.arange(n) / sr

    fmt = C.FORMANTS_ACTIVE.get(phoneme) or C.FORMANTS_ACTIVE["a"]

    nyq = sr / 2.0
    kmax = int(nyq / f0)
    freqs = f0 * np.arange(1, kmax + 1)
    keep = freqs < nyq * 0.95
    freqs = freqs[keep]
    if len(freqs) == 0:
        return np.zeros(n)

    fr_gain = formant_response(freqs, fmt) ** 4.0
    # 共振峰要给到 4 次方才有足够的区分度：一次性方时 u/o 相似度 0.95（听不出），
    # 四次方降到 0.37。雨声包络只占 20%——多了会把共振峰压平，少了就没雨味。
    env_lin = 10 ** (np.interp(np.log(freqs), np.log(env_f), env_db) / 20.0)
    env_lin = env_lin / (env_lin.max() + 1e-12)

    # 谐波幅度 = 共振峰(强化) × 雨声包络(轻) × 自然衰减
    amp = fr_gain * (0.8 + 0.2 * env_lin) / (np.arange(1, len(freqs) + 1) ** tilt)
    amp = amp / (amp.max() + 1e-12)

    y = np.zeros(n)
    for i, f in enumerate(freqs):
        if amp[i] < 1e-3:
            continue
        phase = rng.uniform(0, 2 * np.pi)
        y += amp[i] * np.sin(2 * np.pi * f * t + phase)

    y = y / (np.abs(y).max() + 1e-9)

    # 雨滴瞬态：保住颗粒感
    if buzz > 0:
        hop = max(8, int(round(sr / f0)))
        win = np.exp(-np.linspace(0.0, 6.0, hop))
        atk = max(2, int(hop * 0.08))
        win[:atk] *= np.linspace(0.0, 1.0, atk)
        b = np.zeros(n + hop)
        pos = 0
        off = int(rng.integers(0, max(1, len(y) - hop)))
        while pos < n:
            seg = y[off:off + hop]
            if len(seg) < hop:
                seg = np.pad(seg, (0, hop - len(seg)))
            r = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
            b[pos:pos + hop] += seg / r * win
            pos += hop
        b = b[:n]
        b = b / (np.abs(b).max() + 1e-9)
        y = (1 - buzz) * y + buzz * b

    # 气声底
    if air > 0:
        a = sosfilt(butter(4, 3000.0, btype="lowpass", fs=sr, output="sos"),
                    rng.normal(0, 1, n))
        a = a - a.mean()
        y = y + a / (np.sqrt(np.mean(a ** 2)) + 1e-9) * air * np.sqrt(np.mean(y ** 2))

    a_n = min(int(sr * 0.008), n // 4)
    r_n = min(int(sr * 0.030), n // 4)
    env = np.ones(n)
    if a_n > 0:
        env[:a_n] = np.linspace(0, 1, a_n) ** 0.7
    if r_n > 0:
        env[-r_n:] = np.linspace(1, 0, r_n) ** 1.2
    y = y * env
    y -= y.mean()
    return y / (np.abs(y).max() + 1e-9) * 0.89


DEMOS = {
    "ame": [("a", 60, 1), ("e", 60, 1), ("a", 67, 1), ("e", 67, 1),
            ("u", 69, 1), ("e", 69, 1), ("u", 67, 3)],
    "vowels": [("a", 60, 1), ("i", 62, 1), ("u", 64, 1), ("e", 65, 1),
               ("o", 67, 1), ("a", 69, 1)],
}


def load_rain():
    srcs = sorted(C.NORMALIZED.glob("*.wav"))
    x, sr = sf.read(str(srcs[0]), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != C.SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=C.SR)
    return x


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "ame"
    src = load_rain()
    env_f, env_db = rain_envelope(src)
    print(f"雨声包络: {len(env_f)} 个频段, {env_f[0]:.0f}-{env_f[-1]:.0f} Hz")

    rng = np.random.default_rng(7)
    beat = 60.0 / 96
    parts = []
    for ph, midi, beats in DEMOS[name]:
        f0 = 440.0 * 2 ** ((midi - 69) / 12)
        parts.append(synth_vowel(ph, f0, beat * beats, env_f, env_db, rng))
    y = np.concatenate(parts)
    y = y / (np.abs(y).max() + 1e-9) * 0.9
    out = C.PREVIEW / f"harm_{name}.wav"
    sf.write(str(out), y.astype(np.float32), C.SR, subtype=C.SUBTYPE)
    print(f"[ok] {out}  {len(y)/C.SR:.2f}s")


if __name__ == "__main__":
    main()
