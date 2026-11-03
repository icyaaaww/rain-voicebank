# -*- coding: utf-8 -*-
"""颗粒合成版：雨声颗粒按 f0 周期排列 -> 既是雨，又有音高。

和之前所有方案的根本差别：

  梳状滤波：噪声通过延迟反馈 -> 频谱长出谐波（滤波器视角）
  颗粒合成：雨声颗粒按 1/f0 间隔重复 -> 波形周期性重复（发声源视角）

后者更接近人声的产生机制：声带脉冲（周期 1/f0）→ 声道滤波（共振峰）。
这里把"声带脉冲"换成"雨声颗粒"，每个颗粒保留雨声的短时频谱。

颗粒间隔 = 1/f0  ->  基频与谐波出现  ->  有音高
颗粒内容 = 雨声   ->  每个周期都是雨   ->  有雨味
"""
import importlib.util
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

sys.path.insert(0, os.path.dirname(__file__))
import config as C

_spec = importlib.util.spec_from_file_location(
    "_syn", os.path.join(os.path.dirname(__file__), "02_synthesize.py"))
_syn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_syn)


def load_rain():
    srcs = sorted(C.NORMALIZED.glob("*.wav"))
    x, sr = sf.read(str(srcs[0]), dtype="float64")
    if sr != C.SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=C.SR)
    return x


def granular_excitation(source, f0, n, sr=C.SR, rng=None,
                        cycles=None, jitter=0.0):
    """雨声颗粒按 1/f0 周期排列，Hann 窗 overlap-add。

    cycles 控制每个颗粒跨几个基频周期（越长音色越稳、越短颗粒感越强）。
    """
    rng = rng or np.random.default_rng(0)
    cycles = cycles or C.GRAIN_CYCLES
    hop = max(8, int(round(sr / f0)))
    glen = int(max(64, round(hop * float(cycles))))
    win = np.hanning(glen)

    out = np.zeros(n + glen)
    pos = 0
    while pos < n:
        # 每次从雨声的不同位置取颗粒，避免完全重复（重复会变机械）
        j = rng.normal(0, jitter) * sr / f0 if jitter > 0 else 0.0
        off = int(rng.integers(0, max(1, len(source) - glen - 1)) + j)
        off = int(np.clip(off, 0, len(source) - glen))
        out[pos:pos + glen] += source[off:off + glen] * win
        pos += hop
    return out[:n]


def rain_grain_voice(phoneme, f0, dur, source, rng, sr=C.SR, female=True):
    n = int(sr * dur) + int(sr * 0.12)
    exc = granular_excitation(source, f0, n, sr, rng,
                              jitter=C.GRAIN_JITTER)

    # 去低频隆隆，颗粒的 DC 会堆积
    exc = sosfilt(butter(2, 70.0, btype="highpass", fs=sr, output="sos"), exc)
    exc = exc / (np.abs(exc).max() + 1e-9)

    # 元音共振峰
    table = C.FORMANTS_FEMALE if female else C.FORMANTS
    v = exc
    for f, bw in table.get(phoneme, []):
        v = _syn.resonator(v, f, bw)
    v = v / (np.abs(v).max() + 1e-9)

    # 谐波支撑：纯颗粒的周期性仍不如真声带，掺一点纯谐波稳住音高
    if C.GRAIN_HARMONIC > 0:
        h = _syn.glottal_excitation(f0, n, sr, rng)
        h = h / (np.abs(h).max() + 1e-9)
        hf = h
        for f, bw in table.get(phoneme, []):
            hf = _syn.resonator(hf, f, bw)
        hf = hf / (np.abs(hf).max() + 1e-9)
        v = (1 - C.GRAIN_HARMONIC) * v + C.GRAIN_HARMONIC * hf

    # 慢速起伏（雨一直在下的呼吸感）
    if C.GRAIN_MOVEMENT > 0:
        eq_s, gain_s = _syn.rain_timbre(source, n)
        g = np.interp(np.arange(n) / C.ENV_HOP, np.arange(len(gain_s)), gain_s)
        g = 1.0 + (g - 1.0) * C.GRAIN_MOVEMENT
        v = v * g

    nn = min(len(v), int(sr * dur))
    v = v[:nn]
    t = np.arange(nn) / sr
    atk = np.clip(t / C.INST_ATTACK, 0, 1) ** 0.6
    rel = np.ones(nn)
    rs = max(0.0, dur - C.INST_RELEASE)
    m = t > rs
    rel[m] = np.clip((dur - t[m]) / C.INST_RELEASE, 0, 1) ** 1.3
    v = v * atk * rel
    v -= v.mean()
    return v / (np.abs(v).max() + 1e-9) * 0.9


DEMOS = {
    "ame": [("a", 60, 1), ("e", 60, 1), ("a", 67, 1), ("e", 67, 1),
            ("u", 69, 1), ("e", 69, 1), ("u", 67, 3)],
    "vowels": [("a", 60, 1), ("i", 62, 1), ("u", 64, 1), ("e", 65, 1),
               ("o", 67, 1), ("a", 69, 1), ("o", 67, 1), ("e", 64, 1), ("a", 60, 2)],
}


def render(seq, bpm, out_name):
    src = load_rain()
    rng = np.random.default_rng(11)
    beat = 60.0 / bpm
    parts = []
    for ph, midi, beats in seq:
        f0 = 440.0 * 2 ** ((midi - 69) / 12)
        parts.append(rain_grain_voice(ph, f0, beat * beats, src, rng))
        parts.append(np.zeros(int(C.SR * 0.01)))
    y = np.concatenate(parts)
    y = y / (np.abs(y).max() + 1e-9) * 0.9
    p = C.PREVIEW / out_name
    sf.write(p, y.astype(np.float32), C.SR, subtype=C.SUBTYPE)
    print(f"[ok] {out_name}  {len(y)/C.SR:.2f}s")
    return p


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "ame"
    render(DEMOS[name], 96, f"grain_{name}.wav")
