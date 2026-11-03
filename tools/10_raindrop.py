# -*- coding: utf-8 -*-
"""雨滴声门脉冲版：HPSS 分离 + onset 检测 + 周期重排。

和颗粒版的根本差别：

  颗粒版：从雨声随机位置取 23ms 片段（6 个 f0 周期），Hann 窗 overlap-add。
          片段长度远大于一个周期，内部相位/能量都随机 → 短时起伏 18dB。

  本版：  HPSS 把雨声拆成"雨幕(H)"和"雨滴(P)"，onset 检测出每一滴雨的时刻，
          裁成正好一个 f0 周期长（3-4ms），加 Hann 窗后首尾相接地排列。
          每个周期就是一"滴"，间隔精确等于 1/f0 → 周期性完美，包络不跳。

对应关系：
  雨滴瞬态  = 声带脉冲
  少女共振峰 = 声道
  雨幕      = 气声底
"""
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

sys.path.insert(0, os.path.dirname(__file__))
import config as C


# ---------------------------------------------------------------- 雨滴字典
def extract_drops(source, sr=C.SR, drop_ms=30.0, max_drops=96):
    """HPSS 分离出打击性成分，onset 检测，裁出每一滴雨。"""
    import librosa

    y = source.astype(np.float32)
    if len(y) > sr * 60:
        y = y[: sr * 60]

    print("  HPSS 分离 ...")
    _, P = librosa.effects.hpss(y)
    print("  onset 检测 ...")
    on = librosa.onset.onset_detect(y=P, sr=sr, units="samples", backtrack=True)
    print(f"  检出 {len(on)} 个雨滴")

    L = int(sr * drop_ms / 1000.0)
    drops = []
    for o in on:
        if o + L > len(P):
            continue
        seg = P[o:o + L].astype(np.float64).copy()
        seg -= seg.mean()
        pk = np.abs(seg).max()
        if pk < 1e-6:
            continue
        # 只保留起音快、衰减明显的（真正的瞬态，不是噪声块）
        head = np.abs(seg[: L // 6]).max()
        tail = np.abs(seg[-L // 3:]).max()
        if head < tail * 0.8:
            continue
        drops.append(seg / pk)
    print(f"  有效雨滴 {len(drops)} 个")
    return drops[:max_drops]


# ---------------------------------------------------------------- 激励
def drop_excitation(drops, f0, n, sr=C.SR, rng=None):
    """把雨滴裁成一个 f0 周期的长度，加 Hann 窗，首尾相接排列。"""
    rng = rng or np.random.default_rng(0)
    hop = max(8, int(round(sr / f0)))
    # 不用 Hann：它覆盖整个周期、能量集中在窗心，5ms 窗口一测就是周期性起伏。
    # 真实声门脉冲是'起音极快 + 指数衰减'，所以用一个陡起音 + 快衰减的窗。
    win = np.exp(-np.linspace(0.0, 6.0, hop))
    _atk = max(2, int(hop * 0.08))
    win[:_atk] *= np.linspace(0.0, 1.0, _atk)
    out = np.zeros(n + hop)

    pos = 0
    while pos < n:
        d = drops[int(rng.integers(0, len(drops)))]
        # 把 30ms 的雨滴重采样成 hop（一个周期）长
        g = np.interp(np.linspace(0, len(d) - 1, hop), np.arange(len(d)), d)
        # 归一化必须放在加窗之后：加窗本身会改变能量，先归一再加窗等于白做
        g = g * win
        _r = float(np.sqrt(np.mean(g ** 2))) + 1e-9
        out[pos:pos + hop] += g / _r
        pos += hop
    return out[:n]


def resonator(x, f, bw, sr=C.SR):
    r = np.exp(-np.pi * bw / sr)
    th = 2 * np.pi * f / sr
    a = [1.0, -2 * r * np.cos(th), r * r]
    b = [1.0 - r]
    from scipy.signal import lfilter
    return lfilter(b, a, x)


def formants(x, phoneme, female=True):
    table = C.FORMANTS_FEMALE if female else C.FORMANTS
    y = x
    for f, bw in table.get(phoneme, []):
        y = resonator(y, f, bw)
    return y


def glottal(f0, n, sr=C.SR, rng=None):
    """参考用：纯谐波激励。"""
    t = np.arange(n) / sr
    y = np.zeros(n)
    for k in range(1, int(min(40, sr / 2 / f0)) + 1):
        if k * f0 > sr / 2:
            break
        y += np.sin(2 * np.pi * k * f0 * t) / k ** (C.HARMONIC_TILT / 2)
    return y


# ---------------------------------------------------------------- 合成一个音
def synth(phoneme, f0, dur, drops, source, rng, sr=C.SR,
          drop_mix=0.80, female=True):
    n = int(sr * dur) + int(sr * 0.10)

    exc = drop_excitation(drops, f0, n, sr, rng)
    exc = sosfilt(butter(2, 90.0, btype="highpass", fs=sr, output="sos"), exc)
    exc = exc / (np.abs(exc).max() + 1e-9)
    xd = formants(exc, phoneme, female)
    xd = xd / (np.abs(xd).max() + 1e-9)

    # 谐波支撑：雨滴脉冲本身周期性已经很好，只需要少量
    if drop_mix < 1.0:
        h = glottal(f0, n, sr, rng)
        h = h / (np.abs(h).max() + 1e-9)
        xh = formants(h, phoneme, female)
        xh = xh / (np.abs(xh).max() + 1e-9)
        x = drop_mix * xd + (1 - drop_mix) * xh
    else:
        x = xd

    # 气声底：雨幕（低通后的雨声）
    if C.RAINDROP_AIR > 0:
        air = sosfilt(butter(4, 2500.0, btype="lowpass", fs=sr, output="sos"),
                      source[:n] if len(source) >= n else
                      np.pad(source, (0, n - len(source))))
        air = air - air.mean()
        ar = np.sqrt(np.mean(air ** 2)) + 1e-9
        xr = np.sqrt(np.mean(x ** 2)) + 1e-9
        x = x + air / ar * xr * C.RAINDROP_AIR

    # 共振峰之后做轻度短时压缩：不同雨滴的频谱经不同共振后输出能量不同，
    # 这是逐周期能量起伏的最后来源。压到原唱那个量级（约 2-3dB）。
    _w = max(16, int(sr * 0.020))
    _e = np.sqrt(np.convolve(x ** 2, np.ones(_w) / _w, mode="same") + 1e-12)
    _w2 = max(16, int(sr * 0.100))
    _ref = np.sqrt(np.convolve(x ** 2, np.ones(_w2) / _w2, mode="same") + 1e-12)
    _g = np.clip((np.maximum(_ref, 1e-6) / np.maximum(_e, 1e-6)) ** 0.55, 0.3, 3.5)
    x = x * _g

    x = x[:int(sr * dur)]
    t = np.arange(len(x)) / sr
    # 淡入淡出必须极短：每个音 20/40ms 包络拼起来就是周期性凹陷（11.75dB）。
    # 音符之间的衔接交给调用方的交叉淡化，不在这里做。
    a = min(int(sr * 0.006), len(x) // 4)
    r = min(int(sr * 0.006), len(x) // 4)
    env = np.ones(len(x))
    if a > 0:
        env[:a] = np.linspace(0, 1, a) ** 0.7
    if r > 0:
        env[-r:] = np.linspace(1, 0, r) ** 1.3
    x = x * env
    x -= x.mean()
    return x / (np.abs(x).max() + 1e-9) * 0.89


# ---------------------------------------------------------------- demo
def load_rain():
    srcs = sorted(C.NORMALIZED.glob("*.wav"))
    x, sr = sf.read(str(srcs[0]), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != C.SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=C.SR)
    return x


DEMOS = {
    "ame": [("a", 60, 1), ("e", 60, 1), ("a", 67, 1), ("e", 67, 1),
            ("u", 69, 1), ("e", 69, 1), ("u", 67, 3)],
}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "ame"
    src = load_rain()
    drops = extract_drops(src)
    if not drops:
        raise SystemExit("没有提取到雨滴，检查素材")

    rng = np.random.default_rng(7)
    beat = 60.0 / 96
    parts = []
    for ph, midi, beats in DEMOS[name]:
        f0 = 440.0 * 2 ** ((midi - 69) / 12)
        parts.append(synth(ph, f0, beat * beats, drops, src, rng))
        pass   # 不留间隙：靠包络的极短淡出衔接
    y = np.concatenate(parts)
    y = y / (np.abs(y).max() + 1e-9) * 0.90
    out = C.PREVIEW / f"raindrop_{name}.wav"
    sf.write(str(out), y.astype(np.float32), C.SR, subtype=C.SUBTYPE)
    print(f"[ok] {out}  {len(y)/C.SR:.2f}s")


if __name__ == "__main__":
    main()
