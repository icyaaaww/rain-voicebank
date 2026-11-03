# -*- coding: utf-8 -*-
"""雨在唱歌（乐器路线）：给雨声本身加音高，而不是用人声合成去覆盖它。

核心：梳状滤波（comb filter）
    雨声 -> 延迟反馈 -> 频谱长出等间距谐波峰 -> 有音高，但仍然是雨
对比之前的思路（谐波激励 + 元音共振峰）：那是"造一个人声再染色"，
这是"让雨自己发出音高"。前者永远像合成器，后者像玻璃琴/水琴/风管。

用法:
    python 07_rain_instrument.py
    python 07_rain_instrument.py --demo twinkle
"""
import argparse

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

import config as C


def load_rain():
    srcs = sorted(C.NORMALIZED.glob("*.wav"))
    if not srcs:
        raise SystemExit("[!] build/normalized 为空，先跑 01_ingest.py")
    x, sr = sf.read(str(srcs[0]), dtype="float64")
    if sr != C.SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=C.SR)
    return x


def comb(x, f0, sr=C.SR, feedback=0.82, damp=0.35):
    """带阻尼的梳状滤波。damp 越大，高频衰减越快（像真实管体）。"""
    d = sr / f0
    di = int(np.floor(d))
    frac = d - di
    n = len(x)
    y = np.zeros(n)
    # 分数延迟用一阶全通近似
    prev = 0.0
    for i in range(n):
        if i >= di + 1:
            delayed = y[i - di] * (1 - frac) + y[i - di - 1] * frac
        elif i >= di:
            delayed = y[i - di] * (1 - frac)
        else:
            delayed = 0.0
        # 一阶低通做阻尼
        fb = (1 - damp) * delayed + damp * prev
        prev = fb
        y[i] = x[i] + feedback * fb
    return y


def rain_note(source, f0, dur, rng, sr=C.SR):
    """一个音符：雨声片段 -> 梳状滤波 -> 音符包络。"""
    n = int(sr * dur) + int(sr * 0.15)
    if len(source) <= n:
        reps = int(np.ceil(n / len(source))) + 1
        seg = np.tile(source, reps)[:n]
    else:
        off = int(rng.integers(0, len(source) - n))
        seg = source[off:off + n].copy()

    seg = seg - seg.mean()
    # 先去低频隆隆，梳状滤波在低频会糊
    seg = sosfilt(butter(2, 80.0, btype="highpass", fs=sr, output="sos"), seg)

    y = comb(seg, f0, sr, feedback=C.COMB_FEEDBACK, damp=C.COMB_DAMP)

    # 关键：混一点原始噪声回去。纯 comb 会变成"嗡"，留一点才有雨的颗粒
    y = y / (np.abs(y).max() + 1e-9)
    raw = seg / (np.abs(seg).max() + 1e-9)
    out = (1 - C.RAIN_BLEND) * y + C.RAIN_BLEND * raw

    # 包络：慢起（像雨落在共鸣体上逐步激振），慢收
    nn = int(sr * dur)
    out = out[:nn]
    t = np.arange(nn) / sr
    atk = np.clip(t / C.INST_ATTACK, 0, 1) ** 0.7
    rel = np.ones(nn)
    rs = max(0.0, dur - C.INST_RELEASE)
    m = t > rs
    rel[m] = np.clip((dur - t[m]) / C.INST_RELEASE, 0, 1) ** 1.4
    out = out * atk * rel

    out -= out.mean()
    return out / (np.abs(out).max() + 1e-9) * 0.9


def render(notes, bpm, out_name):
    source = load_rain()
    rng = np.random.default_rng(2026)
    beat = 60.0 / bpm
    parts = []
    for midi, beats in notes:
        f0 = 440.0 * 2 ** ((midi - 69) / 12)
        parts.append(rain_note(source, f0, beat * beats, rng))
        parts.append(np.zeros(int(C.SR * 0.01)))
    y = np.concatenate(parts)
    y = y / (np.abs(y).max() + 1e-9) * 0.9
    p = C.PREVIEW / out_name
    sf.write(p, y.astype(np.float32), C.SR, subtype=C.SUBTYPE)
    print(f"[ok] {out_name}  {len(y)/C.SR:.2f}s  {len(notes)} 个音")
    return p


DEMOS = {
    # 和歌词版同样的旋律，方便直接对比
    "ame": [(60, 1), (60, 1), (67, 1), (67, 1), (69, 1), (69, 1), (67, 3)],
    # 五声音阶铺开，适合听"雨的音色"
    "scale": [(60, 1), (62, 1), (64, 1), (67, 1), (69, 1),
              (72, 1), (69, 1), (67, 1), (64, 1), (60, 2)],
    "twinkle": [(60, 1), (60, 1), (67, 1), (67, 1), (69, 1), (69, 1), (67, 2),
                (65, 1), (65, 1), (64, 1), (64, 1), (62, 1), (62, 1), (60, 2)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", choices=list(DEMOS), default="ame")
    ap.add_argument("--bpm", type=float, default=96)
    args = ap.parse_args()

    p = render(DEMOS[args.demo], args.bpm, f"rain_inst_{args.demo}.wav")
    print(f"     {p}")


if __name__ == "__main__":
    main()
