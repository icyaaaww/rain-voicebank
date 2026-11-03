# -*- coding: utf-8 -*-
"""Step 5 试听：把采样按音符拼成一段旋律，模拟 resampler 的"拉伸+变调"。

不依赖 OpenUtau，先听为敬。输出 build/preview/*.wav
用法:
    python 05_preview.py            # 琶音 + 小星星 + 中文音节
    python 05_preview.py --note-a   # 只出琶音
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import config as C


def load_bank():
    """alias -> (samples, sr)，按需懒加载。"""
    bank = {}
    for w in C.VOICEBANK.glob("*.wav"):
        if w.stem.lower() in ("icon", "portrait"):
            continue
        bank[w.stem] = w
    return bank


def pick_alias(phoneme, midi):
    """选最接近目标音高的 subbank，模拟 OpenUtau 的 subbank 选择。"""
    target = 440.0 * 2 ** ((midi - 69) / 12)
    best = min(C.TONES, key=lambda t: abs(np.log2(t[1] / target)))
    return f"{phoneme}_{best[0]}"


def note_to_audio(phoneme, midi, dur, bank, sr=C.SR):
    alias = pick_alias(phoneme, midi)
    if alias not in bank:
        return np.zeros(int(sr * dur))
    x, src_sr = sf.read(str(bank[alias]), dtype="float64")
    if src_sr != sr:
        x = resample_poly(x, sr, src_sr)

    # 取 target f0 与采样录制音高之比，做重采样变调（=UTAU resampler 的核心动作）
    tone = alias.rpartition("_")[2]
    f0_rec = dict(C.TONES)[tone]
    f0_tgt = 440.0 * 2 ** ((midi - 69) / 12)
    ratio = f0_rec / f0_tgt

    # 先按比例重采样改音高，再线性插值改时长
    n_out = max(1, int(len(x) * ratio))
    x = np.interp(
        np.linspace(0, len(x) - 1, n_out),
        np.arange(len(x)),
        x,
    )

    need = int(sr * dur)
    if len(x) < need:
        # 尾部循环补齐（简化版 wavtool 行为）
        loop_start = int(len(x) * 0.55)
        loop = x[loop_start:]
        reps = int(np.ceil((need - len(x)) / max(1, len(loop)))) + 1
        x = np.concatenate([x, np.tile(loop, reps)])
    x = x[:need]

    # 音符包络，避免拼接爆音
    a = int(sr * 0.008)
    r = int(sr * 0.05)
    env = np.ones(len(x))
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if r > 0 and len(x) > r:
        env[-r:] = np.linspace(1, 0, r)
    return x * env


def render(seq, bank, sr=C.SR, bpm=110):
    """seq: [(phoneme, midi, beats), ...]"""
    beat = 60.0 / bpm
    out = []
    for ph, midi, beats in seq:
        out.append(note_to_audio(ph, midi, beat * beats, bank, sr))
        out.append(np.zeros(int(sr * 0.02)))
    y = np.concatenate(out)
    peak = np.abs(y).max() + 1e-9
    return (y / peak * 0.9).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note-a", action="store_true")
    args = ap.parse_args()

    bank = load_bank()
    if not bank:
        print("[!] voicebank/ 里没有 wav")
        return

    C.PREVIEW.mkdir(parents=True, exist_ok=True)

    # 1) 五度琶音，听音高
    arp = [("a", m, 0.6) for m in (60, 67, 72, 67, 60)] + \
          [("o", m, 0.6) for m in (62, 69, 74, 69, 62)]
    y = render(arp, bank)
    sf.write(C.PREVIEW / "01_arpeggio.wav", y, C.SR, subtype=C.SUBTYPE)
    print("[ok] 01_arpeggio.wav")

    if args.note_a:
        return

    # 2) 小星星（元音当歌词）
    star = [("a", 60, 1), ("a", 60, 1), ("a", 67, 1), ("a", 67, 1),
            ("i", 69, 1), ("i", 69, 1), ("a", 67, 2),
            ("e", 65, 1), ("e", 65, 1), ("o", 64, 1), ("o", 64, 1),
            ("u", 62, 1), ("u", 62, 1), ("a", 60, 2)]
    y = render(star, bank, bpm=120)
    sf.write(C.PREVIEW / "02_twinkle.wav", y, C.SR, subtype=C.SUBTYPE)
    print("[ok] 02_twinkle.wav")

    # 3) 音阶扫，检查各音区衔接
    scale = [("a", m, 0.5) for m in range(60, 73)]
    scale += [("a", m, 0.5) for m in range(72, 59, -1)]
    y = render(scale, bank, bpm=140)
    sf.write(C.PREVIEW / "03_scale.wav", y, C.SR, subtype=C.SUBTYPE)
    print("[ok] 03_scale.wav")

    print(f"\n试听目录: {C.PREVIEW}")


if __name__ == "__main__":
    main()
