# -*- coding: utf-8 -*-
"""Step 6 唱歌词：拼音 -> 音素 -> 渲染成 wav。

声库只有 a/i/u/e/o/n/v 七个韵母，所以：
  · 韵母按「韵腹」映射到最近的元音（ian->a, ing->i, üe->e ...）
  · 声母单独合成：擦音用原始雨声的高频段，爆破音用短促冲击，
    鼻音/边音用低频段。雨声做擦音天生合适，这是这份音源的独门优势。
  · 韵母交给 voicebank 里的采样，按音符音高变调

用法:
    python 06_sing.py                                  # 内置示例
    python 06_sing.py --text "yu luo xia de sheng yin" --notes 60,62,64,67,69,67,64,62,60,62,64,67
    python 06_sing.py --text "xia yu tian" --bpm 72 --out xia_yu.wav
"""
import argparse

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, resample

import config as C

# ------------------------------------------------------------------ 拼音分析
INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
            "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"]

# 韵母 -> 声库里的音素（取韵腹）
FINAL_MAP = {
    "a": "a", "o": "o", "e": "e", "i": "i", "u": "u", "v": "v",
    "ai": "a", "ei": "e", "ao": "a", "ou": "o",
    "an": "a", "en": "e", "ang": "a", "eng": "e", "er": "e",
    "ia": "a", "ie": "e", "iao": "a", "iou": "o", "iu": "o",
    "ian": "a", "in": "i", "iang": "a", "ing": "i", "iong": "o",
    "ua": "a", "uo": "o", "uai": "a", "uei": "e", "ui": "e",
    "uan": "a", "uen": "e", "un": "u", "uang": "a", "ueng": "e",
    "ve": "e", "van": "a", "vn": "v", "ü": "v", "üe": "e",
    "üan": "a", "ün": "v",
}

# 声母类型：(起音方式, 时长ms, 中心频率, 带宽)
INITIAL_KIND = {
    # 擦音：雨声高通，最自然
    "s": ("fric", 85, 6500, 3500), "sh": ("fric", 90, 4200, 3000),
    "x": ("fric", 85, 5200, 3200), "f": ("fric", 80, 3800, 3200),
    "h": ("fric", 75, 2200, 2600), "c": ("fric", 85, 6200, 3400),
    "ch": ("fric", 90, 4000, 3000), "q": ("fric", 88, 5000, 3200),
    "z": ("affr", 55, 6000, 3500), "zh": ("affr", 58, 3800, 3000),
    "j": ("affr", 55, 4800, 3200),
    # 爆破音：短冲击
    "b": ("stop", 18, 900, 700), "p": ("stop", 42, 1100, 900),
    "d": ("stop", 16, 1500, 1100), "t": ("stop", 40, 1700, 1300),
    "g": ("stop", 18, 800, 600), "k": ("stop", 42, 1000, 800),
    # 鼻音/边音/近音：低频
    "m": ("sono", 70, 300, 300), "n": ("sono", 60, 350, 350),
    "l": ("sono", 55, 500, 500), "r": ("sono", 60, 450, 450),
    "y": ("glide", 45, 2200, 1800), "w": ("glide", 45, 700, 700),
}


def split_pinyin(py):
    py = py.strip().lower().replace("ü", "v")
    for ini in INITIALS:
        if py.startswith(ini) and len(py) > len(ini):
            fin = py[len(ini):]
            if fin in FINAL_MAP:
                return ini, fin
    # 没有声母，或者韵母带 y/w 转换
    for i in range(len(py), 0, -1):
        if py[:i] in FINAL_MAP:
            return "", py[:i]
    return "", "a"


def load_source():
    srcs = sorted(C.NORMALIZED.glob("*.wav"))
    if not srcs:
        raise SystemExit("[!] build/normalized 为空，先跑 01_ingest.py")
    data, sr = sf.read(str(srcs[0]), dtype="float64")
    if sr != C.SR:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=C.SR)
    return data


def make_initial(ini, source, rng):
    """合成一个声母起音。返回 (波形, 是否送气)。"""
    if not ini or ini not in INITIAL_KIND:
        return None, False
    kind, dur_ms, fc, bw = INITIAL_KIND[ini]
    n = int(C.SR * dur_ms / 1000.0)
    n = max(n, 32)
    off = rng.integers(0, max(1, len(source) - n - 1))
    seg = source[off:off + n].copy()
    seg -= seg.mean()

    lo = max(60.0, fc - bw / 2)
    hi = min(C.SR * 0.45, fc + bw / 2)
    if hi <= lo * 1.05:
        lo, hi = 100.0, 4000.0
    sos = butter(2, [lo, hi], btype="bandpass", fs=C.SR, output="sos")
    seg = sosfilt(sos, seg)

    t = np.linspace(0, 1, n)
    if kind == "fric":
        seg *= (0.35 + 0.65 * np.exp(-3.2 * t))
    elif kind == "affr":
        sos2 = butter(2, [lo * 0.7, hi], btype="bandpass", fs=C.SR, output="sos")
        seg = sosfilt(sos2, source[off:off + n])
        seg *= np.exp(-8.0 * t)
    elif kind == "stop":
        # 先用一点静默，再接爆破
        gap = np.zeros(int(C.SR * 0.008))
        seg = seg * np.exp(-22.0 * t)
        seg = np.concatenate([gap, seg])
    elif kind == "sono":
        seg *= np.exp(-1.2 * t)
    else:  # glide
        seg *= np.sin(np.pi * t) ** 0.6

    seg = seg - seg.mean()
    peak = np.abs(seg).max()
    if peak < 1e-9:
        return None, False
    return seg / peak, True


def pick_alias(phoneme, midi):
    target = 440.0 * 2 ** ((midi - 69) / 12)
    best = min(C.TONES, key=lambda t: abs(np.log2(t[1] / target)))
    return f"{phoneme}_{best[0]}"


_bank = {}
_f0map = dict(C.TONES)


def load_bank():
    global _bank
    if _bank:
        return _bank
    for w in C.VOICEBANK.glob("*.wav"):
        _bank[w.stem] = w
    return _bank


def vowel_note(phoneme, midi, dur):
    """取一个韵母采样，变调到目标音高，拉伸到目标时长。"""
    bank = load_bank()
    alias = pick_alias(phoneme, midi)
    if alias not in bank:
        return np.zeros(int(C.SR * dur))

    y, sr = sf.read(str(bank[alias]), dtype="float64")
    if sr != C.SR:
        y = resample(y, int(len(y) * C.SR / sr))

    tone = alias.rpartition("_")[2]
    f0_rec = _f0map[tone]
    f0_tgt = 440.0 * 2 ** ((midi - 69) / 12)
    ratio = f0_tgt / f0_rec          # >1 升调

    # 先改音高（重采样），再改时长（插值）
    if abs(np.log2(ratio)) > 0.01:
        y = resample(y, max(64, int(len(y) / ratio)))
    need = max(64, int(C.SR * dur))
    if len(y) < need:
        loop_start = int(len(y) * 0.55)
        loop = y[loop_start:]
        if len(loop) < 64:
            loop = y[-64:]
        reps = int(np.ceil((need - len(y)) / len(loop))) + 1
        y = np.concatenate([y, np.tile(loop, reps)])
    y = y[:need]

    # 音符包络
    a = int(C.SR * 0.015)
    r = min(int(C.SR * 0.08), len(y) // 3)
    win = np.ones(len(y))
    win[:a] = np.linspace(0, 1, a) ** 0.6
    win[-r:] = np.linspace(1, 0, r) ** 1.3
    return y * win


def render(lyrics, notes, bpm=96):
    source = load_source()
    rng = np.random.default_rng(11)
    beat = 60.0 / bpm
    out = []

    syls = lyrics.split()
    # 音符按音节均分，余数优先给靠后的音节（长音收尾）
    n_syl, n_note = len(syls), len(notes)
    if n_note >= n_syl:
        base, rem = divmod(n_note, n_syl)
        spans, idx = [], 0
        for i in range(n_syl):
            k = base + (1 if i >= n_syl - rem else 0)
            spans.append(notes[idx:idx + k])
            idx += k
    else:
        # 音符少于音节：一个音符切给多个音节
        spans = []
        for i in range(n_syl):
            a = int(i * n_note / n_syl)
            b = max(a + 1, int((i + 1) * n_note / n_syl))
            spans.append(notes[a:b])

    for syl, seq in zip(syls, spans):
        ini, fin = split_pinyin(syl)
        ph = FINAL_MAP.get(fin, "a")

        cons, ok = make_initial(ini, source, rng)
        if cons is not None:
            out.append(0.55 * cons)
            out.append(np.zeros(int(C.SR * 0.004)))   # 极短间隔，避免糊成一坨

        for j, midi in enumerate(seq):
            d = beat if len(seq) == 1 else beat * (0.9 if j else 1.0)
            v = vowel_note(ph, midi, d)
            if j == 0 and cons is not None:
                v = v * 0.98
            out.append(v)
        out.append(np.zeros(int(C.SR * 0.03)))

    y = np.concatenate(out) if out else np.zeros(C.SR)
    y -= y.mean()
    peak = np.abs(y).max() + 1e-9
    y = y / peak * 0.92

    # 整体软限幅，去掉声母冲击造成的尖峰
    y = np.tanh(y * 1.15) / np.tanh(1.15)
    return y.astype(np.float32)


DEMO_TEXT = "yu luo xia de sheng yin hai zai wo er bian"
# 五声音阶，11 音正好对上 11 个音节
DEMO_NOTES = [60, 64, 67, 69, 67, 64, 62, 60, 62, 64, 60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=DEMO_TEXT, help="拼音歌词，空格分隔")
    ap.add_argument("--notes", default=None, help="逗号分隔的 MIDI 音符")
    ap.add_argument("--bpm", type=float, default=96)
    ap.add_argument("--out", default="04_lyrics.wav")
    args = ap.parse_args()

    notes = [int(x) for x in args.notes.split(",")] if args.notes else DEMO_NOTES
    y = render(args.text, notes, args.bpm)

    C.PREVIEW.mkdir(parents=True, exist_ok=True)
    out = C.PREVIEW / args.out
    sf.write(out, y, C.SR, subtype=C.SUBTYPE)
    print(f"[ok] {args.out}  {len(y) / C.SR:.2f}s  {len(args.text.split())} 个音节")
    print(f"     歌词: {args.text}")


if __name__ == "__main__":
    main()
