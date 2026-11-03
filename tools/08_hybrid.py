# -*- coding: utf-8 -*-
"""混合路线：雨声 -> 梳状滤波(出音高) -> 元音共振峰(出元音)。

和之前的区别只在激励源：
    之前：人造谐波堆 + 噪声  -> 元音共振峰      （造一个人声再染色）
    混合：雨声 -> comb(f0)   -> 元音共振峰      （让雨自己共振出元音）

保留可唱性（元音、能配词），同时音色来自雨声的随机颗粒。
"""
import importlib.util
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, lfilter

sys.path.insert(0, __import__("os").path.dirname(__file__))
import config as C

_spec = importlib.util.spec_from_file_location(
    "_syn", __import__("os").path.join(__import__("os").path.dirname(__file__), "02_synthesize.py"))
_syn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_syn)


def load_rain():
    srcs = sorted(C.NORMALIZED.glob("*.wav"))
    if not srcs:
        raise SystemExit("[!] build/normalized 为空")
    x, sr = sf.read(str(srcs[0]), dtype="float64")
    if sr != C.SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=C.SR)
    return x


def comb_fast(x, f0, sr=C.SR, feedback=None):
    """用 lfilter 实现梳状滤波，比逐样本循环快两个数量级。

    y[n] = x[n] + fb * y[n-d]，d = sr/f0
    """
    fb = C.COMB_FEEDBACK if feedback is None else feedback
    d = max(1, int(round(sr / f0)))
    a = np.zeros(d + 1)
    a[0] = 1.0
    a[d] = -fb
    return lfilter(np.array([1.0]), a, x)


def rain_vowel(phoneme, f0, dur, source, rng, sr=C.SR, female=False):
    n = int(sr * dur) + int(sr * 0.12)
    if len(source) <= n:
        seg = np.tile(source, int(np.ceil(n / len(source))) + 1)[:n]
    else:
        off = int(rng.integers(0, len(source) - n))
        seg = source[off:off + n].copy()
    seg = seg - seg.mean()
    seg = sosfilt(butter(2, 80.0, btype="highpass", fs=sr, output="sos"), seg)

    pitched = comb_fast(seg, f0, sr)
    # 阻尼：高频滚降，否则梳状滤波的谐波一直延伸到 20k，刺耳
    fc = C.HYB_DAMP_FC * (1.25 if female else 1.0)   # 女声保留更多高频亮度
    pitched = sosfilt(butter(2, min(fc, sr * 0.45), btype="lowpass", fs=sr, output="sos"), pitched)
    pitched = pitched / (np.abs(pitched).max() + 1e-9)

    # 元音共振峰（按性别换表）
    table = C.FORMANTS_FEMALE if female else C.FORMANTS
    v = pitched
    for f, bw in table.get(phoneme, []):
        v = _syn.resonator(v, f, bw)
    v = v / (np.abs(v).max() + 1e-9)

    # 混回一些未共振的雨声，保住颗粒
    raw = seg / (np.abs(seg).max() + 1e-9)
    raw = sosfilt(butter(2, 1000.0, btype="highpass", fs=sr, output="sos"), raw)
    raw = raw / (np.abs(raw).max() + 1e-9)
    mix = C.HYB_RAW_MIX * (1.5 if female else 1.0)    # 少女声气声更多
    mix = min(mix, 0.6)
    out = (1 - mix) * v + mix * raw

    nn = min(len(out), int(sr * dur))
    out = out[:nn]
    t = np.arange(nn) / sr
    atk = np.clip(t / C.INST_ATTACK, 0, 1) ** 0.6
    rel = np.ones(nn)
    rs = max(0.0, dur - C.INST_RELEASE)
    m = t > rs
    rel[m] = np.clip((dur - t[m]) / C.INST_RELEASE, 0, 1) ** 1.3
    out = out * atk * rel
    out -= out.mean()
    return out / (np.abs(out).max() + 1e-9) * 0.9


# 歌词版同样的旋律：a e a e u e u e
DEMOS = {
    "ame": [("a", 60, 1), ("e", 60, 1), ("a", 67, 1), ("e", 67, 1),
            ("u", 69, 1), ("e", 69, 1), ("u", 67, 3)],
    "vowels": [("a", 60, 1), ("i", 62, 1), ("u", 64, 1), ("e", 65, 1),
               ("o", 67, 1), ("a", 69, 1), ("o", 67, 1), ("e", 64, 1), ("a", 60, 2)],
    # 少女音区：整体上移一个八度附近（C5 为主）
    "ame_f": [("a", 72, 1), ("e", 72, 1), ("a", 79, 1), ("e", 79, 1),
              ("u", 81, 1), ("e", 81, 1), ("u", 79, 3)],
    "vowels_f": [("a", 72, 1), ("i", 74, 1), ("u", 76, 1), ("e", 77, 1),
                 ("o", 79, 1), ("a", 81, 1), ("o", 79, 1), ("e", 76, 1), ("a", 72, 2)],
}


def render(seq, bpm, out_name, female=False):
    src = load_rain()
    rng = np.random.default_rng(7)
    beat = 60.0 / bpm
    parts = []
    for ph, midi, beats in seq:
        f0 = 440.0 * 2 ** ((midi - 69) / 12)
        parts.append(rain_vowel(ph, f0, beat * beats, src, rng, female=female))
        parts.append(np.zeros(int(C.SR * 0.01)))
    y = np.concatenate(parts)
    y = y / (np.abs(y).max() + 1e-9) * 0.9
    p = C.PREVIEW / out_name
    sf.write(p, y.astype(np.float32), C.SR, subtype=C.SUBTYPE)
    print(f"[ok] {out_name}  {len(y)/C.SR:.2f}s")
    return p


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "ame"
    female = name.endswith("_f") or "--female" in sys.argv
    render(DEMOS[name], 96, f"hybrid_{name}.wav", female=female)
