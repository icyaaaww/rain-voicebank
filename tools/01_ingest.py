# -*- coding: utf-8 -*-
"""Step 1 素材清理：任何格式 -> 44.1k / 16bit / mono / 去直流 / 去低频隆隆 / 峰值归一。

用法:
    python 01_ingest.py              # 处理 raw/ 下全部音频
    python 01_ingest.py --trim       # 额外做首尾静音裁剪
"""
import argparse
import json

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, sosfiltfilt

import config as C

EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif"}


def load_any(path):
    """优先 soundfile，失败则用 ffmpeg 解码。"""
    try:
        data, sr = sf.read(str(path), always_2d=True, dtype="float64")
        return data.mean(axis=1), sr
    except Exception:
        import subprocess
        import tempfile
        tmp = tempfile.mktemp(suffix=".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(path),
             "-ac", "1", "-ar", str(C.SR), tmp],
            check=True,
        )
        data, sr = sf.read(tmp, dtype="float64")
        Path_unlink(tmp)
        return data, sr


def Path_unlink(p):
    from pathlib import Path
    Path(p).unlink(missing_ok=True)


def resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    import librosa
    return librosa.resample(x, orig_sr=sr_in, target_sr=sr_out, res_type="soxr_hq")


def trim_silence(x, thresh_db=-45.0, pad=0.05):
    win = int(C.SR * 0.02)
    if len(x) < win * 2:
        return x
    frames = len(x) // win
    rms = np.sqrt(np.mean(x[: frames * win].reshape(frames, win) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / (np.abs(x).max() + 1e-12) + 1e-12)
    voiced = np.where(db > thresh_db)[0]
    if len(voiced) == 0:
        return x
    a = max(0, voiced[0] * win - int(pad * C.SR))
    b = min(len(x), (voiced[-1] + 1) * win + int(pad * C.SR))
    return x[a:b]


def process(path, do_trim):
    x, sr = load_any(path)
    x = resample(x, sr, C.SR)
    x = np.nan_to_num(x)

    # 去直流
    x -= x.mean()

    # 高通 60Hz：去掉风声/低频隆隆
    x = sosfilt(butter(4, 60.0, btype="highpass", fs=C.SR, output="sos"), x)

    # 低频轻微提升，让雨声更"厚"（可选，按需改）
    # x = sosfilt(butter(2, [80, 200], btype="bandpass", fs=C.SR, output="sos"), x) * 0.5 + x

    if do_trim:
        x = trim_silence(x)

    peak = np.abs(x).max()
    if peak < 1e-6:
        print(f"  [skip] {path.name}: 静音")
        return None
    x = x / peak * (10 ** (C.PEAK_DBFS / 20.0))

    # 首尾 5ms 淡入淡出，防爆音
    f = int(0.005 * C.SR)
    if len(x) > 2 * f:
        x[:f] *= np.linspace(0, 1, f)
        x[-f:] *= np.linspace(1, 0, f)

    out = C.NORMALIZED / (path.stem + ".wav")
    sf.write(out, x.astype(np.float32), C.SR, subtype=C.SUBTYPE)

    info = {
        "src": str(path.name),
        "out": out.name,
        "sr_in": sr,
        "dur": round(len(x) / C.SR, 3),
        "peak_dbfs": round(20 * np.log10(np.abs(x).max() + 1e-12), 2),
        "rms_dbfs": round(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12), 2),
    }
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trim", action="store_true", help="裁掉首尾静音")
    args = ap.parse_args()

    files = sorted(p for p in C.RAW.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        print(f"[!] {C.RAW} 里没有音频。先放雨声进去，或跑 python 00_demo_rain.py")
        return

    report = []
    for p in files:
        print(f"[ingest] {p.name}")
        r = process(p, args.trim)
        if r:
            report.append(r)
            print(f"         -> {r['dur']}s  peak {r['peak_dbfs']}dBFS  rms {r['rms_dbfs']}dBFS")

    C.BUILD.joinpath("ingest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[ok] {len(report)} 个素材已规范化 -> {C.NORMALIZED}")


if __name__ == "__main__":
    main()
