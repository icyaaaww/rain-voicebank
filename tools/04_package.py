# -*- coding: utf-8 -*-
"""Step 4 元数据 + 打包：character.txt / character.yaml / 图标 / zip。

OpenUtau 靠 character.txt 认出这是歌手；character.yaml 存它自己的扩展信息
（多音阶 subbank、默认 phonemizer、头像等）。打包也可以直接在
OpenUtau 里用 Tools > Singers > ⚙ > Publish Singer 做。
"""
import argparse
import json
import shutil
import zipfile

import numpy as np
import soundfile as sf
import yaml

import config as C

try:
    from PIL import Image, ImageDraw, ImageFilter
    HAS_PIL = True
except Exception:
    HAS_PIL = False


# ------------------------------------------------------------------ 图标
def make_icon(size=128):
    """画一张雨滴图标（无外部素材依赖）。"""
    if not HAS_PIL:
        return None
    S = size * 4
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # 背景圆
    d.ellipse([0, 0, S - 1, S - 1], fill=(24, 38, 56, 255))
    d.ellipse([6, 6, S - 7, S - 7], fill=(38, 62, 92, 255))

    # 雨滴
    cx, cy = S / 2, S * 0.56
    w = S * 0.26
    h = S * 0.34
    pts = []
    for t in np.linspace(0, 1, 60):
        ang = np.pi * t
        pts.append((cx - w / 2 * np.cos(ang) / 1.0, cy + h * t))
    drop = [(cx, cy - h * 1.15)]
    for t in np.linspace(0, 2 * np.pi, 90):
        drop.append((cx + (S * 0.155) * np.sin(t) * (0.75 + 0.25 * np.cos(t)),
                     cy + (S * 0.155) * (1 - np.cos(t)) * 0.95))
    d.polygon(drop, fill=(140, 205, 255, 255))

    # 高光
    d.ellipse([cx - S * 0.06, cy + S * 0.02, cx + S * 0.005, cy + S * 0.10],
              fill=(235, 248, 255, 230))

    # 雨丝
    for i, x in enumerate([S * 0.22, S * 0.32, S * 0.68, S * 0.78]):
        y0 = S * (0.30 + 0.06 * i)
        d.line([(x, y0), (x - S * 0.03, y0 + S * 0.16)],
               fill=(150, 200, 240, 150), width=int(S * 0.016))

    return img.resize((size, size), Image.LANCZOS)


def make_portrait(w=512, h=1024):
    if not HAS_PIL:
        return None
    img = Image.new("RGB", (w, h), (12, 18, 28))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / h
        r = int(12 + 20 * t)
        g = int(20 + 40 * t)
        b = int(34 + 70 * t)
        d.line([(0, y), (w, y)], fill=(r, g, b))

    rng = np.random.default_rng(3)
    for _ in range(700):
        x = rng.uniform(0, w)
        y = rng.uniform(0, h)
        L = rng.uniform(10, 46)
        a = int(rng.uniform(40, 150))
        d.line([(x, y), (x - L * 0.25, y + L)], fill=(120, 180, 230, a),
               width=1 if rng.random() < 0.8 else 2)

    img = img.filter(ImageFilter.GaussianBlur(0.4))
    return img


# ------------------------------------------------------------------ 元数据
def write_character_txt(lang="zh"):
    name = C.VB_NAME if lang == "zh" else C.VB_NAME_JA
    voice = C.VB_VOICE if lang == "zh" else C.VB_VOICE_JA
    lines = [
        f"name={name}",
        "image=icon.png",
        f"author={C.VB_AUTHOR}",
        f"voice={voice}",
        f"version={C.VB_VERSION}",
    ]
    if C.VB_WEB:
        lines.append(f"web={C.VB_WEB}")
    p = C.VOICEBANK / "character.txt"
    p.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    return p


def write_character_yaml(lang="zh"):
    subbanks = []
    for i, (tone, _f0) in enumerate(C.TONES):
        lo_t, hi_t = _tone_range(i)
        subbanks.append({
            "color": "",
            "prefix": "",
            "suffix": f"_{tone}",
            "tone_ranges": [f"{lo_t}-{hi_t}"],
        })

    name = C.VB_NAME if lang == "zh" else C.VB_NAME_JA
    voice = C.VB_VOICE if lang == "zh" else C.VB_VOICE_JA
    phonemizer = C.PHONEMIZER_ZH if lang == "zh" else C.PHONEMIZER_JA

    data = {
        "name": name,
        "localized_names": {"en-US": C.LOCALIZED_NAME_EN, "ja-JP": C.LOCALIZED_NAME_JA},
        "singer_type": "utau",
        "text_file_encoding": "utf-8",
        "image": "icon.png",
        "portrait": "portrait.png" if (C.VOICEBANK / "portrait.png").exists() else "",
        "portrait_opacity": 0.55,
        "portrait_height": 0,
        "author": C.VB_AUTHOR,
        "voice": voice,
        "version": C.VB_VERSION,
        "web": C.VB_WEB,
        "default_phonemizer": phonemizer,
        "subbanks": subbanks,
    }
    p = C.VOICEBANK / "character.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                 encoding="utf-8")
    return p


NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _midi_to_name(m):
    return f"{NOTES[m % 12]}{m // 12 - 1}"


def _tone_range(i):
    """每个 subbank 覆盖到相邻音阶的中点。"""
    pcs = [int(round(69 + 12 * np.log2(f / 440.0))) for _, f in C.TONES]
    if len(pcs) == 1:
        return "C1", "B7"
    lo = pcs[0] - 5 if i == 0 else (pcs[i - 1] + pcs[i]) // 2 + 1
    hi = pcs[-1] + 5 if i == len(pcs) - 1 else (pcs[i] + pcs[i + 1]) // 2
    return _midi_to_name(lo), _midi_to_name(hi)


# ------------------------------------------------------------------ 打包
def pack(lang="zh"):
    name = C.VB_NAME if lang == "zh" else C.VB_NAME_JA
    zpath = C.DIST / f"{name}_v{C.VB_VERSION}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(C.VOICEBANK.rglob("*")):
            if f.is_file():
                z.write(f, f"{name}/{f.relative_to(C.VOICEBANK).as_posix()}")
    return zpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["zh", "ja"], default="zh")
    args = ap.parse_args()
    lang = args.lang

    if HAS_PIL:
        icon = make_icon(128)
        if icon:
            icon.save(C.VOICEBANK / "icon.png")
        por = make_portrait()
        if por:
            por.save(C.VOICEBANK / "portrait.png")
        print("[ok] icon.png / portrait.png")
    else:
        print("[i] 没装 Pillow，跳过图标")

    print("[ok]", write_character_txt(lang).name)
    print("[ok]", write_character_yaml(lang).name)

    zp = pack(lang)
    n = len(list(C.VOICEBANK.glob('*.wav')))
    size = zp.stat().st_size / 1024 / 1024
    print(f"[ok] {zp.name}  ({n} wav, {size:.1f} MB)")
    print(f"\n把 {C.VOICEBANK} 整个文件夹复制到 OpenUtau 的 Singers 目录即可加载。")


if __name__ == "__main__":
    main()
