# -*- coding: utf-8 -*-
"""生成 UTAU 兼容版声库：单音阶 CV + prefix.map + 文件名去特殊字符。

和 OpenUtau 版的区别：
  · 只保留一个音阶（默认 C4）——UTAU 靠 prefix.map 换音阶，单音阶最省事也最好用
  · 文件名里的 '#' 换成 's'（A#4 -> As4）——UTAU 的 oto 解析对 # 不友好
  · 不带 VC/VV——UTAU 原版不支持 CVVC 别名，那些要装 presamp 插件才认
  · 体积从 500MB 降到 25MB 左右

用法:
    python 15_utau_lite.py [--tone C4]
"""
import argparse
import io
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))
import config as C
import japanese as JP


def safe_name(s):
    """UTAU 文件名安全化：升号 # 换 s，空格换 _。"""
    s = s.replace("#", "s").replace(" ", "_")
    s = re.sub(r"[^0-9A-Za-z_\-\u3040-\u30ff\u4e00-\u9fff]", "", s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tone", default="C4")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tone = a.tone
    src = C.VOICEBANK
    dst = C.BUILD / f"utau_lite_{safe_name(tone)}"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # 1. 收集该音阶的 CV 采样
    suffix = f"_{tone}"
    kana = {k for k, _, _ in JP.kana_list(include_yoon=True)}
    copied = 0
    oto_lines = []
    src_oto = (src / "oto.ini").read_text(encoding="utf-8", errors="replace")
    oto_map = {}
    for line in src_oto.splitlines():
        if "=" not in line:
            continue
        fn, rest = line.split("=", 1)
        oto_map[fn] = rest

    for w in sorted(src.glob("*.wav")):
        stem = w.stem
        if not stem.endswith(suffix):
            continue
        base = stem[: -len(suffix)]
        if base not in kana:
            continue
        new_stem = safe_name(base) + "_" + safe_name(tone)
        new_fn = new_stem + ".wav"
        shutil.copy2(w, dst / new_fn)
        params = oto_map.get(w.name)
        if params:
            # alias 部分也要替换（如 あ_C4 -> あ_C4，A#4 -> As4）
            alias, tail = params.split(",", 1)
            new_alias = safe_name(alias) if alias else new_stem
            oto_lines.append(f"{new_fn}={new_alias},{tail}")
        copied += 1

    # 2. oto.ini（CRLF，无 BOM）
    (dst / "oto.ini").write_bytes(
        ("\r\n".join(oto_lines) + "\r\n").encode("utf-8"))
    print(f"oto.ini: {len(oto_lines)} 行")

    # 3. prefix.map —— 单音阶时所有音都指向这个后缀
    tones = ["C1", "C#1", "D1", "D#1", "E1", "F1", "F#1", "G1", "G#1", "A1", "A#1", "B1",
             "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2", "A2", "A#2", "B2",
             "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
             "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
             "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5",
             "C6", "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6", "G#6", "A6", "A#6", "B6",
             "C7", "C#7", "D7", "D#7", "E7", "F7", "F#7", "G7", "G#7", "A7", "A#7", "B7"]
    pm_lines = [f"{t}\t{sfx}" for t in tones if (sfx := safe_name(tone))]
    (dst / "prefix.map").write_bytes(
        ("\r\n".join(pm_lines) + "\r\n").encode("utf-8"))
    print(f"prefix.map: {len(pm_lines)} 行（全部指向 {safe_name(tone)}）")

    # 4. character.txt
    ct = f"""name=雨音レイン
image=icon.png
author=YG
voice=雨音 / rain
version=1.0
"""
    (dst / "character.txt").write_bytes(ct.replace("\n", "\r\n").encode("utf-8"))

    # 5. 复制图片
    for img in ["icon.png", "portrait.png"]:
        s = src / img
        if s.exists():
            shutil.copy2(s, dst / img)

    # 6. 打包
    zip_path = C.DIST / f"雨音レイン_UTAU_{safe_name(tone)}.zip"
    dist = C.DIST if hasattr(C, "DIST") else C.BUILD
    zip_path = dist / f"雨音レイン_UTAU_{safe_name(tone)}.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", dst)
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"[ok] {zip_path}  {size_mb:.1f} MB")
    print(f"     解压后放到 UTAU 的 voice 目录即可")


if __name__ == "__main__":
    main()
