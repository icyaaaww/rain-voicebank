# -*- coding: utf-8 -*-
"""日语假名表：假名 -> (声母, 元音)。

OpenUtau 的 JapaneseCVVCPhonemizer 用**假名本身**做 oto 别名（不是拼音），
所以日语音源的采样文件名就是「か_A3.wav」这种，subbank 后缀照样拼在后面。

音源规模：清音 46 + 浊音 20 + 半浊音 5 = 71 个假名（不含拗音）。
"""

# (声母, 元音)
KANA = {}


def _add(row, consonant, vowels="aiueo"):
    for kana, v in zip(row, vowels):
        KANA[kana] = (consonant, v)


# 元音行
_add("あいうえお", "")
# か行
_add("かきくけこ", "k")
_add("がぎぐげご", "g")
# さ行（し = sh）
_add("さすせそ", "s", "aueo")
KANA["し"] = ("sh", "i")
# ざ行（じ = j）
_add("ざずぜぞ", "z", "aueo")
KANA["じ"] = ("j", "i")
# た行（ち = ch，つ = ts）
_add("たてと", "t", "aeo")
KANA["ち"] = ("ch", "i")
KANA["つ"] = ("ts", "u")
# だ行（ぢ = j，づ = z）
_add("だでど", "d", "aeo")
KANA["ぢ"] = ("j", "i")
KANA["づ"] = ("z", "u")
# な行
_add("なにぬねの", "n")
# は行（ふ = f）
_add("はへほ", "h", "aeo")
KANA["ひ"] = ("h", "i")
KANA["ふ"] = ("f", "u")
# ば行
_add("ばびぶべぼ", "b")
# ぱ行
_add("ぱぴぷぺぽ", "p")
# ま行
_add("まみむめも", "m")
# や行
KANA["や"] = ("y", "a")
KANA["ゆ"] = ("y", "u")
KANA["よ"] = ("y", "o")
# ら行
_add("らりるれろ", "r")
# わ行
KANA["わ"] = ("w", "a")
KANA["を"] = ("", "o")
KANA["ゐ"] = ("", "i")
KANA["ゑ"] = ("", "e")
# 拨音
KANA["ん"] = ("", "n")

# 拗音（可选，加 --yoon 时生成）
YOON = {
    "きゃ": ("ky", "a"), "きゅ": ("ky", "u"), "きょ": ("ky", "o"),
    "ぎゃ": ("gy", "a"), "ぎゅ": ("gy", "u"), "ぎょ": ("gy", "o"),
    "しゃ": ("sh", "a"), "しゅ": ("sh", "u"), "しょ": ("sh", "o"),
    "じゃ": ("j", "a"), "じゅ": ("j", "u"), "じょ": ("j", "o"),
    "ちゃ": ("ch", "a"), "ちゅ": ("ch", "u"), "ちょ": ("ch", "o"),
    "にゃ": ("ny", "a"), "にゅ": ("ny", "u"), "にょ": ("ny", "o"),
    "ひゃ": ("hy", "a"), "ひゅ": ("hy", "u"), "ひょ": ("hy", "o"),
    "びゃ": ("by", "a"), "びゅ": ("by", "u"), "びょ": ("by", "o"),
    "ぴゃ": ("py", "a"), "ぴゅ": ("py", "u"), "ぴょ": ("py", "o"),
    "みゃ": ("my", "a"), "みゅ": ("my", "u"), "みょ": ("my", "o"),
    "りゃ": ("ry", "a"), "りゅ": ("ry", "u"), "りょ": ("ry", "o"),
}

# 声母的合成参数：(方式, 时长ms, 中心频率, 带宽)
#   fric 擦音 / affr 塞擦音 / stop 爆破音 / sono 鼻音流音 / glide 半元音
CONSONANT_SPEC = {
    "k": ("stop", 40, 2600, 2200),   # 日语 k 的爆破在 2-3kHz，设 1500 会撞上元音质心
    "g": ("stop", 22, 1100, 900),
    "s": ("fric", 95, 6200, 3500),
    "z": ("affr", 48, 5000, 3000),
    "t": ("stop", 42, 3200, 2200),
    "d": ("stop", 20, 2000, 1400),
    "n": ("sono", 58, 320, 320),
    "h": ("fric", 78, 2000, 2400),
    "b": ("stop", 20, 900, 700),
    "p": ("stop", 34, 900, 800),     # 双唇爆破，频率低
    "m": ("sono", 60, 300, 300),
    "r": ("sono", 10, 1150, 800),   # 日语ら行弹音：只有一下，10ms 足够，做长了就是"咚"
    "y": ("glide", 48, 2200, 1800),
    "w": ("glide", 45, 700, 700),
    "sh": ("fric", 100, 4200, 3000),
    "ch": ("affr", 65, 4000, 3000),
    "ts": ("affr", 62, 6200, 3400),
    "j": ("affr", 62, 3800, 3000),
    "f": ("fric", 42, 1600, 1500),   # 日语「ふ」是双唇擦音 [ɸ]，很轻，别学英语 f
    "ky": ("stop", 60, 2200, 1800),
    "gy": ("stop", 55, 1800, 1500),
    "ny": ("sono", 62, 400, 400),
    "hy": ("fric", 80, 2600, 2400),
    "by": ("stop", 55, 1000, 800),
    "py": ("stop", 62, 1200, 900),
    "my": ("sono", 62, 320, 320),
    "ry": ("sono", 45, 800, 700),
}


def kana_list(include_yoon=False):
    """返回 [(假名, 声母, 元音)]。"""
    items = [(k, c, v) for k, (c, v) in KANA.items()]
    if include_yoon:
        items += [(k, c, v) for k, (c, v) in YOON.items()]
    return items


# ---------------------------------------------------------------- VC（元音→辅音）
# JapaneseCVVCPhonemizer 里 VC 别名是 `$"{vowel} {consonant}"`，拉丁字母加空格，
# 例如 "a k"、"i s"。和 CV 用假名完全不同，别混。
VC_VOWELS = ["a", "i", "u", "e", "o", "n"]
VC_CONSONANTS = ["k", "g", "s", "z", "t", "d", "n", "h", "b", "p",
                 "m", "r", "y", "w", "sh", "ch", "ts", "j", "f"]


def vc_list():
    """返回 [(别名, 辅音, 元音)]，别名形如 'a k'、'n ts'。"""
    return [(f"{v} {c}", c, v) for v in VC_VOWELS for c in VC_CONSONANTS]


# ---------------------------------------------------------------- VV（元音接元音）
# 下一个音节以元音开头时走 VV 而不是 VC，别名格式 `$"{vow} {currentLyric}"`，
# 例如 "e あ"。缺了它，元音接元音的音符边界会出现很深的电平凹陷（实测 -47 dB）。
VV_KANA = {"あ": "a", "い": "i", "う": "u", "え": "e", "お": "o"}
VV_VOWELS = ["a", "i", "u", "e", "o"]


def vv_list():
    """返回 [(别名, 前一元音, 后一元音)]，别名形如 'e あ'。"""
    return [(f"{v} {k}", v, v2)
            for v in VV_VOWELS for k, v2 in VV_KANA.items()]


if __name__ == "__main__":
    items = kana_list()
    print(f"假名数: {len(items)}")
    missing = sorted({c for _, c, _ in items if c and c not in CONSONANT_SPEC})
    print("缺合成参数的声母:", missing or "无")
    from collections import Counter
    print("声母分布:", Counter(c for _, c, _ in items))
    vcs = vc_list()
    print(f"\nVC 组合数: {len(vcs)}  (元音 {len(VC_VOWELS)} × 辅音 {len(VC_CONSONANTS)})")
    print("示例:", ", ".join(a for a, _, _ in vcs[:8]))
