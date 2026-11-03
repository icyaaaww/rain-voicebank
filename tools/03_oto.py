# -*- coding: utf-8 -*-
"""Step 3 生成 oto.ini，并做一次客观体检（基频/电平/时长）。

oto.ini 单行格式（UTAU 标准）:
    file.wav=alias,offset,preutterance,overlap,consonant,cutoff,breath

    offset       黄线·左空白  发音起点在 wav 中的绝对毫秒
    preutterance 红线·预发声  相对 offset，音符真正开始的位置
    overlap      绿线·重叠    相对 offset，与前一个音符交叉淡化的长度
    consonant    蓝线·固定区  相对 offset，此点之前不循环不拉伸
    cutoff       白线·右空白  通常为负，表示从 wav 末尾倒数的截止位置
    breath       呼吸声参数，一般 0
"""
import argparse
import json

import numpy as np
import soundfile as sf

import config as C


def analyze(path):
    x, sr = sf.read(str(path), dtype="float64")
    peak = np.abs(x).max() + 1e-9
    hop = max(1, int(sr * 0.002))
    n = len(x) // hop
    env = np.abs(x[: n * hop]).reshape(n, hop).max(axis=1)

    voiced = np.where(env > peak * 0.06)[0]
    start_ms = (voiced[0] * hop / sr * 1000.0) if len(voiced) else 0.0

    tail = np.where(env > peak * 0.02)[0]
    end_ms = ((tail[-1] + 1) * hop / sr * 1000.0) if len(tail) else len(x) / sr * 1000.0

    return {
        "file": path.name,
        "dur_ms": len(x) / sr * 1000.0,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "peak": peak,
        "sr": sr,
        "rms_dbfs": 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12),
    }


def f0_estimate(x, sr, expect=0.0):
    """谐波乘积谱估基频。雨声+强共振峰的信号用 pyin 会乱跳，HPS 稳得多。
    若结果正好是 expect 的 1/2、1/3（倍频错误），自动往高修正。"""
    seg = x[int(0.10 * sr): int(0.10 * sr) + int(0.60 * sr)]
    if len(seg) < 4096:
        seg = x[: min(len(x), int(0.6 * sr))]
    seg = seg * np.hanning(len(seg))
    N = 1 << 16
    S = np.abs(np.fft.rfft(seg, n=N))
    f = np.fft.rfftfreq(N, 1 / sr)
    S = S / (S.max() + 1e-12)

    def score(fi):
        acc = 1.0
        for h in range(1, 6):
            j = fi * h
            if j >= len(S):
                break
            acc *= (S[j] + 1e-4)
        return acc

    idx = np.where((f > 60) & (f < 1500))[0]
    if not len(idx):
        return 0.0
    best_i = idx[int(np.argmax([score(i) for i in idx]))]
    f0 = float(f[best_i])

    if expect > 0:
        for mult in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
            if abs(1200 * np.log2(f0 * mult / expect)) < 50:
                f0 *= mult
                break
    return f0


def make_line(info, alias, cons_ms=0.0):
    """cons_ms > 0 表示这是带辅音的 CV 采样，预发声要推到辅音之后。"""
    off = max(0.0, info["start_ms"] - 2.0)
    if cons_ms > 0:
        pre = cons_ms + 15
        ovl = min(C.OTO_OVERLAP, cons_ms * 0.4)
        cons = cons_ms + 40
    else:
        pre = C.OTO_PREUTTERANCE
        ovl = C.OTO_OVERLAP
        cons = max(pre + 30, C.OTO_CONSONANT)
    cutoff = -C.OTO_CUTOFF_MS
    return "%s=%s,{},{},{},{},{},0".format(
        int(round(off)), int(round(pre)), int(round(ovl)),
        int(round(cons)), int(round(cutoff))
    ) % (info["file"], alias)


def make_vc_line(info, alias):
    """VC 样本：前段是元音尾巴，后段是辅音。

    preutterance 指向元音结束处（辅音开始），JapaneseCVVCPhonemizer 用它
    算 vcLength。
    consonant 必须 <= preutterance：固定区比分配时长还长时，resampler
    拉伸会跳过中间一段，留下静音缝（实测 0.55-0.61s 掉了 70ms）。
    """
    pre = C.VC_VOWEL_MS
    cons = pre
    cutoff = -20      # 必须为负（从末尾倒数）；写 0 会被当成绝对位置 0，
                      # OpenUtau 会报 "cutoff before offset" 并整段拒绝渲染
    ovl = min(150, max(10, pre - 10))  # VC/VV 交叉区。50 太短边界会塌(-29dB)，
                                       # 200 太长会把下一音节染成前一个元音
    return "%s=%s,0,{},{},{},{},0".format(
        int(round(pre)), int(round(ovl)),
        int(round(cons)), int(round(cutoff))
    ) % (info["file"], alias)


def make_breath_line(info, alias):
    """呼吸声：几乎没有 preutterance/固定区，整段都是气声。"""
    return "%s=%s,0,15,10,24,-10,0" % (info["file"], alias)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcv", action="store_true",
                    help="额外生成 VCV 别名（前一音素 + 当前音素），用于连音声库")
    args = ap.parse_args()

    wavs = sorted(C.VOICEBANK.glob("*.wav"))
    if not wavs:
        print("[!] voicebank/ 里没有 wav，先跑 02_synthesize.py")
        return

    # 读 02 的清单：拿到真实别名（VC 别名含空格）、类型、辅音长度
    meta_map = {}
    man = C.BUILD / "synth_manifest.json"
    if man.exists():
        try:
            for m in json.loads(man.read_text(encoding="utf-8")):
                meta_map[m.get("file", "")] = m
        except Exception as e:
            print(f"[!] 读清单失败: {e}")

    cons_map = {m["alias"]: float(m.get("cons_ms", 0.0) or 0.0)
                for m in meta_map.values() if "alias" in m}

    lines = []
    stats = []
    for w in wavs:
        info = analyze(w)
        meta = meta_map.get(w.name, {})
        alias = meta.get("alias", w.stem)
        kind = meta.get("kind", "CV")

        if kind == "BR":
            lines.append(make_breath_line(info, alias))
            stats.append({"alias": alias, "kind": kind, "f0_measured": None,
                          "f0_expect": None, "f0_err_cents": None,
                          "dur_ms": round(info["dur_ms"], 1),
                          "start_ms": round(info["start_ms"], 1),
                          "rms_dbfs": round(info["rms_dbfs"], 2)})
            continue

        if kind in ("VC", "VV"):
            lines.append(make_vc_line(info, alias))
            stats.append({"alias": alias, "kind": kind, "f0_measured": None,
                          "f0_expect": None, "f0_err_cents": None,
                          "dur_ms": round(info["dur_ms"], 1),
                          "start_ms": round(info["start_ms"], 1),
                          "rms_dbfs": round(info["rms_dbfs"], 2)})
            continue

        lines.append(make_line(info, alias, cons_map.get(alias, 0.0)))

        ph, _, tone = alias.rpartition("_")
        expect = dict(C.TONES).get(tone, 0.0)
        f0 = f0_estimate(*sf.read(str(w), dtype="float64"), expect=expect)
        stats.append({
            "alias": alias, "f0_measured": round(f0, 1), "f0_expect": expect,
            "f0_err_cents": round(1200 * np.log2(f0 / expect), 1) if f0 > 0 and expect else None,
            "dur_ms": round(info["dur_ms"], 1),
            "start_ms": round(info["start_ms"], 1),
            "rms_dbfs": round(info["rms_dbfs"], 2),
        })

    # 可选：VCV 连音别名（同一样本多别名）
    if args.vcv:
        vowels = C.VOWELS
        extra = []
        for w in wavs:
            alias = w.stem
            ph = alias.rpartition("_")[0]
            if ph not in vowels:
                continue
            for prev in vowels:
                name = f"{prev} {alias}"
                extra.append(f"{w.name}={name},{C.OTO_PREUTTERANCE},{C.OTO_PREUTTERANCE},"
                             f"{C.OTO_OVERLAP},{C.OTO_CONSONANT},{-C.OTO_CUTOFF_MS},0")
        lines += extra
        print(f"[i] 追加 {len(extra)} 条 VCV 别名")

    out = C.VOICEBANK / "oto.ini"
    # 手动写 CRLF 且不经 newline 转换，否则 Windows 上会变成 \r\r\n
    out.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    C.BUILD.joinpath("oto_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    bad = [s for s in stats if s["f0_err_cents"] is not None and abs(s["f0_err_cents"]) > 60]
    n_vc = sum(1 for s in stats if s.get("kind") == "VC")
    print(f"[ok] oto.ini -> {out}  ({len(lines)} 行，其中 VC {n_vc} 行)")
    print(f"[check] 基频偏差 >60 音分的采样: {len(bad)} 个")
    for s in [x for x in stats if x["f0_measured"] is not None][:10]:
        print(f"   {s['alias']:9s} measure {s['f0_measured']:7.1f}Hz "
              f"expect {s['f0_expect']:7.2f}Hz  err {s['f0_err_cents']}cents")


if __name__ == "__main__":
    main()
