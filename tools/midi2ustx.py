# -*- coding: utf-8 -*-
"""MIDI -> USTX（OpenUtau 工程），用雨音レイン声库来唱。

处理要点：
  · tpb 与 USTX 的 480 对齐
  · 过滤人声音域之外的低频残留（basic-pitch 会把伴奏的低音也识别进来）
  · 合并过短音符、修掉重叠（USTX 里音符重叠会报 OverlapError）
  · MIDI 没有歌词，默认按元音轮换填充，可用 --lyric 指定单字

用法:
    python midi2ustx.py <in.mid> <out.ustx> [--bpm auto|数字] [--lyric あ] [--min-ms 60]
"""
import argparse
import math

import mido

NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
VOWELS = ["あ", "い", "う", "え", "お"]


def read_midi(path):
    mid = mido.MidiFile(path)
    bpm = 120.0
    for tr in mid.tracks:
        for m in tr:
            if m.type == "set_tempo":
                bpm = 60e6 / m.tempo
                break
        else:
            continue
        break

    # 找到音符最多的轨（人声主旋律）
    best = None
    for tr in mid.tracks:
        n = sum(1 for m in tr if m.type == "note_on" and m.velocity > 0)
        if best is None or n > best[0]:
            best = (n, tr)
    tr = best[1]

    t = 0
    cur = {}
    notes = []
    for m in tr:
        t += m.time
        if m.type == "note_on" and m.velocity > 0:
            cur.setdefault(m.note, []).append(t)
        elif m.type in ("note_off", "note_on") and (m.type == "note_off" or m.velocity == 0):
            if m.note in cur and cur[m.note]:
                st = cur[m.note].pop(0)
                notes.append([st, t - st, m.note])
    notes.sort(key=lambda x: x[0])
    return mid.ticks_per_beat, bpm, notes


def clean(notes, tpb, bpm, min_ms=60, lo=48, hi=86):
    """过滤音域、合并短音、消除重叠。"""
    sec_per_tick = 60.0 / bpm / tpb
    out = []
    for st, du, nt in notes:
        if not (lo <= nt <= hi):
            continue
        if du * sec_per_tick * 1000.0 < min_ms:
            continue
        out.append([st, du, nt])
    out.sort(key=lambda x: x[0])

    merged = []
    for st, du, nt in out:
        if merged and merged[-1][2] == nt and st - (merged[-1][0] + merged[-1][1]) <= tpb * 0.12:
            merged[-1][1] = max(merged[-1][1], st + du - merged[-1][0])
        else:
            merged.append([st, du, nt])

    # 重叠：前一个截到后一个开始前，至少留 1/32 拍
    for i in range(len(merged) - 1):
        end = merged[i][0] + merged[i][1]
        nxt = merged[i + 1][0]
        if end > nxt:
            merged[i][1] = max(int(tpb * 0.125), nxt - merged[i][0] - 1)
    return merged


def to_ustx(notes, tpb, bpm, lyric=None, name="watashi-wa-ame"):
    scale = 480.0 / tpb
    lines = []
    lines.append(f"name: {name}")
    lines.append('ustx_version: "0.6"')
    lines.append("resolution: 480")
    lines.append("key: 0")
    lines.append("time_signature:")
    lines.append("- bar_position: 0")
    lines.append("  beat_per_bar: 4")
    lines.append("  beat_unit: 4")
    lines.append("tempos:")
    lines.append("- position: 0")
    lines.append("  bpm: %.2f" % bpm)
    lines.append("tracks:")
    lines.append("- singer: 雨音レイン")
    lines.append("  phonemizer: OpenUtau.Plugin.Builtin.JapaneseCVVCPhonemizer")
    lines.append("  track_name: 雨音")
    lines.append("  track_color: Blue")
    lines.append("  mute: false")
    lines.append("  solo: false")
    lines.append("  volume: 0")
    lines.append("voice_parts:")
    lines.append("- name: Part 1")
    lines.append("  track_no: 0")
    lines.append("  position: 0")
    total = int((notes[-1][0] + notes[-1][1]) * scale) + 480 if notes else 480
    lines.append("  duration: %d" % total)
    lines.append("  notes:")
    for i, (st, du, nt) in enumerate(notes):
        ph = lyric if lyric else VOWELS[i % len(VOWELS)]
        pos = int(round(st * scale))
        dur = max(30, int(round(du * scale)))
        lines.append("  - position: %d" % pos)
        lines.append("    duration: %d" % dur)
        lines.append("    tone: %d" % nt)
        lines.append("    lyric: %s" % ph)
        lines.append("    pitch: {data: [], snap_first: false}")
        lines.append("    vibrato: {length: 0, period: 175, depth: 25, in: 10, out: 10, shift: 0, drift: 0, vol_link: 0}")
    lines.append("  curves: []")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("midi")
    ap.add_argument("out")
    ap.add_argument("--lyric", default=None, help="指定单字歌词；不给则按元音轮换")
    ap.add_argument("--min-ms", type=float, default=60)
    ap.add_argument("--lo", type=int, default=48, help="最低音 MIDI 号")
    ap.add_argument("--hi", type=int, default=86, help="最高音 MIDI 号")
    args = ap.parse_args()

    tpb, bpm, notes = read_midi(args.midi)
    print(f"读入: {len(notes)} 音符, tpb={tpb}, BPM={bpm:.1f}")
    cl = clean(notes, tpb, bpm, args.min_ms, args.lo, args.hi)
    print(f"清理后: {len(cl)} 音符")
    if cl:
        ps = [n[2] for n in cl]
        print(f"音域: {NOTE[min(ps)%12]}{min(ps)//12-1} ~ {NOTE[max(ps)%12]}{max(ps)//12-1}")
        print(f"时长: {(cl[-1][0]+cl[-1][1])/tpb/bpm*60:.1f}s")
    txt = to_ustx(cl, tpb, bpm, args.lyric)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"输出: {args.out}")


if __name__ == "__main__":
    main()
