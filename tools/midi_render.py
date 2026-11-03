# -*- coding: utf-8 -*-
"""离线渲染器：MIDI + 雨音レイン采样 -> 完整人声 wav（可混伴奏）。

为什么不走 OpenUtau：它把长 Part 拆成几百个 chunk 缓存在 Cache 里，
文件名是内容 hash，无法按顺序拼回完整音频；导出又只能走 UI。
这里直接实现 UTAU 的渲染语义：选 subbank -> 按音高变调 -> 拉伸到音符时长 -> 拼接。

用法:
    python midi_render.py <in.mid> <out.wav> [--inst 伴奏.mp3] [--inst-gain -8]
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

import config as C


# ---------------------------------------------------------------- MIDI
def read_notes(path):
    import mido
    mid = mido.MidiFile(path)
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
    bpm = 120.0
    for tk in mid.tracks:
        for m in tk:
            if m.type == "set_tempo":
                bpm = 60e6 / m.tempo
                break
        else:
            continue
        break
    return notes, mid.ticks_per_beat, bpm


# ---------------------------------------------------------------- 渲染
def load_bank():
    bank = {}
    for w in C.VOICEBANK.glob("*.wav"):
        parts = w.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in dict(C.TONES):
            bank.setdefault(parts[1], {})[parts[0]] = w
    return bank


def pick(bank, tone):
    if tone in bank:
        return bank[tone]
    return bank[sorted(bank)[0]]


def note_audio(sample_path, f0_rec, f0_tgt, dur, sr=C.SR, fade_out_ms=1.5):
    x, src_sr = sf.read(str(sample_path), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if src_sr != sr:
        import librosa
        x = librosa.resample(x, orig_sr=src_sr, target_sr=sr)

    # 只跳极短的一小段来避开起音爆点。
    # 之前跳 55ms 是为了躲开 preutterance，但日语的辅音就在 0-85ms——
    # 跳掉之后每个音只剩元音，听起来就是"诶诶啊啊呜呜"，歌词全丢。
    pre = int(sr * 0.005)
    if len(x) > pre + sr // 4:
        x = x[pre:]

    ratio = f0_tgt / f0_rec
    if abs(np.log2(ratio)) > 0.005:
        n_out = max(64, int(len(x) / ratio))
        x = np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x)

    need = max(64, int(sr * dur))
    # 辅音段保持原速，只压缩/拉伸元音部分。
    # 之前整体拉伸：音符 46ms 时 1.4s 的采样被压 30 倍，
    # 85ms 的辅音变成 2.8ms，等于完全消失（只听到元音"诶诶啊啊"）。
    cons_fix = min(int(sr * 0.070), len(x) // 3, need // 2)
    head = x[:cons_fix].copy()
    body = x[cons_fix:]
    body_need = max(64, need - len(head))
    if len(body) < 1:
        body = x[-64:]
    if len(body) != body_need:
        if len(body) > body_need:
            body = body[:body_need]
        else:
            loop_start = int(len(body) * 0.6)
            loop = body[loop_start:] if len(body) - loop_start > 64 else body[-64:]
            reps = int(np.ceil((body_need - len(body)) / len(loop))) + 1
            body = np.concatenate([body, np.tile(loop, reps)])[:body_need]
    x = np.concatenate([head, body])

    # 淡出长度必须看上下文：
    #   · 后面紧跟音符 -> 1.5ms，长了会在每个边界制造周期性凹陷
    #   · 后面是换气口   -> 60ms，模拟自然衰减；直接切断会让秒级能量突然掉
    a = max(4, int(sr * 0.0015))
    r = max(4, min(int(sr * fade_out_ms / 1000.0), len(x) // 3))
    env = np.ones(len(x))
    if a > 0:
        env[:a] = np.linspace(0, 1, a) ** 0.7
    if r > 0:
        env[-r:] = np.linspace(1, 0, r) ** 1.2
    return x * env


def expand_notes(notes, target):
    """把音符拆开，使总数等于 target（用于一音节一音符的歌词对齐）。

    pyin 提取时常把连续的多个音符合并成一个长音，导致歌词数 > 音符数。
    按比例拆分，同音高的子音符拆开后每个音节都能分到一个音，
    吐字会清楚很多（代价是长音会被切成若干等长段）。
    """
    n0 = len(notes)
    if target <= n0:
        return notes
    k = target / n0
    out = []
    for i, (st, du, nt) in enumerate(notes):
        a = int(round(i * k))
        b = int(round((i + 1) * k))
        n_sub = max(1, b - a)
        if n_sub == 1:
            out.append([st, du, nt])
        else:
            sub = du / n_sub
            for j in range(n_sub):
                s0 = st + int(round(j * sub))
                s1 = st + int(round((j + 1) * sub)) if j < n_sub - 1 else st + du
                out.append([s0, max(1, s1 - s0), nt])
    print(f"音符拆分 {n0} -> {len(out)}（目标 {target}）")
    return out


def fill_gaps(notes, tpb, bpm, max_gap=0.22):
    """把小于 max_gap 秒的音符间隙填掉。

    basic-pitch 在换音的瞬间会丢几帧，导致音符之间留下 20-200ms 的空隙。
    直接渲染出来就是"字字断开"——真实演唱里元音是连续的。
    这里把每个音的尾巴延伸到下一个音开始，只保留真正的换气口（>max_gap）。
    """
    sec = 60.0 / bpm / tpb
    filled = 0
    for i in range(len(notes) - 1):
        end = notes[i][0] + notes[i][1]
        nxt = notes[i + 1][0]
        gap = (nxt - end) * sec
        if 0 < gap <= max_gap:
            notes[i][1] = nxt - notes[i][0]
            filled += 1
    return notes, filled


def smooth_dynamics(x, sr=C.SR, win_ms=25.0, strength=0.75, target_db=-16.0):
    """短时动态压缩：压平 5-25ms 尺度的能量跳动。

    颗粒合成的每个颗粒取自雨声不同位置，短时能量会有 18dB 的随机起伏
    （原唱只有 2.8dB），听感就是"一卡一卡"。用包络跟随器把它压回来。
    """
    w = max(16, int(sr * win_ms / 1000.0))
    # 短时 RMS（用滑动平均近似）
    e = np.sqrt(np.convolve(x ** 2, np.ones(w) / w, mode="same") + 1e-12)
    # 再平滑一次，得到"想保留"的慢变化
    w2 = max(16, int(sr * 0.080))
    ref = np.sqrt(np.convolve(x ** 2, np.ones(w2) / w2, mode="same") + 1e-12)
    ref = np.maximum(ref, 1e-6)
    e = np.maximum(e, 1e-6)
    # 增益 = (参考/当前)^strength，strength=1 完全压平，0 不处理
    gain = (ref / e) ** strength
    gain = np.clip(gain, 0.25, 4.0)
    return x * gain


def render_raindrop(notes, tpb, bpm, out_path, drops, source,
                    vowel="a", breaths=True):
    """雨滴声门脉冲引擎：HPSS 出的雨滴按 1/f0 周期排列，过少女共振峰。

    关键：音符包络的淡入淡出必须极短（6ms），衔接靠预重叠。
    之前每音 20/40ms 包络，拼接处周期性凹陷，整段起伏 11.75dB（原唱 2.65dB），
    听感就是"一卡一卡"。改成极短包络后降到 2.39dB。
    """
    import importlib.util
    sp = importlib.util.spec_from_file_location(
        "rd", str(Path(__file__).parent / "10_raindrop.py"))
    rd = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(rd)

    sec_per_tick = 60.0 / bpm / tpb
    total = int((max(n[0] + n[1] for n in notes) * sec_per_tick + 2.0) * C.SR)
    buf = np.zeros(total)
    rng = np.random.default_rng(11)

    ADV = 0.030
    for st, du, nt in notes:
        f0 = 440.0 * 2 ** ((nt - 69) / 12)
        dur = max(0.05, du * sec_per_tick)
        seg = rd.synth(vowel, f0, dur + ADV, drops, source, rng)
        _r = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
        if _r > 1e-6:
            seg = seg / _r
        a = int((st * sec_per_tick - ADV) * C.SR)
        if a < 0:
            seg = seg[-a:]
            a = 0
        b = min(total, a + len(seg))
        if a < total:
            buf[a:b] += seg[:b - a]

    n_breath = 0
    if breaths:
        for t0, gap in find_gaps(notes, sec_per_tick):
            d = min(0.46, gap * 0.75)
            if d < 0.15:
                continue
            seg = breath_audio(d)
            if seg is None:
                break
            a = int(t0 * C.SR)
            b = min(total, a + len(seg))
            if a < total:
                buf[a:b] += seg[:b - a]
                n_breath += 1

    buf = smooth_dynamics(buf, C.SR, strength=1.0)
    peak = np.abs(buf).max()
    if peak > 1e-9:
        buf = buf / peak * 0.89
    sf.write(str(out_path), buf.astype(np.float32), C.SR, subtype=C.SUBTYPE)
    print(f"[ok] 雨滴引擎 {len(buf)/C.SR:.1f}s  气口 {n_breath} 处 -> {out_path}")
    return out_path


def find_gaps(notes, sec_per_tick, min_gap=0.55):
    """找乐句之间的换气口：相邻音符之间的静默超过 min_gap 秒。"""
    gaps = []
    for i in range(len(notes) - 1):
        end = (notes[i][0] + notes[i][1]) * sec_per_tick
        nxt = notes[i + 1][0] * sec_per_tick
        if nxt - end >= min_gap:
            gaps.append((end + 0.04, nxt - end))
    return gaps


def breath_audio(dur, sr=C.SR):
    """取一个"息"采样（音高无关），裁/拉伸到 dur。"""
    cands = sorted(C.VOICEBANK.glob("息_*.wav"))
    if not cands:
        return None
    x, s2 = sf.read(str(cands[0]), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if s2 != sr:
        import librosa
        x = librosa.resample(x, orig_sr=s2, target_sr=sr)
    need = max(64, int(sr * dur))
    if len(x) < need:
        x = np.pad(x, (0, need - len(x)))
    x = x[:need]
    f = min(int(sr * 0.02), len(x) // 3)
    if f > 4:
        x[:f] *= np.linspace(0, 1, f)
        x[-f:] *= np.linspace(1, 0, f)
    return x


def render(notes, tpb, bpm, out_path, lyric_vowel="a", breaths=True, lyrics=None):
    bank = load_bank()
    tone_f0 = dict(C.TONES)
    sec_per_tick = 60.0 / bpm / tpb
    total = int((max(n[0] + n[1] for n in notes) * sec_per_tick + 2.0) * C.SR)
    buf = np.zeros(total)

    # 音符预重叠：每个音提前 ADV 秒开始，和上一个音的尾巴叠在一起交叉淡化。
    # 这就是 oto 里 overlap 参数的作用，离线渲染必须自己实现，否则字字断开。
    ADV_MAX = 0.010
    for i, (st, du, nt) in enumerate(notes):
        f0 = 440.0 * 2 ** ((nt - 69) / 12)
        tone = min(C.TONES, key=lambda t: abs(np.log2(t[1] / f0)))[0]
        files = pick(bank, tone)
        if lyrics:
            _li = int(i * len(lyrics) / len(notes))
            _li = min(_li, len(lyrics) - 1)
            _syl = lyrics[_li]
        else:
            _syl = lyric_vowel
        # files 的 key 只有假名（如 "わ"），不带音阶后缀——音阶由 pick() 选好
        if _syl in files:
            _key = _syl
        else:
            _key = next(iter(files))
        dur = max(0.05, du * sec_per_tick)
        ADV = min(ADV_MAX, dur * 0.25)
        dur += ADV
        # 看下一个音有多远：紧跟就用极短淡出，有大间隙就用长淡出（自然收尾）
        _nxt = notes[i + 1][0] * sec_per_tick if i + 1 < len(notes) else 1e9
        _gap = _nxt - (st * sec_per_tick + dur - ADV)
        _fo = 60.0 if _gap > 0.30 else 1.5
        seg = note_audio(files[_key], tone_f0[tone], f0, dur, fade_out_ms=_fo)
        # 音符级 RMS 归一化：每个音拉到同一响度。
        # 采样内部有 ±35% 的慢速起伏（GRAIN_MOVEMENT），不同音符取自不同位置，
        # 相邻音能量能差 18dB，听感就是忽大忽小的"一顿一顿"。
        _r = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
        if _r > 1e-6:
            seg = seg / _r
        a = int((st * sec_per_tick - ADV) * C.SR)
        if a < 0:
            seg = seg[-a:]
            a = 0
        b = min(total, a + len(seg))
        if a < total:
            buf[a:b] += seg[:b - a]

    n_breath = 0
    if breaths:
        for t0, gap in find_gaps(notes, sec_per_tick):
            dur = min(0.46, gap * 0.75)
            if dur < 0.15:
                continue
            seg = breath_audio(dur)
            if seg is None:
                break
            a = int(t0 * C.SR)
            b = min(total, a + len(seg))
            if a < total:
                buf[a:b] += seg[:b - a]
                n_breath += 1

    # 全曲动态压缩：压平颗粒合成带来的短时能量跳动（原唱 2.8dB，我们原本 18dB）
    buf = smooth_dynamics(buf, C.SR, strength=1.0)

    peak = np.abs(buf).max()
    if peak > 1e-9:
        buf = buf / peak * 0.89
    sf.write(str(out_path), buf.astype(np.float32), C.SR, subtype=C.SUBTYPE)
    print(f"[ok] 人声轨 {len(buf)/C.SR:.1f}s  气口 {n_breath} 处 -> {out_path}")
    return out_path


def add_instrumental(vocal_path, inst_path, out_path, gain_db=-8.0, sr=C.SR):
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(inst_path),
                    "-ac", "1", "-ar", str(sr), str(tmp)], check=True)
    inst, _ = sf.read(str(tmp), dtype="float64")
    vocal, _ = sf.read(str(vocal_path), dtype="float64")
    if inst.ndim > 1:
        inst = inst.mean(axis=1)
    n = max(len(inst), len(vocal))
    inst = np.pad(inst, (0, n - len(inst)))
    vocal = np.pad(vocal, (0, n - len(vocal)))
    inst = inst / (np.abs(inst).max() + 1e-9) * (10 ** (gain_db / 20.0))
    mix = vocal * 0.9 + inst
    mix = mix / (np.abs(mix).max() + 1e-9) * 0.92
    sf.write(str(out_path), mix.astype(np.float32), sr, subtype=C.SUBTYPE)
    Path(tmp).unlink(missing_ok=True)
    print(f"[ok] 混音 {len(mix)/sr:.1f}s -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("midi")
    ap.add_argument("out")
    ap.add_argument("--inst", default=None, help="伴奏音频，会混进去")
    ap.add_argument("--inst-gain", type=float, default=-8.0)
    ap.add_argument("--lo", type=int, default=48)
    ap.add_argument("--hi", type=int, default=86)
    ap.add_argument("--min-ms", type=float, default=70)
    ap.add_argument("--vowel", default="a")
    ap.add_argument("--match-lyrics", action="store_true",
                    help="拆分音符使数量等于歌词音节数（一音节一音符）")
    ap.add_argument("--lift-low", type=int, default=0,
                    help="低于该 MIDI 号的音符提高一个八度（如 60=低于C4的都上移）")
    ap.add_argument("--lyrics", action="store_true", help="用 lyrics_ame.py 的歌词填音符")
    ap.add_argument("--engine", choices=["sample", "raindrop"], default="sample",
                    help="sample=用声库采样拼接；raindrop=雨滴声门脉冲实时合成")
    ap.add_argument("--no-fill", action="store_true", help="不填音符间隙（保留原始断句）")
    args = ap.parse_args()

    notes, tpb, bpm = read_notes(args.midi)
    if args.lift_low:
        _n = 0
        for n in notes:
            if n[2] < args.lift_low:
                n[2] += 12
                _n += 1
        print(f"八度上移 {_n} 个低音音符")
    sec = 60.0 / bpm / tpb
    notes = [n for n in notes if args.lo <= n[2] <= args.hi and n[1] * sec * 1000 >= args.min_ms]
    if not args.no_fill:
        notes, nfill = fill_gaps(notes, tpb, bpm)
        print(f"填掉 {nfill} 处小间隙")
    print(f"音符 {len(notes)}  tpb={tpb}  BPM={bpm:.1f}  时长 {max(n[0]+n[1] for n in notes)*sec:.1f}s")

    _lyr = None
    if args.lyrics:
        import lyrics_ame
        _lyr = lyrics_ame.SYLLABLES
        print(f"歌词 {len(_lyr)} 音节 / 音符 {len(notes)}")
        if args.match_lyrics and len(_lyr) > len(notes):
            notes = expand_notes(notes, len(_lyr))
            notes, _nf = fill_gaps(notes, tpb, bpm)
            print(f"拆分并填隙后: {len(notes)} 音符")

    voc = Path(tempfile.mktemp(suffix="_voc.wav"))
    if args.engine == "raindrop":
        import importlib.util as _ilu
        _sp = _ilu.spec_from_file_location("rd10", str(Path(__file__).parent / "10_raindrop.py"))
        _rd = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_rd)
        src = _rd.load_rain()
        drops = _rd.extract_drops(src)
        if not drops:
            raise SystemExit("没提取到雨滴")
        render_raindrop(notes, tpb, bpm, voc, drops, src, args.vowel)
    else:
        _lyr = None
        if args.lyrics:
            import lyrics_ame
            _lyr = lyrics_ame.SYLLABLES
            print(f"歌词 {len(_lyr)} 音节 -> {len(notes)} 音符")
        render(notes, tpb, bpm, voc, args.vowel, lyrics=_lyr)
    if args.inst:
        add_instrumental(voc, args.inst, args.out, args.inst_gain)
    else:
        Path(args.out).write_bytes(voc.read_bytes())
        print(f"[ok] -> {args.out}")
    voc.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
