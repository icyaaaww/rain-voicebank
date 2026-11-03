# -*- coding: utf-8 -*-
"""Step 2 合成：把「雨声」变成「能唱的采样」。

原理（spectral envelope transplant / 雨声嗓音）:
  1. 造一个目标音高的谐波激励（谐波堆 + 少量白噪）
  2. 用雨声的短时幅度谱当作"音色 EQ"去调制激励 —— 雨声的颗粒、起伏、
     频谱倾斜全部搬到谐波上，听起来就是"雨在唱歌"
  3. 再过一串共振峰谐振器，把 a/i/u/e/o 分开
  4. 混一层高通后的原始雨声作为气声，保住真实雨味

输出: voicebank/<alias>.wav  (44100Hz / 16bit / mono)
"""
import argparse
import json

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, get_window

import config as C


# ---------------------------------------------------------------- 基础工具
def frame_signal(x, n_fft, hop):
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]                      # (T, n_fft)


def stft_mag(x, n_fft=C.ENV_N_FFT, hop=C.ENV_HOP):
    win = get_window("hann", n_fft, fftbins=True)
    frames = frame_signal(x, n_fft, hop) * win[None, :]
    return np.abs(np.fft.rfft(frames, axis=1))       # (T, F)


def istft(S, n_fft=C.ENV_N_FFT, hop=C.ENV_HOP, length=None):
    win = get_window("hann", n_fft, fftbins=True)
    frames = np.fft.irfft(S, n=n_fft, axis=1) * win[None, :]
    out_len = hop * (frames.shape[0] - 1) + n_fft
    out = np.zeros(out_len)
    wsum = np.zeros(out_len)
    for i, fr in enumerate(frames):
        a = i * hop
        out[a:a + n_fft] += fr
        wsum[a:a + n_fft] += win ** 2
    out /= np.maximum(wsum, 1e-8)
    if length:
        if len(out) < length:
            out = np.pad(out, (0, length - len(out)))
        out = out[:length]
    return out


# ---------------------------------------------------------------- 激励源
def glottal_excitation(f0, n, sr, rng):
    """声门脉冲串。

    人声不像"一堆独立正弦"，而像一串脉冲：所有谐波相位对齐，时域上形成尖峰，
    经过共振峰滤波后就是人声。相位随机 = 一堆正弦 = 电子琴。
    再叠加颤音、慢速漂移、相位抖动，去掉机械感。
    """
    t = np.arange(n) / sr

    # 颤音
    vib = C.VIBRATO_DEPTH * np.sin(2 * np.pi * C.VIBRATO_RATE * t
                                   + rng.uniform(0, 2 * np.pi))
    # 慢速随机漂移（低通后的噪声）
    drift = np.convolve(rng.normal(0, 1, n),
                        np.ones(1500) / 1500, mode="same")
    drift = drift / (np.abs(drift).max() + 1e-9) * C.DRIFT_AMOUNT

    f_inst = f0 * (1.0 + vib + drift)
    phase = 2 * np.pi * np.cumsum(f_inst) / sr

    out = np.zeros(n)
    fmax = sr * 0.45
    for k in range(1, C.TOP_HARMONICS + 1):
        if k * f0 > fmax:
            break
        # 相位抖动 + 高频额外衰减，模拟声门谱的滚降
        ph = k * phase + rng.normal(0, C.JITTER) * np.sqrt(k)
        amp = 1.0 / (k ** C.HARMONIC_TILT)
        out += amp * np.cos(ph)

    # 收窄脉冲：抬高再压平，让波峰更尖（接近声门波）
    if C.GLOTTAL_CLOSING > 0:
        env = np.abs(out) / (np.abs(out).max() + 1e-9)
        out = out * (1 - C.GLOTTAL_CLOSING + C.GLOTTAL_CLOSING * env)

    # 人声高频自然滚降
    out = sosfilt(butter(2, C.VOICE_LP, btype="lowpass", fs=sr, output="sos"), out)
    return out / (np.abs(out).max() + 1e-9)


def noise_source(n, rng):
    w = rng.standard_normal(n)
    return w / (np.abs(w).max() + 1e-9)


def _old_rain_drops(n, sr=C.SR, rng=None):
    """离散雨滴：随机位置 + 短指数衰减脉冲。

    连续宽带噪声 = 气声；离散短脉冲 = 雨滴。
    雨味的关键在"颗粒"，不在"嘶嘶"。
    """
    rng = rng or np.random.default_rng(0)
    out = np.zeros(n)
    n_drops = max(1, int(n / sr * C.DROP_DENSITY))
    lo = int(sr * C.DROP_MIN_MS / 1000)
    hi = int(sr * C.DROP_MAX_MS / 1000)
    for _ in range(n_drops):
        L = int(rng.integers(lo, max(lo + 1, hi)))
        if L + 1 >= n:
            continue
        pos = int(rng.integers(0, n - L - 1))
        env = np.exp(-np.linspace(0, 7.0, L))
        out[pos:pos + L] += rng.standard_normal(L) * env * rng.uniform(0.25, 1.0)
    # 给雨滴一点高频倾斜，更像水
    out = sosfilt(butter(2, [600.0, min(9000.0, sr * 0.45)],
                         btype="bandpass", fs=sr, output="sos"), out)
    return out


def granular_excitation(source, f0, n, sr=C.SR, rng=None,
                        cycles=None, jitter=None):
    """雨声颗粒按 1/f0 周期排列（PSOLA 式）。

    人声 = 声带脉冲(周期 1/f0) + 声道滤波。这里把"声带脉冲"换成
    "雨声颗粒"：颗粒率决定音高，颗粒内容决定音色。
    颗粒间隔恰好一个基频周期 -> 谐波结构出现 -> 有音高；
    每颗粒取自雨声不同位置 -> 每周期都不一样 -> 还是雨。
    """
    rng = rng or np.random.default_rng(0)
    cycles = C.GRAIN_CYCLES if cycles is None else cycles
    jitter = C.GRAIN_JITTER if jitter is None else jitter

    hop = max(8, int(round(sr / f0)))
    glen = int(max(64, round(hop * float(cycles))))
    win = np.hanning(glen)

    out = np.zeros(n + 2 * glen)
    pos = 0
    while pos < n + glen:
        j = rng.normal(0, jitter) * sr / f0 if jitter > 0 else 0.0
        off = int(rng.integers(0, max(1, len(source) - glen - 1)) + j)
        off = int(np.clip(off, 0, len(source) - glen))
        out[pos:pos + glen] += source[off:off + glen] * win
        pos += hop
    # 从 glen 处截取：开头就是"叠满"的稳态，避免 t=0 只有单个颗粒
    # 导致低频渐堆、听起来像"咚"的一下
    return out[glen:glen + n]


# ---------------------------------------------------------------- 雨滴声门脉冲
_drops_cache = {}


def extract_drops(source, sr=C.SR, drop_ms=30.0, max_drops=96):
    """HPSS 分离雨滴，onset 检测，裁出每一滴。结果按 len(source) 缓存。"""
    key = (len(source), sr)
    if key in _drops_cache:
        return _drops_cache[key]
    import librosa
    y = source.astype(np.float32)
    if len(y) > sr * 60:
        y = y[: sr * 60]
    _, P = librosa.effects.hpss(y)
    on = librosa.onset.onset_detect(y=P, sr=sr, units="samples", backtrack=True)
    L = int(sr * drop_ms / 1000.0)
    drops = []
    for o in on:
        if o + L > len(P):
            continue
        seg = P[o:o + L].astype(np.float64).copy()
        seg -= seg.mean()
        pk = np.abs(seg).max()
        if pk < 1e-6:
            continue
        head = np.abs(seg[: L // 6]).max()
        tail = np.abs(seg[-L // 3:]).max()
        if head < tail * 0.8:
            continue
        drops.append(seg / pk)
    _drops_cache[key] = drops[:max_drops]
    return _drops_cache[key]


def drop_excitation(drops, f0, n, sr=C.SR, rng=None):
    """雨滴裁成一个 f0 周期长，陡起音+指数衰减窗，首尾相接排列。

    这里刻意不用 Hann：Hann 覆盖整个周期、能量集中在窗心，
    5ms 窗口一测就是周期性起伏。声门脉冲是"快起 + 快衰减"。
    """
    rng = rng or np.random.default_rng(0)
    hop = max(8, int(round(sr / f0)))
    win = np.exp(-np.linspace(0.0, 6.0, hop))
    atk = max(2, int(hop * 0.08))
    win[:atk] *= np.linspace(0.0, 1.0, atk)
    out = np.zeros(n + hop)
    pos = 0
    while pos < n:
        d = drops[int(rng.integers(0, len(drops)))]
        # 截取而不是压缩：压缩会把 30ms 的雨滴频谱整体搬高 4 倍，
        # 低频全丢（C3 时 HNR 掉到 -8.3）。截前 hop 个样本，保留原始频谱。
        if len(d) >= hop:
            g = d[:hop].copy()
        else:
            g = np.pad(d, (0, hop - len(d)))
        g = g * win
        r = float(np.sqrt(np.mean(g ** 2))) + 1e-9
        out[pos:pos + hop] += g / r
        pos += hop
    out = out[:n]
    # 预加重：雨滴脉冲只有 3.8ms 宽，能量全压在低频，F2(1000-2800Hz) 几乎没有能量，
    # 导致所有元音的频谱都一样（听不出 a/i/u/e/o）。一阶差分把高频提上来。
    if C.RAINDROP_PREEMPH > 0:
        pe = out.copy()
        pe[1:] = out[1:] - C.RAINDROP_PREEMPH * out[:-1]
        out = pe
    return out


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-12))


def _preemph(x, a=0.93):
    """一阶预加重：H(z)=1-a·z⁻¹，每倍频程提升约 6dB。"""
    from scipy.signal import lfilter
    return lfilter(np.array([1.0, -a]), np.array([1.0]), x)


def mix_rms(x, y, amount):
    """按 RMS 比例混合，而不是峰值。峰值混合会让 crest factor 低的噪声
    偷偷占掉两三倍于标称值的能量。"""
    return (1 - amount) * (x / rms(x)) + amount * (y / rms(y))


# ---------------------------------------------------------------- 雨声音色提取
def rain_timbre(source, n_target, n_fft=C.ENV_N_FFT, hop=C.ENV_HOP):
    """提取雨声的**静态音色曲线** + **慢速起伏**。

    这是修「不像人声」的关键。之前是逐帧用时变随机谱去乘谐波，等于给每条
    谐波各叠了一个独立的随机幅度调制，谱线被抹成噪声带（HNR 只有 3 dB）。
    换成时不变的 EQ 曲线 + <6 Hz 的慢增益，谐波结构完整保留。
    """
    M = stft_mag(source, n_fft, hop)              # (T, F)
    T, F = M.shape
    freqs = np.fft.rfftfreq(n_fft, 1.0 / C.SR)

    # --- 1) 静态频谱：时间中位数，再在对数域用宽核平滑
    lin = np.median(M, axis=0) + 1e-9
    logl = np.log(lin)
    k = 61
    ker = np.hanning(k)
    ker /= ker.sum()
    logl = np.convolve(np.pad(logl, (k // 2, k // 2), mode="edge"), ker, mode="valid")[:F]
    eq = np.exp(logl)

    # 归一化到 400–1200 Hz 均值为 1，避免整体电平漂移
    band = (freqs > 400) & (freqs < 1200)
    eq = eq / (eq[band].mean() + 1e-9) if band.any() else eq

    # 频谱倾斜补偿：近距离亮雨高频过强，会把人声式的低频谐波整个盖掉
    if C.SOURCE_TILT_DB_PER_OCT:
        fq = freqs.copy()
        fq[0] = fq[1]
        eq = eq * (fq / 1000.0) ** (C.SOURCE_TILT_DB_PER_OCT / 6.0206)

    eq = np.clip(eq, C.ENV_FLOOR, 12.0)
    # 与平谱混合：depth 越小越"平"，保证低频谐波有底，不会只剩嘶声
    eq = (1.0 - C.RAIN_ENV_DEPTH) + C.RAIN_ENV_DEPTH * eq

    # --- 2) 慢速增益：全带能量包络，重度平滑后只留 <6 Hz 的起伏
    energy = np.sqrt((M ** 2).sum(axis=1))
    gk = 41
    gker = np.hanning(gk)
    gker /= gker.sum()
    g = np.convolve(np.pad(energy, (gk // 2, gk // 2), mode="edge"), gker, mode="valid")[:T]
    g = g / (g.mean() + 1e-9)
    g = np.clip(g, 1.0 - C.RAIN_MOVEMENT, 1.0 + C.RAIN_MOVEMENT)

    # 拉伸/压缩到目标时长
    T_tgt = 1 + max(0, (n_target - n_fft)) // hop + 1
    src_t = np.linspace(0, 1, len(g))
    tgt_t = np.linspace(0, 1, T_tgt)
    g = np.interp(tgt_t, src_t, g)
    return eq, g


def apply_timbre(x, eq, gain, n_fft=C.ENV_N_FFT, hop=C.ENV_HOP):
    """把静态 EQ 和慢速增益作用到激励上。EQ 每帧相同 -> 不调制谐波。"""
    win = get_window("hann", n_fft, fftbins=True)
    frames = frame_signal(x, n_fft, hop)
    S = np.fft.rfft(frames * win[None, :], axis=1) * eq[None, :]
    y = istft(S, n_fft, hop, length=len(x))
    if len(gain) > 1:
        gs = np.interp(np.arange(len(y)) / hop, np.arange(len(gain)), gain)
        y = y * gs
    return y


# ---------------------------------------------------------------- 共振峰
def resonator(x, f, bw, sr=C.SR):
    """Klatt 风格二阶谐振器：用极点半径控制带宽。"""
    from scipy.signal import lfilter
    r = np.exp(-np.pi * bw / sr)
    theta = 2 * np.pi * f / sr
    a = np.array([1.0, -2.0 * r * np.cos(theta), r * r])
    b = np.array([1.0 - r])          # 零点在 DC，去低频堆积
    return lfilter(b, a, x)


def apply_formants(x, phoneme):
    fmts = C.FORMANTS_ACTIVE.get(phoneme)
    if not fmts:
        return x
    y = x
    for f, bw in fmts:
        y = resonator(y, f, bw)
    return y


def comb_fast(x, f0, sr=C.SR, feedback=None):
    """梳状滤波：延迟 = sr/f0，让噪声长出等间距谐波峰。

    y[n] = x[n] + fb·y[n-d]
    """
    from scipy.signal import lfilter
    fb = C.COMB_FEEDBACK if feedback is None else feedback
    d = max(1, int(round(sr / f0)))
    a = np.zeros(d + 1)
    a[0] = 1.0
    a[d] = -fb
    return lfilter(np.array([1.0]), a, x)


# ---------------------------------------------------------------- 辅音（日语用）
def make_consonant(initial, source, rng, f0=None):
    """从雨声里切一段做出辅音起音。

    传了 f0 就再过一遍梳状滤波：辅音也带上音高，才不会在音符之间
    冒出一小段"嘶"或"噗"。这是 2s/4s 那两处气声的根源。
    """
    import japanese as JP
    spec = JP.CONSONANT_SPEC.get(initial)
    if not spec or not initial:
        return np.zeros(0)

    kind, dur_ms, fc, bw = spec
    n = max(64, int(C.SR * dur_ms / 1000.0))
    if len(source) < n + 8:
        return np.zeros(0)
    off = int(rng.integers(0, len(source) - n - 1))
    seg = source[off:off + n].astype(np.float64).copy()
    seg -= seg.mean()

    lo = max(60.0, fc - bw / 2)
    hi = min(C.SR * 0.45, fc + bw / 2)
    if hi <= lo * 1.05:
        lo, hi = 100.0, 5000.0
    seg = sosfilt(butter(2, [lo, hi], btype="bandpass", fs=C.SR, output="sos"), seg)

    # 给辅音加音高，避免变成一段孤立噪声
    if f0 and f0 > 0:
        p = comb_fast(seg, f0)
        p = p / (np.abs(p).max() + 1e-9)
        seg = seg / (np.abs(seg).max() + 1e-9)
        seg = C.CONS_PITCH_MIX * p + (1 - C.CONS_PITCH_MIX) * seg

    t = np.linspace(0, 1, len(seg))
    if kind == "fric":
        seg = seg * (0.35 + 0.65 * np.exp(-3.0 * t))
    elif kind == "affr":
        seg = seg * np.exp(-7.0 * t)
    elif kind == "stop":
        # 爆破音：极短闭塞 + 一次冲激 + 稍长的衰减。
        # 原来 exp(-16t) 让 45ms 里只有前 9ms 有声音，RMS 比元音低 7dB，
        # 听感上等于没有辅音（只剩元音"啊啊"）。
        gap = np.zeros(int(C.SR * 0.006))
        seg = np.concatenate([gap, seg])[:n]
        seg = seg * np.exp(-9.0 * np.linspace(0, 1, len(seg)))
    elif kind == "sono":
        seg = seg * np.exp(-1.0 * t)
    else:  # glide
        seg = seg * np.sin(np.pi * t) ** 0.6

    peak = np.abs(seg).max()
    return seg / peak if peak > 1e-9 else np.zeros(0)


def pitched_drops(n, f0, sr=C.SR, rng=None):
    """雨滴 = 短噪声脉冲 + 低 Q 共振（模态合成）。

    声学上雨滴撞击声是宽带爆破 + 表面共振，不是正弦。
    用正弦做雨滴得到的是"叮"（电子音），这是之前"尖锐高音"的来源。
    """
    from scipy.signal import lfilter
    rng = rng or np.random.default_rng(0)
    out = np.zeros(n)
    n_drops = max(1, int(n / sr * C.DROP_DENSITY))
    lo = int(sr * C.DROP_MIN_MS / 1000)
    hi = int(sr * C.DROP_MAX_MS / 1000)
    for _ in range(n_drops):
        L = int(rng.integers(lo, max(lo + 1, hi)))
        if L + 1 >= n:
            continue
        pos = int(rng.integers(0, n - L - 1))
        t = np.arange(L) / sr
        # 撞击瞬间：2.5ms 的宽带噪声爆破
        imp = rng.standard_normal(L) * np.exp(-t / 0.0025)
        # 表面共振：并联几个低 Q 谐振器（Q≈8），中心在 f0 及其谐波上
        y = np.zeros(L)
        for k, g in ((1, 1.0), (2, 0.55), (3, 0.28)):
            fk = f0 * k * (1.0 + rng.normal(0, 0.004))   # 轻微失谐，避免机械
            bw = max(60.0, fk / 8.0)
            r = np.exp(-np.pi * bw / sr)
            th = 2 * np.pi * fk / sr
            y += g * lfilter(np.array([1.0 - r]),
                             np.array([1.0, -2 * r * np.cos(th), r * r]), imp)
        # 整体衰减 + 混入一点原始爆破（水花感）
        y *= np.exp(-t / (C.DROP_MAX_MS / 1000.0 * 0.55))
        y = y / (np.abs(y).max() + 1e-9)
        y = (1 - C.DROP_NOISE) * y + C.DROP_NOISE * imp
        out[pos:pos + L] += y * rng.uniform(0.3, 1.0)
    return out


# ---------------------------------------------------------------- VC 过渡
def synth_vc(vowel, cons_name, f0, source, rng):
    """VC 样本 = 元音尾巴 + 下一个音节的辅音起音。

    JapaneseCVVCPhonemizer 用别名 "a k" 找它，作用是把前一个音符的元音
    平滑交接到后一个音节的辅音上，消掉「字字断开」的感觉。
    """
    base, _ = synth_sample(vowel, "", f0, source, rng)
    nv = int(C.SR * C.VC_VOWEL_MS / 1000.0)
    a = int(C.SR * 0.035)                      # 跳过开头 35ms 淡入
    seg = base[a:a + nv].astype(np.float64).copy()
    if len(seg) < nv:
        seg = np.pad(seg, (0, nv - len(seg)))

    c = make_consonant(cons_name, source, rng, f0)
    seg = seg / (np.abs(seg).max() + 1e-9)
    if len(c) == 0:
        out = seg
    else:
        c = c / (np.abs(c).max() + 1e-9) * C.VC_CONS_GAIN
        xf = min(int(C.SR * 0.010), len(seg) // 3, len(c) // 2)
        if xf > 4:
            # 同样要叠加，不能各衰减后硬拼
            ramp = np.linspace(0, 1, xf)
            cross = seg[-xf:] * (1 - ramp) + c[:xf] * ramp
            out = np.concatenate([seg[:-xf], cross, c[xf:]])
        else:
            out = np.concatenate([seg, c])

    # 尾巴淡出，防止接下一个音符爆音
    f = min(int(C.SR * C.VC_FADE_OUT), len(out) // 3)
    if f > 4:
        out[-f:] *= np.linspace(1, 0.30, f) ** 1.2
    out -= out.mean()
    out = out / (np.abs(out).max() + 1e-9) * (10 ** (C.PEAK_DBFS / 20.0))
    return out.astype(np.float32)


def _tv_formant(exc, f1, f2, n, nseg=5, xf_ms=12.0):
    """对同一段激励做时变共振峰：从 f1 线性过渡到 f2。"""
    bounds = np.linspace(0, n, nseg + 1).astype(int)
    xf = int(C.SR * xf_ms / 1000.0)
    out = np.zeros(n)
    for i in range(nseg):
        a, b = bounds[i], bounds[i + 1]
        t = i / (nseg - 1) if nseg > 1 else 0.0
        # 非线性过渡：前元音只占开头一小段，尽快落到后元音上。
        # 50/50 线性的话，「あ」音符的后 190ms 里有一半口型还是「え」，
        # 听起来这个音就是"不一样"的。
        t = t ** C.VV_SNAP
        seg = exc[a:b].copy()
        for (fa, bwa), (fb, bwb) in zip(f1, f2):
            seg = resonator(seg, fa * (1 - t) + fb * t,
                            bwa * (1 - t) + bwb * t)
        # 段间重叠相加，不各自归一化（否则电平会一格一格跳）
        if a > 0 and xf > 4:
            m = min(xf, len(seg))
            ramp = np.linspace(0, 1, m)
            out[a - m:a] = out[a - m:a] * (1 - ramp) + seg[:m] * ramp
            out[a:a + len(seg) - m] = seg[m:]
        else:
            out[a:a + len(seg)] = seg
    return out


def synth_vv(vowel, next_vowel, f0, source, rng, kana=None, tone_name=None):
    """VV 样本：直接拿现成的 CV 采样做"位置对齐"。

    两个坑：
    1) 不能用 synth_sample 重新生成元音 —— 颗粒取自随机位置，两次调用
       得到的是不同信号（实测质心差 947Hz）。
       必须读磁盘上已有的 CV wav，才是同一段音频。
    2) CV 的 preutterance=55ms，VV 的=190ms。要让
          VV文件[190ms] == CV文件[55ms]
       就在 VV 前面补 135ms 的 pad。
    """
    cv_path = None
    if kana and tone_name:
        p = C.VOICEBANK / f"{kana}_{tone_name}.wav"
        if p.exists():
            cv_path = p

    if cv_path is not None:
        x, _ = sf.read(str(cv_path), dtype="float64")
    else:
        x, _ = synth_sample(next_vowel, "", f0, source, rng)
        x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)

    shift = int((C.VC_VOWEL_MS - C.OTO_PREUTTERANCE) / 1000.0 * C.SR)
    if shift > 0:
        pad_s = min(int(C.SR * 0.30), max(0, len(x) - shift - 1))
        pad = x[pad_s:pad_s + shift].copy()   # 用稳态段，不用采样开头（那里有颗粒启动瞬态）
        if len(pad) < shift:
            pad = np.pad(pad, (0, shift - len(pad)))
    else:
        pad = np.zeros(0)
    out = np.concatenate([pad, x])

    n = int(C.SR * 0.60)
    out = out[:n]
    f = min(int(C.SR * C.VC_FADE_OUT), len(out) // 3)
    if f > 4:
        out[-f:] *= np.linspace(1, 0.30, f) ** 1.2
    out -= out.mean()
    out = out / (np.abs(out).max() + 1e-9) * (10 ** (C.PEAK_DBFS / 20.0))
    return out.astype(np.float32)


def synth_breath(kind, source, rng, n_target=None):
    """呼吸声：雨声片段的带通 + 单侧包络。

    呼吸没有基频，所以不需要颗粒合成、不需要共振峰、不需要谐波支撑——
    雨声本身就是宽带噪声，正好是气声的素材。
      kind="in"  吸气：渐强，高通 1.8kHz（气流在齿间摩擦，偏高频）
      kind="out" 呼气：渐弱，带通 0.9-3.2kHz（胸腔共鸣，更低更宽）
    """
    dur_ms = C.BREATH_IN_MS if kind == "in" else C.BREATH_OUT_MS
    n = int(C.SR * (n_target if n_target else dur_ms / 1000.0))
    if len(source) <= n:
        seg = np.tile(source, int(np.ceil(n / len(source))) + 1)[:n].copy()
    else:
        off = int(rng.integers(0, len(source) - n))
        seg = source[off:off + n].astype(np.float64).copy()
    seg = seg - seg.mean()

    if kind == "in":
        lo_i, hi_i = C.BREATH_IN_BP
        seg = sosfilt(butter(4, [lo_i, min(hi_i, C.SR * 0.45)], btype="bandpass",
                             fs=C.SR, output="sos"), seg)
        env = np.linspace(0, 1, n) ** 1.6            # 渐强
    else:
        lo, hi = C.BREATH_BP
        seg = sosfilt(butter(4, [lo, min(hi, C.SR * 0.45)], btype="bandpass",
                             fs=C.SR, output="sos"), seg)
        env = np.linspace(1, 0, n) ** 1.6            # 渐弱

    # 头尾各 15ms 微淡，防爆音
    f = max(4, int(C.SR * 0.015))
    env[:f] *= np.linspace(0, 1, f)
    env[-f:] *= np.linspace(1, 0, f)

    out = seg * env
    out -= out.mean()
    peak = np.abs(out).max()
    if peak < 1e-9:
        return np.zeros(n, dtype=np.float32)
    out = out / peak * (10 ** (C.BREATH_PEAK_DB / 20.0))
    return out.astype(np.float32)


# ---------------------------------------------------------------- 主合成
def synth_sample(phoneme, tone_name, f0, source, rng, mode="formant", consonant=None):
    n = int(C.SR * C.SAMPLE_DUR)

    if mode == "shift":
        # 直通变调模式：直接拿雨声片段做 pitch shift（雨声无基频，听感接近原声）
        import librosa
        seg = source[: int(C.SR * 2.0)]
        if len(seg) < int(C.SR * 0.3):
            seg = np.pad(seg, (0, int(C.SR * 0.5)))
        x = librosa.effects.pitch_shift(seg, sr=C.SR, n_steps=0.0)
        x = x[:n]
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)))
        x = x * np.hanning(len(x)) ** 0.15
    else:
        if C.ENGINE == "raindrop":
            # 雨滴声门脉冲：HPSS 分离出的真实雨滴，按 1/f0 周期排列。
            # 比颗粒合成干净得多——颗粒内部就是随机的。
            drops = extract_drops(source)
            if drops:
                exc = drop_excitation(drops, f0, n, C.SR, rng)
                exc = sosfilt(butter(2, 90.0, btype="highpass",
                                     fs=C.SR, output="sos"), exc)
                exc = exc / (np.abs(exc).max() + 1e-9)
                xd = apply_formants(exc, phoneme)
                xd = xd / (np.abs(xd).max() + 1e-9)

                # 按音高动态调整雨滴/谐波比例：低音脉冲稀疏、周期性差，
                # 必须让谐波占更大比重，否则 C3 的 HNR 只有 -0.3（几乎无音高）
                _b = float(np.clip((350.0 - f0) / 150.0, 0.0, 1.0))
                mix = max(0.15, C.RAINDROP_DROP_MIX - _b * 0.40)
                if mix < 1.0:
                    h = glottal_excitation(f0, n, C.SR, rng)
                    h = h / (np.abs(h).max() + 1e-9)
                    xh = apply_formants(h, phoneme)
                    xh = xh / (np.abs(xh).max() + 1e-9)
                    x = mix * xd + (1 - mix) * xh
                else:
                    x = xd

                if C.RAINDROP_AIR > 0:
                    air = sosfilt(butter(4, 2500.0, btype="lowpass",
                                         fs=C.SR, output="sos"),
                                  source[:n] if len(source) >= n
                                  else np.pad(source, (0, n - len(source))))
                    air = air - air.mean()
                    ar = np.sqrt(np.mean(air ** 2)) + 1e-9
                    xr = np.sqrt(np.mean(x ** 2)) + 1e-9
                    x = x + air / ar * xr * C.RAINDROP_AIR
            else:
                x, _ = synth_sample(phoneme, "", f0, source, rng)
                x = x.astype(np.float64)
        elif C.ENGINE == "grain":
            # 颗粒合成：雨声颗粒按 f0 周期排列 -> 既是雨又有音高
            exc = granular_excitation(source, f0, n, C.SR, rng)
            exc = sosfilt(butter(2, 130.0, btype="highpass",
                                 fs=C.SR, output="sos"), exc)
            exc = exc / (np.abs(exc).max() + 1e-9)

            x = apply_formants(exc, phoneme)
            x = x / (np.abs(x).max() + 1e-9)

            # 谐波支撑：纯颗粒的波形重复性不如真声带，掺一点稳住音高。
            # 低音区要额外多给：颗粒间隔 = 1/f0，低音颗粒更长、随机内容更多。
            _boost = float(np.clip((350.0 - f0) / 150.0, 0.0, 1.0))
            _h = min(0.85, C.GRAIN_HARMONIC + _boost * C.GRAIN_LOW_BOOST)
            if _h > 0:
                h = glottal_excitation(f0, n, C.SR, rng)
                hf = apply_formants(h, phoneme)
                hf = hf / (np.abs(hf).max() + 1e-9)
                x = (1 - _h) * x + _h * hf

            # 雨声慢速起伏 = "雨一直在下"的呼吸感
            if C.GRAIN_MOVEMENT > 0:
                eq_s, gain_s = rain_timbre(source, n)
                if len(gain_s) > 1:
                    g = np.interp(np.arange(len(x)) / C.ENV_HOP,
                                  np.arange(len(gain_s)), gain_s)
                    g = 1.0 + (g - 1.0) * C.GRAIN_MOVEMENT
                    x = x * np.clip(g, 0.5, 1.6)
        elif C.ENGINE == "hybrid":
            # 雨声 -> 梳状滤波(出音高) -> 元音共振峰(出元音)
            # 激励源就是雨本身，所以不再需要 rain_timbre / 气声层 / 空气层
            seg = source[:n] if len(source) >= n else np.tile(
                source, int(np.ceil(n / len(source))) + 1)[:n]
            seg = seg.astype(np.float64).copy()
            seg = seg - seg.mean()
            seg = sosfilt(butter(2, 80.0, btype="highpass", fs=C.SR, output="sos"), seg)

            pitched = comb_fast(seg, f0)
            fc = C.HYB_DAMP_FC * (1.25 if C.VOICE_GENDER == "female" else 1.0)
            pitched = sosfilt(butter(2, min(fc, C.SR * 0.45), btype="lowpass",
                                     fs=C.SR, output="sos"), pitched)
            pitched = pitched / (np.abs(pitched).max() + 1e-9)

            x = apply_formants(pitched, phoneme)
            x = x / (np.abs(x).max() + 1e-9)

            # 混回一点未共振的雨声保住颗粒，比例由 HYB_RAW_MIX 控制
            raw = seg / (np.abs(seg).max() + 1e-9)
            raw = sosfilt(butter(2, 1000.0, btype="highpass", fs=C.SR, output="sos"), raw)
            raw = raw / (np.abs(raw).max() + 1e-9)
            m = C.HYB_RAW_MIX * (1.5 if C.VOICE_GENDER == "female" else 1.0)
            m = min(m, 0.6)
            x = (1 - m) * x + m * raw

            # 离散雨滴：雨味从这里来，而不是从连续底噪来
            if C.DROP_MIX > 0:
                d = pitched_drops(n, f0, C.SR, rng)
                peak = np.abs(d).max()
                if peak > 1e-9:
                    d = d / peak
                    x = x / (np.abs(x).max() + 1e-9)
                    x = (1 - C.DROP_MIX) * x + C.DROP_MIX * d

            # 谐波支撑：纯梳状滤波的 HNR 只有 5.6dB，听感是一团气声。
            # 掺一部分纯谐波（同样过共振峰），音高才立得起来。
            if C.HYBRID_MIX < 1.0:
                h = glottal_excitation(f0, n, C.SR, rng)
                h = apply_formants(h, phoneme)
                h = h / (np.abs(h).max() + 1e-9)
                x = x / (np.abs(x).max() + 1e-9)
                x = C.HYBRID_MIX * x + (1.0 - C.HYBRID_MIX) * h

            # 整体雨味：不用离散雨滴（那会有点状颗粒），改成把雨声的静态频谱
            # 曲线 + 慢速起伏一起染到整个元音上。
            # 慢速起伏（<6Hz）是"雨一直在下"的感觉来源，缺了它就只有音色没有雨。
            if C.HYB_TIMBRE_DEPTH > 0 or C.HYB_MOVEMENT > 0:
                eq_s, gain_s = rain_timbre(source, n)
                eq_soft = ((1.0 - C.HYB_TIMBRE_DEPTH)
                           + C.HYB_TIMBRE_DEPTH * eq_s)
                eq_soft = np.clip(eq_soft, 0.2, 6.0)
                if C.HYB_MOVEMENT > 0 and len(gain_s) > 1:
                    # gain_s 原本是 ±RAIN_MOVEMENT，按 HYB_MOVEMENT 缩放
                    gain_s = 1.0 + (gain_s - 1.0) * (C.HYB_MOVEMENT / max(C.RAIN_MOVEMENT, 1e-6))
                    gain_s = np.clip(gain_s, 0.6, 1.5)
                else:
                    gain_s = np.ones(1)
                x = apply_timbre(x, eq_soft, gain_s)

            # 分频共存：雨声只出现在人声核心频段（300-3000Hz）之外。
            # 这样人声的谐波不被糊住（听得清），雨声又铺满上下两头（听得见雨）。
            if C.RAIN_BANDS_MIX > 0:
                m3 = len(x)
                rs = source[:m3] if len(source) >= m3 else np.pad(
                    source, (0, m3 - len(source)))
                rs = rs - rs.mean()
                hp = sosfilt(butter(4, min(C.RAIN_HP, C.SR * 0.45),
                                    btype="highpass", fs=C.SR, output="sos"), rs)
                lp = sosfilt(butter(4, C.RAIN_LP, btype="lowpass",
                                    fs=C.SR, output="sos"), rs)
                rband = hp * C.RAIN_HP_GAIN + lp * C.RAIN_LP_GAIN
                if C.HYB_MOVEMENT > 0 and len(gain_s) > 1:
                    g3 = np.interp(np.arange(m3) / C.ENV_HOP,
                                   np.arange(len(gain_s)), gain_s)
                    rband = rband * g3
                x = mix_rms(x, rband, C.RAIN_BANDS_MIX)
        else:
            exc = mix_rms(glottal_excitation(f0, n, C.SR, rng),
                          noise_source(n, rng), 0.10)

            # 1) 雨声音色：静态 EQ + 慢速起伏（不破坏谐波结构）
            eq, gain = rain_timbre(source, n)
            x = apply_timbre(exc, eq, gain)

            # 1.5) 预加重：抵消激励 -12dB/oct 的陡降。
            if C.PREEMPH_A > 0:
                x = _preemph(x, C.PREEMPH_A)

            # 1.6) 低频搁架：只把 250Hz 以下压几 dB，让 F1 站住。
            if C.LOWSHELF_MIX > 0:
                hp = sosfilt(butter(2, C.LOWSHELF_FC, btype="highpass", fs=C.SR, output="sos"), x)
                x = C.LOWSHELF_MIX * hp + (1.0 - C.LOWSHELF_MIX) * x

            # 2) 元音共振峰
            x = apply_formants(x, phoneme)

            # 2.5) 基频安全音
            t_f = np.arange(len(x)) / C.SR
            x = mix_rms(x, np.sin(2 * np.pi * f0 * t_f), C.FUND_MIX)

            # 3) 气声层
            if C.BREATH_MIX > 0:
                m2 = len(x)
                b = source[:m2] if len(source) >= m2 else np.pad(source, (0, m2 - len(source)))
                b = b - b.mean()
                b = apply_formants(b, phoneme)
                b = sosfilt(butter(2, C.BREATH_HP, btype="highpass", fs=C.SR, output="sos"), b)
                x = mix_rms(x, b, C.BREATH_MIX)

        # 3.5) 空气层：只留 4kHz 以上的原始雨声，比例极低。
        #      hybrid 模式下激励本身就是雨声，再加空气层雨味会过重，跳过。
        if C.AIR_MIX > 0 and C.ENGINE != "hybrid":
            m = len(x)
            a = source[:m] if len(source) >= m else np.pad(source, (0, m - len(source)))
            a = sosfilt(butter(2, C.AIR_HP, btype="highpass", fs=C.SR, output="sos"), a)
            if C.AIR_FOLLOW_F0:
                # 用基频做幅度调制：纯高频噪声留在谱上是一条平坦底噪，
                # 跟着声门脉冲起伏才像"气声"而不是"嘶声"
                am = 0.35 + 0.65 * (0.5 + 0.5 * np.cos(2 * np.pi * f0 * np.arange(m) / C.SR)) ** 2
                a = a * am
            x = mix_rms(x, a, C.AIR_MIX)

    # 4) 包络：快起、慢收
    t = np.arange(n) / C.SR
    atk = np.clip(t / C.FADE_IN, 0, 1)
    dec = np.ones(n)
    rel_start = C.SAMPLE_DUR - C.FADE_OUT
    m = t > rel_start
    dec[m] = np.clip((C.SAMPLE_DUR - t[m]) / C.FADE_OUT, 0, 1) ** 1.6
    x = x * atk * dec

    # 4.5) 辅音起音（日语 CV）。放在包络之后，免得被淡入压掉。
    # 关键是"替换"而不是"叠加"：清辅音期间声带不振动的，元音本来就没声音。
    # 之前用 += 叠在元音上，元音一强就把辅音盖住，听起来只剩元音（"诶诶啊啊"）。
    cons_ms = 0.0
    if consonant is not None and len(consonant) > 0:
        nc = min(len(consonant), int(n * 0.35))
        cpart = consonant[:nc].copy()
        fi = min(int(C.SR * 0.010), nc // 3)
        if fi > 4:
            cpart[:fi] *= np.linspace(0, 1, fi)
        cpart = cpart / (np.abs(cpart).max() + 1e-9) * 2.2
        # 辅音刻意做得比元音响（2.2 倍峰值）：清辅音能量本来就低，
        # 混音后整体归一化会把它压回去，不给点余量就听不见。
        # 替换：辅音段占主导，元音按剩余比例保留一点（浊辅音的音带振动）
        fade = min(int(C.SR * 0.020), nc // 2)
        mix = np.zeros(nc)
        mix[:] = 1.0
        if fade > 4:
            mix[-fade:] = np.linspace(1.0, C.CONS_GAIN, fade)
        x[:nc] = cpart * mix + x[:nc] * (1.0 - mix)
        cons_ms = nc / C.SR * 1000.0

    # 5) 去直流 + 归一
    x -= x.mean()
    x = x / (np.abs(x).max() + 1e-9) * (10 ** (C.PEAK_DBFS / 20.0))

    # 5.5) 兜底淡入：不管前面怎么叠（辅音、雨滴、共振峰），
    #      采样最开头 15ms 一定从 0 平滑起来，杜绝起音"咔"的一声。
    fi2 = int(C.SR * C.ONSET_FADE)
    if fi2 > 4:
        x[:fi2] *= np.linspace(0, 1, fi2) ** 0.8
    return x.astype(np.float32), cons_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["formant", "shift"], default="formant")
    ap.add_argument("--src", default=None, help="指定单个源 wav，默认用 normalized/ 里全部")
    ap.add_argument("--lang", choices=["zh", "ja"], default="zh",
                    help="zh=元音集（中文/通用），ja=日语假名 CV")
    ap.add_argument("--yoon", action="store_true", help="日语拗音也生成")
    ap.add_argument("--vv", action="store_true",
                    help="生成 VV（元音→元音）过渡，别名如 'e あ'")
    ap.add_argument("--vc", action="store_true",
                    help="生成 VC（元音→辅音）过渡，别名如 'a k'")
    ap.add_argument("--breath", action="store_true",
                    help="生成呼吸声（息/吸），用雨声做气口")
    args = ap.parse_args()

    if args.src:
        srcs = [C.NORMALIZED / args.src]
    else:
        srcs = sorted(C.NORMALIZED.glob("*.wav"))
    if not srcs:
        print(f"[!] {C.NORMALIZED} 为空，先跑 01_ingest.py")
        return

    rng = np.random.default_rng(7)
    manifest = []

    # 音源单位：(名字, 声母, 元音)
    kind = "CV"
    if args.breath:
        # 息 = 吸气，吸 = 同义（JapaneseCVVCPhonemizer 的 nonVowels 都认）
        units = [("息", "in", ""), ("吸", "in", "")]
        kind = "BR"
        print(f"[i] 呼吸声模式：{len(units)} 个别名 × {len(C.TONES)} 音阶")
    elif args.vv:
        import japanese as JP
        units = JP.vv_list()
        kind = "VV"
        print(f"[i] VV 模式：{len(units)} 个「元音→元音」组合"
              f"（别名如 e あ），× {len(C.TONES)} 音阶")
    elif args.vc:
        import japanese as JP
        units = JP.vc_list()
        kind = "VC"
        print(f"[i] VC 模式：{len(units)} 个「元音→辅音」组合"
              f"（别名如 a k / i s），× {len(C.TONES)} 音阶")
    elif args.lang == "ja":
        import japanese as JP
        units = JP.kana_list(include_yoon=args.yoon)
        print(f"[i] 日语模式：{len(units)} 个假名"
              f"{'（含拗音）' if args.yoon else ''}")
    else:
        units = [(ph, "", ph) for ph in (C.VOWELS + C.CONSONANTS)]

    # 先读一遍时长，决定每个采样取多长的源窗口
    probe, _ = sf.read(str(srcs[0]), dtype="float64")
    src_dur = len(probe) / C.SR
    need = int(C.SR * min(3.0, max(1.6, src_dur * 0.5)))
    total = len(C.TONES) * len(units)
    print(f"[i] 源总时长 {src_dur:.1f}s，每个采样取 {need / C.SR:.1f}s 窗口，共 {total} 个")

    for ti, (tone_name, f0) in enumerate(C.TONES):
        for i, (name, ini, vow) in enumerate(units):
            src = srcs[(i + ti) % len(srcs)]
            data, sr = sf.read(str(src), dtype="float64")
            if sr != C.SR:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=C.SR)

            # 确定性分散取样：让采样尽量铺满源文件，减少重复感
            span = max(0, len(data) - need)
            slot = (ti * len(units) + i) / max(1, total)
            off = int(span * ((slot * 7.3) % 1.0))
            data = data[off:off + need]
            if len(data) < need:
                data = np.pad(data, (0, need - len(data)), mode="wrap")

            if kind == "BR":
                alias = f"{name}_{tone_name}"
                fname = alias
                x = synth_breath(ini, data, rng)
                cons_ms = 0.0
            elif kind == "VV":
                # name 形如 "e あ"：前半是前一个元音，后半是后元音的假名
                parts = name.split(" ")
                prev_v = parts[0]
                kana = parts[1] if len(parts) > 1 else ""
                alias = f"{name}_{tone_name}"
                fname = f"{name.replace(' ', '_')}_{tone_name}"
                x = synth_vv(prev_v, vow, f0, data, rng,
                             kana=kana, tone_name=tone_name)
                cons_ms = 0.0
            elif kind == "VC":
                # name 形如 "a k"：前半是元音，后半是辅音
                x = synth_vc(vow, ini, f0, data, rng)
                alias = f"{name}_{tone_name}"                  # "a k_A3"
                fname = f"{name.replace(' ', '_')}_{tone_name}"  # "a_k_A3"
                cons_ms = 0.0
            else:
                cons = make_consonant(ini, data, rng, f0) if ini else None
                x, cons_ms = synth_sample(vow, tone_name, f0, data, rng,
                                          mode=args.mode, consonant=cons)
                alias = f"{name}_{tone_name}"
                fname = alias

            out = C.VOICEBANK / f"{fname}.wav"
            sf.write(out, x, C.SR, subtype=C.SUBTYPE)
            manifest.append({"alias": alias, "kind": kind, "phoneme": name,
                             "initial": ini, "vowel": vow, "tone": tone_name,
                             "f0": f0, "file": out.name,
                             "dur": round(len(x) / C.SR, 3),
                             "cons_ms": round(cons_ms, 1)})
            if (i + 1) % 10 == 0 or i == len(units) - 1:
                print(f"  [{ti + 1}/{len(C.TONES)}] {name}_{tone_name}"
                      f"  ({i + 1}/{len(units)})")

    # 写清单：VC / VV 批次要跟已有条目合并，不能覆盖
    man_path = C.BUILD / "synth_manifest.json"
    merged = list(manifest)
    if kind in ("VC", "VV", "BR") and man_path.exists():
        try:
            old = json.loads(man_path.read_text(encoding="utf-8"))
            merged = [m for m in old if m.get("kind") != kind] + manifest
        except Exception:
            pass
    man_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[ok] {len(manifest)} 个采样 -> {C.VOICEBANK}")
    print(f"[i] 清单共 {len(merged)} 条 -> {man_path.name}")


if __name__ == "__main__":
    main()
