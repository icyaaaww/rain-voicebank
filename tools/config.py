# -*- coding: utf-8 -*-
"""rain-voicebank 全局配置。改这里，全流程跟着变。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW        = ROOT / "raw"          # 原始雨声素材丢这里
BUILD      = ROOT / "build"        # 中间产物
NORMALIZED = BUILD / "normalized"  # 规范化后的素材
RENDERS    = BUILD / "renders"     # 合成出来的 wav
PREVIEW    = BUILD / "preview"     # 试听拼接
VOICEBANK  = ROOT / "voicebank"    # 成品声库目录（可直接丢进 OpenUtau Singers）
DIST       = ROOT / "dist"         # 打包 zip

for _d in (RAW, BUILD, NORMALIZED, RENDERS, PREVIEW, VOICEBANK, DIST):
    _d.mkdir(parents=True, exist_ok=True)

# ---------- 音频格式（UTAU/OpenUtau 惯例） ----------
SR       = 44100
CHANNELS = 1
SUBTYPE  = "PCM_16"
PEAK_DBFS = -3.0

# ---------- 声库元数据 ----------
VB_NAME    = "RainVoice"
VB_AUTHOR  = "YG"
VB_VERSION = "1.0"
VB_WEB     = ""
VB_VOICE   = "雨声 / rain"
LOCALIZED_NAME_EN = "RainVoice (rain source)"
LOCALIZED_NAME_JA = "レインボイス"

# ---------- 日语版 ----------
VB_NAME_JA    = "雨音レイン"
VB_VOICE_JA   = "雨音 / rain"
PHONEMIZER_JA = "OpenUtau.Plugin.Builtin.JapaneseCVVCPhonemizer"
PHONEMIZER_ZH = "OpenUtau.Core.DefaultPhonemizer"

# ---------- 音阶（subbank）----------
# (后缀, 基频Hz)
TONES = [
    ("A3", 220.00),
    ("C4", 261.63),
    ("E4", 329.63),
    ("G4", 392.00),
    ("C5", 523.25),
]

# ---------- 音素集 ----------
# 默认做「元音 + 鼻音」最小可唱集，任何 DefaultPhonemizer / CV 声库都能用。
# 想要中文声母，把 P_CONSONANTS 打开，或换 reclist（见 README）。
VOWELS     = ["a", "i", "u", "e", "o", "n", "v"]   # v = ü
CONSONANTS = []          # 例如 ["b","p","m","f","d","t","n","l","g","k","h","j","q","x"]

# 元音共振峰（F1..F4, 中心频率Hz, 带宽Hz）—— 决定 a/i/u/e/o 的差别
# 带宽要窄：带宽越窄 Q 越高、峰越突出，元音才"立"得起来。
# 带宽 90Hz 时共振增益只有 ~10dB，压不过基频，低音区会嗡成电音。
FORMANTS = {
    "a": [(700, 70), (1220, 85), (2600, 130), (3300, 220)],
    "i": [(300, 55), (2300, 80), (3000, 130), (3700, 220)],
    "u": [(350, 55), (800, 70), (2400, 130), (3300, 220)],
    "e": [(500, 65), (1800, 85), (2500, 130), (3400, 220)],
    "o": [(450, 65), (900, 70), (2400, 130), (3300, 220)],
    "v": [(300, 55), (1750, 85), (2300, 130), (3300, 220)],
    "n": [(250, 60), (1000, 70), (2400, 130), (3300, 220)],
}

# 少女 / 年轻女性：声道更短 -> 共振峰整体上移 15~25%，带宽略窄（更清亮）
# 参考 Peterson & Barney 成年女性 a: F1 850 / F2 1220 / F3 2810，
# 这里再往"二次元少女"方向推一点（F1 900、F2 1500）。
FORMANTS_FEMALE = {
    "a": [(900, 80), (1500, 95), (3000, 140), (3900, 220)],
    "i": [(380, 60), (2800, 95), (3600, 140), (4400, 220)],
    "u": [(420, 60), (1000, 85), (2900, 140), (3800, 220)],
    "e": [(650, 75), (2200, 95), (3000, 140), (3900, 220)],
    "o": [(550, 75), (1100, 85), (2900, 140), (3800, 220)],
    "v": [(380, 60), (2100, 95), (2900, 140), (3900, 220)],
    "n": [(300, 65), (1200, 85), (2900, 140), (3900, 220)],
}

# 少女音域：比男声整体高一个八度附近
TONES_FEMALE = [
    ("C3", 130.81), ("C#3", 138.59), ("D3", 146.83), ("D#3", 155.56),
    ("E3", 164.81), ("F3", 174.61), ("F#3", 185.00), ("G3", 196.00),
    ("G#3", 207.65), ("A3", 220.00), ("A#3", 233.08), ("B3", 246.94),
    ("C4", 261.63), ("C#4", 277.18), ("D4", 293.66), ("D#4", 311.13),
    ("E4", 329.63), ("F4", 349.23), ("F#4", 369.99), ("G4", 392.00),
    ("G#4", 415.30), ("A4", 440.00), ("A#4", 466.16), ("B4", 493.88),
    ("C5", 523.25), ("C#5", 554.37), ("D5", 587.33), ("D#5", 622.25),
    ("E5", 659.26), ("F5", 698.46), ("F#5", 739.99), ("G5", 783.99),
]

# ---------- 合成参数 ----------
SAMPLE_DUR  = 1.40     # 每个采样总长（秒）。UTAU 允许任意长，1.2~1.6 好用
FADE_IN     = 0.030
FADE_OUT    = 0.030    # 别拉太长：每个音符末尾都掉音量，连唱会像在喘气
TOP_HARMONICS = 90
HARMONIC_TILT = 2.00   # 谐波衰减指数，-12dB/oct ≈ 2.0，接近人声声门谱
VIBRATO_RATE  = 5.2    # 颤音频率 Hz
VIBRATO_DEPTH = 0.018  # 颤音深度（相对频率），0.018 ≈ ±30 音分
JITTER        = 0.010  # 各次谐波的相位抖动
DRIFT_AMOUNT  = 0.0035 # 慢速音高漂移
GLOTTAL_CLOSING = 0.55 # 声门脉冲的收窄程度：越大越"实"，0=纯正弦堆
VOICE_LP      = 8000.0 # 人声高频自然滚降

BREATH_MIX    = 0.26   # 气声层：按 **RMS** 混合（不是峰值），0.26 ≈ 6% 能量
BREATH_HP     = 200.0
AIR_MIX       = 0.030  # 空气层：按 RMS 混合，0.03 ≈ 0.1% 能量
AIR_HP        = 5000.0
AIR_FOLLOW_F0 = True   # 空气层跟随基频做幅度调制，否则会留下一条平坦噪声底
FUND_MIX      = 0.0    # 基频"安全音"。谐波已经清楚时不要加：它只会让低频更"嗡"
PREEMPH_A     = 0.0    # 预加重系数。>0.9 会把低频削掉 20dB+，Worldline 解析 F0 会崩，慎用
LOWSHELF_MIX  = 0.55   # 低频搁架：0=不处理；0.55 约把 250Hz 以下衰减 7dB，让 F1 站住
LOWSHELF_FC   = 250.0
RAIN_ENV_DEPTH = 0.55  # 雨声频谱包络移植深度 0~1（1=完全用雨声音色，0=纯平谱谐波）
RAIN_MOVEMENT  = 0.18  # 雨声的慢速起伏幅度（增益 ±18%），只保留 <6Hz 成分
SOURCE_TILT_DB_PER_OCT = -6.0  # 素材频谱倾斜补偿：亮雨用负值压高频，闷雨用正值提亮
ENV_FLOOR      = 0.10  # 包络最小值，防止素材某频段空掉导致该频段被抹平
ENV_N_FFT   = 1024
ENV_HOP     = 256

# ---------- oto.ini 参数 ----------
OTO_PREUTTERANCE = 55
OTO_OVERLAP      = 150
ONSET_FADE       = 0.006   # 每个采样开头强制淡入的时长（秒），杜绝起音咔哒
OTO_CONSONANT    = 90
OTO_CUTOFF_MS    = 25     # 从末尾往前留（写成负值）

# ---------- VC（元音→辅音过渡，A2） ----------
VC_VOWEL_MS   = 190    # VC 样本里元音段的长度
VC_CONS_GAIN  = 0.0   # VC 样本里辅音段的增益（和 CONS_GAIN 对齐，之前漏改，是结尾尖锐的来源）
VC_FADE_OUT   = 0.025

# ---------- 雨声乐器（梳状滤波路线，07_rain_instrument.py） ----------
COMB_FEEDBACK = 0.88   # 梳状滤波反馈量：越大音高越明确、衰减越长
COMB_DAMP     = 0.35   # 阻尼：越大高频衰减越快，越像木/陶管体
RAIN_BLEND    = 0.30   # 混回原始雨声的比例：0=纯"嗡"，1=纯噪声
INST_ATTACK   = 0.030  # 起音时间
INST_RELEASE  = 0.080  # 收尾时间

# ---------- 混合路线（雨声梳状滤波 + 元音共振峰，08_hybrid.py） ----------
HYB_DAMP_FC   = 3500.0 # 梳状滤波后的低通截止，压掉过亮的高次谐波
HYB_RAW_MIX   = 0.02   # 混回未共振雨声：这是"连续气声"的来源，越低越干净
DROP_MIX      = 0.0    # 离散雨滴层：关掉。雨味改用整体频谱色彩（HYB_TIMBRE_DEPTH）
DROP_DENSITY  = 19.0   # 每秒雨滴数（保留配置，DROP_MIX=0 时不生效）
DROP_MIN_MS   = 8.0
DROP_MAX_MS   = 60.0
DROP_NOISE    = 0.30   # 每滴雨里"水声噪声"的比例，其余是 f0 谐波（有音高）

# ---------- 引擎 / 音色 ----------
ENGINE       = "raindrop"   # "formant"=人造谐波激励 | "hybrid"=雨声梳状滤波激励
VOICE_GENDER = "female"   # "male" | "female"，决定共振峰表与音阶
HYBRID_MIX   = 0.62       # hybrid 引擎里"雨声"与"纯谐波"的配比。
                          # 雨声占比要高，否则雨味被谐波盖掉。
CONS_PITCH_MIX = 0.65     # 辅音里梳状滤波(有音高)的占比，其余保留原始噪声
CONS_GAIN      = 0.35     # 辅音叠加到元音上的增益（原来硬编码 0.92，起音太冲）
HYB_TIMBRE_DEPTH = 0.45   # 整体雨味：把雨声的静态频谱曲线染到元音上（0=无，1=完全雨声频谱）
HYB_MOVEMENT  = 0.30   # 雨声的慢速起伏幅度（<6Hz）。这是"雨一直在下"的感觉来源

# ---------- 分频共存（创新方案）----------
# 人耳靠谐波抓音高，但如果谐波所在频段被噪声糊住就抓不住。
# 所以不叠在同频段，而是让雨声只出现在人声核心频段之外：
#   300-3000Hz 留给人声（元音可懂度全靠这里）
#   >4000Hz 给雨声（空气感/嘶声）
#   <180Hz 给雨声（低频涌动）
RAIN_BANDS_MIX = 0.30  # 分频雨声的混入量
RAIN_HP        = 4000.0
RAIN_LP        = 180.0
RAIN_HP_GAIN   = 0.75  # 高频雨声权重（太大会刺）
RAIN_LP_GAIN   = 1.00  # 低频雨声权重

# ---------- 颗粒合成（09_granular.py，雨声颗粒按 f0 周期排列）----------
GRAIN_CYCLES  = 6.0    # 每个颗粒跨几个基频周期：小=颗粒感强/噪，大=音色稳/糊
GRAIN_JITTER  = 0.012   # 颗粒取源位置的随机抖动（0=完全重复变机械）
GRAIN_HARMONIC = 0.68  # 纯谐波支撑比例（纯颗粒的周期性不如真声带）
GRAIN_LOW_BOOST = 0.32 # 低音区额外谐波补偿：颗粒间隔 = 1/f0，低音颗粒更长、
                       # 随机内容更多，周期性被稀释（C4 的 HNR 只有 4.9dB，A4 是 10.6dB）
GRAIN_MOVEMENT = 0.35  # 雨声慢速起伏深度

# ---------- 雨滴声门脉冲（10_raindrop.py）----------
RAINDROP_DROP_MIX = 0.65   # 雨滴脉冲 vs 纯谐波的比例
RAINDROP_AIR      = 0.10   # 雨幕气声底的比例
RAINDROP_PREEMPH  = 0.90   # 激励预加重：雨滴脉冲低频太重，不提高频 F2 出不来
VV_SNAP       = 0.50   # VV 过渡曲线指数：越小越快落到后元音（1.0=线性 50/50）

# ---------- 呼吸声（雨声做气口） ----------
BREATH_IN_MS  = 420    # 吸气时长
BREATH_OUT_MS = 560    # 呼气时长
BREATH_IN_BP  = (1800.0, 5200.0)  # 吸气带通：上限必须卡住，否则雨声的 4-8kHz 会让质心飙到 7k
BREATH_BP     = (900.0, 3200.0)  # 呼气带通范围（更低更宽，像"呼——"）
BREATH_PEAK_DB = -8.0  # 呼吸声峰值（比元音低，否则会抢戏）

# ---- 生效配置（不要在下面再改 TONES / FORMANTS_ACTIVE 之外的东西）----
TONES = TONES_FEMALE if VOICE_GENDER == "female" else TONES
FORMANTS_ACTIVE = FORMANTS_FEMALE if VOICE_GENDER == "female" else FORMANTS
FORMANT_BW_SCALE = 1.0 if VOICE_GENDER == "male" else 1.0
