# 雨音レイン / Rain Voicebank

用雨声合成的一套日语声库（UTAU / OpenUtau 格式）。

不是录音采样，是**程序生成的**——把雨滴的瞬态当成声带脉冲，按音高周期排列，再过少女共振峰。

> **她是一场没有停的雨，学会了用人的形状站着。**
> 角色设定见 [docs/character.md](docs/character.md)

---

## 试听

| 文件 | 内容 |
|---|---|
| [samples/watashi_full.mp3](samples/watashi_full.mp3) | 稲葉曇「わたしは雨」翻唱（人声 + 伴奏） |
| [samples/watashi_vocal.mp3](samples/watashi_vocal.mp3) | 同一首的纯人声 |
| [samples/ame_low.mp3](samples/ame_low.mp3) | 早期 demo「あめあめふれふれ」 |

---

## 两个版本

| | OpenUtau 完整版 | UTAU 兼容版 |
|---|---|---|
| 文件 | `雨音レイン_v1.0.zip`（Release） | `release/雨音レイン_UTAU_C4.zip` |
| 体积 | 486.7 MB | 12.2 MB |
| 音阶 | 32（C3–G5，每半音） | 1（C4，靠变调） |
| 采样 | 7904 | 106 |
| 结构 | CV + VC + VV + 呼吸声 | CV only |
| 安装 | `OpenUtau\Singers\` | `UTAU\voice\` |

**完整版**超过 GitHub 的单文件限制，放在 [Releases](../../releases) 里下载。

---

## 技术原理

```
雨声（4K 视频提取）
  → HPSS 分离「雨幕」和「雨滴」
  → onset 检测出每一滴雨落下的时刻（约 334 滴有效）
  → 裁成正好一个 f0 周期长，陡起音 + 指数衰减窗
  → 按 1/f0 周期排列  ← 这一步把噪声变成有音高的声门脉冲
  → 少女共振峰（F1/F2/F3/F4）
  → 65% 雨滴 + 35% 谐波支撑
  → 音符包络
```

对应关系：

| 人声 | 这套声库 |
|---|---|
| 声带脉冲 | 雨滴瞬态 |
| 声道（共振峰） | IIR 谐振器组 |
| 气声 | 雨幕噪声 |

---

## 已知限制

**唱得出旋律，但听不太清词。**

这不是参数没调好，是物理限制：

```
雨的听感   = 能量平铺在 100 Hz – 10 kHz（宽带噪声）
人声的可懂 = 能量集中在 3-4 个共振峰上（窄带）
```

两者在频谱上互斥。项目里试过 8 种架构（梳状滤波、颗粒合成、雨滴脉冲、谐波包络、频谱整形、倾斜移植、统一激励、因果驱动），**没有一个能同时做到"像人"和"像雨"**。

具体表现：
- a / i / e 三个元音能分开（相似度 0.58–0.88）
- **u / o 分不开**（0.96）——基频 261 Hz 时，两者 F1 只差 130 Hz，不到一个谐波间隔
- 音节间有缝隙（UTAU 版没有 VC/VV 过渡）

**建议用途**：当作一个独特的音色／合成器来用，适合氛围音乐、电子乐、配乐。不适合需要吐字清晰的演唱。

---

## 工具链

`tools/` 里是完整的生成脚本：

| 脚本 | 作用 |
|---|---|
| `config.py` | 所有可调参数 |
| `02_synthesize.py` | 主合成器（CV / VC / VV / 呼吸声） |
| `03_oto.py` | 生成 oto.ini |
| `04_package.py` | 打包 |
| `10_raindrop.py` | 雨滴声门脉冲原型 |
| `11_harmonic.py` | 谐波包络（纯人声路线） |
| `13_rain_tint.py` | 给真人声染上雨的音色 |
| `14_say.py` | 共振峰合成说话 |
| `15_utau_lite.py` | 生成 UTAU 兼容版 |
| `midi_render.py` | MIDI → 人声离线渲染 |
| `midi2ustx.py` | MIDI → OpenUtau 工程 |
| `cons.py` | 辅音合成（Klatt 式并联共振峰） |

### 重建声库

```bash
cd tools
python 02_synthesize.py --lang ja --yoon   # CV（含拗音）
python 02_synthesize.py --vc               # VC 过渡
python 02_synthesize.py --vv               # VV 过渡
python 02_synthesize.py --breath           # 呼吸声
python 03_oto.py                           # oto.ini
python 04_package.py --lang ja             # 打包
```

约 7 分钟，产出 7904 个采样。

---

## 依赖

```
numpy scipy soundfile librosa mido
```

`.wav` 采样不包含在仓库里（体积太大），需要按上面的步骤重新生成，或用 Release 里的包。

---

## 许可

- **声库本体**：可自由用于商业／非商业作品，使用时标注「雨音レイン」
- **代码**：MIT
- **试听里的「わたしは雨」**：词曲版权属稲葉曇 / inabakumori，仅作技术演示

---

## 致谢

- 素材：4K 雨景视频提取的音频
- 参考：Klatt 共振峰合成器、[soundgen 的 transplantFormants](https://search.r-project.org/CRAN/refmans/soundgen/html/transplantFormants.html)、[DDSP 音色处理](https://export.arxiv.org/pdf/2306.10886)
