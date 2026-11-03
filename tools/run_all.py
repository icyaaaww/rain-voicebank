# -*- coding: utf-8 -*-
"""一键跑完整条流水线。

    python run_all.py            # 用 raw/ 里已有的雨声
    python run_all.py --demo     # 没素材时先合成一段假雨跑通
    python run_all.py --vcv      # 附带生成 VCV 连音别名
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

STEPS = ["01_ingest.py", "02_synthesize.py", "03_oto.py", "04_package.py", "05_preview.py"]


def run(script, *args):
    print(f"\n{'=' * 60}\n>>> {script} {' '.join(args)}\n{'=' * 60}")
    r = subprocess.run([PY, str(HERE / script), *args], cwd=HERE)
    if r.returncode != 0:
        print(f"[!] {script} 失败，中止")
        sys.exit(r.returncode)


def main():
    args = sys.argv[1:]
    if "--demo" in args:
        run("00_demo_rain.py")
    for s in STEPS:
        if s == "01_ingest.py":
            run(s, "--trim")
        elif s == "03_oto.py" and "--vcv" in args:
            run(s, "--vcv")
        else:
            run(s)


if __name__ == "__main__":
    main()
