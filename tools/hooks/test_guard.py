#!/usr/bin/env python3
"""guard_commands.py の回帰テスト。ルールを足したらここにケースを足す。

    uv run python tools/hooks/test_guard.py

止めすぎ（正しいコマンドを塞ぐ）は自動運転を壊すので、通ってほしいケースを必ず同数以上入れる。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
GUARD = ROOT / "tools" / "hooks" / "guard_commands.py"

BLOCK = [
    "uv pip install torch",
    "pip install librosa",
    "pip3 install -r requirements.txt",
    "python -m train --config x.yaml",
    "cd foo && python3 script.py",
    "git add .env",
    'git commit -m "x" .env',
    "git push --force origin main",
    "git reset --hard HEAD~1",
    "rm -rf log/",
    "rm -rf data",
    "vastai create instance 123 --image x",
    "vastai destroy instance 5",
    "uv run python -m unittest discover",
    # `-m` だと `_gdrive` の兄弟 import が解決できず必ず ModuleNotFoundError（M3 で実測）
    "uv run python -m preprocess.download_scripts.download_ritsu --voice all",
    "uv run python -m preprocess.download_scripts.download_oniku",
    # 空文字は「未設定」として扱われ CUDA が隠れない。`-1` でないと効かない（M3 で実測）
    'CUDA_VISIBLE_DEVICES="" uv run python tools/nhv_indist.py',
    # 自分のコマンド文字列がパターンに一致し、実行中のシェルごと死ぬ（M3 で実測）
    'pkill -f "python -m train"',
    "pkill -9 -f m3_corpus",
    "ssh host 'pkill -f uv run'",
    "CUDA_VISIBLE_DEVICES= uv run python tools/m3_verify.py --ckpt x.pt",
]

ALLOW = [
    "uv add tensorboard",
    "uv add --optional ops vastai",
    "uv sync --extra train --extra export",
    "uv run python -m unittest test_svc_model",
    "uv run python tools/smoke/run_smoke.py",
    "uv run python tools/vast.py create 123 --yes",
    "uv run python tools/vast.py destroy 5 --yes",
    "git add -A && git commit -m x",
    "git push -u origin feature/svc",
    "rm -rf .smoke/",
    "echo 'python -m foo'",
    "ls && uv run python -c \"import torch\"",
    "python -m train  # guard:allow",
    "uv run python preprocess/download_scripts/download_ritsu.py --voice all",
    "uv run python preprocess/download_scripts/download_natsume.py",
    "CUDA_VISIBLE_DEVICES=-1 uv run python tools/nhv_indist.py --device cpu",
    "uv run python -m preprocess.svc.run --wav-dir download/x --out data/x",
    'pkill -9 -f "python3 -m trai[n]"',        # 角括弧で自己一致を外してある
    "pgrep -fa m3_corpus",                     # 列挙するだけなので害はない
    'pkill -f "python -m train"  # guard:allow',
]


def run(cmd: str) -> int:
    p = subprocess.run([sys.executable, str(GUARD)], cwd=str(ROOT), text=True,
                       input=json.dumps({"tool_name": "Bash", "tool_input": {"command": cmd}}),
                       capture_output=True)
    return p.returncode


def check_local_training() -> int:
    """Windows での学習を **device によらず** 止める。学習はすべて vast.ai で行うと決めている。

    CPU に逃げるのも駄目。遅いうえに実験記録の環境が本番と食い違う。
    Linux（= vast.ai インスタンス）では止めない。同じリポジトリを両方で使うため。
    """
    bad = 0
    win = sys.platform == "win32"
    cases = [
        ("uv run python -m train --config c.yaml --run_name a --device cuda", 2 if win else 0),
        ("uv run python -m train --config c.yaml --run_name a --device cpu", 2 if win else 0),
        ("uv run python -m train --config c.yaml --run_name a", 2 if win else 0),
        ("uv run python train.py --config c.yaml --run_name a", 2 if win else 0),
        ("uv run python -m train --config c.yaml --run_name a --device cuda  # guard:allow", 0),
        ("uv run python tools/smoke/run_smoke.py --device cuda", 0),   # 疎通確認は別
        ("uv run python -m preprocess.svc.run --wav-dir x --out y", 0),  # 前処理は別
    ]
    for cmd, want in cases:
        if run(cmd) != want:
            print(f"NG ローカル学習チェック (want {want}): {cmd}")
            bad += 1
    return bad


def main() -> int:
    bad = 0
    for cmd in BLOCK:
        if run(cmd) != 2:
            print(f"NG 止まるべきなのに通った: {cmd}")
            bad += 1
    for cmd in ALLOW:
        if run(cmd) != 0:
            print(f"NG 通るべきなのに止まった: {cmd}")
            bad += 1

    # 動的チェック: 既存 ckpt がある run へ --init_from を渡すと黙って無視される
    tmp = Path(tempfile.mkdtemp(prefix="guard_"))
    try:
        (tmp / "existing").mkdir()
        (tmp / "existing" / "ckpt_000100.pt").touch()
        win = sys.platform == "win32"
        # Windows では「学習はローカルで回さない」が先に出るので、# guard:allow を付けて
        # check_train 側の判定だけを見る。
        mark = "  # guard:allow" if win else ""
        pairs = [
            (f"uv run python -m train --config c.yaml --out_root {tmp} "
             f"--run_name existing --init_from base.pt --finetune", 2),
            (f"uv run python -m train --config c.yaml --out_root {tmp} --run_name existing{mark}", 0),
            (f"uv run python -m train --config c.yaml --out_root {tmp} "
             f"--run_name fresh --init_from base.pt --finetune{mark}", 0),
        ]
        for cmd, want in pairs:
            if run(cmd) != want:
                print(f"NG train チェック (want {want}): {cmd}")
                bad += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    bad += check_local_training()
    total = len(BLOCK) + len(ALLOW) + 3 + 7
    print(f"{total - bad}/{total} 一致" + ("" if bad else "  （すべて期待どおり）"))
    return bad


if __name__ == "__main__":
    sys.exit(main())
