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
]

ALLOW = [
    "uv add tensorboard",
    "uv add --optional ops vastai",
    "uv sync --extra train --extra export",
    "uv run python -m train --config configs/svc_base.yaml --run_name a",
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
]


def run(cmd: str) -> int:
    p = subprocess.run([sys.executable, str(GUARD)], cwd=str(ROOT), text=True,
                       input=json.dumps({"tool_name": "Bash", "tool_input": {"command": cmd}}),
                       capture_output=True)
    return p.returncode


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
        pairs = [
            (f"uv run python -m train --config c.yaml --out_root {tmp} "
             f"--run_name existing --init_from base.pt --finetune", 2),
            (f"uv run python -m train --config c.yaml --out_root {tmp} --run_name existing", 0),
            (f"uv run python -m train --config c.yaml --out_root {tmp} "
             f"--run_name fresh --init_from base.pt --finetune", 0),
        ]
        for cmd, want in pairs:
            if run(cmd) != want:
                print(f"NG train チェック (want {want}): {cmd}")
                bad += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    total = len(BLOCK) + len(ALLOW) + 3
    print(f"{total - bad}/{total} 一致" + ("" if bad else "  （すべて期待どおり）"))
    return bad


if __name__ == "__main__":
    sys.exit(main())
