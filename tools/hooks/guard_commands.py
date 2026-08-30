#!/usr/bin/env python3
"""PreToolUse hook: このリポジトリで「常に間違い」なコマンドを実行前に止める。

止めるのは、判断の余地なく誤りと言えるものだけ。迷う余地のあるもの（run 名の再利用など）は
止めず、skill 側のチェックリストに任せる。自動運転を邪魔しないため。

stdin に hook の JSON、exit 2 + stderr で拒否（stderr は Claude に返る）、exit 0 で許可。
標準ライブラリのみ。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# (正規表現, 理由, 代わりにどうするか)
RULES: list[tuple[str, str, str]] = [
    (r"(?<![\w./-])(uv\s+pip)\b",
     "`uv pip` はこのリポジトリで禁止されている（pyproject に依存が記録されないため）",
     "依存の追加は `uv add <pkg>`、extra は `uv add --optional <group> <pkg>`、同期は `uv sync`"),

    (r"(?<![\w./-])(pip3?|python3?\s+-m\s+pip)\s+install\b",
     "素の pip は使わない。プロジェクトの .venv と uv.lock が壊れる",
     "`uv add <pkg>` を使う。既存の依存を入れ直すだけなら `uv sync --extra train --extra export`"),

    # 行頭 / パイプ / && の直後に来る素の python。`uv run python` は許可。
    (r"(?:^|[;&|]\s*|\(\s*)(?!uv\s)(python3?|py)\s+(-m\s+|[\w./\\-]+\.py)",
     "素の python では .venv を使わないので torch も librosa も入っていない",
     "`uv run python -m <module>` / `uv run python <script.py>` にする"),

    (r"git\s+(add|commit).*(?<![\w./-])\.env\b",
     ".env には API token が入っている。git に載せてはいけない",
     ".env は .gitignore 済み。個別に add しないこと。値が必要なら tools/vast.py が読む"),

    (r"git\s+push\b[^|;&]*\s(--force|-f)(\s|$)",
     "force push はリモートの履歴を壊す",
     "`git push --force-with-lease` を検討し、それでもユーザーの明示的な指示があるときだけ実行する"),

    (r"git\s+reset\s+--hard\b",
     "reset --hard は未コミットの作業を復元できない形で捨てる",
     "`git stash` で退避するか、何を捨てるのか確認してから実行する"),

    (r"rm\s+-[a-zA-Z]*r[a-zA-Z]*f?[^|;&]*\s(log|data|checkpoints|\.git)(/|\s|$)",
     "log/ data/ checkpoints/ .git/ は再取得や再学習が必要な成果物を含む",
     "消す対象を具体的なパスまで絞る。学習成果物は消す前に退避する"),

    (r"(?<![\w./-])vastai\s+(create|destroy)\s+instance",
     "vastai を直接叩くと料金の確認と --yes の明示が飛ばされる",
     "`uv run python tools/vast.py create <offer_id> --yes` / `... destroy <id> --yes` を使う"),

    (r"(?<![\w./-])unittest\s+discover|python[^|;&]*-m\s+unittest\s+discover",
     "discover は top-level の test_svc_model.py しか拾わず、通っても全体の検証にはならない",
     "SVC の検証は `uv run python -m unittest test_svc_model`、"
     "全経路は `uv run python tools/smoke/run_smoke.py`"),
]

ALLOW_MARK = "# guard:allow"          # どうしても必要なときの明示的な脱出口


def _arg(cmd: str, name: str) -> str | None:
    m = re.search(rf"{re.escape(name)}[=\s]+([^\s]+)", cmd)
    return m.group(1).strip("\"'") if m else None


def check_train(cmd: str) -> tuple[str, str] | None:
    """train.py 固有の、静的な正規表現では見つからない誤り。"""
    if not re.search(r"-m\s+train|train\.py", cmd):
        return None
    run = _arg(cmd, "--run_name")
    if not run:
        return None
    out_root = Path(_arg(cmd, "--out_root") or "log")
    existing = sorted((out_root / run).glob("ckpt_*.pt")) if (out_root / run).is_dir() else []
    if existing and re.search(r"--init_from", cmd):
        # train.py は `if _ckpts: ... elif args.init_from and args.finetune:` なので、
        # 既存 ckpt があると --init_from / --finetune は黙って無視される。
        return (f"{out_root / run} に既に {existing[-1].name} があるため、--init_from は"
                f"**黙って無視され**、その checkpoint からの自動再開になる",
                f"別の --run_name にする（fine-tune は base と別ディレクトリに出す）")
    return None


def check_local_gpu_training(cmd: str) -> tuple[str, str] | None:
    """手元の Windows 機で GPU 学習を始めようとしていないか。

    **決定:** 学習は vast.ai の Linux インスタンスで行い、手元の Windows 機は開発・推論・
    検証に使う（doc/svc-plan.md、doc/svc-data-compute.md）。ローカルで回すと、
    (1) 実験記録の環境が本番と食い違う、(2) 他の作業と GPU を取り合って
    "unspecified launch failure" のような一過性の失敗を起こす。実際に起きた。

    Linux では止めない。同じリポジトリを vast.ai 側でも使うため。
    """
    if sys.platform != "win32":
        return None
    if not re.search(r"-m\s+train(?![\w.])|train\.py", cmd):
        return None
    if not re.search(r"--device[=\s]+cuda", cmd):
        return None
    return ("手元の Windows 機での GPU 学習は行わない決定になっている",
            "vast.ai のインスタンスで実行する（`uv run python tools/vast.py search` -> "
            "`create` -> `tools/vast_bootstrap.sh`）。手元で動かすなら `--device cpu`")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:                                          # noqa: BLE001
        return 0                                               # 解釈できないなら邪魔しない
    cmd = (payload.get("tool_input") or {}).get("command") or ""
    if not cmd or ALLOW_MARK in cmd:
        return 0

    for check in (check_local_gpu_training, check_train):
        hit = check(cmd)
        if hit:
            why, instead = hit
            print(f"このコマンドは止めました。\n"
                  f"  理由: {why}\n"
                  f"  代わりに: {instead}", file=sys.stderr)
            return 2

    for pattern, why, instead in RULES:
        m = re.search(pattern, cmd, re.IGNORECASE)
        if m:
            print(f"このコマンドはリポジトリの規約で止めました。\n"
                  f"  該当: {m.group(0).strip()}\n"
                  f"  理由: {why}\n"
                  f"  代わりに: {instead}\n"
                  f"（どうしても必要なら行末に `{ALLOW_MARK}` を付けると通ります）",
                  file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
