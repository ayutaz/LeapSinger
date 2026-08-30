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
     "discover は収集条件が暗黙で、何件走ったのかが分からない",
     "`uv run python -m unittest test_svc_model test_svc_preprocess test_svc_dataset`、"
     "全経路は `uv run python tools/smoke/run_smoke.py`（unittest ステージが件数を表示する）"),

    # M3 で実測。`-m` 実行だと `_gdrive` の兄弟 import が解決できず必ず失敗する。
    (r"-m\s+preprocess\.download_scripts\.",
     "取得スクリプトは `-m` で実行できない（`ModuleNotFoundError: No module named '_gdrive'`）",
     "スクリプトのパスで実行する: "
     "`uv run python preprocess/download_scripts/download_ritsu.py --voice all`"),

    # M3 で実測。空文字は「未設定」扱いになり CUDA が隠れない。
    (r"CUDA_VISIBLE_DEVICES=(\s|\"\"|'')",
     "`CUDA_VISIBLE_DEVICES` に空文字を入れても CUDA は隠れない（未設定として扱われる）",
     "隠したいなら `CUDA_VISIBLE_DEVICES=-1` にする。"
     "`run_smoke.py --device cpu` は自動で渡す"),
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
                "別の --run_name にする（fine-tune は base と別ディレクトリに出す）")
    return None


def check_local_training(cmd: str) -> tuple[str, str] | None:
    """手元の Windows 機で学習を始めようとしていないか。**device は問わない。**

    **決定:** 学習はすべて vast.ai の Linux インスタンスで行い、手元の Windows 機は
    開発・推論・検証に使う（doc/svc-plan.md、doc/svc-data-compute.md）。

    `--device cpu` に逃げるのも駄目。GPU より桁で遅く（実測で 1 phrase 1200 step が CPU 約 60 分）、
    実験記録の環境が本番と食い違う。GPU をローカルで使うと他の作業と取り合って
    "unspecified launch failure" を起こす（実際に発生）。

    Linux では止めない。同じリポジトリを vast.ai 側でも使うため。
    """
    if sys.platform != "win32":
        return None
    if not re.search(r"-m\s+train(?![\w.])|(?<![\w./-])train\.py", cmd):
        return None
    return ("学習はすべて vast.ai で行う決定になっている（device は問わない。CPU も不可）",
            "`uv run python tools/vast.py search` -> `create` -> `tools/vast_bootstrap.sh` の順に "
            "インスタンスを用意し、そこで学習する")


def check_pkill(cmd: str) -> tuple[str, str] | None:
    """`pkill -f <pattern>` で自分の実行中シェルを殺すのを止める。

    このハーネスのコマンドは `bash -c "<コマンド全文>"` で走るので、**シェル自身の
    cmdline にパターン文字列が含まれます**。`pkill` は自分の PID は除外しますが
    親シェルは除外しないため、`pkill -f "python -m train"` を打つと**そのシェルごと死に、
    後続のコマンドが黙って実行されません**（M3 の vast.ai 作業で実測。config の書き換えが
    実行されず、原因が分かるまで数往復かかった）。

    角括弧で自己一致を外してあれば通します（`trai[n]` は自分の cmdline とは一致しない）。
    `pgrep` は列挙するだけなので対象外です。
    """
    m = re.search(r"(?<![\w./-])(pkill|killall)\s+[^|;&]*?-f\b([^|;&]*)", cmd)
    if not m or "[" in m.group(2):
        return None
    return ("`pkill -f` はこのハーネスでは実行中のシェル自身に一致し、後続のコマンドごと落とす",
            "角括弧で自己一致を外す（例: `pkill -9 -f \"python3 -m trai[n]\"`）。"
            "列挙するだけなら `pgrep -fa <pattern>`")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:                                          # noqa: BLE001
        return 0                                               # 解釈できないなら邪魔しない
    cmd = (payload.get("tool_input") or {}).get("command") or ""
    if not cmd or ALLOW_MARK in cmd:
        return 0

    for check in (check_train, check_local_training, check_pkill):
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
