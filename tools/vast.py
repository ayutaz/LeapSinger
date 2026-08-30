#!/usr/bin/env python3
"""vast.ai の学習インスタンスを操作する薄いラッパー。

REST API を自前で叩かず、公式 CLI (`vastai`, ops extra) に委譲する。このファイルが足すのは
次の 3 点だけ:

  1. API token を `.env` から読む（値は一切表示しない。ただし公式 CLI が受け取る口は
     `--api-key` しかないため、実行中は同一マシンの `ps` から見え得る点は許容する）
  2. 学習用途に合う検索クエリを組み立てる
  3. **課金が発生する操作を `--yes` で明示させる**

    uv sync --extra ops                                   # 初回のみ
    uv run python tools/vast.py status
    uv run python tools/vast.py search --vram 24 --max-price 0.60
    uv run python tools/vast.py create <offer_id> --disk 60 --yes
    uv run python tools/vast.py instances
    uv run python tools/vast.py ssh <instance_id>
    uv run python tools/vast.py destroy <instance_id> --yes

`.env` は .gitignore 対象。次のどれかの名前でトークンを置く:
VASTAI / VAST / VAST_API_KEY / VASTAI_API_KEY / VAST_AI_API_KEY / VAST_TOKEN / VASTAI_TOKEN
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
KEY_NAMES = ("VAST_API_KEY", "VASTAI_API_KEY", "VAST_AI_API_KEY",
             "VAST_TOKEN", "VASTAI_TOKEN", "VAST_AI_TOKEN",
             "VASTAI", "VAST")

# vast.ai 公式イメージ。cu130 の wheel を使うのでホスト CUDA も 13.0 系を選ぶ。
# gcc を含むので Linux では torch.compile(inductor) が使える。
DEFAULT_IMAGE = "vastai/base-image:cuda-13.0.3-auto"
BOOTSTRAP_URL = ("https://raw.githubusercontent.com/ayutaz/LeapSinger/"
                 "feature/svc/tools/vast_bootstrap.sh")


def _parse_env(path: Path) -> dict:
    """.env を読む。値は返すが、呼び出し側は絶対に表示しないこと。"""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        v = v.strip().strip('"').strip("'")
        if k:
            out[k] = v
    return out


def api_key() -> str:
    env = _parse_env(ENV_FILE)
    for name in KEY_NAMES:
        v = os.environ.get(name) or env.get(name)
        if v:
            return v
    found = sorted(env)                                   # 値は出さず名前だけ
    sys.exit(f"API token が見つかりません。{ENV_FILE} に次のどれかの名前で置いてください:\n"
             f"  {' / '.join(KEY_NAMES)}\n"
             f".env にある変数名: {found or '(なし)'}")


def _vastai() -> str:
    exe = shutil.which("vastai")
    if exe:
        return exe
    for cand in (ROOT / ".venv" / "Scripts" / "vastai.exe", ROOT / ".venv" / "bin" / "vastai"):
        if cand.exists():
            return str(cand)
    sys.exit("vastai CLI が見つかりません。`uv sync --extra ops` を実行してください。")


def run(args: list[str], *, raw: bool = False, check: bool = True):
    """vastai を実行。token は表示・ログしない（公式 CLI の受け口が --api-key のため argv には載る）。"""
    key = api_key()
    cmd = [_vastai(), *args] + (["--raw"] if raw else []) + ["--api-key", key]
    print(f"$ vastai {' '.join(args)}{' --raw' if raw else ''} --api-key ***",
          file=sys.stderr)                                # token はログに残さない
    p = subprocess.run(cmd, capture_output=raw, text=True)
    if check and p.returncode != 0:
        if raw and p.stderr:
            print(p.stderr, file=sys.stderr)
        sys.exit(f"vastai が失敗しました (exit {p.returncode})")
    return p.stdout if raw else ""


def _offers(query: str, order: str, limit: int) -> list[dict]:
    out = run(["search", "offers", query, "-o", order, "--limit", str(limit)], raw=True)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        sys.exit(f"検索結果を JSON として読めませんでした:\n{out[:500]}")


def cmd_status(a):
    run(["show", "instances"])


def cmd_search(a):
    q = [f"num_gpus={a.num_gpus}", f"gpu_ram >= {a.vram}", f"dph_total <= {a.max_price}",
         f"disk_space >= {a.disk}", "reliability > 0.98", f"cuda_max_good >= {a.cuda}",
         f"inet_down >= {a.inet}", "rentable=true", "verified=true"]
    if a.gpu:
        q.append(f"gpu_name={a.gpu.replace(' ', '_')}")
    offers = _offers(" ".join(q), a.order, a.limit)
    if not offers:
        print("条件に合う offer がありません。--vram を下げるか --max-price を上げてください。")
        return
    print(f"\n{'offer_id':>10} {'GPU':<22} {'n':>2} {'VRAM':>6} {'$/hr':>7} "
          f"{'disk':>6} {'down':>7} {'rel':>5}  地域")
    for o in offers[:a.limit]:
        print(f"{o.get('id', 0):>10} {str(o.get('gpu_name', ''))[:22]:<22} "
              f"{o.get('num_gpus', 0):>2} {o.get('gpu_ram', 0) / 1000:>5.0f}G "
              f"{o.get('dph_total', 0):>7.3f} {o.get('disk_space', 0):>5.0f}G "
              f"{o.get('inet_down', 0):>6.0f}M "
              f"{o.get('reliability', o.get('reliability2', 0)) or 0:>5.2f}  "
              f"{o.get('geolocation', '')}")
    print(f"\n作成: uv run python tools/vast.py create <offer_id> --disk {a.disk} --yes")


def cmd_create(a):
    matches = _offers(f"id={a.offer_id}", "dph_total", 1)
    o = matches[0] if matches else {}
    price = o.get("dph_total")
    print(f"\noffer {a.offer_id}: {o.get('gpu_name', '?')} x{o.get('num_gpus', '?')} "
          f"VRAM {o.get('gpu_ram', 0) / 1000:.0f}GB  {o.get('geolocation', '?')}")
    if price:
        print(f"料金: ${price:.3f}/hr  ->  24h で ${price * 24:.2f} / 週 ${price * 24 * 7:.2f}"
              f"（ストレージ課金は別）")
    else:
        print("この offer が見つかりません。既に他者に取られたか id が違います。")
        if not a.yes:
            return
    if not a.yes:
        print("\n課金が発生します。実行するには --yes を付けてください。")
        return
    onstart = (f"env >> /etc/environment; "
               f"curl -LsSf {BOOTSTRAP_URL} -o /root/bootstrap.sh && "
               f"RMVPE={'1' if a.rmvpe else '0'} bash /root/bootstrap.sh "
               f"> /root/bootstrap.log 2>&1")
    args = ["create", "instance", str(a.offer_id), "--image", a.image,
            "--disk", str(a.disk), "--ssh", "--direct",
            "--label", a.label, "--onstart-cmd", onstart]
    if a.no_bootstrap:
        args = [x for x in args if x not in ("--onstart-cmd", onstart)]
    run(args)
    print("\n起動後の確認:\n"
          "  uv run python tools/vast.py instances\n"
          "  uv run python tools/vast.py ssh <instance_id>\n"
          "  # bootstrap のログ: インスタンス上の /root/bootstrap.log")


def cmd_instances(a):
    run(["show", "instances"])


def cmd_ssh(a):
    run(["ssh-url", str(a.instance_id)])


def cmd_logs(a):
    run(["logs", str(a.instance_id)])


def cmd_destroy(a):
    if not a.yes:
        print(f"インスタンス {a.instance_id} を破棄します（データは消えます・取り消し不可）。\n"
              "実行するには --yes を付けてください。")
        return
    # vastai は既定で確認プロンプトを出す。stdin が無い環境では中断されるので -y を渡す。
    run(["destroy", "instance", str(a.instance_id), "-y"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="アカウントのインスタンス一覧").set_defaults(fn=cmd_status)

    s = sub.add_parser("search", help="学習に使える offer を探す（課金なし）")
    s.add_argument("--vram", type=int, default=24, help="1 GPU あたりの最低 VRAM (GB)")
    s.add_argument("--num-gpus", type=int, default=1)
    s.add_argument("--max-price", type=float, default=0.60, help="上限 $/hr")
    s.add_argument("--disk", type=int, default=60, help="最低ディスク GB（torch+nvidia で約8GB使う）")
    s.add_argument("--gpu", default=None, help="例: RTX_4090")
    s.add_argument("--cuda", type=float, default=13.0, help="ホストの CUDA 下限（cu130 wheel 用）")
    s.add_argument("--inet", type=int, default=100, help="下り最低 Mbps")
    s.add_argument("--order", default="dph_total", help="並び順。既定は安い順")
    s.add_argument("--limit", type=int, default=15)
    s.set_defaults(fn=cmd_search)

    c = sub.add_parser("create", help="インスタンスを作る（課金が発生する）")
    c.add_argument("offer_id", type=int)
    c.add_argument("--disk", type=int, default=60)
    c.add_argument("--image", default=DEFAULT_IMAGE)
    c.add_argument("--label", default="leapsinger")
    c.add_argument("--rmvpe", action="store_true", help="RMVPE の重みも落とす（前処理を回す場合）")
    c.add_argument("--no-bootstrap", action="store_true", help="onstart を付けない")
    c.add_argument("--yes", action="store_true", help="課金を承知して実行する")
    c.set_defaults(fn=cmd_create)

    sub.add_parser("instances", help="起動中インスタンス").set_defaults(fn=cmd_instances)

    for name, fn, helptext in (("ssh", cmd_ssh, "ssh の接続先を出す"),
                               ("logs", cmd_logs, "インスタンスのログ")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("instance_id", type=int)
        p.set_defaults(fn=fn)

    d = sub.add_parser("destroy", help="インスタンスを破棄する（取り消し不可）")
    d.add_argument("instance_id", type=int)
    d.add_argument("--yes", action="store_true")
    d.set_defaults(fn=cmd_destroy)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
