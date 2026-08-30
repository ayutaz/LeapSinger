#!/usr/bin/env python3
"""M3（multi-singer base pretraining）の素材を用意して shard にする。

    uv run python tools/m3_corpus.py --download --out data --max-hours 0.75
    uv run python tools/m3_corpus.py --out data --write-config log/m3_base/config.yaml
    uv run python tools/m3_corpus.py --plan          # 何をするかだけ出す（課金前の確認用）

[実行計画](../doc/svc-plan.md) M3 と [台帳](../doc/svc-dataset-ledger.md) 6 節の決定に従い、
**base = GTSinger 全 9 言語 + 日本語 3 DB** を使います。

**話者ごとに 1 つの shard ディレクトリを作ります。** `svc_dataset.py` は speaker id を
**ディレクトリ名**から引く（`spk_map`）ので、話者を分けるにはディレクトリを分ける必要が
あります。波音リツだけは 3 音源が同じ歌手なので、3 ディレクトリを同じ speaker id に写します。

**`--max-hours` で歌手ごとの分量を揃えます。** base 事前学習で効くのは総時間より
**話者の多様性**です。GTSinger 80.59 h をそのまま抽出すると 12 時間以上かかるので、
上限を決めて 20 歌手ぶんを現実的な時間で通します。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GTSINGER_REPO = "GTSinger/GTSinger"
GTSINGER_LANGS = ["Chinese", "English", "French", "German", "Italian",
                  "Japanese", "Korean", "Russian", "Spanish"]

# 日本語 3 DB。波音リツは 3 音源が同じ歌手なので speaker を共有する（SVS 側の config と同じ扱い）。
JP_DBS = [
    # (shard 名, WAV ディレクトリ, 同じ歌手としてまとめる key)
    ("ritsu",        "download/ritsu",        "ritsu"),
    ("ritsu_normal", "download/ritsu_normal", "ritsu"),
    ("ritsu_soft",   "download/ritsu_soft",   "ritsu"),
    ("natsume",      "download/natsume",      "natsume"),
    ("oniku",        "download/oniku",        "oniku"),
]


def gtsinger_singers(gts_root: Path) -> list[tuple[str, Path]]:
    """`<言語>/<歌手>` を列挙する。shard 名は歌手名そのもの（`JA-Soprano-1` など）。"""
    out = []
    for lang in GTSINGER_LANGS:
        d = gts_root / lang
        if not d.is_dir():
            continue
        for singer in sorted(p for p in d.iterdir() if p.is_dir()):
            out.append((singer.name.replace("-", "_"), singer))
    return out


def build_plan(args) -> list[dict]:
    """(shard 名, WAV dir, song_parts, speaker key) の一覧。speaker id はここで確定する。"""
    plan: list[dict] = []
    gts_root = Path(args.gtsinger)
    for name, wav_dir in gtsinger_singers(gts_root if gts_root.is_absolute() else ROOT / gts_root):
        # <歌手>/<技法>/<曲>/<Group>/0000.wav なので、曲は相対パスの 1 番目。
        # 技法と Group をまたいで同じ曲を同じ song 名にしないと leakage する。
        plan.append({"name": name, "wav_dir": str(wav_dir), "song_parts": "1",
                     "speaker": name, "max_hours": args.max_hours})
    for name, rel, spk in JP_DBS:
        d = ROOT / rel
        if not d.is_dir():
            print(f"[skip] {name}: {rel} が無い", flush=True)
            continue
        plan.append({"name": name, "wav_dir": str(d), "song_parts": None,
                     "speaker": spk, "max_hours": args.jp_max_hours or args.max_hours})
    keys = sorted({p["speaker"] for p in plan})
    ids = {k: i for i, k in enumerate(keys)}
    for p in plan:
        p["spk_id"] = ids[p["speaker"]]
    return plan


def pick_wavs(all_files, langs: list[str], per_singer: int) -> list[str]:
    """歌手ごとに `per_singer` 本の wav を、**技法と曲をまたいで等間隔**に選ぶ。

    先頭から取ると技法がアルファベット順に偏ります（`Breathy` ばかりになる）。
    パスは `<言語>/<歌手>/<技法>/<曲>/<Group>/NNNN.wav` の順にソートされるので、
    等間隔に間引くと技法・曲・Group がまんべんなく混ざります。
    """
    by_singer: dict[tuple[str, str], list[str]] = {}
    for f in all_files:
        p = f.split("/")
        if f.endswith(".wav") and len(p) > 2 and p[0] in langs:
            by_singer.setdefault((p[0], p[1]), []).append(f)
    picked: list[str] = []
    for key in sorted(by_singer):
        fs = sorted(by_singer[key])
        if per_singer and len(fs) > per_singer:
            step = len(fs) / per_singer
            fs = [fs[int(i * step)] for i in range(per_singer)]
        picked += fs
    return picked


def download_gtsinger(dest: Path, langs: list[str], per_singer: int) -> None:
    """HF から GTSinger の **wav だけ**を、歌手ごとに必要な本数だけ落とす。

    リポジトリは 149,037 ファイル（注釈 json / TextGrid を含む）あり、丸ごと落とすと
    HTTP 429 で律速されて 3 時間半かかります（実測）。**使う wav だけを選んで**落とすと
    1 桁減ります。`--max-hours` は抽出時にさらに正確に切ります。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import HfApi, hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    files = api.list_repo_files(GTSINGER_REPO, repo_type="dataset")
    picked = pick_wavs(files, langs, per_singer)
    print(f"[download] {GTSINGER_REPO}: {len(files)} files -> {len(picked)} wav "
          f"({len(langs)} langs, 歌手あたり {per_singer}) -> {dest}", flush=True)

    def get(rel: str) -> str:
        return hf_hub_download(GTSINGER_REPO, rel, repo_type="dataset",
                               local_dir=str(dest))

    done, t0 = 0, time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(get, f): f for f in picked}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:                                   # noqa: BLE001
                print(f"  [warn] {futures[fut]}: {type(e).__name__}", flush=True)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(picked)}  {time.time() - t0:.0f}s", flush=True)
    print(f"[download] {done} files  {(time.time() - t0) / 60:.1f} min", flush=True)


def download_jp() -> None:
    """日本語 3 DB。波音リツは 3 音源すべて（同じ歌手の別の声質）。"""
    # **スクリプトのパスで実行する。** `-m` だと `_gdrive` の兄弟 import が解決できず
    # `ModuleNotFoundError: No module named '_gdrive'` になる（README もパス形式）。
    jobs = [("download_ritsu.py", ["--voice", "all"]),
            ("download_natsume.py", []), ("download_oniku.py", [])]
    for script, extra in jobs:
        print(f"[download] {script} {' '.join(extra)}", flush=True)
        p = subprocess.run([sys.executable,
                            str(ROOT / "preprocess" / "download_scripts" / script), *extra],
                           cwd=str(ROOT))
        print(f"[download] {script}: exit {p.returncode}", flush=True)


def extract(entry: dict, out_root: Path, args) -> bool:
    out = out_root / entry["name"]
    if (out / "svc_shard.npz").exists() and not args.force:
        print(f"[skip] {entry['name']}: shard が既にある", flush=True)
        return True
    cmd = [sys.executable, "-m", "preprocess.svc.run",
           "--wav-dir", entry["wav_dir"], "--out", str(out),
           "--device", args.device, "--chunk-sec", str(args.chunk_sec),
           "--min-sec", str(args.min_sec), "--min-voiced", str(args.min_voiced),
           "--subset-seed", str(args.subset_seed)]
    if entry["song_parts"]:
        cmd += ["--song-parts", entry["song_parts"]]
    if entry["max_hours"]:
        cmd += ["--max-hours", str(entry["max_hours"])]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT))
    print(f"[{entry['name']}] exit {p.returncode}  {time.time() - t0:.0f}s", flush=True)
    return p.returncode == 0


def write_config(plan: list[dict], out_root: Path, path: Path, base: Path) -> None:
    """recipe + 実際に作った shard から、この run の config を書き出す。"""
    import yaml
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
    built = [e for e in plan if (out_root / e["name"] / "svc_shard.npz").exists()]
    if not built:
        sys.exit("shard が 1 つも無いので config を書けません")
    spk_map = {e["name"]: e["spk_id"] for e in built}
    # 抜けた話者があると id に穴が空く。詰め直して n_speakers と一致させる。
    remap = {old: new for new, old in enumerate(sorted(set(spk_map.values())))}
    spk_map = {k: remap[v] for k, v in spk_map.items()}
    cfg["data"]["spk_map"] = spk_map
    cfg["model"]["n_speakers"] = len(remap)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[config] {len(built)} shards / {len(remap)} speakers -> {path}")
    print("  data_dirs: " + " ".join(str(out_root / e["name"]) for e in built))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", help="shard の出力先（<out>/<話者>/）")
    ap.add_argument("--gtsinger", default="download/gtsinger", help="GTSinger の展開先")
    ap.add_argument("--langs", nargs="*", default=GTSINGER_LANGS)
    ap.add_argument("--download", action="store_true", help="素材を取得してから抽出する")
    ap.add_argument("--plan", action="store_true", help="計画だけ出して何もしない")
    ap.add_argument("--write-config", default=None, help="この run の config を書き出す")
    ap.add_argument("--base-config", default="configs/svc_base_multi.yaml")
    ap.add_argument("--max-hours", type=float, default=0.75,
                    help="GTSinger の 1 歌手あたりの上限（時間）")
    ap.add_argument("--per-singer-files", type=int, default=400,
                    help="GTSinger を落とすときの 1 歌手あたりの wav 本数。"
                         "1 本およそ 10 秒なので 400 本 = 約 1.1 時間ぶん")
    ap.add_argument("--jp-max-hours", type=float, default=0.0,
                    help="日本語 DB の上限（0 なら --max-hours と同じ）")
    ap.add_argument("--chunk-sec", type=float, default=8.0)
    ap.add_argument("--min-sec", type=float, default=2.0)
    ap.add_argument("--min-voiced", type=float, default=0.3)
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="既存 shard も作り直す")
    ap.add_argument("--only", nargs="*", default=None, help="この話者だけ")
    a = ap.parse_args()

    out_root = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    if a.download:
        download_gtsinger(ROOT / a.gtsinger if not Path(a.gtsinger).is_absolute()
                          else Path(a.gtsinger), a.langs, a.per_singer_files)
        download_jp()

    plan = build_plan(a)
    if a.only:
        plan = [e for e in plan if e["name"] in a.only]
    n_spk = len({e["speaker"] for e in plan})
    print(f"[plan] {len(plan)} shards / {n_spk} speakers "
          f"(GTSinger 上限 {a.max_hours}h/歌手)")
    for e in plan:
        print(f"  spk{e['spk_id']:>2}  {e['name']:<16} {e['wav_dir']}")
    if a.plan:
        return 0

    t0, ok = time.time(), 0
    for e in plan:
        ok += bool(extract(e, out_root, a))
    print(f"[extract] {ok}/{len(plan)} shards  {(time.time() - t0) / 60:.1f} min")

    summary = {"plan": plan, "out": str(out_root), "max_hours": a.max_hours,
               "subset_seed": a.subset_seed, "chunk_sec": a.chunk_sec,
               "min_voiced": a.min_voiced, "built": ok}
    (out_root / "m3_corpus.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    if a.write_config:
        write_config(plan, out_root, Path(a.write_config), ROOT / a.base_config)
    return 0 if ok == len(plan) else 1


if __name__ == "__main__":
    raise SystemExit(main())
