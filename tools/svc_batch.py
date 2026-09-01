#!/usr/bin/env python3
"""複数 clip を 1 プロセスで変換する（[実行計画](../doc/svc-plan.md) M5 の測定用）。

    uv run python tools/svc_batch.py --jobs out/m5/leapsvc/jobs.tsv --out out/m5/leapsvc_gpu \
      --ckpt .m0data/m4/ckpt_010000.pt --manifest <manifest.json> --spk-id 22 --device cuda

**律速はモデルの読み込みです。** 26 clip を [`svc_convert.py`](svc_convert.py) で 1 本ずつ
回したとき、GPU 使用率は **5%** しかなく、所要のほとんどは ContentVec と RMVPE を
**clip ごとに読み直す**ぶんでした。ここではモデルを **1 度だけ**読み、全 clip に使い回します。

**変換の中身は `svc_convert.py` と同じ手順**です（`extract_phrase` → `transpose_f0` →
`features_to_item` → `infer_svc_mel` → `mel_to_wav`）。**音量を触らない**、**上限を
`--self-check` と同じ形で出す**、**`--tag` で出力名を分ける**という契約も同じです。

**1 本落ちても残りは続けます。** 26 本の途中で止まると、そこまでの結果を捨てることになります。
失敗した tag と理由は報告に残します。
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.svc_defaults import SVC_NUM_STEPS  # noqa: E402

Job = dict[str, Any]


def run_batch(jobs: Sequence[Job], *, load_models: Callable[[], Any],
              convert_one: Callable[[Job, Any], Any],
              is_done: Callable[[Job], bool] | None = None) -> dict[str, Any]:
    """`jobs` を順に変換する。**モデルは 1 度だけ読み込みます。**

    `is_done` が True を返す job は飛ばします（途中から再開できるように）。
    **全部済んでいればモデルを読み込みません**（数百 MB を無駄に読まない）。
    """
    todo = [j for j in jobs if not (is_done and is_done(j))]
    skipped = [str(j["tag"]) for j in jobs if is_done and is_done(j)]
    report: dict[str, Any] = {"n_total": len(jobs), "converted": [], "failed": [],
                              "skipped": skipped, "errors": {}}
    if not todo:
        return report

    models = load_models()
    for job in todo:
        tag = str(job["tag"])
        try:
            convert_one(job, models)
        except Exception as e:                                   # noqa: BLE001
            # **1 本落ちても続ける。** 途中で止めると、そこまでの結果を捨てることになる。
            report["failed"].append(tag)
            report["errors"][tag] = f"{type(e).__name__}: {e}"
            print(f"  !! {tag}: {type(e).__name__}: {e}", flush=True)
        else:
            report["converted"].append(tag)
    return report


# 全 clip で一致していなければならない条件。**clip ごとに違って当然のもの**
# （transpose / start / seconds / wav / tag / elapsed_sec / 測定値）はここに入れない。
SHARED_FIELDS = ("ckpt", "spk_id", "num_steps", "device", "chunk_sec")


def check_consistent(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """全 clip の変換条件が揃っているかを見る。

    **実際に混ざりました。** 26 clip のうち 15 本を chunk 20 秒、11 本を 10 秒で作ってしまい、
    20 秒の clip では chunk 1 個と 2 個で境界処理が変わります。**測る前にここで気づきます。**

    欠けている項目も「違い」として扱います（古い形式の json が混ざったら気づけるように）。
    """
    if not records:
        raise ValueError("記録が空です")
    differing = []
    for key in SHARED_FIELDS:
        seen = {r.get(key, "<欠落>") for r in records}
        if len(seen) > 1:
            differing.append(key)
    return {
        "consistent": not differing, "differing": differing, "n": len(records),
        "values": {k: sorted({str(r.get(k, "<欠落>")) for r in records})
                   for k in SHARED_FIELDS},
    }


def format_progress(job: Job, *, seconds: float, elapsed: float) -> str:
    """1 clip ぶんの進捗行。

    **transpose は float で来ます**（jobs.tsv から読むため）。ここを `:+3d` で書いたせいで、
    **書き出しの後**に `ValueError` が出て 16 本が「失敗」と記録されました。出力は完全なのに
    失敗と報告されるのが最も危険で、再実行しても `is_done` に飛ばされて気づけません。
    """
    tp = int(round(float(job.get("transpose", 0))))
    return f"  {str(job['tag']):10s} {seconds:5.1f}s  tp={tp:+3d}  {elapsed:5.1f}s"


def _load_models(a) -> dict[str, Any]:
    from infer import load_acoustic, load_vocoder
    from leapsinger.config import MelSpec
    from preprocess.svc.encoders import ContentVecEncoder, RmvpeF0

    ROOT = Path(__file__).resolve().parent.parent
    mel = MelSpec()
    model, cfg = load_acoustic(a.ckpt, device=a.device)
    print(f"[batch] arch={cfg.get('arch')} n_speakers={cfg.get('n_speakers')} "
          f"-> spk_id={a.spk_id} steps={a.num_steps} device={a.device}", flush=True)
    return {
        "mel": mel, "model": model,
        "vocoder": load_vocoder(str(ROOT / a.vocoder), sr=mel.sr),
        "encoder": ContentVecEncoder(a.content_model, layer=a.layer, device=a.device),
        "f0x": RmvpeF0(device=a.device),
    }


def _convert_one(job: Job, m: dict[str, Any], *, a, manifest: dict[str, Any]) -> None:
    import json
    import time

    import numpy as np
    import soundfile as sf

    from infer import infer_svc_mel, mel_to_wav
    from preprocess.svc.extract import _resample, extract_phrase, transpose_f0
    from preprocess.svc.shard import features_to_item
    from tools.audio_metrics import band_profile
    from tools.svc_convert import output_stem

    mel = m["mel"]
    t0 = time.time()
    wav, sr = sf.read(job["path"], dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = _resample(np.ascontiguousarray(wav, dtype=np.float32), sr, mel.sr)
    start, secs = float(job.get("start", 0.0)), float(job.get("seconds", 0.0))
    if start or secs:
        a0 = int(start * mel.sr)
        a1 = a0 + int(secs * mel.sr) if secs else len(wav)
        wav = np.ascontiguousarray(wav[a0:a1])

    # **音量を触らない**（svc_convert.py と同じ契約）。学習は生の音量で特徴を取る。
    step = int(a.chunk_sec * mel.sr)
    pieces, gt_pieces = [], []
    for s in range(0, len(wav), step):
        seg = np.ascontiguousarray(wav[s:s + step])
        if len(seg) < mel.hop * 4:
            break
        feats = extract_phrase(seg, mel.sr, content_encoder=m["encoder"],
                               f0_extract=m["f0x"], mel=mel)
        feats["f0_hz"] = transpose_f0(feats["f0_hz"], float(job.get("transpose", 0)))
        item = features_to_item(feats, manifest)
        item["spk_id"] = int(a.spk_id)
        pred = infer_svc_mel(m["model"], item, num_steps=a.num_steps, device=a.device)
        logf0 = np.log2(np.maximum(feats["f0_hz"], 1.0))
        pieces.append(mel_to_wav(m["vocoder"], pred, logf0, feats["uv"]))
        gt_pieces.append(mel_to_wav(m["vocoder"], feats["mel"], logf0, feats["uv"]))

    if not pieces:
        raise ValueError(f"変換できる長さがありません（{len(wav) / mel.sr:.2f}s）")

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(job["path"], tag=job["tag"])
    conv = np.concatenate(pieces).astype(np.float32)
    gt = np.concatenate(gt_pieces).astype(np.float32)
    sf.write(out_dir / f"{stem}_converted.wav", conv, mel.sr)
    sf.write(out_dir / f"{stem}_vocoder_only.wav", gt, mel.sr)
    peak = float(np.abs(wav).max())
    sf.write(out_dir / f"{stem}_source.wav",
             (wav / peak * 0.95).astype(np.float32) if peak > 1e-6 else wav, mel.sr)
    (out_dir / f"{stem}_convert.json").write_text(json.dumps(
        {"wav": job["path"], "tag": job["tag"], "ckpt": a.ckpt, "spk_id": a.spk_id,
         "num_steps": a.num_steps, "chunk_sec": a.chunk_sec,
         "transpose": job.get("transpose", 0),
         "start": start, "seconds": len(wav) / mel.sr, "device": a.device,
         "elapsed_sec": round(time.time() - t0, 1),
         "source": band_profile(wav, mel.sr), "converted": band_profile(conv, mel.sr),
         "vocoder_only": band_profile(gt, mel.sr)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(format_progress(job, seconds=len(wav) / mel.sr, elapsed=time.time() - t0),
          flush=True)


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True,
                    help="TSV: tag / path / transpose / start / seconds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--spk-id", type=int, default=22)
    ap.add_argument("--num-steps", type=int, default=SVC_NUM_STEPS,
                    help=f"flow の step 数（既定 {SVC_NUM_STEPS}。2026-09-01 の掃引で決定）")
    ap.add_argument("--chunk-sec", type=float, default=20.0,
                    help="この長さごとに分けて変換する。**svc_convert.py と同じ既定にする** "
                         "（違うと境界処理が変わり、同じ test set の中で条件が揃わない）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--content-model", default="lengyue233/content-vec-best")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vocoder", default="checkpoints/nhv_v3.onnx")
    a = ap.parse_args()

    jobs = []
    for line in Path(a.jobs).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tag, path, tp, start, secs = line.split("\t")
        jobs.append({"tag": tag, "path": path, "transpose": float(tp),
                     "start": float(start), "seconds": float(secs)})
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    out_dir = Path(a.out)

    def is_done(job):
        from tools.svc_convert import output_stem
        return (out_dir / f"{output_stem(job['path'], tag=job['tag'])}_converted.wav").exists()

    print(f"[batch] {len(jobs)} clip -> {a.out}", flush=True)
    rep = run_batch(jobs, load_models=lambda: _load_models(a),
                    convert_one=lambda j, m: _convert_one(j, m, a=a, manifest=manifest),
                    is_done=is_done)
    print(f"\n変換 {len(rep['converted'])} / 既存 {len(rep['skipped'])} / "
          f"失敗 {len(rep['failed'])}")
    if rep["failed"]:
        print(f"  失敗した tag: {rep['failed']}")

    # **測る前に条件が揃っているかを見る。** 揃っていない set で測ると、モデルの差なのか
    # 変換条件の差なのかが分からなくなる（実際に 15 本 chunk 20 / 11 本 chunk 10 で混ざった）。
    recs = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(out_dir.glob("*_convert.json"))]
    if recs:
        con = check_consistent(recs)
        rep["consistency"] = con
        if con["consistent"]:
            print(f"  条件は {con['n']} clip すべてで一致")
        else:
            print(f"  ** 条件が揃っていません: {con['differing']} **")
            for k in con["differing"]:
                print(f"     {k}: {con['values'][k]}")
            print("  ** この set で測ると、モデルの差か条件の差か分かりません **")
    (out_dir / "batch_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
