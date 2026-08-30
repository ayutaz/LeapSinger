#!/usr/bin/env python3
"""リポジトリ全経路の疎通確認を 1 コマンドで回す。

依存の更新（torch / librosa / Python 本体）や環境の移行（Windows -> Linux）のあとに走らせて、
「学習が起動する / 再開する / 推論できる / WAV が出る / ONNX が出る」までを機械的に確かめる。
**入力は合成波形なので品質の検証にはならない。** 配線が壊れていないことだけを見る。

    uv run python tools/smoke/run_smoke.py                 # 全部
    uv run python tools/smoke/run_smoke.py --device cpu    # GPU なしの環境
    uv run python tools/smoke/run_smoke.py --skip preprocess export
    uv run python tools/smoke/run_smoke.py --keep          # 作業ディレクトリを残す

終了コードは失敗したステージ数。0 なら全部通っている。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PY = sys.executable


class Fail(Exception):
    pass


_ENV: dict[str, str] = {}          # 全ステージのサブプロセスに渡す追加の環境変数


def sh(args: list[str], *, cwd: Path = ROOT, timeout: int = 3600) -> str:
    """サブプロセスを回して stdout+stderr を返す。失敗したら Fail。"""
    p = subprocess.run([str(a) for a in args], cwd=str(cwd), text=True, timeout=timeout,
                       env={**os.environ, **_ENV} if _ENV else None,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        tail = "\n".join(p.stdout.splitlines()[-25:])
        raise Fail(f"exit {p.returncode}\n{tail}")
    return p.stdout


def need(out: str, marker: str, what: str) -> None:
    if marker not in out:
        raise Fail(f"{what}: 期待する出力 {marker!r} が見つからない")


def write_configs(work: Path, device: str) -> None:
    import yaml
    def load(name):
        return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
    voc = str(ROOT / "checkpoints" / "nhv_v3.onnx")
    fast = dict(save_interval=20, log_interval=5, eval_interval=20, eval_items=2,
                num_workers=0, vocoder=voc)

    svc = load("svc_base.yaml")
    svc["train"].update(**fast)
    svc["data"]["eval_songs"] = 1
    (work / "svc.yaml").write_text(yaml.safe_dump(svc, sort_keys=False), encoding="utf-8")

    svs = load("3speaker_gan2d.yaml")
    svs["model"].update(n_speakers=1, hidden=128, backbone_ch=128)
    svs["train"].update(**fast, max_batch_frames=30000, max_batch_size=8)
    svs["gan"].update(gan_start_step=10, d_warmup_steps=2, gan_ramp_steps=2)  # GAN 経路も踏む
    svs["data"] = {"spk_map": {"oniku": 0}, "eval_songs": 1, "min_sec": 0.3,
                   "silence": True, "silence_fade_sec": 0.05}
    (work / "svs.yaml").write_text(yaml.safe_dump(svs, sort_keys=False), encoding="utf-8")

    svs_pp = json.loads(json.dumps(svs))
    svs_pp["data"]["spk_map"] = {"smokedb": 0}
    svs_pp["gan"]["enabled"] = False
    (work / "svs_pp.yaml").write_text(yaml.safe_dump(svs_pp, sort_keys=False), encoding="utf-8")


# ── ステージ ───────────────────────────────────────────────────────────────────
def st_libs(w: Path, dev: str) -> str:
    out = sh([PY, "-c", (
        "import numpy as np, librosa, scipy, soundfile as sf, pyloudnorm as pyln, "
        "matplotlib; matplotlib.use('Agg');"
        "import matplotlib.pyplot as plt, onnxruntime as ort, torch;"
        "from scipy.signal import resample;"
        "from torch.utils.tensorboard import SummaryWriter;"
        "from leapsinger.mel import wav_to_mel_nhv, wav_to_mag_nhv_torch;"
        "t=np.arange(44100)/44100;"
        "wav=(0.3*np.sin(2*np.pi*220*t)).astype('float32');"
        "m=wav_to_mel_nhv(wav);"
        "g=wav_to_mag_nhv_torch(torch.from_numpy(wav))[0].numpy();"
        "assert np.isfinite(m).all() and m.shape[0]==128;"
        "assert resample(wav,16000).shape==(16000,);"
        "print('LIBS', librosa.__version__, np.__version__, scipy.__version__, "
        "ort.__version__, torch.__version__, matplotlib.__version__)")])
    need(out, "LIBS", "library API")
    return out.strip().splitlines()[-1]


def st_gen(w: Path, dev: str) -> str:
    out = sh([PY, str(ROOT / "tools" / "smoke" / "gen_synth_data.py"), str(w)])
    need(out, "[gen]", "合成データ生成")
    return out.strip().splitlines()[-1]


def _train(w: Path, cfg: str, data: str, run: str, dev: str, steps: int) -> str:
    return sh([PY, "-m", "train", "--config", str(w / cfg), "--data_dirs", str(w / data),
               "--run_name", run, "--out_root", str(w / "log"),
               "--device", dev, "--max_updates", str(steps)])


def st_svc_train(w: Path, dev: str) -> str:
    out = _train(w, "svc.yaml", "data/svc_target", "svc", dev, 20)
    need(out, "model harmonic_svc", "SVC モデル構築")
    need(out, "ckpt_000020.pt", "checkpoint 保存")
    return "SVC 20 step + ckpt"


def st_svc_resume(w: Path, dev: str) -> str:
    out = _train(w, "svc.yaml", "data/svc_target", "svc", dev, 40)
    need(out, "[resume]", "自動再開")      # torch.load 既定 (weights_only) 経路
    need(out, "ckpt_000040.pt", "再開後の checkpoint")
    return "step 20 -> 40 で自動再開"


def st_svc_infer(w: Path, dev: str) -> str:
    out = sh([PY, str(ROOT / "tools" / "smoke" / "_infer_check.py"), str(w), "svc", dev])
    need(out, "INFER-OK", "SVC 推論 + ボコーダー")
    return out.strip().splitlines()[-1]


def st_svc_pp(w: Path, dev: str) -> str:
    """SVC 前処理の CLI（実行計画 M1）。2 段目だけを合成 cache から回す。

    1 段目は ContentVec と RMVPE を落とすので、ここでは踏まない（実モデルは
    `test_svc_preprocess_integration.py` の担当）。ここで見るのは CLI・`build_shard`・
    データ契約・**再実行で bit 一致**（M1 ゴール 4）と、実 loader が読めること（ゴール 2）。
    """
    args = [PY, "-m", "preprocess.svc.run", "--from-cache", str(w / "svc_cache"),
            "--n-dims", "32", "--subset-seed", "0"]
    out = sh([*args, "--out", str(w / "data" / "svc_pp")])
    need(out, "[shard]", "shard 生成")
    sh([*args, "--out", str(w / "data" / "svc_pp2")])
    a = (w / "data" / "svc_pp" / "svc_shard.npz").read_bytes()
    b = (w / "data" / "svc_pp2" / "svc_shard.npz").read_bytes()
    if a != b:
        raise Fail("同じ cache・同じ設定なのに shard が bit 一致しない")

    check = ("import json,sys;from svc_dataset import SVCFeatureDataset;"
             f"ds=SVCFeatureDataset([r'{w / 'data' / 'svc_pp'}'],split='train',eval_songs=0);"
             "it=ds[0];assert it['content'].shape[1]==32;"
             "m=json.load(open(r'" + str(w / 'data' / 'svc_pp' / 'manifest.json') + "'));"
             "assert m['loudness_normalization']=='dataset_zscore';"
             "print('PP-OK',len(ds),it['content'].shape[1])")
    got = sh([PY, "-c", check])
    need(got, "PP-OK", "loader が読めること")
    return f"{got.strip().splitlines()[-1]} / 再実行で bit 一致"


def st_svs_train(w: Path, dev: str) -> str:
    out = _train(w, "svs.yaml", "data/oniku", "svs", dev, 20)
    need(out, "[gan] enabled=True", "GAN 有効")
    need(out, "ckpt_000020.pt", "checkpoint 保存")
    ev = sorted((w / "log" / "svs").glob("events.out.tfevents.*"))
    if not ev:
        raise Fail("TensorBoard の events が無い")
    return "SVS 20 step (GAN on)"


def st_preprocess(w: Path, dev: str) -> str:
    out = sh([PY, "-m", "preprocess.run", "--recipe", str(w / "recipe.yaml"),
              "--out_root", str(w / "data_pp")], timeout=3600)
    need(out, "shard.npz", "shard 生成")
    meta = json.loads((w / "data_pp" / "smokedb" / "metadata.json").read_text(encoding="utf-8"))
    if not meta.get("phrases"):
        raise Fail("phrase が 0 件")
    return f"{len(meta['phrases'])} phrases (RMVPE 込み)"


def st_svs_pp(w: Path, dev: str) -> str:
    out = sh([PY, "-m", "train", "--config", str(w / "svs_pp.yaml"),
              "--data_dirs", str(w / "data_pp" / "smokedb"), "--run_name", "svs_pp",
              "--out_root", str(w / "log"), "--device", dev, "--max_updates", "20"])
    need(out, "ckpt_000020.pt", "前処理出力での学習")
    return "前処理出力 -> 学習"


def st_svs_infer(w: Path, dev: str) -> str:
    out = sh([PY, str(ROOT / "tools" / "smoke" / "_infer_check.py"), str(w), "svs", dev])
    need(out, "INFER-OK", "SVS 推論 + ボコーダー")
    return out.strip().splitlines()[-1]


def st_export(w: Path, dev: str) -> str:
    res = []
    for name, extra in (("fp32", []), ("fp16", ["--fp16"])):
        out = sh([PY, "-m", "export.cli", "--ckpt", str(w / "log" / "svs" / "ckpt_000020.pt"),
                  "--out", str(w / f"export_{name}"), "--model-name", f"smoke_{name}",
                  "--variant", "diffsinger", "--hop", "512", "--speaker", "embed",
                  "--verify", *extra], timeout=1800)
        need(out, "[export] final", f"{name} 書き出し")
        need(out, "finite=True", f"{name} の ORT 検証")
        m = re.search(r"\[export\] final -> .*? \(([\d.]+ MB)\)", out)
        res.append(f"{name} {m.group(1) if m else '?'}")
    return " / ".join(res)


def st_unittest(w: Path, dev: str) -> str:
    # 重いモデルを使わない単体テストを全部。統合テスト（実 ContentVec / RMVPE）は
    # LEAPSINGER_INTEGRATION=1 のときだけ走るので、ここでは自動的に skip される。
    files = sorted(p.stem for p in ROOT.glob("test_*.py") if "integration" not in p.stem)
    out = sh([PY, "-m", "unittest", *files])
    need(out, "OK", "unittest")
    m = re.search(r"Ran (\d+) tests", out)
    return f"{m.group(1) if m else '?'} tests OK ({len(files)} files)"


STAGES = [
    ("libs", st_libs, "3rd-party API（librosa/scipy/onnxruntime/matplotlib/TB）"),
    ("gen", st_gen, "合成データ生成"),
    ("svc-train", st_svc_train, "SVC 学習"),
    ("svc-resume", st_svc_resume, "SVC 自動再開（torch.load 既定）"),
    ("svc-infer", st_svc_infer, "SVC 推論 -> mel -> WAV"),
    ("svc-pp", st_svc_pp, "SVC 前処理 CLI（2 段目・bit 一致・loader）"),
    ("svs-train", st_svs_train, "SVS 学習（GAN 有効）"),
    ("preprocess", st_preprocess, "preprocess.run（RMVPE 込み・初回 181MB DL）"),
    ("svs-pp", st_svs_pp, "前処理出力で学習"),
    ("svs-infer", st_svs_infer, "SVS 推論 -> mel -> WAV"),
    ("export", st_export, "ONNX 書き出し fp32/fp16 + ORT 検証"),
    ("unittest", st_unittest, "単体テスト"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None, help="cuda / cpu（既定は使えるほう）")
    ap.add_argument("--work", default=str(ROOT / ".smoke"))
    ap.add_argument("--skip", nargs="*", default=[], help="飛ばすステージ名")
    ap.add_argument("--only", nargs="*", default=None, help="このステージだけ")
    ap.add_argument("--keep", action="store_true", help="作業ディレクトリを消さない")
    a = ap.parse_args()

    if a.device is None:
        # `torch.cuda.is_available()` はドライバの有無しか見ないので、**実際に確保**して確かめる。
        # driver は生きているのに context 生成が `devices busy or unavailable` で失敗する状態が
        # 実在し、そこで cuda を選ぶと全ステージが落ちる（このリポジトリの Windows 機で実測）。
        probe = subprocess.run(
            [PY, "-c", "import torch;torch.zeros(1,device='cuda');print('CUDA-OK')"],
            cwd=str(ROOT), capture_output=True, text=True)
        a.device = "cuda" if "CUDA-OK" in probe.stdout else "cpu"

    if a.device == "cpu":
        # **CPU 実行では CUDA を完全に隠す。** torch 2.13 の optimizer は step() のたびに
        # `torch.accelerator.current_stream()` を呼ぶので、CPU の tensor しか無くても
        # 壊れた CUDA に触りにいって落ちる。`""` では効かず `-1` が要る（実測）。
        _ENV["CUDA_VISIBLE_DEVICES"] = "-1"

    work = Path(a.work).resolve()
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    write_configs(work, a.device)

    print(f"device={a.device}  work={work}\n")
    rows, failed = [], 0
    for name, fn, desc in STAGES:
        if name in a.skip or (a.only and name not in a.only):
            rows.append(("SKIP", name, desc, "-", 0.0))
            continue
        t0 = time.time()
        try:
            note = fn(work, a.device)
            rows.append(("PASS", name, desc, note, time.time() - t0))
            print(f"  PASS {name:<11} {note}")
        except Exception as e:                                 # noqa: BLE001
            failed += 1
            rows.append(("FAIL", name, desc, str(e).splitlines()[0], time.time() - t0))
            print(f"  FAIL {name:<11} {e}\n")

    print("\n" + "=" * 78)
    for status, name, desc, note, sec in rows:
        print(f"{status:<5} {name:<12} {sec:>6.1f}s  {desc}")
        if status != "SKIP":
            print(f"{'':18}{note}")
    print("=" * 78)
    print(f"{'全ステージ通過' if failed == 0 else f'{failed} ステージ失敗'}"
          f"（合成音声での配線確認。品質の検証ではない）")

    if not a.keep and failed == 0:
        shutil.rmtree(work, ignore_errors=True)
    elif failed:
        print(f"作業ディレクトリを残した: {work}")
    return failed


if __name__ == "__main__":
    sys.exit(main())
