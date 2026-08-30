#!/usr/bin/env python3
"""疎通確認用の合成データを作る。実歌唱ではないので品質評価には使えない。

作るもの（すべて出力先ディレクトリ配下）:

  data/svc_target/{metadata.json,svc_shard.npz}   SVC 学習用（content は mel の線形射影で代用）
  svc_cache/*.npz                                 SVC 前処理 2 段目（build-shard）の入力
  data/oniku/{metadata.json,shard.npz}            SVS 学習用（phoneme_ids / dur_sec つき）
  db/{wav/take01.wav,mono_label/take01.lab}       preprocess.run にかける生データ
  recipe.yaml                                     その DB の recipe

mel は実際に leapsinger.mel.wav_to_mel_nhv（librosa）で計算するので、librosa 経路も踏む。

    uv run python tools/smoke/gen_synth_data.py <out_dir>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from leapsinger.mel import wav_to_mel_nhv  # noqa: E402

SR, HOP, N_FFT, WIN, N_MELS, FMIN, FMAX = 44100, 256, 2048, 2048, 128, 40.0, 16000.0
# shard に書く幅は `configs/svc_base.yaml` の `model.content_dim` と一致していなければならない。
# 食い違うと train.py が「model.content_dim=... but dataset content_dim=...」で止まる。
# ここを定数にしていたせいで、config を 768 -> 256 に変えたとき smoke の SVC 学習・再開・
# 推論の 3 ステージが黙って落ちるようになっていた。**config から読む。**
CONTENT_DIM = int(yaml.safe_load(
    (ROOT / "configs" / "svc_base.yaml").read_text(encoding="utf-8"))["model"]["content_dim"])
SSL_DIM = 768          # 1 段目の cache に残る生の ContentVec 幅（2 段目が 256 へ削る）


def _sing(dur_sec: float, base_hz: float, seed: int):
    """倍音 + ビブラート + 無声区間つきの疑似歌声 -> (wav, f0[サンプル], uv[サンプル])"""
    r = np.random.default_rng(seed)
    n = int(dur_sec * SR)
    t = np.arange(n) / SR
    seg = n // 4
    semis = np.concatenate([np.full(seg, s) for s in r.integers(-4, 8, size=4)])[:n]
    semis = np.pad(semis, (0, n - len(semis)), mode="edge").astype(np.float64)
    f0 = base_hz * 2 ** ((semis + 0.25 * np.sin(2 * np.pi * 5.5 * t)) / 12.0)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    wav = sum((0.5 / k) * np.sin(k * phase) for k in range(1, 25))
    wav += 0.01 * r.standard_normal(n)
    uv = np.ones(n)
    for a in (int(0.15 * n), int(0.62 * n)):                 # 無声（息）区間
        b = a + int(0.06 * n)
        wav[a:b] = 0.05 * r.standard_normal(b - a)
        uv[a:b] = 0.0
    wav *= np.hanning(n) ** 0.25
    return (wav / (np.abs(wav).max() + 1e-9) * 0.8).astype(np.float32), f0, uv


def _frames(x: np.ndarray, T: int) -> np.ndarray:
    """サンプル列 -> ちょうど T フレーム（mel のフレーム中心をサンプリング）"""
    return x[np.clip(np.arange(T) * HOP + HOP // 2, 0, len(x) - 1)]


def build_shards(out: Path) -> dict:
    names = [(f"song{s}_{i:04d}", 1.2 + 0.3 * i) for s in ("A", "B", "C") for i in range(3)]
    built, meta = {}, {}
    for i, (name, dur) in enumerate(names):
        wav, f0s, uvs = _sing(dur, 180.0 + 40 * (i % 3), 100 + i)
        mel = wav_to_mel_nhv(wav, sr=SR, n_fft=N_FFT, hop=HOP, win=WIN,
                             n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
        T = mel.shape[1]
        built[name] = dict(wav=wav, mel=mel, T=T,
                           f0=_frames(f0s, T).astype(np.float32),
                           uv=(_frames(uvs, T) > 0.5).astype(np.float32))
        meta[name] = T

    rng = np.random.default_rng(11)
    # ── SVC shard ────────────────────────────────────────────────────────────
    svc_dir = out / "data" / "svc_target"
    svc_dir.mkdir(parents=True, exist_ok=True)
    proj = np.random.default_rng(7).standard_normal((N_MELS, CONTENT_DIM)).astype(np.float32) * 0.05
    arrays = {}
    for name, d in built.items():
        T = d["T"]
        content = (d["mel"].T @ proj) + 0.01 * rng.standard_normal((T, CONTENT_DIM)).astype(np.float32)
        rms = np.sqrt(np.maximum(1e-10, _frames(d["wav"].astype(np.float64) ** 2, T)))
        arrays[f"{name}|content"] = content.astype(np.float32)
        arrays[f"{name}|f0_interp"] = d["f0"]
        arrays[f"{name}|uv"] = d["uv"]
        arrays[f"{name}|loudness"] = np.log(rms).astype(np.float32)
        arrays[f"{name}|mel"] = d["mel"]
    np.savez(svc_dir / "svc_shard.npz", **arrays)
    (svc_dir / "metadata.json").write_text(json.dumps(
        {"content_dim": CONTENT_DIM, "frame_rate": SR / HOP, "phrases": meta}, indent=1),
        encoding="utf-8")

    # ── SVC 前処理 2 段目の入力 cache ────────────────────────────────────────
    # 1 段目（ContentVec / RMVPE）は重いので合成で代用し、CLI の 2 段目だけを踏む。
    # content は **SSL の 50 Hz グリッド**のままにする（整列は 2 段目の仕事）。
    cache = out / "svc_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for name, d in built.items():
        T = d["T"]
        t_ssl = max(1, round(T * 50.0 / (SR / HOP)))
        np.savez(cache / f"{name}.npz",
                 content=rng.standard_normal((t_ssl, SSL_DIM)).astype(np.float32),
                 f0_hz=d["f0"], uv=d["uv"],
                 loudness=np.log(np.maximum(1e-5, np.sqrt(np.maximum(
                     1e-10, _frames(d["wav"].astype(np.float64) ** 2, T))))).astype(np.float32),
                 mel=d["mel"])

    # ── SVS shard ────────────────────────────────────────────────────────────
    phonemes = [l.split("#")[0].strip() for l in
                (ROOT / "dict" / "ja.phonemes").read_text(encoding="utf-8").splitlines()]
    phonemes = [p for p in phonemes if p]
    vowels = [phonemes.index(v) for v in ("a", "i", "u", "e", "o") if v in phonemes]
    cons = [phonemes.index(c) for c in ("k", "s", "t", "n", "m") if c in phonemes]
    svs_dir = out / "data" / "oniku"
    svs_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for j, (name, d) in enumerate(built.items()):
        T = d["T"]
        r = np.random.default_rng(200 + j)
        ids = [0] + [int(r.choice(cons if k % 2 else vowels)) for k in range(6)] + [0]
        w = r.random(len(ids)) + 0.4
        arrays[f"{name}|phoneme_ids"] = np.array(ids, dtype=np.int64)
        arrays[f"{name}|dur_sec"] = (w / w.sum() * (T * HOP / SR)).astype(np.float32)
        arrays[f"{name}|f0_interp"] = d["f0"]
        arrays[f"{name}|uv"] = d["uv"]
        arrays[f"{name}|mel"] = d["mel"]
    np.savez(svs_dir / "shard.npz", **arrays)
    (svs_dir / "metadata.json").write_text(json.dumps(
        {"frame_rate": SR / HOP, "phrases": meta}, indent=1), encoding="utf-8")
    return meta


def build_db(out: Path) -> None:
    """preprocess.run にかける生の WAV + lab（lab と波形が一致している）。"""
    db = out / "db"
    (db / "wav").mkdir(parents=True, exist_ok=True)
    (db / "mono_label").mkdir(parents=True, exist_ok=True)
    score = [("pau", 0.5, None), ("a", 0.45, 0), ("k", 0.10, None), ("i", 0.45, 2),
             ("s", 0.10, None), ("u", 0.45, 4), ("e", 0.50, 5), ("pau", 0.80, None),
             ("o", 0.50, 7), ("t", 0.10, None), ("a", 0.45, 5), ("n", 0.12, 4),
             ("i", 0.50, 3), ("m", 0.10, None), ("e", 0.55, 0), ("pau", 0.60, None)]
    r = np.random.default_rng(3)
    wav, lab, t0 = [], [], 0.0
    for ph, dur, semi in score:
        n = int(dur * SR)
        t = np.arange(n) / SR
        if semi is None:
            s = np.zeros(n) if ph == "pau" else 0.004 * r.standard_normal(n)
            if ph in ("k", "t"):
                s[: int(0.01 * SR)] += 0.2 * r.standard_normal(int(0.01 * SR))
            if ph == "s":
                s = s + 0.05 * r.standard_normal(n)
        else:
            f0 = 240.0 * 2 ** ((semi + 0.2 * np.sin(2 * np.pi * 5.0 * t)) / 12.0)
            acc = 2 * np.pi * np.cumsum(f0) / SR
            s = sum((0.5 / k) * np.sin(k * acc) for k in range(1, 30))
            s = s * np.minimum(1.0, np.minimum(t, dur - t) / 0.02) + 0.005 * r.standard_normal(n)
        wav.append(np.asarray(s, dtype=np.float64))
        lab.append(f"{t0:.6f} {t0 + dur:.6f} {ph}")
        t0 += dur
    w = np.concatenate(wav)
    sf.write(db / "wav" / "take01.wav", (w / (np.abs(w).max() + 1e-9) * 0.7).astype(np.float32), SR)
    (db / "mono_label" / "take01.lab").write_text("\n".join(lab) + "\n", encoding="utf-8")
    (out / "recipe.yaml").write_text(yaml.safe_dump(
        {"name": "smokedb", "spk_id": 0, "f0_min": 100.0, "f0_max": 700.0,
         "lab_unit": "sec", "save_wav": False, "layout": "type_split",
         "db_root": str(db), "wav_dir": "wav", "wav": "wav/{song}.wav",
         "lab": "mono_label/{song}.lab", "name_prefix": "smoke",
         "phon_norm": {}, "exclude": []}, sort_keys=False, allow_unicode=True),
        encoding="utf-8")


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / ".smoke").resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = build_shards(out)
    build_db(out)
    print(f"[gen] {len(meta)} phrases  frames={sorted(set(meta.values()))}  -> {out}")


if __name__ == "__main__":
    main()
