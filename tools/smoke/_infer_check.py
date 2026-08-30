#!/usr/bin/env python3
"""checkpoint -> mel -> NHVSing -> WAV の疎通。run_smoke.py から呼ばれる。

    uv run python tools/smoke/_infer_check.py <work_dir> {svc|svs} <device>
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from infer import infer_mel, infer_svc_mel, load_acoustic, load_vocoder, mel_to_wav  # noqa: E402

work, kind, dev = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
run = "svc" if kind == "svc" else "svs_pp"
ck = sorted(glob.glob(str(work / "log" / run / "ckpt_*.pt")))[-1]

# checkpoint が weights_only=True（torch 2.6+ の既定）で読めること
safe = torch.load(ck, map_location="cpu", weights_only=True)
assert "model" in safe and "config" in safe, "checkpoint の中身が想定と違う"

model, cfg = load_acoustic(ck, device=dev)
steps = int(cfg.get("num_steps", 1))

if kind == "svc":
    d = work / "data" / "svc_target"
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    z = np.load(d / "svc_shard.npz")
    name = sorted(meta["phrases"])[0]
    f0, uv = z[f"{name}|f0_interp"], z[f"{name}|uv"]
    logf0 = np.log2(np.maximum(f0, 1.0)).astype(np.float32)
    item = {"content": z[f"{name}|content"], "f0_logf0": logf0,
            "uv": uv.astype(np.float32), "loudness": z[f"{name}|loudness"]}
    mel = infer_svc_mel(model, item, num_steps=steps, device=dev)
    gt = z[f"{name}|mel"]
    assert mel.shape == gt.shape, (mel.shape, gt.shape)
else:
    d = work / "data_pp" / "smokedb"
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    z = np.load(d / "shard.npz")
    name = sorted(meta["phrases"])[0]
    f0, uv = z[f"{name}|f0_interp"], z[f"{name}|uv"]
    logf0 = np.log2(np.maximum(f0, 1.0)).astype(np.float32)
    T = z[f"{name}|mel"].shape[1]
    dur = z[f"{name}|dur_sec"].astype(np.float64)
    fb = np.round(np.concatenate([[0.0], np.cumsum(dur)]) * meta["frame_rate"]).astype(int)
    fb[-1] = T
    item = {"ph_ids": z[f"{name}|phoneme_ids"].astype(np.int64),
            "ph_durs": np.maximum(np.diff(fb), 1).astype(np.int64),
            "f0_logf0": logf0, "uv": uv.astype(np.float32),
            "spk_id": np.int64(0), "style_id": np.int64(0)}
    mel = infer_mel(model, item, num_steps=steps, device=dev)

assert np.isfinite(mel).all(), "mel に非有限値がある"
assert mel.shape[0] == 128, mel.shape

voc = load_vocoder(str(ROOT / "checkpoints" / "nhv_v3.onnx"))
n = mel.shape[1]
wav = mel_to_wav(voc, mel, logf0[:n], uv[:n].astype(np.float32))
assert np.isfinite(wav).all(), "WAV に非有限値がある"
peak = float(np.abs(wav).max())
assert peak > 1e-3, f"WAV が無音 (peak={peak})"
out = work / f"{kind}_out.wav"
sf.write(out, wav, 44100)

print(f"INFER-OK {kind} arch={cfg['arch']} mel={mel.shape} "
      f"wav={len(wav) / 44100:.2f}s peak={peak:.3f} -> {out.name}")
