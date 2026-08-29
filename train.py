"""Train the LeapSinger acoustic model.

Two stages in one run: the rectified-flow + reconstruction loss first learns the base voice, then
a lightweight mel-spectrogram discriminator (GAN) is switched on partway through to sharpen the
texture. The schedule — when the GAN turns on, its strength, and so on — is set in the config's
`gan:` section; with `gan.enabled: false` it is plain flow + reconstruction training.

    python -m train --config configs/3speaker_gan2d.yaml \
        --data_dirs data/oniku data/natsume data/ritsu \
        --run_name 3speaker_gan2d --out_root log --device cuda

Re-running the same command resumes from the latest checkpoint.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from leapsinger.config import MelSpec
from dataset import (LeapSingerDataset, acoustic_collate_fn, FrameBasedBatchSampler)
from svc_dataset import SVCFeatureDataset, svc_collate_fn
from infer import infer_mel, infer_svc_mel, load_vocoder, mel_to_wav
from leapsinger.modules.discriminators import (
    JCUMelDiscriminator, Mel2DDiscriminator, d_loss_jcu, g_adv_fm_jcu, laplacian_var_ratio)
from preprocess.vocab import Vocab
from leapsinger.models.acoustic import HarmonicAcousticModel, HarmonicAcousticModelMultiSpk
from leapsinger.models.svc import HarmonicSVCModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _mel_fig(mel, title, vmin=-11.5, vmax=2.0):
    """mel [n_mels, T] -> a magma-colormapped matplotlib figure (low freq at the bottom), for
    TensorBoard add_figure. cmap='magma', origin='lower', and a fixed vmin/vmax so GT and
    prediction share one colour scale."""
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(np.asarray(mel), origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax, cmap="magma")
    ax.set_title(title, fontsize=8)
    ax.set_ylabel("Mel bin")
    ax.set_xlabel("Frame")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    return fig


def build_model(cfg: dict, n_phonemes: int):
    m, e, mel = cfg["model"], cfg["excitation"], cfg["mel"]
    common = dict(
        hidden=m.get("hidden", 256), mel_bins=mel["n_mels"],
        mel_vmin=m.get("mel_vmin", -11.5), mel_vmax=m.get("mel_vmax", 2.0),
        backbone_ch=m.get("backbone_ch", 256), n_cycles=m.get("n_cycles", 3),
        dilation_schedule=m.get("dilation_schedule", "pow2_15"),
        n_styles=m.get("n_styles", 0),
        flow_loss=m.get("flow_loss", "l2"), use_uv=m.get("use_uv", True),
    )
    harm = dict(
        n_harm=e.get("n_harm", 50),
        noise_ratio=e.get("noise_ratio", 0.05), exc_scale=e.get("exc_scale", 0.15),
        harm_decay=e.get("harm_decay", 1.0), exc_hop=mel["hop"],
    )
    spk_dim = int(m.get("spk_dim", 0))
    if m.get("arch") == "svc":
        model = HarmonicSVCModel(
            **common, **harm, content_dim=int(m["content_dim"]),
            content_layers=int(m.get("content_layers", 2)),
            content_dropout=float(m.get("content_dropout", 0.1)),
            n_speakers=int(m.get("n_speakers", 0)), spk_dim=spk_dim,
        )
        arch = "harmonic_svc"
    elif spk_dim > 0:
        model = HarmonicAcousticModelMultiSpk(
            **common, **harm, n_phonemes=n_phonemes,
            n_speakers=int(m["n_speakers"]), spk_dim=spk_dim)
        arch = "harmonic_multispk"
    else:
        model = HarmonicAcousticModel(
            **common, **harm, n_phonemes=n_phonemes,
            n_speakers=int(m.get("n_speakers", 0)))
        arch = "harmonic"
    return model, arch


def resolve_vocab(phonemes_path, data_dirs):
    """Training phoneme vocab. Explicit --phonemes wins; else the phoneme list stored in the
    shard metadata (so it always matches how the data was preprocessed); else the bundled
    Japanese default. The chosen list is saved into the checkpoint (export/infer reuse it)."""
    if phonemes_path:
        return Vocab.load(phonemes_path)
    for d in data_dirs:
        mp = os.path.join(d, "metadata.json")
        if os.path.exists(mp):
            meta = json.load(open(mp, encoding="utf-8"))
            if meta.get("phonemes"):
                return Vocab(meta["phonemes"])
    return Vocab.load()                              # bundled Japanese


def ckpt_config(cfg: dict, arch: str, n_phonemes: int, phonemes: list | None = None) -> dict:
    m, e, mel, tr = cfg["model"], cfg["excitation"], cfg["mel"], cfg["train"]
    out = {
        "arch": arch, "n_phonemes": n_phonemes, "phonemes": phonemes,
        "hidden": m.get("hidden", 256), "mel_bins": mel["n_mels"],
        "mel_vmin": m.get("mel_vmin", -11.5), "mel_vmax": m.get("mel_vmax", 2.0),
        "backbone_ch": m.get("backbone_ch", 256), "n_cycles": m.get("n_cycles", 3),
        "dilation_schedule": m.get("dilation_schedule", "pow2_15"),
        "n_speakers": int(m.get("n_speakers", 0)), "n_styles": m.get("n_styles", 0),
        "spk_dim": int(m.get("spk_dim", 0)),
        "flow_loss": m.get("flow_loss", "l2"), "use_uv": m.get("use_uv", True),
        "n_harm": e.get("n_harm", 50), "noise_ratio": e.get("noise_ratio", 0.05),
        "exc_scale": e.get("exc_scale", 0.15), "harm_decay": e.get("harm_decay", 1.0),
        "exc_hop": mel["hop"], "hop": mel["hop"], "sample_rate": mel["sr"],
        "recon_weight": tr.get("recon_weight", 1.0), "infer_steps": tr.get("num_steps", 10),
    }
    if arch == "harmonic_svc":
        out.update(
            content_dim=int(m["content_dim"]),
            content_layers=int(m.get("content_layers", 2)),
            content_dropout=float(m.get("content_dropout", 0.1)),
        )
    return out


def _forward(model, b, device, recon_weight):
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
    if isinstance(model, HarmonicSVCModel):
        out = model(
            b["content"], b["f0_logf0"], b["uv"], b["loudness"], b["target_mel"],
            content_mask=b["content_mask"], spk_id=b["spk_id"], style_id=b["style_id"],
            frame_mask=b["frame_mask"], harm_wave=b.get("harm_wave"))
    else:
        out = model(
            b["ph_ids"], b["ph_durs"], b["f0_logf0"], b["uv"], b["target_mel"],
            padding_mask=b["ph_mask"], spk_id=b["spk_id"], style_id=b["style_id"],
            frame_mask=b["frame_mask"], harm_wave=b.get("harm_wave"))
    loss = out["flow"]
    if recon_weight > 0 and "recon" in out:
        loss = loss + recon_weight * out["recon"]
    return loss, out


def _forward_flow_gan(model, b, flow_loss_type: str):
    """_forward の GAN 版: flow 損失 + GAN ペア素材を単一 velocity_fn forward で返す。
    数式は mel_dilated_rectified_flow.compute_loss と同一（同じ mask / t 分布）。
    ★x0 は励起（randn ではない）。RNG 順（_encode dropout → 励起 randn → t rand）も compute_loss と一致。"""
    flow = model.flow
    if isinstance(model, HarmonicSVCModel):
        cond = model._encode(
            b["content"], b["f0_logf0"], b["uv"], b["loudness"], b["content_mask"],
            max_frames=b["target_mel"].shape[2], spk_id=b["spk_id"],
            style_id=(b["style_id"] if getattr(model, "style_emb", None) is not None else None))
    else:
        cond = model._encode(
            b["ph_ids"], b["ph_durs"], b["f0_logf0"], b["uv"], b["ph_mask"],
            max_frames=b["target_mel"].shape[2], spk_id=b["spk_id"],
            style_id=(b["style_id"] if getattr(model, "style_emb", None) is not None else None))
    x1 = flow._norm(b["target_mel"])
    x0 = model._excitation_x0(b["f0_logf0"], b["uv"], harm_wave=b.get("harm_wave"))   # ★励起
    t = torch.rand(x1.shape[0], device=x1.device)
    x_t = x0 + t[:, None, None] * (x1 - x0)
    v = flow.velocity_fn(x_t, t, cond)
    diff = v - (x1 - x0)
    mask = (~b["frame_mask"]) if "frame_mask" in b and b["frame_mask"] is not None else None
    if mask is None:                                   # compute_loss と同一の縮約
        loss = diff.abs().mean() if flow_loss_type == "l1" else diff.pow(2).mean()
    else:
        e = diff.abs() if flow_loss_type == "l1" else diff.pow(2)
        m = mask[:, None, :].to(e.dtype)
        loss = (e * m).sum() / m.sum().clamp(min=1.0) / e.shape[1]
    x1_pred_n = x_t + (1.0 - t[:, None, None]) * v     # 正規化域の 1-step clean 予測
    vmask = mask[:, None, :] if mask is not None else torch.ones_like(x1[:, :1, :], dtype=torch.bool)
    mel_l1 = ((x1_pred_n - x1).abs() * vmask).sum() / (vmask.sum().clamp(min=1) * x1.shape[1])  # = _recon_loss
    return {"flow": loss, "mel_l1": mel_l1, "cond": cond, "t": t, "x_t": x_t,
            "x1_n": x1, "x1_pred_n": x1_pred_n, "x1_pred": flow._denorm(x1_pred_n)}


def _adaptive_adv_weight(recon_ref, gan_term, last_w, cap: float):
    """VQGAN 式: 生成器最終層重みでの ∥∇recon∥/∥∇GAN∥ を毎 step 計測し、GAN 勾配が
    再構成の錨を上回らないよう自動スケール（cap で上限）。GAN 暴走への構造的対処。"""
    g_r = torch.autograd.grad(recon_ref, last_w, retain_graph=True)[0].norm()
    g_a = torch.autograd.grad(gan_term, last_w, retain_graph=True)[0].norm()
    return (g_r / (g_a + 1e-8)).clamp(max=cap).detach()


def _rand_window(tensors, win: int):
    """時間軸で長さ win の同一ランダム窓を切る（全サンプル共有）。win<=0 or T<=win なら無変更。
    tensors: [B, C, T] のリスト。GAN-TTS 流に「全体を見せない」ための短窓。"""
    T = tensors[0].shape[-1]
    if win <= 0 or T <= win:
        return tensors
    s = int(torch.randint(0, T - win + 1, (1,)))
    return [x[..., s:s + win] for x in tensors]


def main():
    ap = argparse.ArgumentParser(description="Train the LeapSinger acoustic model with a light GAN.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--out_root", default="log")
    ap.add_argument("--max_updates", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--phonemes", default=None,
                    help="phoneme list file (default: shard metadata / bundled Japanese)")
    ap.add_argument("--init_from", default=None,
                    help="finetune 起点の flow ckpt（G 重みのみ読み込み）")
    ap.add_argument("--finetune", action="store_true",
                    help="init_from の G 重みだけ読み、step/optimizer をリセット・D は新規（GAN post-filter）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg.setdefault("mel", {}); cfg.setdefault("model", {})
    cfg.setdefault("excitation", {}); cfg.setdefault("train", {})
    cfg["mel"] = MelSpec.from_dict(cfg["mel"]).to_dict()
    tr = cfg["train"]
    g = cfg.get("gan", {})
    if args.max_updates:
        tr["max_updates"] = args.max_updates
    device = torch.device(args.device)

    # ── data（データ供給の足場）────────────────────────────
    dcfg = cfg.get("data", {})
    spk_map = dict(dcfg.get("spk_map") or {})
    style_map = dict(dcfg.get("style_map") or {})
    for d in args.data_dirs:
        dbname = os.path.basename(os.path.normpath(d))
        rp = os.path.join("configs", "recipes", f"{dbname}.yaml")
        if os.path.exists(rp):
            rr = yaml.safe_load(open(rp, encoding="utf-8"))
            spk_map.setdefault(dbname, int(rr.get("spk_id", 0)))
            style_map.setdefault(dbname, int(rr.get("style_id", 0)))
    print(f"[spk_map] {spk_map}\n[style_map] {style_map}")
    is_svc = cfg["model"].get("arch") == "svc"
    if is_svc and tr.get("pitch_aug", False):
        raise SystemExit("SVC feature training does not support online pitch_aug; augment before extraction")
    if is_svc:
        train_ds = SVCFeatureDataset(
            args.data_dirs, "train", eval_songs=dcfg.get("eval_songs", 2),
            min_sec=dcfg.get("min_sec", 0.3), spk_map=spk_map, style_map=style_map)
    else:
        train_ds = LeapSingerDataset(
            args.data_dirs, "train", eval_songs=dcfg.get("eval_songs", 2),
            min_sec=dcfg.get("min_sec", 0.3), pitch_aug=tr.get("pitch_aug", False),
            silence=dcfg.get("silence", True),
            silence_fade_sec=dcfg.get("silence_fade_sec", 0.05),
            spk_map=spk_map, style_map=style_map)
    if "n_speakers" not in cfg["model"]:
        cfg["model"]["n_speakers"] = (max(spk_map.values()) + 1) if spk_map else 1

    if is_svc:
        configured_dim = int(cfg["model"].get("content_dim", train_ds.content_dim))
        if configured_dim != train_ds.content_dim:
            raise SystemExit(
                f"model.content_dim={configured_dim} but dataset content_dim={train_ds.content_dim}"
            )
        cfg["model"]["content_dim"] = configured_dim
        vocab = None
        n_phonemes = 0
    else:
        vocab = resolve_vocab(args.phonemes, args.data_dirs)
        n_phonemes = vocab.n_phonemes
        print(f"[vocab] {vocab.n_phonemes} phonemes")
    model, arch = build_model(cfg, n_phonemes)
    model = model.to(device)
    print(f"model {arch}  {sum(p.numel() for p in model.parameters())/1e6:.2f}M params  "
          f"n_speakers={cfg['model']['n_speakers']}")

    if not tr.get("pitch_aug", False) and hasattr(train_ds, "warm_harm_cache"):
        train_ds.warm_harm_cache(n_harm=model.n_harm, harm_decay=model.harm_decay,
                                 exc_hop=model.exc_hop, use_uv=model.use_uv, device=device)

    _balance_by = tr.get("balance_by") or ("speaker" if tr.get("balance_speakers", False) else None)
    _weights = None
    if _balance_by:
        _keys = {"speaker": train_ds.speaker_ids, "db": train_ds.db_ids}.get(_balance_by)
        if _keys is None:
            raise SystemExit(f"balance_by='{_balance_by}' は不正（speaker | db）")
        _cnt = collections.Counter(_keys)
        _weights = [1.0 / _cnt[k] for k in _keys]
        print(f"[dataset] balanced sampling by '{_balance_by}'")
    sampler = FrameBasedBatchSampler(train_ds.frame_counts, tr.get("max_batch_frames", 60000),
                                     tr.get("max_batch_size", 16), shuffle=True, weights=_weights)
    _nw = tr.get("num_workers", 2)
    _dl_kw = dict(pin_memory=True)
    if _nw > 0:
        _dl_kw.update(persistent_workers=True, prefetch_factor=4)
    collate_fn = svc_collate_fn if is_svc else acoustic_collate_fn
    loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_fn,
                        num_workers=_nw, **_dl_kw)

    opt_name = tr.get("optimizer", "radam")
    Opt = torch.optim.RAdam if opt_name == "radam" else torch.optim.AdamW
    opt = Opt(model.parameters(), lr=tr.get("lr", 1e-4))

    # ── 判別器 + D optimizer（軽量 1D JCU、TTUR）───────────────────────────────
    gan_enabled = bool(g.get("enabled", True))
    n_cond = int(cfg["model"]["n_speakers"]) if g.get("cond", "speaker") == "speaker" else 0
    d_type = g.get("d_type", "jcu")                    # jcu(1D, DiffGAN-TTS原典) | mel2d(2D patch)
    _DISC = {"jcu": JCUMelDiscriminator, "mel2d": Mel2DDiscriminator}
    if d_type not in _DISC:
        raise SystemExit(f"gan.d_type='{d_type}' は不正（jcu | mel2d）")
    _dkw = dict(mel_bins=cfg["mel"]["n_mels"], n_cond=n_cond)
    if d_type == "mel2d":
        _dkw["spectral_norm"] = bool(g.get("d_spectral_norm", True))   # Lipschitz拘束で発散抑制
    disc = _DISC[d_type](**_dkw).to(device)
    d_optimizer = torch.optim.Adam(disc.parameters(), lr=float(g.get("d_lr", 2.0e-4)),
                                   betas=tuple(g.get("d_betas", [0.5, 0.9])))
    last_w = model.flow.velocity_fn.out[-1].weight     # VQGAN 適応重み用（生成器の最終層）
    print(f"[gan] enabled={gan_enabled}  d_type={d_type}  "
          f"disc {sum(p.numel() for p in disc.parameters())/1e6:.2f}M  "
          f"d_lr={g.get('d_lr',2e-4)}  window={g.get('window',64)}  cond={g.get('cond','speaker')}  "
          f"fm={'dynamic' if g.get('fm_dynamic',True) else g.get('fm_weight',10.0)}")

    out_dir = Path(args.out_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_out = ckpt_config(
        cfg, arch, n_phonemes, (None if vocab is None else vocab.phonemes))
    recon_weight = tr.get("recon_weight", 1.0)
    max_updates = tr.get("max_updates", 60000)
    save_interval = tr.get("save_interval", 5000)
    log_interval = tr.get("log_interval", 50)
    eval_interval = tr.get("eval_interval", 2000)
    eval_n = int(tr.get("eval_items", 4))
    grad_clip = tr.get("grad_clip", 5.0)

    def save(step):
        p = out_dir / f"ckpt_{step:06d}.pt"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "disc": disc.state_dict(), "d_optimizer": d_optimizer.state_dict(),
                    "step": step, "config": cfg_out,
                    "n_speakers": cfg["model"]["n_speakers"]}, p)
        tqdm.write(f"  saved {p}")

    # ── TensorBoard eval──────────────────────────────
    writer = SummaryWriter(str(out_dir))
    if is_svc:
        eval_ds = SVCFeatureDataset(
            args.data_dirs, "eval", eval_songs=dcfg.get("eval_songs", 2),
            min_sec=dcfg.get("min_sec", 0.3), spk_map=spk_map, style_map=style_map)
    else:
        eval_ds = LeapSingerDataset(
            args.data_dirs, "eval", eval_songs=dcfg.get("eval_songs", 2),
            min_sec=dcfg.get("min_sec", 0.3),
            silence=dcfg.get("silence", True),
            silence_fade_sec=dcfg.get("silence_fade_sec", 0.05),
            spk_map=spk_map, style_map=style_map)
    _picks: dict = {}
    for _idx, (_sh, _nm, _db) in enumerate(eval_ds.files):
        _s = int(spk_map.get(os.path.basename(os.path.normpath(_db)), 0))
        _picks.setdefault(_s, [])
        if len(_picks[_s]) < eval_n:
            _picks[_s].append(_idx)
    eval_samples = [(s, k, eval_ds[idx]) for s in sorted(_picks) for k, idx in enumerate(_picks[s])]
    sr = int(cfg["mel"]["sr"])
    voc, vc = None, tr.get("vocoder")
    if vc:
        try:
            voc = load_vocoder(vc); print(f"[eval] vocoder loaded ({vc})")
        except Exception as e:                        # noqa: BLE001
            print(f"[eval] vocoder load failed ({e}); mel only")
    _gt_done: set = set()
    mvmin = float(cfg["model"].get("mel_vmin", -11.5))
    mvmax = float(cfg["model"].get("mel_vmax", 2.0))

    def log_eval(step):
        model.eval()
        with torch.no_grad():
            torch.manual_seed(1234)
            eb = collate_fn([it for _, _, it in eval_samples])
            eloss, eout = _forward(model, eb, device, recon_weight)
            writer.add_scalar("eval/loss", eloss.item(), step)
            writer.add_scalar("eval/flow", eout["flow"].item(), step)
            writer.add_scalar("eval/recon", float(eout.get("recon", 0)), step)
            _vsum = 0.0
            for s, k, it in eval_samples:
                tag = f"spk{s}/s{k}"
                pred = ((infer_svc_mel if is_svc else infer_mel)(
                    model, it, num_steps=tr.get("num_steps", 10), device=str(device)))
                writer.add_figure(f"{tag}_pred_mel",
                                  _mel_fig(pred, f"{tag} pred @ {step}", mvmin, mvmax), step)
                _vsum += laplacian_var_ratio(torch.as_tensor(pred)[None],
                                             torch.as_tensor(it["target_mel"])[None])
                if (s, k) not in _gt_done:
                    writer.add_figure(f"{tag}_gt_mel",
                                      _mel_fig(it["target_mel"], f"{tag} GT", mvmin, mvmax), step)
                if voc is not None:
                    vuv = it["uv"] if getattr(model, "use_uv", True) else np.ones_like(it["uv"])
                    writer.add_audio(f"{tag}_pred", mel_to_wav(voc, pred, it["f0_logf0"], vuv),
                                     step, sample_rate=sr)
                    if (s, k) not in _gt_done:
                        writer.add_audio(f"{tag}_gt",
                                         mel_to_wav(voc, it["target_mel"], it["f0_logf0"], vuv),
                                         step, sample_rate=sr)
                _gt_done.add((s, k))
            writer.add_scalar("eval/varL", _vsum / max(1, len(eval_samples)), step)  # GT比(1.0=同等)
        model.train()

    # ── resume / finetune ──────────────────────────────────────────────────────
    step = 0
    _ckpts = sorted(out_dir.glob("ckpt_*.pt"))
    if _ckpts:                                        # 同 run を継続（G+opt+D+d_opt+step）
        ck = torch.load(_ckpts[-1], map_location=device)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            try: opt.load_state_dict(ck["optimizer"])
            except Exception as e: print(f"[resume] optimizer skipped ({e})")  # noqa: BLE001
        if "disc" in ck:                              # arch不一致(D差し替え)なら新規Dにフォールバック
            try: disc.load_state_dict(ck["disc"])
            except Exception as e: print(f"[resume] disc arch mismatch -> fresh D ({e})")  # noqa: BLE001
        if "d_optimizer" in ck:
            try: d_optimizer.load_state_dict(ck["d_optimizer"])
            except Exception as e: print(f"[resume] d_optimizer skipped ({e})")  # noqa: BLE001
        step = int(ck.get("step", 0))
        for pg in opt.param_groups: pg["lr"] = tr.get("lr", 1e-4)
        print(f"[resume] {_ckpts[-1].name} -> step {step}")
    elif args.init_from and args.finetune:            # GAN post-filter: G 重みのみ・D 新規・step 0
        ck = torch.load(args.init_from, map_location=device)
        model.load_state_dict(ck["model"])
        print(f"[finetune] G weights <- {args.init_from} (D fresh, step 0)")

    d_ema = None
    gan_start = int(g.get("gan_start_step", 0))
    d_warmup = int(g.get("d_warmup_steps", 1000))
    gan_ramp = int(g.get("gan_ramp_steps", 1000))
    d_gate_floor = float(g.get("d_gate_floor", 0.3))
    adv_weight = float(g.get("adv_weight", 1.0))
    fm_dynamic = bool(g.get("fm_dynamic", True))
    fm_weight = float(g.get("fm_weight", 10.0))
    window = int(g.get("window", 64))
    grad_cap = float(g.get("grad_cap", 1.0))
    adaptive_adv = bool(g.get("adaptive_adv", True))
    gan_strength = float(g.get("gan_strength", 1.0))   # 適応重みへの倍率。収束済みfinetuneは∇recon極小
    #  なので 1.0(=パリティ)だとGANが弱すぎる。>1 で「GAN勾配=gan_strength×∇recon」に自己較正
    #  （reconに紐づくので暴走しない）。無制御にしたい時のみ adaptive_adv:false（発散注意）。

    print(f"training -> {out_dir}  (max_updates={max_updates}, start {step}, recon_weight={recon_weight})")
    model.train()
    pbar = tqdm(total=max_updates, initial=step, desc=args.run_name, unit="step", dynamic_ncols=True)
    _done = False
    while step < max_updates:
        for b in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            losses = _forward_flow_gan(model, b, tr.get("flow_loss", cfg["model"].get("flow_loss", "l2")))
            g_total = losses["flow"]
            d_loss = None; d_logits = {}; g_adv = g_fm = None; fm_w = 0.0; w_ad = 1.0

            gan_on = gan_enabled and step >= gan_start
            _g = step - gan_start - d_warmup
            d_only = gan_on and _g < 0
            ramp = 0.0 if (not gan_on or d_only) else min(1.0, (_g + 1) / max(1, gan_ramp))

            if gan_on:
                cond_id = b["spk_id"] if n_cond > 0 else None
                x_t, t = losses["x_t"], losses["t"]
                real_j = losses["x1_n"]
                fake_j = losses["x1_pred_n"].clamp(-1.0, 1.0)
                x_t_w, real_w, fake_w = _rand_window([x_t, real_j, fake_j], window)
                # D step（fake はヘルパー内で detach）
                d_optimizer.zero_grad()
                d_loss, d_logits = d_loss_jcu(disc, x_t_w, real_w, fake_w, t, cond_id)
                gate_open = (d_gate_floor <= 0.0 or d_ema is None or d_ema >= d_gate_floor)
                if gate_open:
                    d_loss.backward()
                    nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                    d_optimizer.step()
                d_ema = d_loss.item() if d_ema is None else 0.99 * d_ema + 0.01 * d_loss.item()
                # G step（warmup 後・ランプ付き）
                if not d_only:
                    g_adv, g_fm = g_adv_fm_jcu(disc, x_t_w, real_w, fake_w, t, cond_id)
                    fm_w = (losses["mel_l1"].item() / max(g_fm.item(), 1e-8)) if fm_dynamic else fm_weight
                    gan_term = adv_weight * g_adv + fm_w * g_fm
                    recon_ref = losses["flow"] + recon_weight * losses["mel_l1"]
                    w_ad = gan_strength * (_adaptive_adv_weight(recon_ref, gan_term, last_w, grad_cap)
                                           if adaptive_adv else 1.0)
                    g_total = losses["flow"] + ramp * w_ad * gan_term

            g_total = g_total + recon_weight * losses["mel_l1"]    # 土台の recon（常時ON）
            opt.zero_grad()
            g_total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            step += 1
            pbar.update(1)

            if step % log_interval == 0:
                pbar.set_postfix(flow=f"{losses['flow'].item():.4f}",
                                 recon=f"{losses['mel_l1'].item():.4f}",
                                 d=f"{d_loss.item():.3f}" if d_loss is not None else "-")
                writer.add_scalar("train/flow", losses["flow"].item(), step)
                writer.add_scalar("train/recon", losses["mel_l1"].item(), step)
                writer.add_scalar("train/varL_ratio",
                                  laplacian_var_ratio(losses["x1_pred"], model.flow._denorm(losses["x1_n"])), step)
                if d_loss is not None:
                    writer.add_scalar("train/d_loss", d_loss.item(), step)
                    writer.add_scalar("train/d_real_logit", d_logits["d_real_logit"], step)
                    writer.add_scalar("train/d_fake_logit", d_logits["d_fake_logit"], step)
                    writer.add_scalar("train/gan_ramp", ramp, step)
                if g_adv is not None:
                    writer.add_scalar("train/adv", g_adv.item(), step)
                    writer.add_scalar("train/fm", g_fm.item(), step)
                    writer.add_scalar("train/fm_w", fm_w, step)
                    writer.add_scalar("train/adaptive_w", float(w_ad), step)
            if step % eval_interval == 0:
                log_eval(step)
            if step % save_interval == 0:
                save(step)
            if step >= max_updates:
                _done = True; break
        if _done:
            break
    pbar.close()
    save(step)
    log_eval(step)
    writer.close()
    print("done.")


if __name__ == "__main__":
    main()
