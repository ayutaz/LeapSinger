"""Re-synthesis: (phonemes, durations, F0, uv, speaker, style) -> mel -> waveform.

Rebuilds the acoustic model straight from a checkpoint's config (no app framework), runs
the harmonic flow, then renders the mel with the bundled NHVSing vocoder ONNX
(checkpoints/nhv_v3.onnx) via onnxruntime — self-contained, no external dependency.

    from infer import load_acoustic, infer_mel, load_vocoder, mel_to_wav
    model, cfg = load_acoustic("ckpt.pt")
    mel = infer_mel(model, item, num_steps=cfg["infer_steps"])
    voc = load_vocoder("checkpoints/nhv_v3.onnx")
    wav = mel_to_wav(voc, mel, item["f0_logf0"], item["uv"])
"""
from __future__ import annotations

import os

import numpy as np
import torch

from leapsinger.models.acoustic import HarmonicAcousticModel, HarmonicAcousticModelMultiSpk
from leapsinger.models.svc import HarmonicSVCModel
from tools.svc_defaults import SVC_NUM_STEPS


def _build_from_config(cfg: dict, device):
    common = dict(
        hidden=cfg.get("hidden", 256),
        mel_bins=cfg["mel_bins"], mel_vmin=cfg.get("mel_vmin", -11.5),
        mel_vmax=cfg.get("mel_vmax", 2.0), backbone_ch=cfg.get("backbone_ch", 256),
        n_cycles=cfg.get("n_cycles", 3), dilation_schedule=cfg.get("dilation_schedule", "pow2_15"),
        n_styles=cfg.get("n_styles", 0), flow_loss=cfg.get("flow_loss", "l2"),
        use_uv=cfg.get("use_uv", True),
    )
    harm = dict(
        n_harm=cfg.get("n_harm", 50),
        noise_ratio=cfg.get("noise_ratio", 0.05), exc_scale=cfg.get("exc_scale", 0.15),
        harm_decay=cfg.get("harm_decay", 1.0), exc_hop=cfg.get("exc_hop", cfg.get("hop", 256)),
    )
    if cfg.get("arch") == "harmonic_svc":
        model = HarmonicSVCModel(
            **common, **harm, content_dim=cfg["content_dim"],
            content_layers=cfg.get("content_layers", 2),
            content_dropout=cfg.get("content_dropout", 0.1),
            n_speakers=cfg.get("n_speakers", 0), spk_dim=cfg.get("spk_dim", 0),
        )
    elif cfg.get("arch") == "harmonic_multispk":
        model = HarmonicAcousticModelMultiSpk(
            **common, **harm, n_phonemes=cfg["n_phonemes"],
            n_speakers=cfg["n_speakers"], spk_dim=cfg["spk_dim"])
    else:
        model = HarmonicAcousticModel(
            **common, **harm, n_phonemes=cfg["n_phonemes"],
            n_speakers=cfg.get("n_speakers", 0))
    return model.to(device)


def load_acoustic(ckpt_path: str, device: str = "cpu"):
    """Load a checkpoint, rebuild the model from its stored config. Returns (model, config)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = _build_from_config(cfg, torch.device(device))
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, cfg


@torch.no_grad()
def infer_mel(model, item: dict, num_steps: int = 10, device: str = "cpu", seed: int = 0):
    """Run the acoustic model on one item -> mel [n_mels, T]. `item` needs at least
    ph_ids, ph_durs, f0_logf0, uv (T frames), and optionally spk_id/style_id."""
    torch.manual_seed(seed)
    dev = torch.device(device)

    def t(a, dt):
        return torch.as_tensor(np.asarray(a), dtype=dt).unsqueeze(0).to(dev)

    T = int(len(item["f0_logf0"]))
    kw = dict(
        phoneme_ids=t(item["ph_ids"], torch.long),
        ph_durations=t(item["ph_durs"], torch.long),
        f0_logf0=t(item["f0_logf0"], torch.float32),
        uv=t(item["uv"], torch.float32),
        n_frames=T, num_steps=num_steps, algorithm="euler",
    )
    if hasattr(model, "spk_bank") or getattr(model, "spk_emb", None) is not None:
        kw["spk_id"] = torch.tensor([int(item.get("spk_id", 0))], device=dev)
    if getattr(model, "style_emb", None) is not None:
        kw["style_id"] = torch.tensor([int(item.get("style_id", 0))], device=dev)
    mel = model.infer(**kw)
    return mel[0].cpu().numpy()


@torch.no_grad()
def infer_svc_mel(model, item: dict, num_steps: int = SVC_NUM_STEPS,
                  device: str = "cpu", seed: int = 0):
    """Run a HarmonicSVCModel on one frame-aligned feature item.

    Required keys are ``content`` [T,C], ``f0_logf0`` [T], ``uv`` [T], and
    ``loudness`` [T]. Content encoder and pitch extractor execution deliberately
    live outside this function so the same contract can serve offline training
    and a future streaming frontend.

    ``num_steps`` defaults to :data:`tools.svc_defaults.SVC_NUM_STEPS` (16), chosen by a
    pre-registered rule over a 20-clip sweep: it recovers 58.9% -> 68.6% of the way toward
    the target speaker for -0.019 content cosine, and the flow costs 2.1x rather than 16x
    because most of the time is call overhead. The SVS path (:func:`infer_mel`) is
    unchanged -- that route was not measured here.
    """
    if not isinstance(model, HarmonicSVCModel):
        raise TypeError("infer_svc_mel requires HarmonicSVCModel")
    torch.manual_seed(seed)
    dev = torch.device(device)

    content = np.asarray(item["content"], dtype=np.float32)
    if content.ndim != 2:
        raise ValueError(f"item['content'] must be [T,C], got {content.shape}")
    frames = content.shape[0]

    def frame(name):
        value = np.asarray(item[name], dtype=np.float32).reshape(-1)
        if len(value) != frames:
            raise ValueError(f"item['{name}'] must have {frames} frames, got {len(value)}")
        return torch.as_tensor(value, device=dev)[None]

    kw = dict(
        content_features=torch.as_tensor(content, device=dev)[None],
        f0_logf0=frame("f0_logf0"),
        uv=frame("uv"),
        loudness=frame("loudness"),
        n_frames=frames,
        num_steps=num_steps,
        algorithm="euler",
    )
    if model.spk_n > 0:
        kw["spk_id"] = torch.tensor([int(item.get("spk_id", 0))], device=dev)
    if model.style_emb is not None:
        kw["style_id"] = torch.tensor([int(item.get("style_id", 0))], device=dev)
    mel = model.infer(**kw)
    return mel[0].cpu().numpy()


@torch.no_grad()
def infer_mel_uvfree(model, tokens, durations, f0_hz, *, num_steps: int = 10,
                         spk_embed=None, device: str = "cpu", seed: int = 0):
    """DiffSinger-contract adapter: drive a **uv-free** model with ONLY the inputs OpenUTAU's
    DiffSinger acoustic runtime provides — `tokens`, `durations`, and a continuous (gap-less)
    `f0_hz` — and return mel [n_mels, T]. No uv, no variance (uv is the ignored arg for a
    use_uv=False model). This is the PyTorch analog of the eventual ONNX wrapper.

        tokens     [Np] int   phoneme ids
        durations  [Np] int   per-phoneme FRAME counts (sum = T)
        f0_hz      [T]  float  interpolated (never-zero) F0 in Hz — DiffSinger's pitch curve
        spk_embed  optional [H] speaker embedding vector (host-mixed); added to the condition
                   for a multi-speaker model in place of the discrete spk_bank lookup.
    """
    assert getattr(model, "use_uv", True) is False, \
        "infer_mel_uvfree requires a uv-free model (use_uv=False)"
    torch.manual_seed(seed)
    dev = torch.device(device)
    T = int(sum(int(d) for d in durations))
    tok = torch.as_tensor(np.asarray(tokens), dtype=torch.long, device=dev)[None]
    dur = torch.as_tensor(np.asarray(durations), dtype=torch.long, device=dev)[None]
    f0 = torch.as_tensor(np.asarray(f0_hz)[:T], dtype=torch.float32, device=dev)[None]
    f0_logf0 = torch.log2(f0.clamp(min=1.0))
    uv = torch.ones(1, T, device=dev)                       # ignored (use_uv=False); kept for the signature
    kw = dict(phoneme_ids=tok, ph_durations=dur, f0_logf0=f0_logf0, uv=uv,
              n_frames=T, num_steps=num_steps, algorithm="euler")
    if spk_embed is not None:                               # host-mixed speaker vector (OpenUTAU spk_embed)
        v = torch.as_tensor(np.asarray(spk_embed), dtype=torch.float32, device=dev).view(-1)
        kw["spk_id"] = None
        cond_add = v[None, :, None]                         # [1,H,1] broadcast over T — matches _spk_add
        # inject by temporarily overriding _spk_add so the provided vector is added, not a bank lookup
        orig = getattr(model, "_spk_add", None)
        if orig is not None:
            model._spk_add = lambda cond, _sid, _v=cond_add: cond + _v
            try:
                mel = model.infer(**kw)
            finally:
                model._spk_add = orig
            return mel[0].cpu().numpy()
    mel = model.infer(**kw)
    return mel[0].cpu().numpy()


def build_item(phonemes, ph_dur_frames, f0_hz, *, spk_id=0, style_id=0,
               vuv_mode="f0", vuv_dict=None, vocab=None):
    """Assemble one model input from the raw duration + F0 that the (separate) duration and
    F0 models produce — no shard needed.

      phonemes       : list[str] canonical phonemes (see preprocess.vocab.CANONICAL_PHONEMES)
      ph_dur_frames  : list[int] frames per phoneme (at the model's hop); sum = T
      f0_hz          : array [T] Hz per frame; <=1 / nan / 0 marks unvoiced
      vuv_mode       : 'f0'      -> voiced where f0>1 (from the F0 model)               [default]
                       'all'     -> every frame voiced (gap-less interp-F0, no unvoiced gate)
                       'phoneme' -> voicing forced per-phoneme from `vuv_dict`
      vuv_dict       : {phoneme: bool voiced} — required for vuv_mode='phoneme'
    """
    from preprocess.vocab import DEFAULT_VOCAB
    p2i = (vocab or DEFAULT_VOCAB).phoneme2id      # custom-vocab: pass Vocab(ckpt config['phonemes'])

    ph_ids = np.asarray([p2i[p] for p in phonemes], np.int64)
    ph_durs = np.asarray(ph_dur_frames, np.int64)
    T = int(ph_durs.sum())
    f0 = np.asarray(f0_hz, np.float32).reshape(-1)[:T]
    if len(f0) < T:
        f0 = np.pad(f0, (0, T - len(f0)))

    voiced = np.isfinite(f0) & (f0 > 1.0)
    if vuv_mode == "all":
        voiced[:] = True
    elif vuv_mode == "phoneme":
        if not vuv_dict:
            raise ValueError("vuv_mode='phoneme' needs vuv_dict")
        pos = 0
        # 長さは一致する想定（音素 1 つに duration 1 つ）。linter 導入で実行時の挙動を
        # 変えないよう strict=False のまま明示する。厳格化は別途判断する。
        for p, d in zip(phonemes, ph_durs, strict=False):
            if p in vuv_dict:
                voiced[pos:pos + d] = bool(vuv_dict[p])
            pos += d

    logf0 = np.zeros(T, np.float32)                       # log2-F0, unvoiced gaps interpolated
    src = np.isfinite(f0) & (f0 > 1.0)                    # real pitch samples to interp from
    if src.any():
        idx = np.arange(T)
        logf0 = np.interp(idx, idx[src], np.log2(f0[src])).astype(np.float32)
    return dict(
        ph_ids=ph_ids, ph_durs=ph_durs, f0_logf0=logf0, uv=voiced.astype(np.float32),
        spk_id=int(spk_id), style_id=int(style_id))


class _OnnxVocoder:
    """Holder around the bundled NHVSing vocoder ONNX session (self-contained, no external NHVSing)."""
    __slots__ = ("session", "sr")


def load_vocoder(onnx_path: str, sr: int = 44100):
    """Load the bundled NHVSing vocoder ONNX (mel -> waveform) via onnxruntime. Self-contained: no
    external NHVSing checkout. ONNX I/O: mel[B,T,128], f0[B,1,T] (Hz), uv[B,1,T] (1 = unvoiced)
    -> waveform[B,1,256*T] (44.1 kHz, hop 256). `onnx_path` = the bundled checkpoints/nhv_v3.onnx."""
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.abspath(os.path.expanduser(onnx_path)),
                                providers=["CPUExecutionProvider"])
    v = _OnnxVocoder()
    v.session, v.sr = sess, int(sr)
    return v


def mel_to_wav(vocoder, mel: np.ndarray, f0_logf0: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """mel [n_mels, T] + F0/uv -> waveform via the bundled NHVSing ONNX vocoder.
    mel -> [1,T,128], f0 (Hz = 2**f0_logf0) -> [1,1,T], uv -> [1,1,T] in the vocoder's convention
    (1 = unvoiced), i.e. `1 - voiced`. Our F0 is already gap-less, so no interpolation is needed."""
    mel_t = np.ascontiguousarray(np.asarray(mel).T[None], dtype=np.float32)            # [1,T,128]
    T = mel_t.shape[1]

    def _fit(a):
        a = np.asarray(a, dtype=np.float32).reshape(-1)[:T]
        return np.pad(a, (0, max(0, T - len(a))))

    f0hz = (2.0 ** _fit(f0_logf0))[None, None].astype(np.float32)                      # [1,1,T]
    uv_in = (1.0 - _fit(uv))[None, None].astype(np.float32)                            # [1,1,T] (1=unvoiced)
    wav = np.asarray(vocoder.session.run(
        ["waveform"], {"mel": mel_t, "f0": f0hz, "uv": uv_in})[0]).reshape(-1)
    peak = float(np.abs(wav).max())
    return wav / peak * 0.95 if peak > 1e-6 else wav
