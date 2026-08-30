"""Parity checks for the ONNX export.

  verify_excitation(): HarmonicNoiseExcitationONNX(deterministic) vs the training
      harmonic_noise_mel_torch(noise_ratio=0) on a real F0 curve — the excitation is the flow's
      x0, so it must reproduce training numerics (not "improve" on them).
  verify_full_graph(): PyTorch wrapper vs the exported ONNX in ORT, on two different lengths, to
      confirm export fidelity AND that the token/frame axes are dynamic.

Deterministic (noise=0) is used for parity; the deployment graph draws noise internally and is
only smoke-checked (shape/finite) since it is not reproducible across runtimes.
"""
from __future__ import annotations

import numpy as np
import torch


def verify_excitation(f0_hz=None, n_harm=50, hop=256, mel_bins=128) -> dict:
    """MAE between the ONNX-friendly excitation and the training excitation (both noise-free)."""
    from leapsinger.modules.harmonic_excitation import harmonic_noise_mel_torch

    from .excitation_onnx import HarmonicNoiseExcitationONNX
    if f0_hz is None:
        t = np.linspace(0, 3, 400)
        f0_hz = (250 + 60 * np.sin(2 * np.pi * 0.8 * t)).astype(np.float32)
    f0l = torch.log2(torch.clamp(torch.as_tensor(f0_hz, dtype=torch.float32), min=1.0))[None]
    T = f0l.shape[1]
    uv = torch.ones(1, T)
    orig = harmonic_noise_mel_torch(f0l, uv, noise_ratio=0.0, n_harm=n_harm, hop=hop,
                                    scale=0.15, harm_decay=1.0)
    exc = HarmonicNoiseExcitationONNX(n_harm=n_harm, exc_hop=hop, n_mels=mel_bins,
                                      deterministic=True).eval()
    with torch.no_grad():
        new = exc(f0l, uv)
    Tc = min(orig.shape[2], new.shape[2])
    d = (orig[:, :, :Tc] - new[:, :, :Tc]).abs()
    return {"mae": float(d.mean()), "max": float(d.max()), "T": int(Tc)}


def _synth_item(Np, seed, vocab=50):
    g = np.random.default_rng(seed)
    tokens = g.integers(1, max(2, vocab), size=Np).astype(np.int64)
    durs = g.integers(4, 14, size=Np).astype(np.int64)
    T = int(durs.sum())
    f0 = (250 + 60 * np.sin(np.linspace(0, 6, T)) + g.normal(0, 3, T)).clip(120, 600).astype(np.float32)
    return tokens, durs, f0


def verify_full_graph(wrapper, onnx_path: str, input_names, *, fp16: bool = False,
                      lengths=((1, 24), (7, 40))) -> list:
    """Run the PyTorch wrapper vs ORT on several lengths. Requires wrapper.exc.deterministic=True
    for a meaningful MAE; otherwise only shape/finite is trustworthy. Returns per-length dicts."""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    sess_inputs = {i.name for i in sess.get_inputs()}
    det = bool(getattr(wrapper.exc, "deterministic", False))
    variant_full = "uv" in input_names
    speaker_embed = "spk_embed" in input_names
    enc = getattr(wrapper.model, "phoneme_encoder", None)
    vocab = enc.embed.num_embeddings if enc is not None and getattr(enc, "embed", None) is not None \
        else wrapper.model.phoneme_embed.num_embeddings
    out = []
    for seed, Np in lengths:
        tokens, durs, f0 = _synth_item(Np, seed, vocab)
        T = int(durs.sum())
        feeds = {"tokens": tokens[None], "durations": durs[None], "f0": f0[None]}
        targs = [torch.as_tensor(tokens)[None], torch.as_tensor(durs)[None], torch.as_tensor(f0)[None]]
        if variant_full:
            uv = np.ones((1, T), np.float32); feeds["uv"] = uv; targs.append(torch.as_tensor(uv))
        if speaker_embed:
            se = np.zeros((1, wrapper.hidden), np.float32); feeds["spk_embed"] = se
            targs.append(torch.as_tensor(se))
        if "speedup" in sess_inputs:                       # OpenUTAU input, accepted-but-ignored
            feeds["speedup"] = np.array([1], np.int64)     # (not a wrapper arg — ORT-only)
        with torch.no_grad():
            tmel = wrapper(*targs)[0].numpy()
        omel = sess.run(None, feeds)[0][0]
        # tmel and omel share the wrapper's layout (WrapperA emits [1,T,mel], WrapperB [1,mel,T]),
        # so compare whole arrays; equal shapes also confirm the ORT frame axis stayed dynamic.
        same_shape = tuple(tmel.shape) == tuple(omel.shape)
        mae = float(np.abs(tmel - omel).mean()) if (det and same_shape) else None
        rec = {"Np": Np, "T": T, "torch_shape": tmel.shape, "ort_shape": omel.shape,
               "mae": mae, "finite": bool(np.isfinite(omel).all()),
               "T_dynamic_ok": same_shape}
        out.append(rec)
        tag = "fp16" if fp16 else "fp32"
        maes = f"MAE={mae:.4f}" if mae is not None else "MAE=n/a(stochastic)"
        print(f"[verify {tag}] Np={Np:2d} T={T:4d} torch{tmel.shape} ort{omel.shape} "
              f"{maes} finite={rec['finite']} dyn_ok={rec['T_dynamic_ok']}")
    return out


if __name__ == "__main__":
    r = verify_excitation()
    print(f"[verify] excitation  MAE={r['mae']:.6f} MAX={r['max']:.4f} T={r['T']}")
