"""LeapSinger acoustic-model ONNX exporter.

    uv run python -m export.cli --ckpt log/3speaker_gan2d/ckpt_050000.pt --out export/3speaker \
        --model-name leapsinger_oniku --variant diffsinger --hop 512      # fp32 (default)
    #   add --fp16 for a ~32% smaller file (lossless, but not faster on CPU; helps GPU/mobile)

Variants (chosen to match the trained ckpt; the only weight-level split is use_uv):
    diffsinger (A): use_uv=False, n_styles=0. Inputs [tokens, durations, f0 (+spk_embed)].
                    --hop 512 = pseudo-512 (×2 upsample in / pairwise-average out, OpenUTAU grid) | 256 = native.
    full       (B): use_uv=True (and/or styles). Inputs [tokens, durations, f0, uv (+spk_embed)].
                    Native hop. A style-bearing model runs at its base style (ONNX style input not wired).

Speaker: none (single-spk) | embed (spk_embed graph input, .emb per speaker) | bake (freeze one
speaker into the graph; acoustic file becomes {model}.{spk}.onnx).

fp32 by default (simple `torch.onnx.export`, the public-release default — none of the fp16
machinery runs). Pass --fp16 for the ~32% smaller partial-fp16 build (see postprocess.py).

Output dir gets: {model}.onnx (fp32 unless --fp16), dsconfig.yaml, {model}.phonemes.txt,
{model}.leapsinger.json, and (embed) {model}.{spk}.emb per speaker.
"""
from __future__ import annotations

import argparse
import os

import torch

from infer import load_acoustic

from . import dsconfig as dscfg
from . import postprocess as post
from . import spk_embed as spk
from .wrappers import AcousticExportWrapperA, AcousticExportWrapperB


def _vocab_size(model):
    """Number of phoneme-embedding rows (attn: phoneme_encoder.embed; conv: phoneme_embed)."""
    enc = getattr(model, "phoneme_encoder", None)
    if enc is not None and getattr(enc, "embed", None) is not None:
        return enc.embed.num_embeddings
    return model.phoneme_embed.num_embeddings


def _validate(cfg, model, variant, speaker):
    """Hard checks: variant must match the ckpt's weight-level capabilities."""
    use_uv = bool(cfg.get("use_uv", True))
    has_style = getattr(model, "style_emb", None) is not None
    if variant == "diffsinger":
        if use_uv:
            raise SystemExit("variant=diffsinger needs a uv-free ckpt (use_uv=False); this ckpt "
                             "has use_uv=True — export it as --variant full, or train uv-free.")
        if has_style:
            raise SystemExit("variant=diffsinger needs n_styles=0; this ckpt has a style bank.")
    elif variant == "full":
        if not use_uv and not has_style:
            print("[export] note: --variant full on a uv-free, style-less ckpt is just the "
                  "native-hop A graph with a v/uv input that is ignored.")
    if speaker in ("embed", "bake") and not spk.has_speakers(model):
        raise SystemExit(f"speaker={speaker} needs a multi-speaker ckpt (spk_bank); this ckpt is "
                         f"single-speaker. Use --speaker none.")
    if speaker == "none" and spk.has_speakers(model):
        print("[export] note: multi-speaker ckpt exported with --speaker none "
              "(speaker 0's bank vector is unused; cond gets no speaker term).")


def _build_wrapper(model, variant, hop, num_steps, speaker, frozen_spk):
    if variant == "diffsinger":
        return AcousticExportWrapperA(model, num_steps=num_steps, hop=hop,
                                      speaker=speaker, frozen_spk=frozen_spk)
    return AcousticExportWrapperB(model, num_steps=num_steps, speaker=speaker,
                                  frozen_spk=frozen_spk)


def _example_and_axes(variant, speaker, hidden, vocab_size):
    """Return (input_names, example_tensors, dynamic_axes) for torch.onnx.export.
    Trace tokens are bounded by vocab_size (a ckpt may have n_phonemes < the current 50)."""
    Np, T = 8, 40
    tokens = torch.randint(1, max(2, vocab_size), (1, Np), dtype=torch.long)
    durs = torch.full((1, Np), T // Np, dtype=torch.long)
    T = int(durs.sum())
    f0 = torch.linspace(200, 300, T)[None]
    names = ["tokens", "durations", "f0"]
    args = [tokens, durs, f0]
    axes = {"tokens": {1: "n_tokens"}, "durations": {1: "n_tokens"}, "f0": {1: "n_frames"},
            "mel": {2: "n_frames"}}
    if variant == "full":
        names.append("uv"); args.append(torch.ones(1, T)); axes["uv"] = {1: "n_frames"}
    if speaker == "embed":
        names.append("spk_embed"); args.append(torch.zeros(1, hidden))
    return names, tuple(args), axes


def main():
    ap = argparse.ArgumentParser(description="Export a LeapSinger acoustic ckpt to ONNX.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--variant", choices=["diffsinger", "full"], default="diffsinger")
    ap.add_argument("--hop", type=int, choices=[256, 512], default=512,
                    help="diffsinger only: 512=pseudo-512 (OpenUTAU) | 256=native")
    ap.add_argument("--speaker", choices=["none", "embed", "bake"], default="none")
    ap.add_argument("--spk-name", default=None, help="bake: speaker name; embed: unused")
    ap.add_argument("--spk-id", type=int, default=0, help="bake: which speaker id to freeze")
    ap.add_argument("--num-steps", type=int, default=None, help="flow Euler steps (default ckpt infer_steps)")
    ap.add_argument("--opset", type=int, default=17)
    # fp32 is the public-release default (simple export, faster on CPU, lossless). --fp16 opts into
    # the ~32%-smaller partial-fp16 build (helps GPU/mobile size & speed, no CPU speedup).
    ap.add_argument("--fp16", dest="fp16", action="store_true", default=False)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--simplify", dest="simplify", action="store_true", default=True)
    ap.add_argument("--no-simplify", dest="simplify", action="store_false")
    # Replace the ~17 MB DFT-as-matmul excitation with a native ONNX `DFT` node (lossless, -17 MB,
    # a little faster single-threaded). Default on: the forward `DFT` op is supported by the
    # onnxruntime CPU EP (>= ~1.14), and OpenUTAU bundles onnxruntime 1.23. Use --no-native-dft for
    # the matmul build, which runs on any ORT (e.g. a very old runtime without the DFT kernel).
    ap.add_argument("--native-dft", dest="native_dft", action="store_true", default=True)
    ap.add_argument("--no-native-dft", dest="native_dft", action="store_false")
    # Declare OpenUTAU's required `speedup` input (accepted but ignored — our flow is fixed-step).
    # On by default for the diffsinger variant (the OpenUTAU-targeting one).
    ap.add_argument("--speedup-input", dest="speedup_input", action="store_true", default=True)
    ap.add_argument("--no-speedup-input", dest="speedup_input", action="store_false")
    ap.add_argument("--deterministic", action="store_true",
                    help="zero the excitation noise (reproducible; for verification, not deployment)")
    ap.add_argument("--vocoder-name", default="nhvsing_v6_44k",
                    help="dsconfig 'vocoder' — the NHVSing vocoder package the host installs")
    ap.add_argument("--verify", action="store_true", help="run ORT parity vs PyTorch after export")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model, cfg = load_acoustic(args.ckpt, device="cpu")
    num_steps = args.num_steps or int(cfg.get("infer_steps", 10))
    _validate(cfg, model, args.variant, args.speaker)

    frozen_spk = None
    speakers_for_cfg = None
    acoustic_file = f"{args.model_name}.onnx"
    if args.speaker == "bake":
        frozen_spk = spk.speaker_vector(model, args.spk_id)
        spk_name = args.spk_name or f"spk{args.spk_id}"
        acoustic_file = f"{args.model_name}.{spk_name}.onnx"
    elif args.speaker == "embed":
        # write one .emb per speaker (names default to spk{i})
        names = {i: (args.spk_name if i == args.spk_id and args.spk_name else f"spk{i}")
                 for i in range(model.spk_n)}
        speakers_for_cfg = spk.export_speaker_embeds(model, args.out, args.model_name, names)

    wrapper = _build_wrapper(model, args.variant, args.hop, num_steps,
                             args.speaker, frozen_spk).eval()
    wrapper.exc.deterministic = bool(args.deterministic)

    input_names, example, axes = _example_and_axes(
        args.variant, args.speaker, model.hidden, _vocab_size(model))

    fp32_path = os.path.join(args.out, acoustic_file.replace(".onnx", ".fp32.onnx"))
    final_path = os.path.join(args.out, acoustic_file)
    torch.onnx.export(wrapper, example, fp32_path, opset_version=args.opset,
                      input_names=input_names, output_names=["mel"],
                      dynamic_axes=axes, do_constant_folding=True, dynamo=False)
    print(f"[export] fp32 -> {fp32_path} ({os.path.getsize(fp32_path)/1e6:.1f} MB)")

    add_speedup = args.speedup_input and args.variant == "diffsinger"
    post.finalize(fp32_path, final_path, fp16=args.fp16, do_simplify=args.simplify,
                  native_dft=args.native_dft, add_speedup=add_speedup)
    print(f"[export] final -> {final_path} ({os.path.getsize(final_path)/1e6:.1f} MB) "
          f"fp16={args.fp16} simplify={args.simplify}")
    if final_path != fp32_path and os.path.exists(fp32_path):
        os.remove(fp32_path)

    dscfg.write_phonemes(args.out, args.model_name, phonemes=cfg.get("phonemes"))   # phonemes.txt (SP/AP)
    dscfg.write_dsconfig(args.out, args.model_name, hop=(args.hop if args.variant == "diffsinger" else cfg.get("hop", 256)),
                         hidden_size=model.hidden, vocoder_name=args.vocoder_name,
                         speakers=speakers_for_cfg, acoustic_file=acoustic_file)
    dscfg.write_leapsinger_meta(args.out, args.model_name, variant=args.variant,
                                hop=(args.hop if args.variant == "diffsinger" else cfg.get("hop", 256)),
                                use_uv=bool(cfg.get("use_uv", True)), infer_steps=num_steps,
                                speaker_mode=args.speaker)
    from . import openutau_assets as oua
    oua.write_dsdict(args.out)                                      # dsdict.yaml (kana -> phonemes) for the phonemizer
    print(f"[export] wrote dsconfig.yaml, {args.model_name}.phonemes.txt, dsdict.yaml, "
          f"{args.model_name}.leapsinger.json"
          + (f", {len(speakers_for_cfg)} .emb" if speakers_for_cfg else ""))

    if args.verify:
        from .verify import verify_full_graph
        verify_full_graph(wrapper, final_path, input_names, fp16=args.fp16)


if __name__ == "__main__":
    main()
