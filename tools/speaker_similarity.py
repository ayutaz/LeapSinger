#!/usr/bin/env python3
"""target similarity を測る（[実行計画](../doc/svc-plan.md) M4 ゴール 2）。

    uv run python tools/speaker_similarity.py --converted out/m4/verify_ft20000_tp12 \
      --target download/ritsu/DATABASE --unrelated .m0data/vocalset_wav --device cpu

**単独の cos 値には意味がありません。** 話者照合の埋め込みは、同一話者の別クリップどうしでも
1.0 にはならず、無関係な話者どうしでも 0 にはなりません。[`m3_verify.py`](m3_verify.py) の
content cos と同じく、必ず 2 つの基準と並べます。

| 基準 | 意味 |
|---|---|
| **上限** | target の別クリップどうしの cos。同一話者でも届く上限 |
| 変換 | 変換後の音声 対 target のクリップ |
| **下限** | target 対 無関係な話者。話者性が乗っていないときに落ちる先 |

回復率 = (変換 − 下限) / (上限 − 下限)。

埋め込みモデルは差し替えられます（`embed` を引数で受け取る）。同梱の `XVectorEncoder` は
`transformers` の `WavLMForXVector` で、**新しい依存は増えません**（ContentVec で既に
transformers を使っているため）。**重みのライセンスは未確認**です。

**確認済み（2026-09-01）: ECAPA-TDNN を 12 秒以上のクリップで使うこと。** 較正は
[`speaker_calibrate.py`](speaker_calibrate.py) で再実行できます（VocalSet 20 歌手・
excerpts/straight・各 3 本）。**合格条件は走らせる前に決めた `MAX_OVERLAP = 0.20`** です。

| encoder | クリップ長 | 同一話者 | 別話者・同性 | 別話者・異性 | 重なり | 判定 |
|---|---:|---|---|---|---:|:--:|
| wavlm-base-plus-sv | 6 s | 0.8307 ± 0.0832 | 0.7713 ± 0.1104 | 0.5199 ± 0.1296 | 83.3% | 不合格 |
| wavlm-base-plus-sv | 12 s | 0.8856 ± 0.0670 | 0.8196 ± 0.0943 | 0.5314 ± 0.1323 | 77.0% | 不合格 |
| ECAPA-TDNN | 6 s | 0.5415 ± 0.1331 | 0.3452 ± 0.1325 | 0.1816 ± 0.0976 | 56.9% | 不合格 |
| **ECAPA-TDNN** | **12 s** | 0.6632 ± 0.0995 | 0.3891 ± 0.1369 | 0.1966 ± 0.1013 | **19.8%** | **合格** |
| **ECAPA-TDNN** | **20 s** | 0.7096 ± 0.1037 | 0.4040 ± 0.1386 | 0.2109 ± 0.1096 | **17.3%** | **合格** |

「重なり」は、同一話者ペアの下位 5% を超える別話者・同性ペアの割合です。

**encoder と長さの両方が要ります。** 長さだけでは足りず（x-vector は 12 秒でも 77.0%）、
ECAPA だけでも足りません（6 秒では 56.9%）。**6 秒で測ったことが、以前
「話者類似度は測れない」と結論した原因の半分でした。**

そのため `similarity_report` は `MIN_SECONDS` 秒未満のクリップを**拒否します**。較正の外だと
承知のうえで測るときだけ `min_seconds=0` を明示してください。

**差し替えたら必ずこの較正をやり直すこと**（同一話者 / 別話者・同性 / 別話者・異性の 3 群で、
重なりが十分小さいこと）。**重みのライセンスは未確認**です。
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

Embed = Callable[[np.ndarray, int], np.ndarray]      # (wav16, sr) -> [D]

# **較正が成り立つ最短のクリップ長。** VocalSet 20 歌手の較正（tools/speaker_calibrate.py）で、
# ECAPA-TDNN の重なりは 6 秒だと 56.9%、12 秒で 19.8%、20 秒で 17.3% でした。**短いクリップでは
# 事前登録した 20% を満たしません。** ここを下回る素材で数値を出しても読めないので拒否します。
MIN_SECONDS = 12.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    n = (np.linalg.norm(x) + 1e-12) * (np.linalg.norm(y) + 1e-12)
    return float(np.dot(x, y) / n)


def _agg(vals: Sequence[float]) -> dict[str, Any] | None:
    if not len(vals):
        return None
    return {"mean": float(np.mean(vals)), "median": float(np.median(vals)), "n": int(len(vals))}


def _check_lengths(groups: dict[str, Sequence[np.ndarray]], sr: int, min_seconds: float) -> None:
    """較正が成り立つ長さか。**足りなければ黙って測らず、どこが短いかを言って止まります。**"""
    if min_seconds <= 0:
        return
    need = int(min_seconds * sr)
    for name, wavs in groups.items():
        for k, w in enumerate(wavs):
            if len(w) < need:
                raise ValueError(
                    f"{name} の {k} 本目が {len(w) / sr:.1f} 秒しかありません。この指標は "
                    f"{MIN_SECONDS} 秒以上で較正しています（6 秒では重なり 56.9% で通りません）。"
                    "較正の外だと承知のうえで測るなら min_seconds=0 を明示してください")


def similarity_report(converted: Sequence[np.ndarray], target_refs: Sequence[np.ndarray],
                      unrelated: Sequence[np.ndarray] | None = None, *,
                      embed: Embed, sr: int,
                      min_seconds: float = MIN_SECONDS) -> dict[str, Any]:
    """変換 / 上限 / 下限の 3 つを返す。

    **target の参照クリップは 2 本以上**必要です。1 本では「同一話者でもここまでしか
    届かない」という上限が作れず、変換の数値を読めません。黙って上限なしの数値を返しません。

    **クリップは `MIN_SECONDS` 秒以上**必要です。較正はその長さでしか通っていません。
    """
    if len(target_refs) < 2:
        raise ValueError(
            f"target の参照クリップが {len(target_refs)} 本です。上限（別クリップどうしの cos）"
            "を作るのに 2 本以上要ります")
    if not len(converted):
        raise ValueError("変換後のクリップが空です")
    _check_lengths({"converted": converted, "target_refs": target_refs,
                    "unrelated": unrelated or []}, sr, min_seconds)
    # **1 クリップにつき 1 回だけ埋め込む。** ペアごとに呼ぶと実モデルで組合せ爆発する。
    ref_e = [embed(w, sr) for w in target_refs]
    conv_e = [embed(w, sr) for w in converted]
    other_e = [embed(w, sr) for w in (unrelated or [])]

    ceiling = [cosine(ref_e[i], ref_e[j])
               for i in range(len(ref_e)) for j in range(i + 1, len(ref_e))]
    conv = [cosine(c, r) for c in conv_e for r in ref_e]
    floor = [cosine(r, o) for r in ref_e for o in other_e]

    # **較正は長さに依存する。** 後から読む人が「較正の内側で出た値か」を判断できるよう残す。
    seen = [len(w) / sr for w in (list(converted) + list(target_refs) + list(unrelated or []))]
    out: dict[str, Any] = {"converted_vs_target": _agg(conv),
                           "target_self_ceiling": _agg(ceiling),
                           "unrelated_floor": _agg(floor), "recovery": None,
                           "clip_seconds": {"min": float(min(seen)), "max": float(max(seen)),
                                            "min_required": float(min_seconds)}}
    if floor:
        span = out["target_self_ceiling"]["mean"] - out["unrelated_floor"]["mean"]
        if abs(span) > 1e-9:
            out["recovery"] = float(
                (out["converted_vs_target"]["mean"] - out["unrelated_floor"]["mean"]) / span)
    return out


class XVectorEncoder:
    """x-vector 話者埋め込み。16 kHz の 1 本の波形 -> 埋め込み [D]。

    `AutoModelForAudioXVector` なので `--model` でモデルを差し替えられます。**差し替えたら
    必ず較正をやり直すこと**（module の docstring）。
    """

    def __init__(self, model_id: str = "microsoft/wavlm-base-plus-sv",
                 device: str = "cpu", revision: str | None = None):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
        self.model_id, self.device, self.revision = model_id, device, revision
        self.fe = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)
        self.model = (AutoModelForAudioXVector
                      .from_pretrained(model_id, revision=revision).to(device).eval())
        self._torch = torch

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if int(sr) != 16000:
            raise ValueError(f"16 kHz を渡すこと（{sr} が来ました）")
        x = self.fe(wav, sampling_rate=16000, return_tensors="pt")
        with self._torch.no_grad():
            e = self.model(**{k: v.to(self.device) for k, v in x.items()}).embeddings
        return e.squeeze(0).cpu().numpy().astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        return {"speaker_encoder": self.model_id,
                "speaker_encoder_revision": self.revision or "main",
                "speaker_encoder_sr": 16000,
                "speaker_encoder_license": "未確認（内部指標としてのみ使う）"}


class EcapaEncoder:
    """ECAPA-TDNN 話者埋め込み（speechbrain）。16 kHz の 1 本の波形 -> 埋め込み [192]。

    x-vector（`XVectorEncoder`）が歌声の同性ペアで較正を通らなかったため追加しました。
    **これも通るとは限りません。** 使う前に必ず
    [`speaker_calibrate.py`](speaker_calibrate.py) を走らせ、重なりを確認すること。

    重みは初回に HuggingFace から取得し、`~/.cache/speechbrain/` へ置きます
    （リポジトリを汚さないため。speechbrain の既定は `./pretrained_models/`）。
    **重みのライセンスは未確認**なので、内部指標としてのみ使います。
    """

    def __init__(self, model_id: str = "speechbrain/spkrec-ecapa-voxceleb",
                 device: str = "cpu", savedir: str | None = None):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        self.model_id, self.device = model_id, device
        self._savedir = savedir or str(Path.home() / ".cache" / "speechbrain"
                                       / model_id.replace("/", "--"))
        self.model = EncoderClassifier.from_hparams(
            source=model_id, savedir=self._savedir, run_opts={"device": device})
        self.model.eval()
        self._torch = torch

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if int(sr) != 16000:
            raise ValueError(f"16 kHz を渡すこと（{sr} が来ました）")
        x = self._torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
        with self._torch.no_grad():
            e = self.model.encode_batch(x.to(self.device))
        return e.reshape(-1).cpu().numpy().astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        return {"speaker_encoder": self.model_id,
                "speaker_encoder_kind": "ecapa-tdnn (speechbrain)",
                "speaker_encoder_sr": 16000,
                "speaker_encoder_license": "未確認（内部指標としてのみ使う）"}


def _load(paths: Sequence[Path], sr: int, seconds: float) -> list[np.ndarray]:
    import soundfile as sf

    from preprocess.svc.extract import _resample
    out = []
    for p in paths:
        w, s = sf.read(p, dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
        w = _resample(np.ascontiguousarray(w, dtype=np.float32), s, sr)
        if len(w) < sr // 2:
            continue
        if seconds and len(w) > int(seconds * sr):
            a = (len(w) - int(seconds * sr)) // 2
            w = np.ascontiguousarray(w[a:a + int(seconds * sr)])
        out.append(w)
    return out


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--converted", required=True, help="変換後の WAV ディレクトリ")
    ap.add_argument("--target", required=True, help="target singer の生 WAV ディレクトリ")
    ap.add_argument("--unrelated", default=None, help="無関係な話者の WAV ディレクトリ（下限）")
    ap.add_argument("--out", default=None, help="report の書き出し先 JSON")
    ap.add_argument("--n-clips", type=int, default=16, help="各群から使うクリップ数")
    ap.add_argument("--seconds", type=float, default=20.0,
                    help=f"各クリップの長さ。較正は {MIN_SECONDS} 秒以上でのみ通っている")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--encoder", default="ecapa", choices=("ecapa", "xvector"),
                    help="既定は ECAPA-TDNN（較正を通った唯一の encoder）")
    ap.add_argument("--model", default=None, help="model id を上書きする")
    ap.add_argument("--min-seconds", type=float, default=MIN_SECONDS,
                    help="0 にすると較正外の長さでも測る（読めない値になる点に注意）")
    a = ap.parse_args()

    def pick(root: str | None) -> list[Path]:
        if not root:
            return []
        # 変換結果のディレクトリには source も混ざるので除く
        ps = [p for p in sorted(Path(root).rglob("*.wav")) if "_source" not in p.name]
        return ps[:a.n_clips]

    conv_p, ref_p, other_p = pick(a.converted), pick(a.target), pick(a.unrelated)
    if not conv_p or not ref_p:
        sys.exit(f"WAV が足りません: converted {len(conv_p)} / target {len(ref_p)}")
    if a.encoder == "ecapa":
        enc = EcapaEncoder(a.model or "speechbrain/spkrec-ecapa-voxceleb", device=a.device)
    else:
        enc = XVectorEncoder(a.model or "microsoft/wavlm-base-plus-sv", device=a.device)
    rep = similarity_report(_load(conv_p, 16000, a.seconds), _load(ref_p, 16000, a.seconds),
                            _load(other_p, 16000, a.seconds), embed=enc, sr=16000,
                            min_seconds=a.min_seconds)
    rep["manifest"] = enc.manifest()
    rep["files"] = {"converted": [str(p) for p in conv_p], "target": [str(p) for p in ref_p],
                    "unrelated": [str(p) for p in other_p]}
    c, ceil, floor = (rep["converted_vs_target"], rep["target_self_ceiling"],
                      rep["unrelated_floor"])
    print("=== target similarity ===")
    print(f"  変換 対 target : {c['mean']:.4f}  (n={c['n']})")
    print(f"  上限（target 同士）: {ceil['mean']:.4f}  (n={ceil['n']})")
    if floor:
        print(f"  下限（無関係）   : {floor['mean']:.4f}  (n={floor['n']})")
        print(f"  -> 回復率        : {rep['recovery'] * 100:.1f}%")
    if a.out:
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
