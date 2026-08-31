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

**確認済み（2026-08-31）: 既定の `microsoft/wavlm-base-plus-sv` は歌声では使えません。**
VocalSet 20 歌手 × 同一技法（arpeggios / straight）4 clip = 80 本で較正した結果:

| ペア | cos |
|---|---|
| 同一話者 | 0.7533 ± 0.1379 (n=120) |
| 別話者・同性 | 0.6961 ± 0.1476 (n=1456) |
| 別話者・異性 | 0.6813 ± 0.1323 (n=1584) |

同一話者の下位 5%（0.490）を**別話者・同性ペアの 88.3%** が超えます。差 0.057 に対して
ばらつきが 0.14 あり、**同一話者と別話者を判別できていません**。話し声で学習された
話者照合モデルを歌声にそのまま当てた結果です。

**したがって、この既定 encoder の出力を target similarity として報告してはいけません。**
上限・下限・回復率という枠組み自体は正しいので、**歌声で較正を通る encoder に差し替えて**
から使ってください。差し替えたら必ずこの較正（同一話者 / 別話者・同性 / 別話者・異性）を
やり直し、分離が出ることを確かめること。
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

Embed = Callable[[np.ndarray, int], np.ndarray]      # (wav16, sr) -> [D]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    n = (np.linalg.norm(x) + 1e-12) * (np.linalg.norm(y) + 1e-12)
    return float(np.dot(x, y) / n)


def _agg(vals: Sequence[float]) -> dict[str, Any] | None:
    if not len(vals):
        return None
    return {"mean": float(np.mean(vals)), "median": float(np.median(vals)), "n": int(len(vals))}


def similarity_report(converted: Sequence[np.ndarray], target_refs: Sequence[np.ndarray],
                      unrelated: Sequence[np.ndarray] | None = None, *,
                      embed: Embed, sr: int) -> dict[str, Any]:
    """変換 / 上限 / 下限の 3 つを返す。

    **target の参照クリップは 2 本以上**必要です。1 本では「同一話者でもここまでしか
    届かない」という上限が作れず、変換の数値を読めません。黙って上限なしの数値を返しません。
    """
    if len(target_refs) < 2:
        raise ValueError(
            f"target の参照クリップが {len(target_refs)} 本です。上限（別クリップどうしの cos）"
            "を作るのに 2 本以上要ります")
    if not len(converted):
        raise ValueError("変換後のクリップが空です")
    # **1 クリップにつき 1 回だけ埋め込む。** ペアごとに呼ぶと実モデルで組合せ爆発する。
    ref_e = [embed(w, sr) for w in target_refs]
    conv_e = [embed(w, sr) for w in converted]
    other_e = [embed(w, sr) for w in (unrelated or [])]

    ceiling = [cosine(ref_e[i], ref_e[j])
               for i in range(len(ref_e)) for j in range(i + 1, len(ref_e))]
    conv = [cosine(c, r) for c in conv_e for r in ref_e]
    floor = [cosine(r, o) for r in ref_e for o in other_e]

    out: dict[str, Any] = {"converted_vs_target": _agg(conv),
                           "target_self_ceiling": _agg(ceiling),
                           "unrelated_floor": _agg(floor), "recovery": None}
    if floor:
        span = out["target_self_ceiling"]["mean"] - out["unrelated_floor"]["mean"]
        if abs(span) > 1e-9:
            out["recovery"] = float(
                (out["converted_vs_target"]["mean"] - out["unrelated_floor"]["mean"]) / span)
    return out


class XVectorEncoder:
    """WavLM の x-vector。16 kHz の 1 本の波形 -> 埋め込み [D]。"""

    def __init__(self, model_id: str = "microsoft/wavlm-base-plus-sv",
                 device: str = "cpu", revision: str | None = None):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.model_id, self.device, self.revision = model_id, device, revision
        self.fe = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)
        self.model = WavLMForXVector.from_pretrained(model_id, revision=revision).to(device).eval()
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
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="microsoft/wavlm-base-plus-sv")
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
    enc = XVectorEncoder(a.model, device=a.device)
    rep = similarity_report(_load(conv_p, 16000, a.seconds), _load(ref_p, 16000, a.seconds),
                            _load(other_p, 16000, a.seconds), embed=enc, sr=16000)
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
