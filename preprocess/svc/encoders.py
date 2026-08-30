"""実物の content encoder と F0 抽出器を、[`extract.py`](extract.py) が受け取る形に合わせる薄い層。

ここだけが重いモデルに触れます。[`extract.py`](extract.py) と [`shard.py`](shard.py) は
呼び出し可能オブジェクトを受け取るだけなので、単体テストはモデルもネットワークも要りません。

**content encoder の決定:** ContentVec（`lengyue233/content-vec-best`、MIT、768 次元、layer 12）。
根拠は [content encoder の選定](../../doc/svc-content-encoder.md)。話者情報の除去を目的に
追加学習されており、SVC で最大のリスクである timbre 漏れに直接効きます。

**F0 の決定:** RMVPE 固定。リポジトリの `preprocess/f0_rmvpe.py` は mel と同じフレーム数を
返すので（実測で確認済み）、そのまま使えます。
"""
from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_CONTENTVEC = "lengyue233/content-vec-best"
CONTENTVEC_SR = 16000
CONTENTVEC_STRIDE = 320          # 16 kHz / 320 = 50 Hz


class ContentVecEncoder:
    """ContentVec の指定層の隠れ状態を `[T_ssl, C]` で返す。

    `extract_phrase(content_encoder=...)` にそのまま渡せます。**16 kHz 前提**で、
    それ以外の sample rate を渡されたら例外にします（黙って resample すると、
    どの rate で抽出したのかが manifest と食い違うため）。
    """

    def __init__(self, model_id: str = DEFAULT_CONTENTVEC, *, layer: int = 12,
                 device: str = "cpu", revision: str | None = None):
        import torch
        from transformers import HubertModel

        self.model_id = model_id
        self.layer = int(layer)
        self.device = device
        self.revision = revision
        self._torch = torch
        self.model = HubertModel.from_pretrained(model_id, revision=revision).to(device).eval()
        n_layers = int(getattr(self.model.config, "num_hidden_layers", 0))
        if not 0 <= self.layer <= n_layers:
            raise ValueError(f"layer {self.layer} は 0..{n_layers} の範囲外です")

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if int(sr) != CONTENTVEC_SR:
            raise ValueError(f"ContentVec は {CONTENTVEC_SR} Hz 前提です; got {sr}")
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim != 1:
            raise ValueError(f"wav must be mono 1-D; got shape {wav.shape}")
        with self._torch.no_grad():
            x = self._torch.from_numpy(wav)[None].to(self.device)
            out = self.model(x, output_hidden_states=True)
        # hidden_states[0] は embedding 出力、hidden_states[i] が i 層目の出力。
        return out.hidden_states[self.layer][0].float().cpu().numpy().astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        """M1 ゴール 5 が要求する再現情報。抽出条件は後段の全実験の比較基盤になる。"""
        return {
            "content_encoder": self.model_id,
            "content_encoder_revision": self.revision or "main",
            "content_encoder_layer": self.layer,
            "content_encoder_sr": CONTENTVEC_SR,
            "content_encoder_stride": CONTENTVEC_STRIDE,
            "content_encoder_hidden": int(self.model.config.hidden_size),
        }


class RmvpeF0:
    """`extract_phrase(f0_extract=...)` の形に合わせた RMVPE の薄い包み。

    **確認済み:** `extract_f0_rmvpe` は mel と同じフレーム数を返します（44,100 / 30,011 /
    65,537 サンプルで実測、差 0）。`interpolate=False` で無声を 0 のまま返し、
    `uv` を別に受け取ります。
    """

    def __init__(self, *, f0_min: float = 65.0, f0_max: float = 1100.0,
                 device: str = "cpu"):
        self.f0_min, self.f0_max, self.device = float(f0_min), float(f0_max), device

    def __call__(self, wav: np.ndarray, sr: int, hop: int):
        from preprocess.f0_rmvpe import extract_f0_rmvpe
        f0, uv = extract_f0_rmvpe(wav, int(sr), int(hop), self.f0_min, self.f0_max,
                                  device=self.device, interpolate=False)
        return np.asarray(f0, np.float32), np.asarray(uv, np.float32)

    def manifest(self) -> dict[str, Any]:
        return {"f0_extractor": "rmvpe", "f0_min": self.f0_min, "f0_max": self.f0_max}
