"""WAV から cache（生の特徴）を作る、抽出の 1 段目。

[実行計画](../../doc/svc-plan.md) M1。重い処理（ContentVec と RMVPE）はここに集約し、
整列・正規化・次元削減は 2 段目（[`shard.py`](shard.py)）に置きます。こうしておくと、
補間方法や部分集合 seed を変えた ablation を 2 段目の再実行だけで回せます。

**ContentVec と F0 抽出器は引数で受け取ります。** 関数の内側で `from_pretrained` すると
単体テストが重いモデルとネットワークに依存してしまうためです。実物を差し込む薄い adapter は
[`encoders.py`](encoders.py) にあります。

出力は `f0_hz` / `uv` / `loudness` / `mel` が同じフレーム数で、`content` だけが SSL の
frame grid（50 Hz）のままです。**整列は 2 段目の仕事**です。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from leapsinger.config import MelSpec
from leapsinger.mel import wav_to_mel_nhv

from .loudness import frame_log_rms

ContentEncoder = Callable[[np.ndarray, int], np.ndarray]      # (wav, sr) -> [T_ssl, C]
F0Extract = Callable[[np.ndarray, int, int], "tuple[np.ndarray, np.ndarray]"]


def _resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if int(src_sr) == int(dst_sr):
        return wav
    from scipy.signal import resample_poly
    return resample_poly(wav, int(dst_sr), int(src_sr)).astype(np.float32)


def transpose_f0(f0_hz: np.ndarray, semitones: float) -> np.ndarray:
    """F0 を半音単位で移調する。**無声（0）はそのまま 0 に残す。**

    content と loudness には触りません。変換時に F0 だけを動かせるようにしておくと、
    「入力の F0 が低いと出力が暗くなる」現象を F0 単独で切り分けられます。男女をまたいで
    変換するときにも使います。
    """
    out = np.asarray(f0_hz, dtype=np.float32).copy()
    if float(semitones) == 0.0:
        return out
    voiced = out > 0.0
    out[voiced] = (out[voiced] * (2.0 ** (float(semitones) / 12.0))).astype(np.float32)
    return out


def extract_phrase(wav: np.ndarray, sr: int, *, content_encoder: ContentEncoder,
                   f0_extract: F0Extract, mel: MelSpec,
                   encoder_sr: int = 16000) -> dict[str, Any]:
    """1 クリップから `{content, f0_hz, uv, loudness, mel}` を作る。

    入口で `mel.sr` へ resample します。手元の素材は 44.1 k / 48 k / 96 kHz が混在しますが、
    `mel` セクションは前処理・loader・励起で共有され常に一致していなければならないためです。
    """
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim != 1:
        raise ValueError(f"wav must be mono 1-D; got shape {wav.shape}")
    if wav.size == 0:
        raise ValueError("wav is empty")

    wav = _resample(wav, sr, mel.sr)

    mel_db = wav_to_mel_nhv(wav, sr=mel.sr, n_fft=mel.n_fft, hop=mel.hop, win=mel.win,
                            n_mels=mel.n_mels, fmin=mel.fmin, fmax=mel.fmax)
    frames = int(mel_db.shape[1])

    f0_hz, uv = f0_extract(wav, mel.sr, mel.hop)
    f0_hz = np.asarray(f0_hz, dtype=np.float32)
    uv = np.asarray(uv, dtype=np.float32)
    for name, arr in (("f0", f0_hz), ("uv", uv)):
        if arr.ndim != 1 or int(arr.shape[0]) != frames:
            # 黙って切り詰めたり伸ばしたりしない。前処理の取り違えをここで露出させる。
            raise ValueError(f"{name} must be 1-D with {frames} frames; got shape {arr.shape}")

    loudness = frame_log_rms(wav, hop=mel.hop, n_fft=mel.n_fft)
    if int(loudness.shape[0]) != frames:
        raise ValueError(f"loudness frames {loudness.shape[0]} != mel frames {frames}")

    # ContentVec は 16 kHz を前提にする。44.1 kHz のまま渡すと無意味な特徴になる。
    content = np.asarray(content_encoder(_resample(wav, mel.sr, encoder_sr), int(encoder_sr)),
                         dtype=np.float32)
    if content.ndim != 2:
        raise ValueError(f"content_encoder must return [T_ssl, C]; got shape {content.shape}")

    return {"content": content, "f0_hz": f0_hz, "uv": uv,
            "loudness": loudness, "mel": mel_db.astype(np.float32)}
