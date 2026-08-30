"""長い曲を phrase へ切る。

SVS 側（`preprocess/phrase_cut.py`）は `.lab` の音素境界と pau を使って切りますが、**SVC は
音素ラベルを使わない**ので、ここでは固定長で切ります。素材によっては無音で切るほうが自然
ですが、固定長は**再現が容易で、境界が入力に依存しない**という利点があります。

**未実装:** 無音区間を優先した分割。M2 で境界の artifact が問題になったら検討します。

phrase 名は `{song}_{NNNN}` にしてください。`dataset.py` の `_song_of()` が曲単位の
train/eval 分割に使うので、この命名を崩すと leakage 防止が効かなくなります。
"""
from __future__ import annotations


def chunk_spans(n_samples: int, sr: int, *, chunk_sec: float,
                min_sec: float) -> list[tuple[int, int]]:
    """`[(start, end), ...]` をサンプル単位で返す。隙間も重複もありません。

    末尾が `min_sec` に満たなければ捨てます。短すぎる断片は F0 も content も不安定で、
    学習の役に立たないためです。
    """
    n_samples, sr = int(n_samples), int(sr)
    chunk_sec, min_sec = float(chunk_sec), float(min_sec)
    if chunk_sec <= 0:
        raise ValueError(f"chunk_sec must be positive; got {chunk_sec}")
    if min_sec < 0:
        raise ValueError(f"min_sec must be non-negative; got {min_sec}")
    if min_sec > chunk_sec:
        raise ValueError(f"min_sec ({min_sec}) が chunk_sec ({chunk_sec}) より長いと "
                         f"すべての chunk が捨てられます")

    step = int(chunk_sec * sr)
    floor = int(min_sec * sr)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n_samples:
        end = min(start + step, n_samples)
        if end - start >= floor:
            spans.append((start, end))
        start = end
    return spans


def voiced_ratio(uv) -> float:
    """有声フレームの割合を返す。無声だけの chunk を弾くために使います。

    **実データで見つかった問題:** 曲を先頭から固定長で切ると、イントロや間奏が丸ごと無声の
    phrase になります。波音リツ 1 曲を 3 秒で切ったところ **89 phrase 中 35 件（39%）が完全に
    無声**で、先頭 9 個（27 秒）は連続して無声でした。これを学習に入れると「無音を出す」ことを
    学びます。M2 の overfit がまさにそれで、生成された WAV が無音になりました。
    """
    import numpy as np

    uv = np.asarray(uv)
    if uv.ndim != 1:
        raise ValueError(f"uv must be 1-D; got shape {uv.shape}")
    if uv.size == 0:
        return 0.0
    return float((uv > 0.5).mean())
