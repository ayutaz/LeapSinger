"""loudness 特徴量（フレーム単位の log-RMS）と、その dataset 統計での正規化。

**フレーム数は mel と完全に一致させます。** `leapsinger/mel.py` の `wav_to_mel_nhv` は
`center=False` + 事前 reflect pad `(n_fft-hop)//2` という framing なので、素直に窓を切ると必ず
ずれます。ここでは同じ pad を当ててから同じ窓長・同じ hop で RMS を取ります。データ契約は
`content` / `f0_interp` / `uv` / `loudness` / `mel` の `T` 完全一致を要求し、loader は暗黙に
直さないので、ずれは学習開始前に例外になります。

正規化は **dataset 統計**で行います（`doc/svc.md` の決定事項）。phrase 単位で正規化すると
phrase 間の強弱差が消え、歌の表情が平坦になるためです。
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

FLOOR = 1e-5           # 無音（RMS = 0）で -inf にしないための下限。mel 側の clamp と同じ値


def loudness_manifest(*, hop: int, n_fft: int, floor: float = FLOOR) -> dict:
    """loudness の**定義**を manifest 用の dict で返す（実行計画 M1 ゴール 5）。

    mean / std だけでは、後から窓幅や floor を変えたときに古い shard と見分けがつきません。
    抽出条件は以降の全実験の比較基盤になるので、窓幅・hop・floor・定義そのものを残します。
    """
    return {
        "loudness_definition": "frame log-RMS (natural log), mel framing "
                               "(center=False + reflect pad (n_fft-hop)//2)",
        "loudness_n_fft": int(n_fft),
        "loudness_hop": int(hop),
        "loudness_floor": float(floor),
    }


def frame_log_rms(wav: np.ndarray, *, hop: int, n_fft: int,
                  floor: float = FLOOR) -> np.ndarray:
    """`wav` -> フレームごとの自然対数 RMS `[T]`。`T` は同じ設定の mel と一致します。

    `floor` は無音（RMS = 0）で `-inf` にならないための下限で、mel 側の clamp と同じ値です。
    """
    import librosa

    if wav.ndim != 1:
        raise ValueError(f"wav must be mono 1-D; got shape {wav.shape}")
    hop, n_fft = int(hop), int(n_fft)
    pad = (n_fft - hop) // 2
    padded = np.pad(np.asarray(wav, dtype=np.float64), (pad, pad), mode="reflect")
    rms = librosa.feature.rms(y=padded, frame_length=n_fft, hop_length=hop, center=False)[0]
    return np.log(np.maximum(floor, rms)).astype(np.float32)


def loudness_match_gain(wav: np.ndarray, manifest, *, hop: int, n_fft: int,
                        floor: float = FLOOR) -> float:
    """入力の loudness を学習分布の平均へ寄せる倍率を返す。

    配信用に整えられた音源は学習素材よりずっと大きく、条件が学習分布の外へ出ます
    （実測: 自作の YouTube 音源は **+1.40 sigma**、波音リツ DB は peak 0.107）。
    その状態で変換すると高域が削れます（spectral centroid が上限比 **-47%**。
    合わせると **-22%** まで戻りました）。

    `frame_log_rms` は自然対数なので、倍率 g をかけると平均が log(g) だけ動きます。
    したがって `g = exp(学習平均 - 現在の平均)` です。

    **推論時に波形を勝手に加工しないのが原則**なので、これは呼び出し側が明示的に使う
    ための関数です（`svc_convert.py --match-loudness`）。無音は寄せようがないので 1.0 を返します。
    """
    if "loudness_mean" not in manifest:
        raise ValueError("manifest に loudness_mean がありません。shard の manifest.json を渡してください")
    current = float(frame_log_rms(wav, hop=hop, n_fft=n_fft, floor=floor).mean())
    if current <= float(np.log(floor)) + 1e-6:       # 実質すべて無音。寄せる先が無い
        return 1.0
    return float(np.exp(float(manifest["loudness_mean"]) - current))


def dataset_stats(arrays: Sequence[np.ndarray] | Iterable[np.ndarray]) -> tuple[float, float]:
    """全 phrase を通した (mean, std) を返す。

    phrase ごとの平均ではなく**全フレームを一緒にした**統計です。長い phrase はその分だけ
    重く効きます。これは意図した挙動で、収録時間に比例した重み付けになります。
    """
    flat = [np.asarray(a, dtype=np.float64).ravel() for a in arrays]
    flat = [a for a in flat if a.size]
    if not flat:
        raise ValueError("dataset_stats needs at least one non-empty array")
    all_values = np.concatenate(flat)
    return float(all_values.mean()), float(all_values.std())


def normalize_with_stats(x: np.ndarray, mean: float, std: float,
                         eps: float = 1e-8) -> np.ndarray:
    """`(x - mean) / std`。`std` が 0 でも NaN を返しません。

    全フレームが同じ値の dataset（無音のみなど）で std = 0 になります。そこで NaN を作ると
    shard に混入して学習が静かに壊れるため、その場合は 1.0 で割ります。
    """
    scale = float(std) if float(std) > eps else 1.0
    return ((np.asarray(x, dtype=np.float32) - np.float32(mean)) / np.float32(scale)).astype(np.float32)
