"""cache（生の特徴）から `svc_shard.npz` を作る、抽出の 2 段目。

[実行計画](../../doc/svc-plan.md) M1。抽出を 2 段に分けているのは、**補間方法と 256 次元
部分集合の ablation を、この段の再実行だけで回せるようにする**ためです。ContentVec と RMVPE を
回し直さずに済み、vast.ai の時間課金に直接効きます。

この段が行うのは 3 つだけです。

1. content を SSL の frame grid から **mel の frame grid へ left（直前保持）で整列**する
2. ContentVec 768 次元から **固定ランダムの部分集合**を切り出す
3. loudness を **dataset 統計**で正規化する

出力は `svc_dataset.py` の契約どおり、全配列の `T` が完全一致した npz です。loader は暗黙に
直さないので、ここでずれていれば学習の開始時に例外になります。
"""
from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .align import align_left
from .loudness import dataset_stats, normalize_with_stats
from .subset import apply_subset, subset_indices

REQUIRED = ("content", "f0_hz", "uv", "loudness", "mel")
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)          # zip の既定は現在時刻。固定して bit 一致させる


def _savez_deterministic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """`np.savez` と同じ形式で、**バイト単位で再現する** npz を書く。

    `np.savez` は zip の各エントリに現在時刻を書き込むため、同じ配列でも実行ごとに
    バイト列が変わります。データ契約は「同じ入力・同じ設定で再実行したら一致」を
    要求するので、タイムスタンプを固定し、エントリ順も名前順に固定します。
    """
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for key in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[key]),
                                      allow_pickle=False)
            zf.writestr(zipfile.ZipInfo(f"{key}.npy", date_time=_FIXED_TIME), buf.getvalue())


def _validate(name: str, phrase: Mapping[str, Any], content_dim_in: int) -> int:
    missing = [k for k in REQUIRED if k not in phrase]
    if missing:
        raise ValueError(f"{name}: 足りないキー {missing}（必要: {list(REQUIRED)}）")

    mel = np.asarray(phrase["mel"])
    if mel.ndim != 2:
        raise ValueError(f"{name}: mel は [n_mels, T] であること; got {mel.shape}")
    frames = int(mel.shape[1])
    if frames == 0:
        raise ValueError(f"{name}: mel のフレーム数が 0")

    content = np.asarray(phrase["content"])
    if content.ndim != 2:
        raise ValueError(f"{name}: content は [T_ssl, C] であること; got {content.shape}")
    if int(content.shape[1]) != content_dim_in:
        raise ValueError(f"{name}: content の幅が揃っていません "
                         f"({content.shape[1]} != {content_dim_in})")

    for key in ("f0_hz", "uv", "loudness"):
        arr = np.asarray(phrase[key])
        if arr.ndim != 1 or int(arr.shape[0]) != frames:
            raise ValueError(f"{name}: {key} は mel と同じ長さの 1-D であること "
                             f"（{arr.shape} vs {frames} frames）")
    return frames


def build_shard(phrases: Mapping[str, Mapping[str, Any]], out_dir, *,
                n_dims: int, subset_seed: int, frame_rate: float,
                manifest_extra: Mapping[str, Any] | None = None) -> dict:
    """cache の phrase 群から `svc_shard.npz` / `metadata.json` / `manifest.json` を書く。

    `phrases` は `{phrase 名: {content, f0_hz, uv, loudness, mel}}`。`content` は SSL の
    frame rate（50 Hz）のままで構いません。ここで mel の frame 数へ整列します。
    戻り値は manifest（再現に要る情報一式）です。
    """
    if not phrases:
        raise ValueError("build_shard needs at least one phrase")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    first = phrases[sorted(phrases)[0]]
    if "content" not in first or np.asarray(first["content"]).ndim != 2:
        raise ValueError("content は [T_ssl, C] であること")
    content_dim_in = int(np.asarray(first["content"]).shape[1])

    frames = {name: _validate(name, phrases[name], content_dim_in) for name in sorted(phrases)}

    # 部分集合は先に決める。n_dims が content の幅を超えていればここで落ちる。
    indices = subset_indices(content_dim_in, n_dims, seed=subset_seed)

    # loudness は **全 phrase を通した統計**で正規化する（phrase 単位ではない）。
    mean, std = dataset_stats([np.asarray(phrases[n]["loudness"]) for n in sorted(phrases)])

    arrays: dict[str, np.ndarray] = {}
    for name in sorted(phrases):
        phrase, n_frames = phrases[name], frames[name]
        content = align_left(np.asarray(phrase["content"], dtype=np.float32), n_frames)
        arrays[f"{name}|content"] = apply_subset(content, indices).astype(np.float32)
        arrays[f"{name}|f0_interp"] = np.asarray(phrase["f0_hz"], dtype=np.float32)
        arrays[f"{name}|uv"] = np.asarray(phrase["uv"], dtype=np.float32)
        arrays[f"{name}|loudness"] = normalize_with_stats(
            np.asarray(phrase["loudness"]), mean, std)
        arrays[f"{name}|mel"] = np.asarray(phrase["mel"], dtype=np.float32)

    _savez_deterministic(out_dir / "svc_shard.npz", arrays)
    (out_dir / "metadata.json").write_text(json.dumps(
        {"content_dim": int(n_dims), "frame_rate": float(frame_rate), "phrases": frames},
        ensure_ascii=False, indent=1), encoding="utf-8")

    manifest: dict[str, Any] = {
        "content_dim_in": content_dim_in,
        "n_dims": int(n_dims),
        "subset_seed": int(subset_seed),
        # seed だけでなく index 自体を残す。numpy の Generator は分布メソッドのストリームが
        # バージョン間で変わり得るため、seed からの再生成は保証されない。
        "subset_indices": [int(i) for i in indices],
        "interpolation": "left",
        "frame_rate": float(frame_rate),
        "loudness_mean": float(mean),
        "loudness_std": float(std),
        # 「正規化単位」（M1 ゴール 5）。mean/std だけでは phrase 単位か dataset 単位か
        # 分からない。決定は dataset 統計なので値として残す。
        "loudness_normalization": "dataset_zscore",
        "n_phrases": len(frames),
        "total_frames": int(sum(frames.values())),
    }
    if manifest_extra:
        manifest.update(dict(manifest_extra))
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def features_to_item(features: Mapping[str, Any],
                     manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """1 段目の出力 + manifest -> `infer_svc_mel` に渡せる item。

    学習は shard を読むので正規化済みですが、**新しい source WAV から推論するときは
    manifest に記録した統計と部分集合を同じように当てる必要があります。**
    それを人手に任せると必ずずれるので、`build_shard` と同じ手順をここに閉じ込めます
    （[実行計画](../../doc/svc-plan.md) M2 ゴール 5）。

    同じ特徴と manifest を与えれば、shard に入っている値と **1 bit も違いません**。
    """
    for key in ("subset_indices", "loudness_mean", "loudness_std", "content_dim_in"):
        if key not in manifest:
            raise ValueError(f"manifest に {key} がありません。shard を作った manifest.json を渡してください")

    content = np.asarray(features["content"], dtype=np.float32)
    if content.ndim != 2:
        raise ValueError(f"content は [T_ssl, C] であること; got {content.shape}")
    if int(content.shape[1]) != int(manifest["content_dim_in"]):
        raise ValueError(f"content の幅が manifest と違います "
                         f"({content.shape[1]} != {manifest['content_dim_in']})")

    frames = (int(np.asarray(features["mel"]).shape[1]) if "mel" in features
              else int(np.asarray(features["f0_hz"]).shape[0]))
    aligned = align_left(content, frames)
    indices = np.asarray(manifest["subset_indices"], dtype=np.int64)
    f0_hz = np.asarray(features["f0_hz"], dtype=np.float32)
    return {
        "content": apply_subset(aligned, indices).astype(np.float32),
        # loader (svc_dataset) が f0_logf0 を渡すので、推論側も同じ表現にする。
        "f0_logf0": np.log2(np.maximum(f0_hz, 1.0)).astype(np.float32),
        "uv": np.asarray(features["uv"], dtype=np.float32),
        "loudness": normalize_with_stats(np.asarray(features["loudness"]),
                                         manifest["loudness_mean"], manifest["loudness_std"]),
    }
