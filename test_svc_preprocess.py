"""SVC 特徴抽出前処理のテスト。

    uv run python -m unittest test_svc_preprocess -v

重い事前学習モデル（ContentVec / RMVPE）は使わない。整列・契約・決定性をテストする。
実モデルを使う統合テストは別途 test_svc_preprocess_integration.py に置く。
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from leapsinger.mel import wav_to_mel_nhv
from preprocess.svc.align import align_left
from preprocess.svc.shard import build_shard
from preprocess.svc.loudness import dataset_stats, frame_log_rms, normalize_with_stats
from preprocess.svc.subset import apply_subset, subset_indices


class AlignLeftTests(unittest.TestCase):
    """SSL の 50 Hz を mel grid の 172.265625 Hz へ合わせる整列。

    比 3.4453125 は整数にならないので、端数の扱いが契約になる。
    doc/svc-content-encoder.md 6 節で left（直前保持）を採用した。
    """

    # 実際に使う比。1 秒ぶんに相当する長さ。
    T_SRC, T_TGT = 50, 172

    def test_returns_exactly_the_requested_frame_count(self):
        # データ契約が T の完全一致を要求するので、端数が出ても必ず target ちょうどにする。
        for t_src, t_tgt in ((self.T_SRC, self.T_TGT), (3, 11), (7, 24), (1, 4), (200, 689)):
            with self.subTest(t_src=t_src, t_tgt=t_tgt):
                out = align_left(np.zeros((t_src, 5), dtype=np.float32), t_tgt)
                self.assertEqual(out.shape, (t_tgt, 5))

    def test_every_output_frame_is_an_unmodified_input_frame(self):
        # left を選んだ理由の 1 つ。linear と違い、実在しないベクトルを作らない。
        src = np.random.default_rng(0).standard_normal((self.T_SRC, 8)).astype(np.float32)
        out = align_left(src, self.T_TGT)
        for t in range(self.T_TGT):
            self.assertTrue(any(np.array_equal(out[t], row) for row in src),
                            f"frame {t} が入力に存在しないベクトルになっている")

    def test_maps_each_output_frame_to_the_floor_of_its_source_position(self):
        # so-vits-svc の repeat_expand_2d_left と同じ index 演算であること。
        src = np.arange(self.T_SRC, dtype=np.float32).reshape(-1, 1)
        out = align_left(src, self.T_TGT)
        for t in range(self.T_TGT):
            expected = min(math.floor(t * self.T_SRC / self.T_TGT), self.T_SRC - 1)
            self.assertEqual(float(out[t, 0]), float(expected), f"frame {t}")

    def test_never_reads_a_future_source_frame(self):
        # left を選んだ最大の理由。M6 の streaming で lookahead を増やさない。
        src = np.arange(self.T_SRC, dtype=np.float32).reshape(-1, 1)
        out = align_left(src, self.T_TGT)
        for t in range(self.T_TGT):
            elapsed_src_frames = t * self.T_SRC / self.T_TGT
            self.assertLessEqual(float(out[t, 0]), elapsed_src_frames,
                                 f"frame {t} が未来の source frame を参照している")

    def test_preserves_dtype_and_channel_width(self):
        src = np.zeros((self.T_SRC, 768), dtype=np.float32)
        out = align_left(src, self.T_TGT)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape[1], 768)

    def test_upsamples_and_downsamples_with_the_same_rule(self):
        # 通常は上げる側だが、規則を分岐させない（分岐は再現性の敵）。
        src = np.arange(10, dtype=np.float32).reshape(-1, 1)
        self.assertEqual(align_left(src, 5)[:, 0].tolist(), [0.0, 2.0, 4.0, 6.0, 8.0])

    def test_rejects_input_that_is_not_2d(self):
        with self.assertRaises(ValueError):
            align_left(np.zeros(self.T_SRC, dtype=np.float32), self.T_TGT)

    def test_rejects_non_positive_target_length(self):
        for bad in (0, -1):
            with self.subTest(target=bad):
                with self.assertRaises(ValueError):
                    align_left(np.zeros((self.T_SRC, 4), dtype=np.float32), bad)

    def test_rejects_empty_source(self):
        with self.assertRaises(ValueError):
            align_left(np.zeros((0, 4), dtype=np.float32), self.T_TGT)


class SubsetIndicesTests(unittest.TestCase):
    """ContentVec 768 から学習に使う 256 次元を選ぶ。

    doc/svc-content-encoder.md 3 節。ランダムな部分集合を選んで固定するだけで、
    生の 768 次元より timbre 漏れが減る（SMOS 3.512 -> 4.038）。
    再現性のため、seed と選んだ index の両方を manifest に記録する。
    """

    TOTAL, N = 768, 256

    def test_returns_the_requested_number_of_indices(self):
        idx = subset_indices(self.TOTAL, self.N, seed=0)
        self.assertEqual(len(idx), self.N)

    def test_indices_are_sorted_and_unique(self):
        # manifest に載せて人が読むもの。並びが安定しないと差分が読めない。
        idx = subset_indices(self.TOTAL, self.N, seed=0)
        self.assertEqual(len(set(idx.tolist())), self.N)
        self.assertEqual(idx.tolist(), sorted(idx.tolist()))

    def test_indices_are_within_range(self):
        idx = subset_indices(self.TOTAL, self.N, seed=3)
        self.assertGreaterEqual(int(idx.min()), 0)
        self.assertLess(int(idx.max()), self.TOTAL)

    def test_same_seed_gives_the_same_indices(self):
        # 決定性。shard を作り直しても同じにならないと実験が比較できない。
        self.assertEqual(subset_indices(self.TOTAL, self.N, seed=0).tolist(),
                         subset_indices(self.TOTAL, self.N, seed=0).tolist())

    def test_different_seed_gives_different_indices(self):
        # M2 で seed 0 と seed 1 を比較すると決めたので、実際に違う必要がある。
        self.assertNotEqual(subset_indices(self.TOTAL, self.N, seed=0).tolist(),
                            subset_indices(self.TOTAL, self.N, seed=1).tolist())

    def test_rejects_selecting_more_dimensions_than_available(self):
        with self.assertRaises(ValueError):
            subset_indices(self.TOTAL, self.TOTAL + 1, seed=0)

    def test_rejects_non_positive_count(self):
        for bad in (0, -1):
            with self.subTest(n=bad):
                with self.assertRaises(ValueError):
                    subset_indices(self.TOTAL, bad, seed=0)

    def test_selecting_all_dimensions_is_the_identity_order(self):
        self.assertEqual(subset_indices(8, 8, seed=5).tolist(), list(range(8)))


class ApplySubsetTests(unittest.TestCase):
    def test_keeps_the_selected_columns_in_index_order(self):
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        out = apply_subset(x, np.array([1, 3]))
        self.assertEqual(out.shape, (3, 2))
        self.assertEqual(out[:, 0].tolist(), [1.0, 5.0, 9.0])
        self.assertEqual(out[:, 1].tolist(), [3.0, 7.0, 11.0])

    def test_preserves_frame_count_and_dtype(self):
        x = np.zeros((172, 768), dtype=np.float32)
        out = apply_subset(x, subset_indices(768, 256, seed=0))
        self.assertEqual(out.shape, (172, 256))
        self.assertEqual(out.dtype, np.float32)

    def test_rejects_indices_out_of_range(self):
        with self.assertRaises(ValueError):
            apply_subset(np.zeros((3, 4), dtype=np.float32), np.array([0, 4]))

    def test_rejects_input_that_is_not_2d(self):
        with self.assertRaises(ValueError):
            apply_subset(np.zeros(4, dtype=np.float32), np.array([0, 1]))


class FrameLogRmsTests(unittest.TestCase):
    """loudness をフレーム単位で出す。

    最重要の契約は「mel とフレーム数がちょうど一致すること」。mel は center=False +
    reflect pad (n_fft-hop)//2 という独特な framing なので、素直に窓を切ると必ずずれる。
    loader は暗黙に直さないので、ここがずれると学習が始まらない。
    """

    SR, HOP, N_FFT = 44100, 256, 2048

    def _wav(self, n):
        t = np.arange(n) / self.SR
        return (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def test_frame_count_matches_the_mel_grid_exactly(self):
        # 端数が出る長さを含めて、実際の mel 計算と突き合わせる。
        for n in (4096, 11025, 22050, 44100, 30011, 65537):
            with self.subTest(samples=n):
                wav = self._wav(n)
                mel = wav_to_mel_nhv(wav, sr=self.SR, n_fft=self.N_FFT, hop=self.HOP,
                                     win=self.N_FFT, n_mels=128, fmin=40.0, fmax=16000.0)
                out = frame_log_rms(wav, hop=self.HOP, n_fft=self.N_FFT)
                self.assertEqual(out.shape, (mel.shape[1],))

    def test_is_finite_even_for_digital_silence(self):
        # 無音は log(0) になる。floor が効いていないと -inf が shard に入る。
        out = frame_log_rms(np.zeros(22050, dtype=np.float32), hop=self.HOP, n_fft=self.N_FFT)
        self.assertTrue(np.isfinite(out).all())

    def test_louder_audio_gives_larger_values(self):
        quiet = frame_log_rms(self._wav(22050) * 0.1, hop=self.HOP, n_fft=self.N_FFT)
        loud = frame_log_rms(self._wav(22050), hop=self.HOP, n_fft=self.N_FFT)
        self.assertGreater(float(loud.mean()), float(quiet.mean()))

    def test_is_deterministic(self):
        wav = self._wav(22050)
        a = frame_log_rms(wav, hop=self.HOP, n_fft=self.N_FFT)
        b = frame_log_rms(wav, hop=self.HOP, n_fft=self.N_FFT)
        self.assertTrue(np.array_equal(a, b))

    def test_returns_float32(self):
        out = frame_log_rms(self._wav(22050), hop=self.HOP, n_fft=self.N_FFT)
        self.assertEqual(out.dtype, np.float32)


class DatasetStatsTests(unittest.TestCase):
    """loudness は **dataset 統計**で正規化する（phrase 単位ではない）。

    doc/svc.md の決定事項。phrase 単位だと phrase 間の強弱差が消えて歌の表情が平坦になる。
    """

    def test_statistics_are_taken_over_all_phrases_together(self):
        # phrase ごとではなく全体の平均・標準偏差であること。
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([4.0, 4.0, 4.0], dtype=np.float32)
        mean, std = dataset_stats([a, b])
        self.assertAlmostEqual(mean, 2.0, places=5)
        self.assertAlmostEqual(std, 2.0, places=5)

    def test_weights_phrases_by_their_frame_count(self):
        # 長い phrase のほうが多くのフレームを持つ。単純な phrase 平均にしない。
        short = np.array([0.0], dtype=np.float32)
        long = np.array([3.0] * 9, dtype=np.float32)
        mean, _ = dataset_stats([short, long])
        self.assertAlmostEqual(mean, 2.7, places=5)

    def test_rejects_an_empty_dataset(self):
        with self.assertRaises(ValueError):
            dataset_stats([])

    def test_normalization_centres_and_scales(self):
        x = np.array([0.0, 2.0, 4.0], dtype=np.float32)
        out = normalize_with_stats(x, 2.0, 2.0)
        self.assertEqual(out.tolist(), [-1.0, 0.0, 1.0])

    def test_normalization_survives_a_constant_dataset(self):
        # 全フレームが同じ値だと std=0。ゼロ除算で NaN を shard に入れない。
        mean, std = dataset_stats([np.full(5, 1.5, dtype=np.float32)])
        out = normalize_with_stats(np.full(5, 1.5, dtype=np.float32), mean, std)
        self.assertTrue(np.isfinite(out).all())

    def test_normalization_returns_float32(self):
        out = normalize_with_stats(np.zeros(4, dtype=np.float32), 0.0, 1.0)
        self.assertEqual(out.dtype, np.float32)


class BuildShardTests(unittest.TestCase):
    """cache（生の特徴）-> `svc_shard.npz` の 2 段目。

    実行計画 M1。**この段だけを再実行すれば補間方法と 256 次元 seed の ablation ができる**
    ように、重い抽出（ContentVec / RMVPE）とは分けています。
    """

    SR, HOP, N_MELS = 44100, 256, 128
    C_IN, N_DIMS = 768, 256
    FR = SR / HOP

    def setUp(self):
        self.out = Path(tempfile.mkdtemp(prefix="shard_"))
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)

    def _phrase(self, t_mel, seed=0):
        """mel フレーム数 t_mel の 1 phrase。content は SSL の 50 Hz なので短い。"""
        r = np.random.default_rng(seed)
        t_ssl = max(1, round(t_mel * 50.0 / self.FR))
        return {
            "content": r.standard_normal((t_ssl, self.C_IN)).astype(np.float32),
            "f0_hz": (220.0 + 10 * r.standard_normal(t_mel)).astype(np.float32),
            "uv": (r.random(t_mel) > 0.3).astype(np.float32),
            "loudness": (-3.0 + r.standard_normal(t_mel)).astype(np.float32),
            "mel": r.standard_normal((self.N_MELS, t_mel)).astype(np.float32) - 5.0,
        }

    def _phrases(self, sizes=(206, 258, 310)):
        return {f"song{i//2:02d}_{i:04d}": self._phrase(t, seed=i)
                for i, t in enumerate(sizes)}

    def _build(self, phrases=None, **kw):
        phrases = phrases if phrases is not None else self._phrases()
        opts = dict(n_dims=self.N_DIMS, subset_seed=0, frame_rate=self.FR)
        opts.update(kw)
        return build_shard(phrases, self.out, **opts)

    def test_writes_the_shard_and_metadata(self):
        self._build()
        self.assertTrue((self.out / "svc_shard.npz").exists())
        self.assertTrue((self.out / "metadata.json").exists())

    def test_every_array_of_a_phrase_has_the_mel_frame_count(self):
        # 契約の中核。loader は暗黙に直さないので、ここがずれると学習が始まらない。
        phrases = self._phrases()
        self._build(phrases)
        z = np.load(self.out / "svc_shard.npz")
        for name, p in phrases.items():
            t = p["mel"].shape[1]
            self.assertEqual(z[f"{name}|content"].shape, (t, self.N_DIMS), name)
            for key in ("f0_interp", "uv", "loudness"):
                self.assertEqual(z[f"{name}|{key}"].shape, (t,), f"{name}|{key}")
            self.assertEqual(z[f"{name}|mel"].shape, (self.N_MELS, t), name)

    def test_the_shard_is_readable_by_the_real_loader(self):
        # 最も強い検証: 実際の SVCFeatureDataset が例外なく読めること。
        from svc_dataset import SVCFeatureDataset
        self._build()
        ds = SVCFeatureDataset([str(self.out)], split="train", eval_songs=0)
        self.assertGreater(len(ds), 0)
        item = ds[0]
        self.assertEqual(item["content"].shape[1], self.N_DIMS)

    def test_metadata_declares_the_written_content_dim(self):
        self._build()
        meta = json.loads((self.out / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["content_dim"], self.N_DIMS)
        self.assertAlmostEqual(meta["frame_rate"], self.FR, places=6)
        self.assertEqual(sorted(meta["phrases"]), sorted(self._phrases()))

    def test_manifest_records_what_is_needed_to_reproduce(self):
        m = self._build()
        for key in ("subset_indices", "subset_seed", "n_dims", "interpolation",
                    "loudness_mean", "loudness_std", "frame_rate", "content_dim_in"):
            self.assertIn(key, m, key)
        self.assertEqual(m["interpolation"], "left")
        self.assertEqual(len(m["subset_indices"]), self.N_DIMS)

    def test_loudness_is_normalised_with_dataset_statistics(self):
        # phrase 単位ではない。全 phrase を通した平均が 0・標準偏差が 1 に近づく。
        self._build()
        z = np.load(self.out / "svc_shard.npz")
        vals = np.concatenate([z[k] for k in z.files if k.endswith("|loudness")])
        self.assertAlmostEqual(float(vals.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(vals.std()), 1.0, places=4)

    def test_is_deterministic(self):
        a = self.out / "svc_shard.npz"
        self._build(self._phrases())
        first = a.read_bytes()
        self._build(self._phrases())
        self.assertEqual(first, a.read_bytes())

    def test_rejects_a_phrase_whose_frame_arrays_disagree_with_the_mel(self):
        phrases = self._phrases()
        name = sorted(phrases)[0]
        phrases[name]["uv"] = phrases[name]["uv"][:-1]
        with self.assertRaises(ValueError):
            self._build(phrases)

    def test_rejects_content_with_an_unexpected_width(self):
        phrases = self._phrases()
        name = sorted(phrases)[0]
        phrases[name]["content"] = phrases[name]["content"][:, : self.C_IN - 1]
        with self.assertRaises(ValueError):
            self._build(phrases)

    def test_rejects_more_dims_than_the_content_has(self):
        with self.assertRaises(ValueError):
            self._build(n_dims=self.C_IN + 1)

    def test_rejects_an_empty_phrase_set(self):
        with self.assertRaises(ValueError):
            self._build({})

    def test_a_different_subset_seed_changes_the_written_content(self):
        # ablation が 2 段目の再実行だけで回せること。
        self._build(subset_seed=0)
        a = np.load(self.out / "svc_shard.npz")
        first = {k: a[k].copy() for k in a.files if k.endswith("|content")}
        self._build(subset_seed=1)
        b = np.load(self.out / "svc_shard.npz")
        name = sorted(first)[0]
        self.assertFalse(np.array_equal(first[name], b[name]))


if __name__ == "__main__":
    unittest.main()
