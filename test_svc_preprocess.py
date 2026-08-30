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

from leapsinger.config import MelSpec
from leapsinger.mel import wav_to_mel_nhv
from preprocess.svc.align import align_left
from preprocess.svc.chunk import chunk_spans, voiced_ratio
from preprocess.svc.extract import extract_phrase
from preprocess.svc.shard import build_shard, features_to_item
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

    def test_manifest_records_the_loudness_normalisation_unit(self):
        # M1 ゴール 5 の「正規化単位」。mean/std だけでは phrase 単位か dataset 単位か
        # 分からない。決定は dataset 統計なので、その旨を値として残す。
        m = self._build()
        self.assertEqual(m["loudness_normalization"], "dataset_zscore")

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


class ExtractPhraseTests(unittest.TestCase):
    """WAV -> cache の 1 段目。

    重い ContentVec / RMVPE は**引数で受け取り**ます。関数の内側で `from_pretrained` すると
    単体テストが書けなくなるためです（`leapsinger-tdd` skill）。ここでテストするのは
    「整列と契約」であって encoder の中身ではありません。
    """

    MEL = MelSpec()          # 44,100 / hop 256 / 128 mel
    C_IN = 768
    ENCODER_SR = 16000

    def _wav(self, seconds=2.0, sr=None):
        sr = sr or self.MEL.sr
        t = np.arange(int(seconds * sr)) / sr
        return (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def _fake_encoder(self, record=None):
        """16 kHz・stride 320 の SSL を模した偽 encoder。値は本物を真似ない。"""
        def enc(wav16, sr):
            if record is not None:
                record.append((len(wav16), sr))
            frames = max(1, len(wav16) // 320)
            return np.tile(np.arange(frames, dtype=np.float32)[:, None], (1, self.C_IN))
        return enc

    def _fake_f0(self, frames_for):
        def f0(wav, sr, hop):
            n = frames_for(wav, sr, hop)
            return (np.full(n, 220.0, np.float32), np.ones(n, np.float32))
        return f0

    def _mel_frames(self, wav, sr, hop):
        return wav_to_mel_nhv(wav, sr=self.MEL.sr, n_fft=self.MEL.n_fft, hop=self.MEL.hop,
                              win=self.MEL.win, n_mels=self.MEL.n_mels,
                              fmin=self.MEL.fmin, fmax=self.MEL.fmax).shape[1]

    def _extract(self, wav=None, sr=None, **kw):
        wav = self._wav() if wav is None else wav
        opts = dict(content_encoder=self._fake_encoder(), f0_extract=self._fake_f0(self._mel_frames),
                    mel=self.MEL, encoder_sr=self.ENCODER_SR)
        opts.update(kw)
        return extract_phrase(wav, sr or self.MEL.sr, **opts)

    def test_returns_every_key_build_shard_needs(self):
        out = self._extract()
        for key in ("content", "f0_hz", "uv", "loudness", "mel"):
            self.assertIn(key, out)

    def test_frame_arrays_all_match_the_mel(self):
        out = self._extract()
        frames = out["mel"].shape[1]
        for key in ("f0_hz", "uv", "loudness"):
            self.assertEqual(out[key].shape, (frames,), key)

    def test_content_stays_on_the_ssl_grid(self):
        # 整列は 2 段目の仕事。ここでは SSL のフレーム数のまま返す。
        out = self._extract()
        self.assertEqual(out["content"].shape[1], self.C_IN)
        self.assertLess(out["content"].shape[0], out["mel"].shape[1])

    def test_resamples_to_the_encoder_rate_before_calling_it(self):
        # ContentVec は 16 kHz を前提にする。44.1 kHz のまま渡すと無意味な特徴になる。
        seen = []
        self._extract(content_encoder=self._fake_encoder(seen))
        self.assertEqual(len(seen), 1)
        got_len, got_sr = seen[0]
        self.assertEqual(got_sr, self.ENCODER_SR)
        self.assertAlmostEqual(got_len / self.ENCODER_SR, 2.0, places=1)

    def test_resamples_a_48k_input_to_the_mel_rate(self):
        # 素材は 44.1k / 48k / 96k が混在する。mel 設定は共有なので入口で揃える。
        out = self._extract(wav=self._wav(sr=48000), sr=48000)
        self.assertEqual(out["mel"].shape[1], self._mel_frames(self._wav(), self.MEL.sr, self.MEL.hop))

    def test_output_feeds_build_shard(self):
        # 1 段目と 2 段目が実際につながることの確認。
        out = self._extract()
        d = Path(tempfile.mkdtemp(prefix="ex_")); self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        m = build_shard({"songA_0000": out}, d, n_dims=64, subset_seed=0,
                        frame_rate=self.MEL.frame_rate)
        self.assertEqual(m["n_dims"], 64)
        z = np.load(d / "svc_shard.npz")
        self.assertEqual(z["songA_0000|content"].shape, (out["mel"].shape[1], 64))

    def test_is_deterministic(self):
        a, b = self._extract(), self._extract()
        for k in ("content", "f0_hz", "uv", "loudness", "mel"):
            self.assertTrue(np.array_equal(a[k], b[k]), k)

    def test_rejects_a_non_mono_input(self):
        with self.assertRaises(ValueError):
            self._extract(wav=np.zeros((2, 1000), dtype=np.float32))

    def test_rejects_an_f0_extractor_that_returns_the_wrong_length(self):
        # 契約違反は黙って直さずに落とす。
        bad = lambda wav, sr, hop: (np.zeros(3, np.float32), np.zeros(3, np.float32))
        with self.assertRaises(ValueError):
            self._extract(f0_extract=bad)

    def test_rejects_an_encoder_that_returns_a_non_2d_array(self):
        bad = lambda wav16, sr: np.zeros(10, dtype=np.float32)
        with self.assertRaises(ValueError):
            self._extract(content_encoder=bad)


class ChunkSpansTests(unittest.TestCase):
    """長い曲を phrase へ切る。

    SVS 側は `.lab` の音素境界で切りますが、SVC は音素ラベルを使わないので固定長で切ります。
    名前は `{song}_{NNNN}` にする必要があります（`dataset.py` の `_song_of()` が曲単位の
    分割に使うため。崩すと leakage 防止が効かなくなります）。
    """

    SR = 44100

    def test_covers_the_signal_in_order_without_gaps(self):
        spans = chunk_spans(self.SR * 10, self.SR, chunk_sec=3.0, min_sec=0.5)
        self.assertEqual(spans[0][0], 0)
        for (a, b), (c, d) in zip(spans, spans[1:]):
            self.assertEqual(b, c, "隙間や重複がある")
            self.assertLess(a, b)

    def test_every_chunk_is_the_requested_length_except_the_tail(self):
        spans = chunk_spans(self.SR * 10, self.SR, chunk_sec=3.0, min_sec=0.5)
        for a, b in spans[:-1]:
            self.assertEqual(b - a, int(3.0 * self.SR))

    def test_drops_a_tail_shorter_than_the_minimum(self):
        # 10.2 秒を 3 秒で切ると末尾 1.2 秒。min_sec 2.0 なら捨てる。
        spans = chunk_spans(int(self.SR * 10.2), self.SR, chunk_sec=3.0, min_sec=2.0)
        self.assertEqual(len(spans), 3)
        self.assertEqual(spans[-1][1], int(3.0 * self.SR) * 3)

    def test_keeps_a_tail_at_or_above_the_minimum(self):
        spans = chunk_spans(int(self.SR * 10.2), self.SR, chunk_sec=3.0, min_sec=1.0)
        self.assertEqual(len(spans), 4)
        self.assertEqual(spans[-1][1], int(self.SR * 10.2))

    def test_returns_nothing_for_a_signal_shorter_than_the_minimum(self):
        self.assertEqual(chunk_spans(int(self.SR * 0.2), self.SR, chunk_sec=3.0, min_sec=0.5), [])

    def test_a_signal_shorter_than_a_chunk_but_long_enough_is_one_span(self):
        n = int(self.SR * 1.5)
        self.assertEqual(chunk_spans(n, self.SR, chunk_sec=3.0, min_sec=0.5), [(0, n)])

    def test_is_deterministic(self):
        a = chunk_spans(self.SR * 37, self.SR, chunk_sec=4.0, min_sec=1.0)
        b = chunk_spans(self.SR * 37, self.SR, chunk_sec=4.0, min_sec=1.0)
        self.assertEqual(a, b)

    def test_rejects_non_positive_chunk_length(self):
        for bad in (0.0, -1.0):
            with self.subTest(chunk_sec=bad):
                with self.assertRaises(ValueError):
                    chunk_spans(self.SR, self.SR, chunk_sec=bad, min_sec=0.5)

    def test_rejects_a_minimum_longer_than_the_chunk(self):
        with self.assertRaises(ValueError):
            chunk_spans(self.SR * 10, self.SR, chunk_sec=1.0, min_sec=2.0)


class FeaturesToItemTests(unittest.TestCase):
    """1 段目の出力 + manifest -> `infer_svc_mel` に渡せる item。

    実行計画 M2 ゴール 5「学習時と推論時で特徴量の正規化が同一である」を**構造で保証する**
    ための関数です。学習は shard を読むので正規化済みですが、新しい source WAV から推論する
    ときは manifest に記録した統計と部分集合を同じように当てる必要があります。
    それを人手に任せると必ずずれます。
    """

    SR, HOP, N_MELS = 44100, 256, 128
    C_IN, N_DIMS = 64, 16
    FR = SR / HOP

    def setUp(self):
        self.out = Path(tempfile.mkdtemp(prefix="fti_"))
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)

    def _features(self, t_mel, seed=0):
        r = np.random.default_rng(seed)
        t_ssl = max(1, round(t_mel * 50.0 / self.FR))
        return {
            "content": r.standard_normal((t_ssl, self.C_IN)).astype(np.float32),
            "f0_hz": (220.0 + 10 * r.standard_normal(t_mel)).astype(np.float32),
            "uv": (r.random(t_mel) > 0.3).astype(np.float32),
            "loudness": (-3.0 + r.standard_normal(t_mel)).astype(np.float32),
            "mel": r.standard_normal((self.N_MELS, t_mel)).astype(np.float32) - 5.0,
        }

    def _built(self):
        phrases = {"songA_0000": self._features(206, 0), "songA_0001": self._features(258, 1)}
        manifest = build_shard(phrases, self.out, n_dims=self.N_DIMS, subset_seed=0,
                               frame_rate=self.FR)
        return phrases, manifest

    def test_reproduces_exactly_what_the_shard_holds(self):
        # これが本題。同じ特徴から作った item が、shard の中身と 1 bit も違わないこと。
        phrases, manifest = self._built()
        z = np.load(self.out / "svc_shard.npz")
        for name, feats in phrases.items():
            item = features_to_item(feats, manifest)
            self.assertTrue(np.array_equal(item["content"], z[f"{name}|content"]), name)
            self.assertTrue(np.array_equal(item["loudness"], z[f"{name}|loudness"]), name)
            self.assertTrue(np.array_equal(item["uv"], z[f"{name}|uv"]), name)

    def test_converts_f0_to_log2(self):
        # loader (svc_dataset) が f0_logf0 を渡すので、推論側も同じ表現にする。
        feats = self._features(206)
        _, manifest = self._built()
        item = features_to_item(feats, manifest)
        self.assertTrue(np.allclose(item["f0_logf0"],
                                    np.log2(np.maximum(feats["f0_hz"], 1.0)), atol=1e-6))

    def test_returns_the_keys_infer_svc_mel_needs(self):
        feats = self._features(206)
        _, manifest = self._built()
        item = features_to_item(feats, manifest)
        for key in ("content", "f0_logf0", "uv", "loudness"):
            self.assertIn(key, item)

    def test_content_is_aligned_and_reduced(self):
        feats = self._features(206)
        _, manifest = self._built()
        item = features_to_item(feats, manifest)
        self.assertEqual(item["content"].shape, (feats["mel"].shape[1], self.N_DIMS))

    def test_rejects_a_manifest_whose_indices_do_not_fit_the_content(self):
        feats = self._features(206)
        _, manifest = self._built()
        manifest = {**manifest, "content_dim_in": self.C_IN + 8}
        with self.assertRaises(ValueError):
            features_to_item(feats, manifest)

    def test_rejects_a_manifest_missing_the_normalisation(self):
        feats = self._features(206)
        _, manifest = self._built()
        broken = {k: v for k, v in manifest.items() if k != "loudness_mean"}
        with self.assertRaises(ValueError):
            features_to_item(feats, broken)


class VoicedRatioTests(unittest.TestCase):
    """無声だけの chunk を弾くための判定。

    実データで見つかった問題: 曲を先頭から固定長で切ると、イントロや間奏が丸ごと
    無声の phrase になる。波音リツ 1 曲を 3 秒で切ったところ **89 phrase 中 35 件（39%）が
    完全に無声**で、先頭 9 個（27 秒）は連続して無声だった。これを学習に入れると
    「無音を出す」ことを学ぶ。
    """

    def test_returns_the_fraction_of_voiced_frames(self):
        self.assertAlmostEqual(voiced_ratio(np.array([1, 1, 0, 0], dtype=np.float32)), 0.5)
        self.assertAlmostEqual(voiced_ratio(np.array([1, 1, 1, 1], dtype=np.float32)), 1.0)
        self.assertAlmostEqual(voiced_ratio(np.zeros(10, dtype=np.float32)), 0.0)

    def test_treats_values_above_half_as_voiced(self):
        # uv は 0/1 で入るが、float なので閾値を明示しておく。
        self.assertAlmostEqual(voiced_ratio(np.array([0.6, 0.4], dtype=np.float32)), 0.5)

    def test_returns_zero_for_an_empty_array(self):
        self.assertEqual(voiced_ratio(np.zeros(0, dtype=np.float32)), 0.0)

    def test_rejects_a_non_1d_array(self):
        with self.assertRaises(ValueError):
            voiced_ratio(np.zeros((2, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()


class LoudnessManifestTests(unittest.TestCase):
    """loudness の**定義**を manifest に残す。

    実行計画 M1 ゴール 5 は「loudness の定義（窓幅・floor・正規化単位）」を manifest に
    要求します。mean/std だけでは、窓幅や floor を後から変えたときに古い shard と
    見分けがつきません。抽出条件は以降の全実験の比較基盤なので、値そのものを残します。
    """

    def test_records_the_window_hop_and_floor(self):
        from preprocess.svc.loudness import loudness_manifest
        m = loudness_manifest(hop=256, n_fft=2048)
        self.assertEqual(m["loudness_n_fft"], 2048)
        self.assertEqual(m["loudness_hop"], 256)
        self.assertIn("loudness_floor", m)
        self.assertIn("loudness_definition", m)

    def test_the_recorded_floor_is_the_one_frame_log_rms_actually_uses(self):
        # manifest が実装と食い違ったら、記録の意味が無くなる。
        from preprocess.svc.loudness import loudness_manifest
        floor = loudness_manifest(hop=256, n_fft=2048)["loudness_floor"]
        silence = frame_log_rms(np.zeros(22050, dtype=np.float32), hop=256, n_fft=2048)
        self.assertAlmostEqual(float(silence[0]), math.log(floor), places=5)

    def test_records_an_explicit_floor_when_one_is_given(self):
        from preprocess.svc.loudness import loudness_manifest
        m = loudness_manifest(hop=256, n_fft=2048, floor=1e-4)
        self.assertAlmostEqual(m["loudness_floor"], 1e-4)


class RmvpeManifestTests(unittest.TestCase):
    """F0 抽出器の **version** を manifest に残す（M1 ゴール 5）。

    `"rmvpe"` という名前だけでは version になりません。RMVPE の重みは初回実行時に
    HuggingFace から落ちてくるので、**重みそのものの checksum と入手元**を残します。
    重みが未取得の環境でも manifest 生成が落ちないこと（記録は残す）。
    """

    def test_records_the_weight_source_and_checksum(self):
        from preprocess.svc.encoders import RmvpeF0
        m = RmvpeF0().manifest()
        self.assertEqual(m["f0_extractor"], "rmvpe")
        self.assertIn("f0_extractor_weight_url", m)
        self.assertIn("f0_extractor_weight_sha256", m)

    def test_the_checksum_matches_the_weight_file_when_it_exists(self):
        import hashlib
        from preprocess.svc.encoders import RMVPE_WEIGHT_PATH, RmvpeF0
        if not RMVPE_WEIGHT_PATH.exists():
            self.skipTest("RMVPE の重みが未取得")
        h = hashlib.sha256(RMVPE_WEIGHT_PATH.read_bytes()).hexdigest()
        self.assertEqual(RmvpeF0().manifest()["f0_extractor_weight_sha256"], h)


class SongNameTests(unittest.TestCase):
    """WAV のパス -> phrase 名の `{song}` 部分（M1 ゴール 3）。

    `dataset.py` の `_song_of()` は末尾の `_NNNN` を落として曲名にします。`song_name()` の
    出力が `_NNNN` で終わると曲名の一部まで削られ、**別の曲が同じ曲名に潰れて
    train/eval 分割の leakage 防止が効かなくなります**。往復することを契約にします。
    """

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="songname_"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _name(self, rel: str) -> str:
        from preprocess.svc.run import song_name
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return song_name(path, self.root)

    def test_the_phrase_name_round_trips_through_song_of(self):
        from dataset import _song_of
        for rel in ("song_a/vocal.wav", "01_intro.wav", "track_0001/vocal.wav",
                    "曲 名 (2)/x.wav", "sample_12345.wav"):
            with self.subTest(rel=rel):
                song = self._name(rel)
                self.assertEqual(_song_of(f"{song}_0000"), song)

    def test_uses_the_parent_directory_when_the_file_sits_in_one(self):
        self.assertEqual(self._name("kokoronashi/vocal.wav"), "kokoronashi")

    def test_uses_the_file_stem_at_the_root(self):
        self.assertEqual(self._name("kokoronashi.wav"), "kokoronashi")

    def test_different_songs_do_not_collide(self):
        # 潰れると別の曲が同じ曲名になり、曲単位 split が leakage する。
        a = self._name("song_0001/vocal.wav")
        b = self._name("song_0002/vocal.wav")
        self.assertNotEqual(a, b)

    def test_strips_characters_that_are_unsafe_in_an_npz_key(self):
        # shard の key は `<name>|content`。区切り文字・空白・記号は落とす。
        # **CJK は落とさない。** ASCII だけに削ると日本語題の曲がすべて同じ名前に潰れ、
        # 曲単位 split が効かなくなる（実測: GTSinger 1,922 ファイル中 1,723 件が 1 名に潰れた）。
        name = self._name("さよならの夏 (歌集バージョン)/vocal.wav")
        self.assertNotRegex(name, r"[|/\\\s()~]")
        self.assertIn("さよならの夏", name)

    def test_titles_that_differ_only_in_cjk_stay_distinct(self):
        self.assertNotEqual(self._name("心做し/vocal.wav"), self._name("愛してる/vocal.wav"))

    def test_an_ascii_title_is_unchanged_by_the_cjk_support(self):
        # 既存の Ritsu の名前を変えないこと（`+` は落ちる、が従来どおり）。
        self.assertEqual(self._name("1st_color+3_normal/x.wav"), "1st_color_3_normal")


class PhraseNameCollisionTests(unittest.TestCase):
    """入れ子のコーパスで phrase 名が衝突しないこと（M1 ゴール 3 の続き）。

    GTSinger は `<言語>/<歌手>/<技法>/<曲>/<Group>/0000.wav` という深い構造で、
    ファイル名は曲をまたいで `0000.wav` から振り直されます。`song_name()` が親ディレクトリ
    （`Control_Group`）を返し、連番がファイルごとに 0 へ戻る実装だと、**別の曲の phrase が
    同じ名前になって cache を黙って上書き**します（実測: 40 ファイルが 3 名に潰れた）。

    上書きは例外にもならず、shard のフレーズ数が減るだけなので気づけません。
    名前の一意性を契約としてテストします。
    """

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="collide_"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _touch(self, rel: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        return p

    def test_song_name_can_select_path_components(self):
        # GTSinger の 1 歌手ぶんを root にすると、相対パスは <技法>/<曲>/<Group>/x.wav。
        # 曲は 1 番目の要素。技法と Group をまたいで**同じ曲は同じ song 名**にする
        # （別々にすると同じ曲が train と eval に分かれて leakage する）。
        from preprocess.svc.run import song_name
        a = self._touch("Breathy/kokoronashi/Breathy_Group/0000.wav")
        b = self._touch("Vibrato/kokoronashi/Control_Group/0003.wav")
        self.assertEqual(song_name(a, self.root, parts=(1,)),
                         song_name(b, self.root, parts=(1,)))
        self.assertEqual(song_name(a, self.root, parts=(1,)), "kokoronashi")

    def test_song_name_can_join_several_components(self):
        from preprocess.svc.run import song_name
        p = self._touch("Breathy/kokoronashi/Breathy_Group/0000.wav")
        # 表記ゆれを畳むため casefold される（SongNameNormalisationTests 参照）。
        self.assertEqual(song_name(p, self.root, parts=(0, 1)), "breathy_kokoronashi")

    def test_song_name_without_parts_keeps_the_old_behaviour(self):
        # 既存の Ritsu 配置（<曲>/<曲>.wav）を壊さない。
        from preprocess.svc.run import song_name
        p = self._touch("1st_color/1st_color.wav")
        self.assertEqual(song_name(p, self.root), "1st_color")

    def test_the_phrase_counter_continues_across_files_of_the_same_song(self):
        # ここが上書きの原因。ファイルごとに 0 へ戻すと 2 つ目のファイルが 1 つ目を潰す。
        from preprocess.svc.run import phrase_name
        counters: dict[str, int] = {}
        first = [phrase_name(counters, "kokoronashi") for _ in range(3)]
        second = [phrase_name(counters, "kokoronashi") for _ in range(2)]
        self.assertEqual(first, ["kokoronashi_0000", "kokoronashi_0001", "kokoronashi_0002"])
        self.assertEqual(second, ["kokoronashi_0003", "kokoronashi_0004"])

    def test_different_songs_count_independently(self):
        from preprocess.svc.run import phrase_name
        counters: dict[str, int] = {}
        self.assertEqual(phrase_name(counters, "a"), "a_0000")
        self.assertEqual(phrase_name(counters, "b"), "b_0000")
        self.assertEqual(phrase_name(counters, "a"), "a_0001")

    def test_the_generated_names_still_round_trip_through_song_of(self):
        from dataset import _song_of
        from preprocess.svc.run import phrase_name, song_name
        counters: dict[str, int] = {}
        p = self._touch("Breathy/kokoronashi/Breathy_Group/0000.wav")
        song = song_name(p, self.root, parts=(1,))
        self.assertEqual(_song_of(phrase_name(counters, song)), song)

    def test_a_whole_nested_tree_produces_unique_names(self):
        # 実データと同じ形。40 ファイルが 3 名に潰れていた退行を止める。
        from preprocess.svc.run import phrase_name, song_name
        paths = [self._touch(f"{tech}/{song}/{tech}_Group/{i:04d}.wav")
                 for tech in ("Breathy", "Vibrato")
                 for song in ("songA", "songB")
                 for i in range(5)]
        counters: dict[str, int] = {}
        names = [phrase_name(counters, song_name(p, self.root, parts=(1,))) for p in paths]
        self.assertEqual(len(set(names)), len(paths), "phrase 名が衝突している")


class NonAsciiPhraseNameTests(unittest.TestCase):
    """日本語の曲名を持つ phrase が shard を往復できること。

    曲名を ASCII に削ると日本語題の曲が 1 つの名前に潰れ、曲単位 split が効きません。
    そこで CJK を残すことにしたので、**npz の key と loader がそれを扱えること**を契約にします。
    """

    SR, HOP, N_MELS = 44100, 256, 128
    C_IN, N_DIMS = 32, 8
    FR = SR / HOP

    def setUp(self):
        self.out = Path(tempfile.mkdtemp(prefix="cjk_"))
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)

    def _phrase(self, t_mel, seed=0):
        r = np.random.default_rng(seed)
        t_ssl = max(1, round(t_mel * 50.0 / self.FR))
        return {
            "content": r.standard_normal((t_ssl, self.C_IN)).astype(np.float32),
            "f0_hz": (220.0 + 10 * r.standard_normal(t_mel)).astype(np.float32),
            "uv": (r.random(t_mel) > 0.3).astype(np.float32),
            "loudness": (-3.0 + r.standard_normal(t_mel)).astype(np.float32),
            "mel": r.standard_normal((self.N_MELS, t_mel)).astype(np.float32) - 5.0,
        }

    def test_a_japanese_song_name_survives_the_shard_and_the_loader(self):
        from svc_dataset import SVCFeatureDataset
        phrases = {"さよならの夏_0000": self._phrase(206, 0),
                   "さよならの夏_0001": self._phrase(258, 1),
                   "心做し_0000": self._phrase(310, 2)}
        build_shard(phrases, self.out, n_dims=self.N_DIMS, subset_seed=0, frame_rate=self.FR)
        z = np.load(self.out / "svc_shard.npz")
        self.assertIn("さよならの夏_0000|content", z.files)
        ds = SVCFeatureDataset([str(self.out)], split="train", eval_songs=0)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0]["content"].shape[1], self.N_DIMS)

    def test_the_split_still_groups_by_song(self):
        # 日本語題でも `_song_of()` が曲名を取り出せること。ここが崩れると leakage する。
        from dataset import _song_of
        self.assertEqual(_song_of("さよならの夏_0012"), "さよならの夏")


class SongNameNormalisationTests(unittest.TestCase):
    """曲名の表記ゆれを畳む。

    GTSinger の JA-Tenor-1 には `Heartful_Song` と `Heartful_song` が同居しています
    （実測）。別の曲として扱うと、同じ曲が train と eval に分かれて leakage します。
    M0 でも「曲名を正規化しないと leakage する」ことを実データで確認しています。
    """

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="songnorm_"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _name(self, rel: str, parts=None) -> str:
        from preprocess.svc.run import song_name
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        return song_name(p, self.root, parts)

    def test_case_variants_of_the_same_title_collapse(self):
        self.assertEqual(self._name("Breathy/Heartful_Song/G/0000.wav", (1,)),
                         self._name("Vibrato/Heartful_song/G/0000.wav", (1,)))

    def test_cjk_titles_are_unaffected(self):
        self.assertEqual(self._name("心做し/x.wav"), "心做し")


class SelectByBudgetTests(unittest.TestCase):
    """歌手ごとに使う分量を揃えて選ぶ（M3 の base 事前学習）。

    base 事前学習で効くのは総時間より**話者の多様性**です。GTSinger 全 9 言語 80.59 h を
    そのまま抽出すると 12 時間以上かかるので、1 歌手あたりの上限を決めて、
    **曲をまたいでラウンドロビン**で選びます。先頭から順に取ると、技法や曲が
    アルファベット順に偏ります。
    """

    def _entries(self, spec):
        return [(song, f"{song}/{i:04d}.wav", sec)
                for song, n, sec in spec for i in range(n)]

    def test_stays_within_the_budget(self):
        from preprocess.svc.run import select_by_budget
        entries = self._entries([("a", 10, 10.0), ("b", 10, 10.0)])
        picked = select_by_budget(entries, 50.0)
        self.assertLessEqual(sum(e[2] for e in picked), 50.0)
        self.assertGreater(len(picked), 0)

    def test_spreads_across_songs_instead_of_taking_the_first_ones(self):
        from preprocess.svc.run import select_by_budget
        entries = self._entries([("a", 10, 10.0), ("b", 10, 10.0), ("c", 10, 10.0)])
        songs = {e[0] for e in select_by_budget(entries, 60.0)}
        self.assertEqual(songs, {"a", "b", "c"}, "曲が偏っている")

    def test_returns_everything_when_the_budget_is_not_set(self):
        from preprocess.svc.run import select_by_budget
        entries = self._entries([("a", 3, 10.0)])
        self.assertEqual(select_by_budget(entries, 0), entries)
        self.assertEqual(select_by_budget(entries, None), entries)

    def test_returns_everything_when_the_budget_exceeds_the_material(self):
        from preprocess.svc.run import select_by_budget
        entries = self._entries([("a", 3, 10.0), ("b", 2, 10.0)])
        self.assertEqual(len(select_by_budget(entries, 10_000.0)), 5)

    def test_is_deterministic(self):
        from preprocess.svc.run import select_by_budget
        entries = self._entries([("a", 7, 3.0), ("b", 5, 4.0), ("c", 9, 2.0)])
        self.assertEqual(select_by_budget(entries, 40.0), select_by_budget(entries, 40.0))

    def test_keeps_the_original_order_of_the_selection(self):
        # cache のファイル名は採番順に決まる。並びが安定しないと bit 一致しない。
        from preprocess.svc.run import select_by_budget
        entries = self._entries([("a", 4, 5.0), ("b", 4, 5.0)])
        picked = select_by_budget(entries, 30.0)
        self.assertEqual(picked, [e for e in entries if e in picked])


class LoudnessMatchGainTests(unittest.TestCase):
    """入力の音量を学習分布へ合わせる係数。

    配信用に整えられた音源は学習素材よりずっと大きく、loudness 条件が学習分布の外に出ます
    （実測: YouTube から取った自作音源は **+1.40σ**、波音リツ DB は peak 0.107）。
    その状態で変換すると高域が削れます（spectral centroid が上限比 −47%。合わせると −22%）。

    **推論時に波形を勝手に加工しない**のが原則ですが、これは「学習分布へ合わせる」ための
    明示的な操作で、無加工が正しい前提そのものを崩さないよう **opt-in** にします。
    """

    SR, HOP, N_FFT = 44100, 256, 2048

    def _wav(self, amp, n=44100):
        t = np.arange(n) / self.SR
        return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def _manifest(self, mean, std=1.69):
        return {"loudness_mean": mean, "loudness_std": std}

    def test_returns_one_when_the_level_already_matches(self):
        from preprocess.svc.loudness import frame_log_rms, loudness_match_gain
        w = self._wav(0.1)
        mean = float(frame_log_rms(w, hop=self.HOP, n_fft=self.N_FFT).mean())
        g = loudness_match_gain(w, self._manifest(mean), hop=self.HOP, n_fft=self.N_FFT)
        self.assertAlmostEqual(g, 1.0, places=3)

    def test_scales_a_too_loud_input_down(self):
        from preprocess.svc.loudness import frame_log_rms, loudness_match_gain
        quiet, loud = self._wav(0.05), self._wav(0.9)
        mean = float(frame_log_rms(quiet, hop=self.HOP, n_fft=self.N_FFT).mean())
        g = loudness_match_gain(loud, self._manifest(mean), hop=self.HOP, n_fft=self.N_FFT)
        self.assertLess(g, 1.0)

    def test_the_scaled_input_lands_on_the_training_mean(self):
        # これが本題。合わせたあとの loudness が学習分布の平均へ寄ること。
        from preprocess.svc.loudness import frame_log_rms, loudness_match_gain
        target = self._wav(0.08)
        mean = float(frame_log_rms(target, hop=self.HOP, n_fft=self.N_FFT).mean())
        loud = self._wav(0.95)
        g = loudness_match_gain(loud, self._manifest(mean), hop=self.HOP, n_fft=self.N_FFT)
        got = float(frame_log_rms((loud * g).astype(np.float32),
                                  hop=self.HOP, n_fft=self.N_FFT).mean())
        self.assertAlmostEqual(got, mean, places=2)

    def test_is_deterministic(self):
        from preprocess.svc.loudness import loudness_match_gain
        w, m = self._wav(0.5), self._manifest(-4.6)
        self.assertEqual(loudness_match_gain(w, m, hop=self.HOP, n_fft=self.N_FFT),
                         loudness_match_gain(w, m, hop=self.HOP, n_fft=self.N_FFT))

    def test_survives_digital_silence(self):
        from preprocess.svc.loudness import loudness_match_gain
        g = loudness_match_gain(np.zeros(22050, dtype=np.float32), self._manifest(-4.6),
                                hop=self.HOP, n_fft=self.N_FFT)
        self.assertTrue(np.isfinite(g) and g > 0)

    def test_rejects_a_manifest_without_the_statistics(self):
        from preprocess.svc.loudness import loudness_match_gain
        with self.assertRaises(ValueError):
            loudness_match_gain(self._wav(0.5), {}, hop=self.HOP, n_fft=self.N_FFT)
