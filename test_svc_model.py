import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import torch

from infer import _build_from_config, infer_svc_mel
from leapsinger.models.svc import HarmonicSVCModel
from leapsinger.modules.encoders.content_adapter import ContentAdapter
from svc_dataset import SVCFeatureDataset, svc_collate_fn


class ContentAdapterTests(unittest.TestCase):
    def test_masks_padding_and_keeps_valid_frames_independent(self):
        torch.manual_seed(7)
        adapter = ContentAdapter(6, 8, n_layers=2, dropout=0.0).eval()
        features = torch.randn(2, 5, 6)
        mask = torch.tensor([
            [False, False, False, True, True],
            [False, False, False, False, False],
        ])

        changed_padding = features.clone()
        changed_padding[0, 3:] = 1000.0
        actual = adapter(features, mask)
        changed = adapter(changed_padding, mask)

        self.assertEqual((2, 8, 5), tuple(actual.shape))
        torch.testing.assert_close(actual[0, :, :3], changed[0, :, :3])
        self.assertTrue(torch.count_nonzero(actual[0, :, 3:]) == 0)

    def test_rejects_wrong_feature_width(self):
        adapter = ContentAdapter(6, 8)
        with self.assertRaisesRegex(ValueError, "feature dim"):
            adapter(torch.zeros(1, 4, 5))


class HarmonicSVCModelTests(unittest.TestCase):
    def _model(self):
        return HarmonicSVCModel(
            content_dim=6,
            content_layers=1,
            content_dropout=0.0,
            hidden=8,
            mel_bins=128,
            backbone_ch=8,
            n_cycles=1,
            dilation_schedule="pow2_15",
            n_speakers=2,
            spk_dim=4,
            n_harm=1,
            noise_ratio=0.0,
        ).eval()

    def test_encodes_frame_aligned_svc_conditioning(self):
        torch.manual_seed(11)
        model = self._model()
        content = torch.randn(2, 5, 6)
        f0 = torch.randn(2, 5)
        uv = torch.ones(2, 5)
        loudness = torch.randn(2, 5)
        mask = torch.tensor([
            [False, False, False, True, True],
            [False, False, False, False, False],
        ])

        cond = model._encode(
            content, f0, uv, loudness, mask,
            max_frames=5, spk_id=torch.tensor([0, 1]),
        )

        self.assertEqual((2, 8, 5), tuple(cond.shape))
        self.assertTrue(torch.count_nonzero(cond[0, :, 3:]) == 0)

    def test_rejects_unaligned_content(self):
        model = self._model()
        with self.assertRaisesRegex(ValueError, "aligned to 6 mel frames"):
            model._encode(
                torch.zeros(1, 5, 6),
                torch.zeros(1, 5),
                torch.ones(1, 5),
                torch.zeros(1, 5),
                max_frames=6,
            )

    def test_forward_and_infer_reuse_flow_with_svc_conditioning(self):
        model = self._model()
        content = torch.randn(1, 4, 6)
        f0 = torch.full((1, 4), 8.0)
        uv = torch.ones(1, 4)
        loudness = torch.zeros(1, 4)
        target = torch.zeros(1, 128, 4)

        # Isolate the SVC-to-flow wiring from the separately tested harmonic
        # excitation implementation and its librosa-backed mel filter setup.
        model._excitation_x0 = lambda f0_logf0, uv, harm_wave=None: torch.zeros(
            f0_logf0.shape[0], 128, f0_logf0.shape[1]
        )
        out = model(content, f0, uv, loudness, target, spk_id=torch.tensor([1]))
        pred = model.infer(
            content, f0, uv, loudness, num_steps=1, spk_id=torch.tensor([1])
        )

        self.assertTrue(torch.isfinite(out["flow"]))
        self.assertTrue(torch.isfinite(out["recon"]))
        self.assertEqual((1, 128, 4), tuple(out["x1_pred"].shape))
        self.assertEqual((1, 128, 4), tuple(pred.shape))

    def test_checkpoint_config_rebuilds_svc_model(self):
        cfg = {
            "arch": "harmonic_svc",
            "content_dim": 6,
            "content_layers": 1,
            "content_dropout": 0.0,
            "hidden": 8,
            "mel_bins": 128,
            "backbone_ch": 8,
            "n_cycles": 1,
            "dilation_schedule": "pow2_15",
            "n_speakers": 2,
            "spk_dim": 4,
            "n_harm": 1,
            "noise_ratio": 0.0,
            "exc_scale": 0.15,
            "harm_decay": 1.0,
            "exc_hop": 256,
            "use_uv": True,
            "flow_loss": "l2",
        }
        model = _build_from_config(cfg, torch.device("cpu"))
        self.assertIsInstance(model, HarmonicSVCModel)
        self.assertEqual(6, model.content_dim)

    def test_existing_svs_checkpoint_config_still_rebuilds(self):
        cfg = {
            "arch": "harmonic",
            "n_phonemes": 12,
            "hidden": 8,
            "mel_bins": 128,
            "backbone_ch": 8,
            "n_cycles": 1,
            "dilation_schedule": "pow2_15",
            "n_speakers": 0,
            "n_harm": 1,
            "noise_ratio": 0.0,
            "exc_scale": 0.15,
            "harm_decay": 1.0,
            "exc_hop": 256,
            "use_uv": True,
            "flow_loss": "l2",
        }
        model = _build_from_config(cfg, torch.device("cpu"))
        self.assertNotIsInstance(model, HarmonicSVCModel)
        self.assertIsNotNone(model.phoneme_encoder)

    def test_single_item_inference_contract(self):
        model = self._model()
        model._excitation_x0 = lambda f0_logf0, uv, harm_wave=None: torch.zeros(
            f0_logf0.shape[0], 128, f0_logf0.shape[1]
        )
        item = {
            "content": torch.randn(4, 6).numpy(),
            "f0_logf0": torch.full((4,), 8.0).numpy(),
            "uv": torch.ones(4).numpy(),
            "loudness": torch.zeros(4).numpy(),
            "spk_id": 1,
        }
        mel = infer_svc_mel(model, item, num_steps=1)
        self.assertEqual((128, 4), mel.shape)


class SVCCollateTests(unittest.TestCase):
    def test_pads_frame_aligned_features_and_builds_masks(self):
        def item(frames):
            return {
                "content": torch.randn(frames, 6).numpy(),
                "f0_logf0": torch.zeros(frames).numpy(),
                "uv": torch.ones(frames).numpy(),
                "loudness": torch.zeros(frames).numpy(),
                "target_mel": torch.zeros(128, frames).numpy(),
                "spk_id": 0,
                "style_id": 0,
                "item_name": f"item-{frames}",
            }

        batch = svc_collate_fn([item(3), item(5)])
        self.assertEqual((2, 5, 6), tuple(batch["content"].shape))
        self.assertEqual((2, 128, 5), tuple(batch["target_mel"].shape))
        self.assertEqual([False, False, False, True, True], batch["frame_mask"][0].tolist())
        torch.testing.assert_close(batch["content_mask"], batch["frame_mask"])

    def test_dataset_reads_declared_feature_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = {
                "phrases": {"song_a_0000": 3, "song_b_0000": 4},
                "frame_rate": 172.265625,
                "content_dim": 6,
            }
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            payload = {}
            for name, frames in metadata["phrases"].items():
                payload[f"{name}|content"] = np.zeros((frames, 6), np.float32)
                payload[f"{name}|f0_interp"] = np.full(frames, 220.0, np.float32)
                payload[f"{name}|uv"] = np.ones(frames, np.float32)
                payload[f"{name}|loudness"] = np.zeros(frames, np.float32)
                payload[f"{name}|mel"] = np.zeros((128, frames), np.float32)
            np.savez(root / "svc_shard.npz", **payload)

            dataset = SVCFeatureDataset([root], eval_songs=0)
            self.assertEqual(2, len(dataset))
            self.assertEqual(6, dataset.content_dim)
            self.assertEqual((3, 6), dataset[0]["content"].shape)
            dataset.close()


if __name__ == "__main__":
    unittest.main()


class LoaderKwargsTests(unittest.TestCase):
    """DataLoader の追加引数。**pin_memory は CUDA のときだけ**。

    `--device cpu` でも `pin_memory=True` にすると、バッチを取り出す瞬間に
    `torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable`
    で落ちます（GPU が他の作業と取り合っている Windows 機で実測）。CPU 実行と
    GPU の無い環境は `tools/smoke/run_smoke.py --device cpu` が通る前提なので、
    ここが壊れると疎通確認そのものが回らなくなります。
    """

    def test_pin_memory_is_off_for_cpu(self):
        from train import _loader_kwargs
        self.assertFalse(_loader_kwargs("cpu", 0)["pin_memory"])

    def test_pin_memory_is_on_for_cuda(self):
        from train import _loader_kwargs
        self.assertTrue(_loader_kwargs("cuda", 0)["pin_memory"])

    def test_accepts_a_torch_device_too(self):
        from train import _loader_kwargs
        self.assertFalse(_loader_kwargs(torch.device("cpu"), 0)["pin_memory"])

    def test_worker_options_only_when_workers_are_used(self):
        from train import _loader_kwargs
        self.assertNotIn("persistent_workers", _loader_kwargs("cpu", 0))
        self.assertTrue(_loader_kwargs("cpu", 2)["persistent_workers"])
        self.assertEqual(_loader_kwargs("cpu", 2)["prefetch_factor"], 4)


class PerfSnapshotTests(unittest.TestCase):
    """学習の実測値（実行計画 M3 ゴール 4）。

    「最初の 100 / 1,000 update で examples/sec、frames/sec、peak VRAM、checkpoint size、
    validation time を実測し、見積もりを実測値へ更新してある」ことがゴールです。
    tqdm の step/s は画面に出るだけで記録に残らないので、値として出せるようにします。
    vast.ai は時間課金なので、この数字はそのまま料金の見積もりになります。
    """

    def test_computes_rates_from_the_counters(self):
        from train import perf_snapshot
        s = perf_snapshot(step=100, seconds=50.0, examples=800, frames=3_000_000)
        self.assertAlmostEqual(s["steps_per_sec"], 2.0)
        self.assertAlmostEqual(s["examples_per_sec"], 16.0)
        self.assertAlmostEqual(s["frames_per_sec"], 60_000.0)
        self.assertEqual(s["step"], 100)

    def test_survives_a_zero_elapsed_time(self):
        # 最初の 1 step で呼ばれるとゼロ除算になり得る。落とさない。
        from train import perf_snapshot
        s = perf_snapshot(step=1, seconds=0.0, examples=8, frames=30_000)
        self.assertTrue(all(np.isfinite(v) for k, v in s.items()
                            if isinstance(v, float)))

    def test_reports_peak_vram_when_a_value_is_given(self):
        from train import perf_snapshot
        s = perf_snapshot(step=100, seconds=10.0, examples=100, frames=10,
                          peak_vram_bytes=2 * 1024 ** 3)
        self.assertAlmostEqual(s["peak_vram_gb"], 2.0)

    def test_omits_peak_vram_on_cpu(self):
        from train import perf_snapshot
        self.assertNotIn("peak_vram_gb", perf_snapshot(step=1, seconds=1.0,
                                                       examples=1, frames=1))

    def test_formats_a_single_readable_line(self):
        from train import perf_line, perf_snapshot
        line = perf_line(perf_snapshot(step=100, seconds=50.0, examples=800,
                                       frames=3_000_000, peak_vram_bytes=1024 ** 3))
        self.assertIn("step 100", line)
        self.assertIn("2.00 step/s", line)
        self.assertIn("1.00 GB", line)


class BandProfileTests(unittest.TestCase):
    """帯域エネルギー分布と spectral centroid。

    M3 の検証は content cos / F0 相関 / V/UV だけで、**高域の欠落を検知できませんでした**。
    推論側の loudness 条件がずれて centroid が 620 → 368 Hz へ落ちたとき、content cos は
    0.8217 → 0.8096 としか動きませんでした（実測）。耳では明らかに「こもった」音です。
    音の明るさを数値にして、この種の劣化を検証に載せます。
    """

    SR = 44100

    def _tone(self, hz, seconds=1.0, harmonics=1):
        t = np.arange(int(seconds * self.SR)) / self.SR
        w = sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, harmonics + 1))
        return (w / np.abs(w).max() * 0.8).astype(np.float32)

    def test_bands_sum_to_one(self):
        from tools.audio_metrics import band_profile
        p = band_profile(self._tone(220, harmonics=8), self.SR)
        self.assertAlmostEqual(sum(p["bands"].values()), 1.0, places=3)

    def test_a_low_tone_puts_its_energy_in_the_lowest_band(self):
        from tools.audio_metrics import band_profile
        p = band_profile(self._tone(200), self.SR)
        self.assertGreater(p["bands"]["0-1k"], 0.95)
        self.assertAlmostEqual(p["centroid_hz"], 200, delta=40)

    def test_centroid_rises_with_the_tone(self):
        from tools.audio_metrics import band_profile
        low = band_profile(self._tone(200), self.SR)["centroid_hz"]
        high = band_profile(self._tone(3000), self.SR)["centroid_hz"]
        self.assertGreater(high, low * 5)

    def test_removing_the_high_band_lowers_the_centroid(self):
        # これが検知したい劣化そのもの。高調波を削ると centroid が下がること。
        from tools.audio_metrics import band_profile
        bright = band_profile(self._tone(220, harmonics=16), self.SR)
        dull = band_profile(self._tone(220, harmonics=2), self.SR)
        self.assertLess(dull["centroid_hz"], bright["centroid_hz"])
        self.assertGreater(dull["bands"]["0-1k"], bright["bands"]["0-1k"])

    def test_is_deterministic(self):
        from tools.audio_metrics import band_profile
        w = self._tone(220, harmonics=8)
        self.assertEqual(band_profile(w, self.SR), band_profile(w, self.SR))

    def test_returns_empty_for_a_signal_shorter_than_the_window(self):
        from tools.audio_metrics import band_profile
        self.assertEqual(band_profile(np.zeros(128, dtype=np.float32), self.SR), {})


class SpeakerSimilarityTests(unittest.TestCase):
    """M4 ゴール 2 の target similarity。

    **単独の cos 値には意味がありません。** 話者照合の埋め込みは、同一話者の別クリップ
    どうしでも 1.0 にはならず、無関係な話者どうしでも 0 にはなりません。`m3_verify.py` の
    content cos と同じく、**上限（target の別クリップどうし）と下限（無関係な話者）を
    必ず並べ**、回復率で読みます。

    埋め込みモデルは引数で受け取ります（`extract_phrase` と同じ理由。単体テストを重い
    事前学習モデルとネットワークに依存させないため）。
    """

    SR = 16000

    def _wavs(self, n, seed=0, seconds=12.0):
        # 較正が成り立つ長さ（12 秒以上）で作る。短いと report が拒否する。
        r = np.random.default_rng(seed)
        return [r.standard_normal(int(self.SR * seconds)).astype(np.float32) for _ in range(n)]

    def _embed_from(self, table):
        """wav の先頭サンプルを key にして、決められた埋め込みを返す偽 encoder。"""
        def embed(wav, sr):
            return np.asarray(table[float(wav[0])], dtype=np.float32)
        return embed

    def test_report_records_the_clip_lengths_it_used(self):
        # 較正は長さに依存する（6 秒では通らない）。あとから報告を読む人が、
        # その数値が較正の内側で出たのかを判断できなければならない。
        from tools.speaker_similarity import similarity_report
        refs, conv = self._wavs(2, seed=1, seconds=15.0), self._wavs(1, seed=2, seconds=13.0)
        table = {float(w[0]): [1.0, 0.0] for w in refs + conv}
        rep = similarity_report(conv, refs, embed=self._embed_from(table), sr=self.SR)
        self.assertEqual(rep["clip_seconds"]["min"], 13.0)
        self.assertEqual(rep["clip_seconds"]["min_required"], 12.0)

    def test_report_refuses_clips_shorter_than_the_calibrated_length(self):
        # 較正（tools/speaker_calibrate.py）は 6 秒では通らず 12 秒で通った。
        # 較正が成り立たない長さで数値を出すと、読めない値を報告することになる。
        from tools.speaker_similarity import similarity_report
        refs = self._wavs(2, seed=1)
        short = self._wavs(1, seed=2, seconds=6.0)
        table = {float(w[0]): [1.0, 0.0] for w in refs + short}
        with self.assertRaises(ValueError):
            similarity_report(short, refs, embed=self._embed_from(table), sr=self.SR)

    def test_report_refuses_a_short_target_reference_too(self):
        from tools.speaker_similarity import similarity_report
        refs = self._wavs(2, seed=1, seconds=6.0)
        conv = self._wavs(1, seed=2)
        table = {float(w[0]): [1.0, 0.0] for w in refs + conv}
        with self.assertRaises(ValueError):
            similarity_report(conv, refs, embed=self._embed_from(table), sr=self.SR)

    def test_the_length_error_names_the_calibrated_minimum(self):
        from tools.speaker_similarity import MIN_SECONDS, similarity_report
        refs = self._wavs(2, seed=1)
        short = self._wavs(1, seed=2, seconds=3.0)
        table = {float(w[0]): [1.0, 0.0] for w in refs + short}
        with self.assertRaises(ValueError) as cm:
            similarity_report(short, refs, embed=self._embed_from(table), sr=self.SR)
        self.assertIn(str(MIN_SECONDS), str(cm.exception))

    def test_the_length_guard_can_be_lifted_explicitly(self):
        # 較正外だと分かったうえで測る場合は、明示的に外させる（黙って通さない）。
        from tools.speaker_similarity import similarity_report
        refs = self._wavs(2, seed=1, seconds=1.0)
        conv = self._wavs(1, seed=2, seconds=1.0)
        table = {float(w[0]): [1.0, 0.0] for w in refs + conv}
        rep = similarity_report(conv, refs, embed=self._embed_from(table), sr=self.SR,
                                min_seconds=0.0)
        self.assertEqual(rep["converted_vs_target"]["n"], 2)

    def test_cosine_of_identical_vectors_is_one(self):
        from tools.speaker_similarity import cosine
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.assertAlmostEqual(cosine(v, v), 1.0, places=6)

    def test_cosine_of_orthogonal_vectors_is_zero(self):
        from tools.speaker_similarity import cosine
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(cosine(a, b), 0.0, places=6)

    def test_the_ceiling_uses_distinct_reference_pairs_only(self):
        # 自分自身との組を数えると上限が 1.0 に張り付き、基準として使えなくなる。
        from tools.speaker_similarity import similarity_report
        refs = self._wavs(3, seed=1)
        table = {float(w[0]): [1.0, 0.0] for w in refs}
        conv = self._wavs(1, seed=2)
        table[float(conv[0][0])] = [1.0, 0.0]
        rep = similarity_report(conv, refs, embed=self._embed_from(table), sr=self.SR)
        self.assertEqual(rep["target_self_ceiling"]["n"], 3)      # 3 本なら 3 組

    def test_compares_every_converted_clip_against_every_reference(self):
        from tools.speaker_similarity import similarity_report
        refs, conv = self._wavs(3, seed=1), self._wavs(2, seed=2)
        table = {float(w[0]): [1.0, 0.0] for w in refs + conv}
        rep = similarity_report(conv, refs, embed=self._embed_from(table), sr=self.SR)
        self.assertEqual(rep["converted_vs_target"]["n"], 6)

    def test_recovery_is_one_when_the_conversion_matches_the_target(self):
        from tools.speaker_similarity import similarity_report
        refs, conv, other = self._wavs(2, seed=1), self._wavs(2, seed=2), self._wavs(2, seed=3)
        table = {float(w[0]): [1.0, 0.0] for w in refs + conv}
        table.update({float(w[0]): [0.0, 1.0] for w in other})
        rep = similarity_report(conv, refs, other, embed=self._embed_from(table), sr=self.SR)
        self.assertAlmostEqual(rep["recovery"], 1.0, places=5)

    def test_recovery_is_zero_when_the_conversion_matches_the_unrelated_speaker(self):
        from tools.speaker_similarity import similarity_report
        refs, conv, other = self._wavs(2, seed=1), self._wavs(2, seed=2), self._wavs(2, seed=3)
        table = {float(w[0]): [1.0, 0.0] for w in refs}
        table.update({float(w[0]): [0.0, 1.0] for w in conv + other})
        rep = similarity_report(conv, refs, other, embed=self._embed_from(table), sr=self.SR)
        self.assertAlmostEqual(rep["recovery"], 0.0, places=5)

    def test_rejects_a_single_reference_clip(self):
        # 1 本では上限が作れない。黙って上限なしの数値を返さない。
        from tools.speaker_similarity import similarity_report
        refs, conv = self._wavs(1, seed=1), self._wavs(1, seed=2)
        table = {float(w[0]): [1.0, 0.0] for w in refs + conv}
        with self.assertRaises(ValueError):
            similarity_report(conv, refs, embed=self._embed_from(table), sr=self.SR)

    def test_embeds_each_clip_exactly_once(self):
        # ペアごとに呼ぶと実モデルでは組合せ爆発する。
        from tools.speaker_similarity import similarity_report
        refs, conv, other = self._wavs(3, seed=1), self._wavs(2, seed=2), self._wavs(2, seed=3)
        table = {float(w[0]): [1.0, 0.0] for w in refs + conv + other}
        calls = []
        base = self._embed_from(table)

        def counting(wav, sr):
            calls.append(float(wav[0]))
            return base(wav, sr)
        similarity_report(conv, refs, other, embed=counting, sr=self.SR)
        self.assertEqual(len(calls), 7)


class SpeakerCalibrationTests(unittest.TestCase):
    """話者照合 encoder が「歌声で使えるか」を判定する較正（M4 ゴール 2 / M5 ゴール 2）。

    **cos の絶対値ではなく、3 群の分離で判定します。** 同一話者ペアが高く、別話者ペアが
    低いだけでは足りません。**分布が重なっていれば、個々の数値を読めません。**

    - **同一話者**（別クリップどうし）— 上限の分布
    - **別話者・同性** — ここと重なると target similarity を主張できない
    - **別話者・異性** — 最も分かれやすい。ここだけで判定しない

    判定は「重なり」（同一話者ペアの下位 5% を超える別話者・同性ペアの割合）で行います。
    **閾値は encoder を走らせる前に決めます**（M4 の checkpoint 選択規則と同じ理由）。

    埋め込みは引数で受け取るので、このテストは重いモデルもネットワークも使いません。
    """

    def test_gender_is_read_from_the_vocalset_directory_name(self):
        from tools.speaker_calibrate import gender_of
        self.assertEqual(gender_of("female1"), "female")
        self.assertEqual(gender_of("male10"), "male")

    def test_unknown_speaker_naming_is_rejected_rather_than_guessed(self):
        # 性別を取り違えると「異性は分かれる」という結論ごと壊れる。黙って推測しない。
        from tools.speaker_calibrate import gender_of
        with self.assertRaises(ValueError):
            gender_of("singer07")

    def test_pairs_are_split_into_the_three_groups(self):
        from tools.speaker_calibrate import pair_groups
        speakers = ["female1", "female1", "female2", "male1"]
        g = pair_groups(speakers)
        self.assertEqual(g["same"], [(0, 1)])
        self.assertEqual(g["diff_same_gender"], [(0, 2), (1, 2)])
        self.assertEqual(sorted(g["diff_cross_gender"]), [(0, 3), (1, 3), (2, 3)])

    def test_a_clip_is_never_paired_with_itself(self):
        # 自己ペアを混ぜると同一話者群が 1.0 に張り付き、上限が上振れする。
        from tools.speaker_calibrate import pair_groups
        for pairs in pair_groups(["female1", "female1", "male1"]).values():
            for i, j in pairs:
                self.assertNotEqual(i, j)

    def test_overlap_is_zero_when_different_speakers_all_fall_below(self):
        from tools.speaker_calibrate import overlap_fraction
        same = [0.90, 0.91, 0.92, 0.93, 0.94]
        self.assertEqual(overlap_fraction(same, [0.10, 0.20, 0.30]), 0.0)

    def test_overlap_is_one_when_every_different_speaker_pair_exceeds(self):
        from tools.speaker_calibrate import overlap_fraction
        same = [0.50, 0.51, 0.52, 0.53, 0.54]
        self.assertEqual(overlap_fraction(same, [0.80, 0.90]), 1.0)

    def test_overlap_is_measured_against_the_fifth_percentile_of_same_speaker(self):
        # 平均どうしを比べると、ばらつきの大きい encoder を通してしまう。
        from tools.speaker_calibrate import overlap_fraction
        same = list(np.linspace(0.0, 1.0, 101))          # 5 パーセンタイル = 0.05
        self.assertAlmostEqual(overlap_fraction(same, [0.04, 0.06, 0.07, 0.08]), 0.75, places=6)

    def test_report_returns_all_three_groups_with_spread(self):
        from tools.speaker_calibrate import calibration_report
        embs = [np.array([1.0, 0.0]), np.array([1.0, 0.0]),
                np.array([0.0, 1.0]), np.array([0.0, 1.0])]
        rep = calibration_report(embs, ["female1", "female1", "female2", "male1"])
        for key in ("same", "diff_same_gender", "diff_cross_gender"):
            self.assertIn("mean", rep[key])
            self.assertIn("sd", rep[key])
            self.assertGreater(rep[key]["n"], 0)

    def test_report_refuses_when_there_is_no_same_speaker_pair(self):
        # 上限が作れないのに数値だけ返すと、読めない報告になる。
        from tools.speaker_calibrate import calibration_report
        embs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        with self.assertRaises(ValueError):
            calibration_report(embs, ["female1", "female2"])

    def test_report_refuses_when_no_same_gender_pair_exists(self):
        # 同性ペアが無いと、この encoder の一番きつい条件を測っていない。
        from tools.speaker_calibrate import calibration_report
        embs = [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        with self.assertRaises(ValueError):
            calibration_report(embs, ["female1", "female1", "male1"])

    def test_a_perfectly_separating_encoder_passes(self):
        from tools.speaker_calibrate import calibration_report, passes_calibration
        embs = [np.array([1.0, 0.0]), np.array([1.0, 0.01]),
                np.array([0.0, 1.0]), np.array([0.01, 1.0]),
                np.array([1.0, 1.0]), np.array([1.0, 1.01])]
        rep = calibration_report(embs, ["female1", "female1", "female2", "female2",
                                        "male1", "male1"])
        self.assertTrue(passes_calibration(rep, max_overlap=0.20))

    def test_an_encoder_that_cannot_separate_same_gender_singers_fails(self):
        # 2026-08-31 に実測した wavlm/unispeech の形（重なり 83%）を再現する。
        from tools.speaker_calibrate import calibration_report, passes_calibration
        rng = np.random.default_rng(0)
        embs = [rng.standard_normal(8) * 0.02 + np.array([1.0] + [0.0] * 7) for _ in range(8)]
        rep = calibration_report(embs, ["female1"] * 2 + ["female2"] * 2
                                 + ["female3"] * 2 + ["male1"] * 2)
        self.assertFalse(passes_calibration(rep, max_overlap=0.20))

    def test_collect_takes_the_same_number_of_clips_from_every_speaker(self):
        # 本数が偏ると、多く入った話者が群の平均を支配する。
        import soundfile as sf

        from tools.speaker_calibrate import _collect
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for spk, n in (("female1", 5), ("female2", 3)):
                (root / spk).mkdir()
                for i in range(n):
                    sf.write(root / spk / f"{i}.wav", np.zeros(16000, np.float32), 16000)
            _, speakers = _collect(root, clips_per_speaker=2, seconds=0.5, sr=16000)
            self.assertEqual(speakers.count("female1"), 2)
            self.assertEqual(speakers.count("female2"), 2)

    def test_collect_skips_a_speaker_that_cannot_form_a_same_speaker_pair(self):
        import soundfile as sf

        from tools.speaker_calibrate import _collect
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "female1").mkdir()
            sf.write(root / "female1" / "a.wav", np.zeros(16000, np.float32), 16000)
            (root / "female2").mkdir()
            for i in range(2):
                sf.write(root / "female2" / f"{i}.wav", np.zeros(16000, np.float32), 16000)
            _, speakers = _collect(root, clips_per_speaker=3, seconds=0.5, sr=16000)
            self.assertNotIn("female1", speakers)
            self.assertEqual(speakers.count("female2"), 2)

    def test_collect_restricts_the_material_with_a_glob(self):
        # 「どのクリップで較正したか」は結論そのもの。歌唱と朗読を混ぜると別の実験になる。
        import soundfile as sf

        from tools.speaker_calibrate import _collect
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for sub in ("excerpts/straight", "arpeggios"):
                (root / "female1" / sub).mkdir(parents=True)
                for i in range(2):
                    sf.write(root / "female1" / sub / f"{i}.wav",
                             np.zeros(16000, np.float32), 16000)
            wavs, _ = _collect(root, clips_per_speaker=9, seconds=0.5, sr=16000,
                               glob="excerpts/straight/*.wav")
            self.assertEqual(len(wavs), 2)

    def test_the_verdict_records_the_threshold_it_was_judged_against(self):
        # 後から閾値を動かせないよう、判定に使った値を報告へ残す。
        from tools.speaker_calibrate import calibration_report, passes_calibration
        embs = [np.array([1.0, 0.0]), np.array([1.0, 0.01]),
                np.array([0.0, 1.0]), np.array([0.01, 1.0]),
                np.array([1.0, 1.0]), np.array([1.0, 1.01])]
        rep = calibration_report(embs, ["female1", "female1", "female2", "female2",
                                        "male1", "male1"])
        passes_calibration(rep, max_overlap=0.20)
        self.assertEqual(rep["verdict"]["max_overlap"], 0.20)
        self.assertIn("passed", rep["verdict"])


class ToolHelpTests(unittest.TestCase):
    """`tools/*.py --help` が落ちないこと。

    **実際に 2 本落ちていました。** `svc_convert.py` は help 文字列の `%` を argparse が
    書式指定子として解釈して `ValueError`、`nhv_indist.py` は em-dash（U+2014）が
    日本語 Windows の cp932 コンソールに書けず `UnicodeEncodeError` でした。

    どちらも「動かしてみるまで分からない」種類の壊れ方で、しかも `--help` は道具を
    使い始める最初の一歩です。文字の使い方まで含めて固定します。
    """

    @property
    def TOOLS(self):
        """`main()` を持つ `tools/*.py` を**自動で拾う**。

        手書きの一覧にすると、後から足した道具が黙って漏れます（`audio_metrics.py` の
        ようにライブラリしか無いものは CLI が無いので対象外）。
        """
        import importlib
        names = []
        for f in sorted(Path(__file__).parent.glob("tools/*.py")):
            if f.name.startswith("_"):
                continue
            if hasattr(importlib.import_module(f"tools.{f.stem}"), "main"):
                names.append(f.stem)
        self.assertGreaterEqual(len(names), 8, "tools の収集に失敗している")
        return names

    def _help_text(self, name):
        import contextlib
        import importlib
        import io
        mod = importlib.import_module(f"tools.{name}")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            with unittest.mock.patch.object(sys, "argv", [name, "--help"]):
                mod.main()
        self.assertEqual(cm.exception.code, 0, f"tools/{name}.py --help が異常終了した")
        return buf.getvalue()

    def test_every_tool_renders_its_help(self):
        for name in self.TOOLS:
            with self.subTest(tool=name):
                self.assertIn("usage", self._help_text(name).lower())

    def test_no_tool_source_contains_characters_cp932_cannot_encode(self):
        """**help だけでは足りません。** 実行時の print に em-dash を書いて落ちました
        （`guard_rail.py`。`nhv_indist.py` で同じ誤りを直した後です）。開発機は日本語
        Windows なので、cp932 で書けない文字はソースのどこにあっても実行時に落ちます。
        """
        bad = []
        for f in sorted(Path(__file__).parent.glob("tools/**/*.py")):
            text = f.read_text(encoding="utf-8")
            for i, line in enumerate(text.split("\n"), 1):
                try:
                    line.encode("cp932")
                except UnicodeEncodeError as e:
                    ch = line[e.start:e.end]
                    bad.append(f"{f.name}:{i} {ch!r}")
        self.assertEqual(bad, [], "cp932 で書けない文字がある（実行時に落ちます）")

    def test_help_text_survives_a_cp932_console(self):
        # 開発機は日本語 Windows。cp932 で書けない文字を help に入れると、使う瞬間に落ちる。
        for name in self.TOOLS:
            with self.subTest(tool=name):
                try:
                    self._help_text(name).encode("cp932")
                except UnicodeEncodeError as e:
                    self.fail(f"tools/{name}.py の help に cp932 で書けない文字がある: {e}")

