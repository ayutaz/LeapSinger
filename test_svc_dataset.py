"""M0（データ確定）の道具のテスト。

    uv run python -m unittest test_svc_dataset -v

素材の品質検査と split 作成。実音声も重いモデルも使わない。
"""
from __future__ import annotations

import unittest

import numpy as np

from preprocess.svc.audit import AuditThresholds, audit_clip, effective_bandwidth_hz
from preprocess.svc.report import build_report
from preprocess.svc.coverage import label_seconds, pitch_band_seconds, voiced_range
from preprocess.svc.split import split_by_group


def _tone(n, sr=44100, hz=220.0, amp=0.5, seed=0):
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


class AuditClipTests(unittest.TestCase):
    """1 クリップを検査し、除外理由のリストを返す。

    実行計画 M0 ゴール 2「除外したものは reject reason つきで残してある」。
    合格なら空リスト。理由は人が読んで判断できる文字列にする。
    """

    SR = 44100

    def test_accepts_a_clean_clip(self):
        self.assertEqual(audit_clip(_tone(self.SR * 3), self.SR, expected_sr=self.SR), [])

    def test_rejects_a_sample_rate_mismatch(self):
        # mel 設定は前処理・loader・励起で共有されるので、sr の取り違えは静かに全部を壊す。
        reasons = audit_clip(_tone(22050 * 3, sr=22050), 22050, expected_sr=self.SR)
        self.assertTrue(any(r.startswith("sample_rate") for r in reasons), reasons)

    def test_rejects_a_clipped_clip(self):
        wav = _tone(self.SR, amp=1.5).clip(-1.0, 1.0)      # 振り切って潰れた波形
        reasons = audit_clip(wav, self.SR, expected_sr=self.SR)
        self.assertTrue(any(r.startswith("clipping") for r in reasons), reasons)

    def test_accepts_a_clip_that_merely_touches_full_scale(self):
        # 1 サンプルだけ 1.0 に触れるのは clipping ではない。閾値が緩すぎても厳しすぎても困る。
        wav = _tone(self.SR)
        wav[0] = 1.0
        self.assertEqual(audit_clip(wav, self.SR, expected_sr=self.SR), [])

    def test_rejects_a_mostly_silent_clip(self):
        wav = np.zeros(self.SR * 3, dtype=np.float32)
        wav[: self.SR // 10] = _tone(self.SR // 10)
        reasons = audit_clip(wav, self.SR, expected_sr=self.SR)
        self.assertTrue(any(r.startswith("silence") for r in reasons), reasons)

    def test_rejects_a_dc_offset(self):
        reasons = audit_clip(_tone(self.SR) + 0.2, self.SR, expected_sr=self.SR)
        self.assertTrue(any(r.startswith("dc_offset") for r in reasons), reasons)

    def test_rejects_a_clip_that_is_too_short(self):
        reasons = audit_clip(_tone(self.SR // 100), self.SR, expected_sr=self.SR)
        self.assertTrue(any(r.startswith("too_short") for r in reasons), reasons)

    def test_rejects_non_finite_samples(self):
        wav = _tone(self.SR).copy()
        wav[100] = np.nan
        reasons = audit_clip(wav, self.SR, expected_sr=self.SR)
        self.assertTrue(any(r.startswith("non_finite") for r in reasons), reasons)

    def test_reports_every_applicable_reason_not_just_the_first(self):
        # 1 つ直したら次が出る、では検査が何往復もする。まとめて返す。
        wav = np.zeros(22050 * 3, dtype=np.float32) + 0.9
        reasons = audit_clip(wav, 22050, expected_sr=self.SR)
        self.assertGreaterEqual(len(reasons), 2, reasons)

    def test_reasons_carry_the_measured_value(self):
        # 「clipping」だけでは閾値を調整できない。実測値を添える。
        wav = _tone(self.SR, amp=1.5).clip(-1.0, 1.0)
        reason = next(r for r in audit_clip(wav, self.SR, expected_sr=self.SR)
                      if r.startswith("clipping"))
        self.assertIn("=", reason)

    def test_thresholds_are_configurable(self):
        wav = _tone(self.SR)
        strict = AuditThresholds(max_dc_offset=0.0)
        self.assertNotEqual(audit_clip(wav, self.SR, expected_sr=self.SR, thresholds=strict), [])

    def test_rejects_a_non_mono_clip(self):
        with self.assertRaises(ValueError):
            audit_clip(np.zeros((2, 1000), dtype=np.float32), self.SR, expected_sr=self.SR)


class BandwidthTests(unittest.TestCase):
    """低い sample rate から上げただけの素材を見抜く。

    `mel.fmax` は 16,000 Hz。24 kHz 音源（Nyquist 12 kHz）を 44.1 kHz へ上げても
    12〜16 kHz は空のままで、混ぜて学習するとその帯域を「無い」と学習してこもった出力になる。
    JVS-MuSiC が配布版 24 kHz なのはまさにこの例で、GTSinger 自身も
    check_valid_bandwidth.py を同梱している。実在する問題なので検査する。
    """

    SR = 44100

    def _noise(self, n=44100, seed=0):
        return np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.1

    def _band_limited(self, cutoff_hz, n=44100):
        # cutoff より上を落とした白色雑音（= その sr から上げた素材の模擬）
        rng = np.random.default_rng(1)
        spec = np.fft.rfft(rng.standard_normal(n))
        freqs = np.fft.rfftfreq(n, 1 / self.SR)
        spec[freqs > cutoff_hz] = 0.0
        y = np.fft.irfft(spec, n)
        return (y / (np.abs(y).max() + 1e-9) * 0.5).astype(np.float32)

    def test_full_band_noise_reaches_near_nyquist(self):
        bw = effective_bandwidth_hz(self._noise(), self.SR)
        self.assertGreater(bw, 0.9 * self.SR / 2)

    def test_detects_the_cutoff_of_band_limited_audio(self):
        bw = effective_bandwidth_hz(self._band_limited(12000.0), self.SR)
        self.assertAlmostEqual(bw, 12000.0, delta=800.0)

    def test_audit_flags_band_limited_audio_when_a_minimum_is_set(self):
        th = AuditThresholds(min_bandwidth_hz=16000.0)
        reasons = audit_clip(self._band_limited(12000.0), self.SR,
                             expected_sr=self.SR, thresholds=th)
        self.assertTrue(any(r.startswith("band_limited") for r in reasons), reasons)

    def test_audit_accepts_full_band_audio_when_a_minimum_is_set(self):
        th = AuditThresholds(min_bandwidth_hz=16000.0)
        reasons = audit_clip(self._noise(), self.SR, expected_sr=self.SR, thresholds=th)
        self.assertEqual([r for r in reasons if r.startswith("band_limited")], [])

    def test_bandwidth_check_is_off_by_default(self):
        # 純音のような合成信号は高域が無くて当然。既定で有効にすると誤検出だらけになる。
        reasons = audit_clip(self._band_limited(3000.0), self.SR, expected_sr=self.SR)
        self.assertEqual([r for r in reasons if r.startswith("band_limited")], [])


class SplitByGroupTests(unittest.TestCase):
    """曲・収録セッション単位で train/eval/test を分ける。

    実行計画 M0 ゴール 4。フレーズ単位で切ると同じ曲が train と test の両方に入り、
    leakage で性能を過大評価する。
    """

    def _names(self, n_groups=10, per_group=4):
        return {f"song{g:02d}_{i:04d}": f"song{g:02d}"
                for g in range(n_groups) for i in range(per_group)}

    def test_no_group_appears_in_two_splits(self):
        s = split_by_group(self._names(), seed=0, eval_groups=2, test_groups=2)
        for a, b in (("train", "eval"), ("train", "test"), ("eval", "test")):
            ga = {self._names()[n] for n in s[a]}
            gb = {self._names()[n] for n in s[b]}
            self.assertEqual(ga & gb, set(), f"{a} と {b} に同じ group がある")

    def test_every_name_lands_in_exactly_one_split(self):
        names = self._names()
        s = split_by_group(names, seed=0, eval_groups=2, test_groups=2)
        allocated = s["train"] + s["eval"] + s["test"]
        self.assertEqual(sorted(allocated), sorted(names))
        self.assertEqual(len(allocated), len(set(allocated)))

    def test_requested_group_counts_are_honoured(self):
        names = self._names()
        s = split_by_group(names, seed=0, eval_groups=3, test_groups=2)
        self.assertEqual(len({names[n] for n in s["eval"]}), 3)
        self.assertEqual(len({names[n] for n in s["test"]}), 2)

    def test_same_seed_gives_the_same_split(self):
        names = self._names()
        a = split_by_group(names, seed=7, eval_groups=2, test_groups=2)
        b = split_by_group(names, seed=7, eval_groups=2, test_groups=2)
        self.assertEqual(a, b)

    def test_different_seed_gives_a_different_split(self):
        names = self._names(n_groups=20)
        a = split_by_group(names, seed=0, eval_groups=3, test_groups=3)
        b = split_by_group(names, seed=1, eval_groups=3, test_groups=3)
        self.assertNotEqual(a, b)

    def test_train_keeps_at_least_one_group(self):
        with self.assertRaises(ValueError):
            split_by_group(self._names(n_groups=4), seed=0, eval_groups=2, test_groups=2)

    def test_rejects_an_empty_dataset(self):
        with self.assertRaises(ValueError):
            split_by_group({}, seed=0, eval_groups=1, test_groups=1)

    def test_names_within_a_split_are_sorted(self):
        # split list はファイルに書いて差分を読むもの。順序が安定しないと差分が意味を失う。
        s = split_by_group(self._names(), seed=0, eval_groups=2, test_groups=2)
        for k in ("train", "eval", "test"):
            self.assertEqual(s[k], sorted(s[k]))


class CoverageTests(unittest.TestCase):
    """音域の滞在時間を集計する（実行計画 M0 ゴール 3）。

    高音・裏声が薄い素材で学習すると、そこだけ崩れる。学習前に分布を見るための集計。
    """

    FR = 44100 / 256   # 172.265625 Hz

    def test_counts_only_voiced_frames(self):
        f0 = np.array([220.0, 220.0, 220.0, 220.0], dtype=np.float32)
        uv = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        total = sum(pitch_band_seconds(f0, uv, frame_rate=self.FR, edges_hz=[300.0]).values())
        self.assertAlmostEqual(total, 2 / self.FR, places=6)

    def test_assigns_frames_to_bands_by_the_given_edges(self):
        f0 = np.array([100.0, 250.0, 600.0], dtype=np.float32)
        uv = np.ones(3, dtype=np.float32)
        bands = pitch_band_seconds(f0, uv, frame_rate=self.FR, edges_hz=[200.0, 400.0])
        self.assertEqual(len(bands), 3)
        for seconds in bands.values():
            self.assertAlmostEqual(seconds, 1 / self.FR, places=6)

    def test_a_value_on_an_edge_goes_to_the_upper_band(self):
        f0 = np.array([200.0], dtype=np.float32)
        uv = np.ones(1, dtype=np.float32)
        bands = pitch_band_seconds(f0, uv, frame_rate=self.FR, edges_hz=[200.0])
        labels = list(bands)
        self.assertAlmostEqual(bands[labels[0]], 0.0, places=9)
        self.assertAlmostEqual(bands[labels[1]], 1 / self.FR, places=6)

    def test_all_unvoiced_gives_zero_everywhere_without_raising(self):
        f0 = np.array([220.0, 330.0], dtype=np.float32)
        uv = np.zeros(2, dtype=np.float32)
        bands = pitch_band_seconds(f0, uv, frame_rate=self.FR, edges_hz=[300.0])
        self.assertEqual(set(bands.values()), {0.0})

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            pitch_band_seconds(np.zeros(3, dtype=np.float32), np.zeros(2, dtype=np.float32),
                               frame_rate=self.FR, edges_hz=[300.0])

    def test_rejects_unsorted_edges(self):
        with self.assertRaises(ValueError):
            pitch_band_seconds(np.array([220.0], dtype=np.float32),
                               np.ones(1, dtype=np.float32),
                               frame_rate=self.FR, edges_hz=[400.0, 200.0])

    def test_voiced_range_reports_percentiles_in_hz(self):
        f0 = np.linspace(100.0, 500.0, 101).astype(np.float32)
        uv = np.ones(101, dtype=np.float32)
        r = voiced_range(f0, uv, frame_rate=self.FR)
        self.assertAlmostEqual(r["p50_hz"], 300.0, delta=1.0)
        self.assertLess(r["p05_hz"], r["p50_hz"])
        self.assertGreater(r["p95_hz"], r["p50_hz"])

    def test_voiced_range_reports_the_span_in_semitones(self):
        # 1 オクターブ = 12 半音。音域の広さは Hz より半音のほうが読める。
        f0 = np.array([220.0, 440.0], dtype=np.float32)
        uv = np.ones(2, dtype=np.float32)
        r = voiced_range(f0, uv, frame_rate=self.FR)
        self.assertAlmostEqual(r["span_semitones"], 12.0, places=3)

    def test_voiced_range_reports_voiced_seconds(self):
        f0 = np.full(10, 220.0, dtype=np.float32)
        uv = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        r = voiced_range(f0, uv, frame_rate=self.FR)
        self.assertAlmostEqual(r["voiced_sec"], 3 / self.FR, places=6)

    def test_voiced_range_survives_a_fully_unvoiced_clip(self):
        r = voiced_range(np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32),
                         frame_rate=self.FR)
        self.assertEqual(r["voiced_sec"], 0.0)
        self.assertTrue(all(v == 0.0 or np.isnan(v) is False for v in r.values()))


class StratifiedSplitTests(unittest.TestCase):
    """held-out に偏りが出ないよう層で均す。

    実データ（VocalSet 20 名）で確認した問題: 歌手単位でランダムに held-out を選ぶと、
    seed 0 では eval も test も全員男性、seed 3 では全員女性になった。SVC は異性間の変換が
    難しい方なので、held-out が片方の性別だけだと評価がその難所を素通りする。
    """

    def _names(self, n_each=8):
        names, strata = {}, {}
        for i in range(n_each):
            for sex in ("f", "m"):
                g = f"{sex}{i}"
                strata[g] = sex
                for k in range(2):
                    names[f"{g}_{k:04d}"] = g
        return names, strata

    def test_each_split_gets_a_mix_of_strata(self):
        names, strata = self._names()
        for seed in range(5):
            with self.subTest(seed=seed):
                s = split_by_group(names, seed=seed, eval_groups=2, test_groups=2,
                                   strata=strata)
                for key in ("eval", "test"):
                    got = {strata[names[n]] for n in s[key]}
                    self.assertEqual(got, {"f", "m"}, f"{key} が片方の層だけ (seed={seed})")

    def test_still_honours_the_requested_group_counts(self):
        names, strata = self._names()
        s = split_by_group(names, seed=0, eval_groups=2, test_groups=2, strata=strata)
        self.assertEqual(len({names[n] for n in s["eval"]}), 2)
        self.assertEqual(len({names[n] for n in s["test"]}), 2)

    def test_no_group_appears_in_two_splits(self):
        names, strata = self._names()
        s = split_by_group(names, seed=1, eval_groups=2, test_groups=2, strata=strata)
        for a, b in (("train", "eval"), ("train", "test"), ("eval", "test")):
            ga = {names[n] for n in s[a]}
            gb = {names[n] for n in s[b]}
            self.assertEqual(ga & gb, set())

    def test_is_deterministic(self):
        names, strata = self._names()
        a = split_by_group(names, seed=2, eval_groups=2, test_groups=2, strata=strata)
        b = split_by_group(names, seed=2, eval_groups=2, test_groups=2, strata=strata)
        self.assertEqual(a, b)

    def test_falls_back_when_a_stratum_runs_out(self):
        # 層の数より held-out が多いときも、要求した group 数は満たす。
        names, strata = self._names(n_each=4)
        s = split_by_group(names, seed=0, eval_groups=3, test_groups=1, strata=strata)
        self.assertEqual(len({names[n] for n in s["eval"]}), 3)

    def test_rejects_a_stratum_map_missing_a_group(self):
        names, strata = self._names()
        del strata["f0"]
        with self.assertRaises(ValueError):
            split_by_group(names, seed=0, eval_groups=2, test_groups=2, strata=strata)


class LabelSecondsTests(unittest.TestCase):
    """ラベルごとの滞在秒数（実行計画 M0 ゴール 3 の「発声スタイルの coverage」）。

    技法ラベルを持つ corpus（GTSinger / VocalSet）なら、これで発声スタイルの偏りが分かる。
    区間の長さで重み付けする。区間数を数えても、短い区間が多いだけで多いことになってしまう。
    """

    def test_sums_durations_per_label(self):
        out = label_seconds(["belt", "breathy", "belt"], [1.0, 2.0, 0.5])
        self.assertAlmostEqual(out["belt"], 1.5)
        self.assertAlmostEqual(out["breathy"], 2.0)

    def test_orders_by_descending_seconds(self):
        # 偏りを見るための集計なので、多い順に並んでいないと読めない。
        out = label_seconds(["a", "b", "c"], [1.0, 3.0, 2.0])
        self.assertEqual(list(out), ["b", "c", "a"])

    def test_breaks_ties_by_label_for_determinism(self):
        out = label_seconds(["z", "a"], [1.0, 1.0])
        self.assertEqual(list(out), ["a", "z"])

    def test_returns_empty_for_no_segments(self):
        self.assertEqual(label_seconds([], []), {})

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            label_seconds(["a", "b"], [1.0])

    def test_rejects_negative_durations(self):
        with self.assertRaises(ValueError):
            label_seconds(["a"], [-1.0])


class BuildReportTests(unittest.TestCase):
    """M0 の成果物（reject list / coverage / split list / manifest）を 1 度に作る。

    実行計画 M0 の成果物は「dataset ledger、split list、reject list、coverage 集計」。
    道具が関数として在るだけでは、素材が届いたときに毎回つなぎを書くことになる。
    音声の読み込みは差し替え可能にして、重いモデルもファイル I/O も使わずにテストする。
    """

    SR = 44100

    def _loader(self, bad: set[str]):
        """name -> (wav, sr) を返す偽のローダー。bad に入れた名前は無音（= 弾かれる）。"""
        def load(name):
            if name in bad:
                return np.zeros(self.SR * 2, dtype=np.float32), self.SR
            return _tone(self.SR * 2, sr=self.SR), self.SR
        return load

    def _names(self, n_groups=6, per_group=3):
        return {f"song{g:02d}_{i:04d}": f"song{g:02d}"
                for g in range(n_groups) for i in range(per_group)}

    def test_records_every_rejected_clip_with_its_reasons(self):
        names = self._names()
        bad = {"song00_0000", "song03_0001"}
        r = build_report(names, self._loader(bad), expected_sr=self.SR,
                         seed=0, eval_groups=1, test_groups=1)
        self.assertEqual(sorted(r["rejects"]), sorted(bad))
        for name in bad:
            self.assertTrue(r["rejects"][name], f"{name} に理由が付いていない")

    def test_rejected_clips_never_appear_in_any_split(self):
        # 弾いた素材が split に残っていると、学習が起動時に落ちる。
        names = self._names()
        bad = {"song00_0000"}
        r = build_report(names, self._loader(bad), expected_sr=self.SR,
                         seed=0, eval_groups=1, test_groups=1)
        for key in ("train", "eval", "test"):
            self.assertNotIn("song00_0000", r["split"][key])

    def test_accepted_clips_are_all_placed(self):
        names = self._names()
        bad = {"song00_0000"}
        r = build_report(names, self._loader(bad), expected_sr=self.SR,
                         seed=0, eval_groups=1, test_groups=1)
        placed = r["split"]["train"] + r["split"]["eval"] + r["split"]["test"]
        self.assertEqual(sorted(placed), sorted(set(names) - bad))

    def test_manifest_records_what_is_needed_to_reproduce(self):
        r = build_report(self._names(), self._loader(set()), expected_sr=self.SR,
                         seed=3, eval_groups=1, test_groups=1)
        m = r["manifest"]
        for key in ("seed", "eval_groups", "test_groups", "expected_sr", "thresholds"):
            self.assertIn(key, m)
        self.assertEqual(m["seed"], 3)

    def test_reports_accepted_and_rejected_durations(self):
        names = self._names(n_groups=2, per_group=2)
        r = build_report(names, self._loader({"song00_0000"}), expected_sr=self.SR,
                         seed=0, eval_groups=1, test_groups=0)
        self.assertAlmostEqual(r["totals"]["accepted_sec"], 3 * 2.0, places=3)
        self.assertAlmostEqual(r["totals"]["rejected_sec"], 1 * 2.0, places=3)

    def test_is_deterministic_for_the_same_seed(self):
        names = self._names()
        a = build_report(names, self._loader(set()), expected_sr=self.SR,
                         seed=5, eval_groups=1, test_groups=1)
        b = build_report(names, self._loader(set()), expected_sr=self.SR,
                         seed=5, eval_groups=1, test_groups=1)
        self.assertEqual(a["split"], b["split"])

    def test_raises_when_every_clip_is_rejected(self):
        # 全滅は「検査が厳しすぎる」か「素材が壊れている」。黙って空の split を返さない。
        names = self._names(n_groups=3)
        with self.assertRaises(ValueError):
            build_report(names, self._loader(set(names)), expected_sr=self.SR,
                         seed=0, eval_groups=1, test_groups=1)

    def test_a_group_losing_every_clip_drops_out_of_the_split(self):
        names = self._names(n_groups=4, per_group=2)
        bad = {"song00_0000", "song00_0001"}
        r = build_report(names, self._loader(bad), expected_sr=self.SR,
                         seed=0, eval_groups=1, test_groups=1)
        placed = r["split"]["train"] + r["split"]["eval"] + r["split"]["test"]
        self.assertEqual([n for n in placed if n.startswith("song00")], [])


if __name__ == "__main__":
    unittest.main()
