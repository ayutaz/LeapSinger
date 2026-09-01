"""M5 ゴール 2 が要求する客観指標のうち、道具が無かった 4 つ。

**timing / CER / 信号品質 / 推論 RTF。** どれも「変換後の数値だけ」では読めないので、
このリポジトリの他の指標（`m3_verify.py` の content cos、`speaker_similarity.py` の回復率）
と同じく、**上限（GT mel をボコーダーに通した再合成）や source と並べて**読む形にします。

重いモデル（ASR / SQUIM）は**引数で受け取り**ます。単体テストはネットワークも GPU も使いません。
"""

import unittest

import numpy as np


class TimingMetricsTests(unittest.TestCase):
    """onset のずれ（M5 ゴール 2 の timing）。

    **「ずれの平均」だけでは読めません。** onset が消えた・増えた場合はペアが作れず、
    平均からは黙って抜け落ちます。**対応が付いた割合**を必ず併せて返します。
    """

    def test_identical_onsets_have_zero_deviation(self):
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([0.5, 1.0, 1.5], [0.5, 1.0, 1.5], tol=0.05)
        self.assertAlmostEqual(r["median_abs_dev"], 0.0, places=9)
        self.assertEqual(r["matched"], 3)
        self.assertAlmostEqual(r["matched_ratio"], 1.0)

    def test_a_constant_shift_is_reported_as_that_shift(self):
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([0.5, 1.0, 1.5], [0.52, 1.02, 1.52], tol=0.05)
        self.assertAlmostEqual(r["median_abs_dev"], 0.02, places=6)
        self.assertEqual(r["matched"], 3)

    def test_signed_bias_separates_early_from_late(self):
        # 遅れと進みが打ち消し合うと「ずれていない」に見える。符号つきも返す。
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([1.0, 2.0], [1.03, 1.97], tol=0.05)
        self.assertAlmostEqual(r["median_abs_dev"], 0.03, places=6)
        self.assertAlmostEqual(r["median_signed_dev"], 0.0, places=6)

    def test_a_dropped_onset_lowers_the_matched_ratio(self):
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([0.5, 1.0, 1.5], [0.5, 1.5], tol=0.05)
        self.assertEqual(r["matched"], 2)
        self.assertEqual(r["missed"], 1)
        self.assertAlmostEqual(r["matched_ratio"], 2 / 3)

    def test_an_inserted_onset_is_counted_separately(self):
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([0.5, 1.0], [0.5, 0.75, 1.0], tol=0.05)
        self.assertEqual(r["spurious"], 1)
        self.assertEqual(r["matched"], 2)

    def test_an_onset_outside_the_tolerance_is_not_matched(self):
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([1.0], [1.2], tol=0.05)
        self.assertEqual(r["matched"], 0)
        self.assertEqual(r["missed"], 1)
        self.assertEqual(r["spurious"], 1)

    def test_two_source_onsets_cannot_share_one_converted_onset(self):
        # **source 2 本に対し変換側が 1 本しかない場合。** 同じ onset を使い回すと
        # matched=2 になり、消えた onset（missed=1）が見えなくなる。
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([1.00, 1.02], [1.01], tol=0.05)
        self.assertEqual(r["matched"], 1)
        self.assertEqual(r["missed"], 1)
        self.assertEqual(r["spurious"], 0)

    def test_a_converted_onset_is_consumed_by_the_first_match(self):
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([1.00, 1.02, 1.04], [1.01, 1.03], tol=0.05)
        self.assertEqual(r["matched"], 2)
        self.assertEqual(r["missed"], 1)
        self.assertEqual(r["spurious"], 0)

    def test_matching_prefers_the_closest_candidate(self):
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([1.0], [1.04, 1.001], tol=0.05)
        self.assertAlmostEqual(r["median_abs_dev"], 0.001, places=6)

    def test_no_onsets_at_all_is_reported_not_crashed(self):
        # 無声だけのクリップは実在する（chunk の 39/89 が無声だった）。0 除算で落とさない。
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([], [], tol=0.05)
        self.assertEqual(r["matched"], 0)
        self.assertIsNone(r["median_abs_dev"])
        self.assertIsNone(r["matched_ratio"])

    def test_deviation_below_the_hop_resolution_cannot_be_resolved(self):
        # 実測でずれの中央値がちょうど 1 フレーム（5.8 ms）だった。分解能の下限であって
        # 「ほぼずれていない」ではない。この関係を明示しておく。
        from tools.timing_metrics import hop_resolution_seconds
        self.assertAlmostEqual(hop_resolution_seconds(44100, 256), 256 / 44100, places=9)

    def test_report_states_its_own_resolution(self):
        # 読む人が「5.8 ms のずれ」を分解能と区別できるようにする。
        from tools.timing_metrics import timing_report
        sr = 22050
        wav = np.zeros(sr, dtype=np.float32)
        wav[sr // 2: sr // 2 + 200] = np.hanning(200).astype(np.float32)
        r = timing_report(wav, wav, sr, hop=256)
        self.assertAlmostEqual(r["resolution_seconds"], 256 / sr, places=9)

    def test_detect_onsets_finds_a_pulse_train(self):
        from tools.timing_metrics import detect_onsets
        sr = 22050
        wav = np.zeros(sr * 2, dtype=np.float32)
        for t in (0.25, 0.75, 1.25, 1.75):
            i = int(t * sr)
            wav[i:i + 200] = np.hanning(200).astype(np.float32)
        got = detect_onsets(wav, sr)
        self.assertGreaterEqual(len(got), 3)
        for t in (0.25, 0.75, 1.25):
            self.assertTrue(any(abs(g - t) < 0.05 for g in got), f"{t}s の onset が出ていない")


class CerTests(unittest.TestCase):
    """明瞭度（M5 ゴール 2 の CER）。

    **歌唱の ASR は当てになりません。** したがって「変換後の CER が X%」には意味がなく、
    必ず 2 つの基準と並べます。

    | 基準 | 意味 |
    |---|---|
    | **source** | 参照テキスト。歌詞は未知なので、source の書き起こしを参照にする |
    | **上限** | GT mel をボコーダーに通した再合成。**ASR とボコーダー由来の誤りの下駄** |
    | 変換 | 変換後 |

    ASR は引数で受け取ります（単体テストで重いモデルを落とさないため）。
    """

    def test_identical_strings_have_zero_cer(self):
        from tools.asr_cer import cer
        self.assertAlmostEqual(cer("あいうえお", "あいうえお"), 0.0)

    def test_one_substitution_in_five_characters(self):
        from tools.asr_cer import cer
        self.assertAlmostEqual(cer("あいうえお", "あいうえか"), 0.2)

    def test_insertions_and_deletions_are_counted(self):
        from tools.asr_cer import cer
        self.assertAlmostEqual(cer("あいう", "あいうえ"), 1 / 3)
        self.assertAlmostEqual(cer("あいう", "あい"), 1 / 3)

    def test_cer_can_exceed_one(self):
        # 幻聴で長く書き起こすと 1 を超える。1 に丸めると「全滅」と区別できなくなる。
        from tools.asr_cer import cer
        self.assertGreater(cer("あ", "あいうえお"), 1.0)

    def test_an_empty_reference_is_rejected_rather_than_dividing_by_zero(self):
        from tools.asr_cer import cer
        with self.assertRaises(ValueError):
            cer("", "あいう")

    def test_normalization_ignores_spaces_and_punctuation(self):
        from tools.asr_cer import normalize_ja
        self.assertEqual(normalize_ja("あい、うえ お。"), "あいうえお")

    def test_normalization_folds_fullwidth_to_halfwidth(self):
        from tools.asr_cer import normalize_ja
        self.assertEqual(normalize_ja("ＡＢＣ１２３"), "abc123")

    def test_normalization_keeps_kana_distinct(self):
        # カナを潰すと別語が同じ扱いになり、CER が甘くなる。
        from tools.asr_cer import normalize_ja
        self.assertNotEqual(normalize_ja("シャツ"), normalize_ja("シヤツ"))

    def test_report_needs_the_ceiling_to_be_readable(self):
        # 上限が無いと、ASR とボコーダー由来の誤りを模型のせいにしてしまう。
        from tools.asr_cer import cer_report
        with self.assertRaises(ValueError):
            cer_report("s.wav", "c.wav", ceiling_wav=None,
                       transcribe=lambda p: "あいうえお")

    def test_report_puts_the_conversion_against_source_and_ceiling(self):
        from tools.asr_cer import cer_report
        texts = {"s.wav": "あいうえお", "c.wav": "あいうえか", "g.wav": "あいうえお"}
        r = cer_report("s.wav", "c.wav", ceiling_wav="g.wav",
                       transcribe=lambda p: texts[p])
        self.assertAlmostEqual(r["cer_converted"], 0.2)
        self.assertAlmostEqual(r["cer_ceiling"], 0.0)

    def test_report_flags_an_asr_that_produced_nothing(self):
        # 歌唱で ASR が空を返すことは実際に起こる。0% と区別できないと嘘になる。
        from tools.asr_cer import cer_report
        texts = {"s.wav": "", "c.wav": "あ", "g.wav": "あ"}
        r = cer_report("s.wav", "c.wav", ceiling_wav="g.wav", transcribe=lambda p: texts[p])
        self.assertTrue(r["asr_failed"])
        self.assertIsNone(r["cer_converted"])

    def test_an_unusable_ceiling_suppresses_the_difference(self):
        # 実測: Whisper が歌を全く書き起こせず、上限 CER も変換 CER も 1.0 になった。
        # そのとき差は 0.0 になり、「音響モデルは劣化させていない」と読めてしまう。
        # **判別できていないだけ**なので、差を出さずに旗を立てる。
        from tools.asr_cer import cer_report
        texts = {"s.wav": "あいうえお", "c.wav": "かきくけこ", "g.wav": "さしすせそ"}
        r = cer_report("s.wav", "c.wav", ceiling_wav="g.wav", transcribe=lambda p: texts[p])
        self.assertTrue(r["ceiling_unusable"])
        self.assertIsNone(r["cer_excess_over_ceiling"])

    def test_a_usable_ceiling_still_yields_the_difference(self):
        from tools.asr_cer import cer_report
        texts = {"s.wav": "あいうえおかきくけこ", "c.wav": "あいうえおかきくけさ",
                 "g.wav": "あいうえおかきくけこ"}
        r = cer_report("s.wav", "c.wav", ceiling_wav="g.wav", transcribe=lambda p: texts[p])
        self.assertFalse(r["ceiling_unusable"])
        self.assertAlmostEqual(r["cer_excess_over_ceiling"], 0.1, places=6)

    def test_the_usability_threshold_is_recorded(self):
        from tools.asr_cer import CEILING_MAX_CER, cer_report
        texts = {"s.wav": "あいうえお", "c.wav": "あいうえお", "g.wav": "あいうえお"}
        r = cer_report("s.wav", "c.wav", ceiling_wav="g.wav", transcribe=lambda p: texts[p])
        self.assertEqual(r["ceiling_max_cer"], CEILING_MAX_CER)

    def test_report_transcribes_each_clip_exactly_once(self):
        from tools.asr_cer import cer_report
        calls = []

        def tr(p):
            calls.append(p)
            return "あいうえお"
        cer_report("s.wav", "c.wav", ceiling_wav="g.wav", transcribe=tr)
        self.assertEqual(sorted(calls), ["c.wav", "g.wav", "s.wav"])

    def test_report_keeps_the_transcripts_for_inspection(self):
        # 数字だけ見て「明瞭度が落ちた」と言えない。何と聞こえたかを残す。
        from tools.asr_cer import cer_report
        texts = {"s.wav": "あいうえお", "c.wav": "あいうえか", "g.wav": "あいうえお"}
        r = cer_report("s.wav", "c.wav", ceiling_wav="g.wav", transcribe=lambda p: texts[p])
        self.assertEqual(r["transcripts"]["converted"], "あいうえか")


class SignalQualityTests(unittest.TestCase):
    """信号品質（M5 ゴール 2）。

    **歌声への妥当性は未検証です。** 使えるモデル（SQUIM / DNSMOS 系）はどれも話し声で
    学習されています。したがって**絶対値を品質として主張しません**。上限（GT mel を
    ボコーダーに通した再合成）と並べ、**そこからの差**だけを読みます。

    採点器は引数で受け取ります。
    """

    def _scorer(self, table):
        return lambda wav, sr: dict(table[float(wav[0])])

    def test_report_requires_the_ceiling(self):
        # 話し声モデルの絶対値には意味がない。上限が無いなら数値を出さない。
        from tools.signal_quality import quality_report
        wav = np.array([1.0, 0.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            quality_report(wav, ceiling=None, sr=16000,
                           score=self._scorer({1.0: {"mos": 3.0}}))

    def test_report_returns_the_gap_from_the_ceiling(self):
        from tools.signal_quality import quality_report
        cnv = np.array([1.0, 0.0], dtype=np.float32)
        ceil = np.array([2.0, 0.0], dtype=np.float32)
        r = quality_report(cnv, ceiling=ceil, sr=16000,
                           score=self._scorer({1.0: {"mos": 3.2}, 2.0: {"mos": 3.8}}))
        self.assertAlmostEqual(r["converted"]["mos"], 3.2)
        self.assertAlmostEqual(r["ceiling"]["mos"], 3.8)
        self.assertAlmostEqual(r["gap"]["mos"], -0.6, places=6)

    def test_every_shared_metric_gets_a_gap(self):
        from tools.signal_quality import quality_report
        cnv = np.array([1.0, 0.0], dtype=np.float32)
        ceil = np.array([2.0, 0.0], dtype=np.float32)
        r = quality_report(cnv, ceiling=ceil, sr=16000, score=self._scorer(
            {1.0: {"mos": 3.0, "stoi": 0.80}, 2.0: {"mos": 3.5, "stoi": 0.90}}))
        self.assertEqual(sorted(r["gap"]), ["mos", "stoi"])

    def test_a_metric_missing_from_one_side_is_not_silently_dropped(self):
        # 片側にしか無い指標の差を 0 と書くと、測れていないことが見えなくなる。
        from tools.signal_quality import quality_report
        cnv = np.array([1.0, 0.0], dtype=np.float32)
        ceil = np.array([2.0, 0.0], dtype=np.float32)
        r = quality_report(cnv, ceiling=ceil, sr=16000, score=self._scorer(
            {1.0: {"mos": 3.0, "stoi": 0.8}, 2.0: {"mos": 3.5}}))
        self.assertNotIn("stoi", r["gap"])
        self.assertIn("stoi", r["unpaired"])

    def test_report_carries_the_domain_caveat(self):
        # この数値を読む人が、話し声モデルだと知らずに引用しないようにする。
        from tools.signal_quality import quality_report
        cnv = np.array([1.0, 0.0], dtype=np.float32)
        ceil = np.array([2.0, 0.0], dtype=np.float32)
        r = quality_report(cnv, ceiling=ceil, sr=16000,
                           score=self._scorer({1.0: {"mos": 3.0}, 2.0: {"mos": 3.5}}))
        self.assertIn("caveat", r)
        self.assertIn("話し声", r["caveat"])

    def test_each_clip_is_scored_exactly_once(self):
        from tools.signal_quality import quality_report
        cnv = np.array([1.0, 0.0], dtype=np.float32)
        ceil = np.array([2.0, 0.0], dtype=np.float32)
        calls = []

        def score(wav, sr):
            calls.append(float(wav[0]))
            return {"mos": 3.0}
        quality_report(cnv, ceiling=ceil, sr=16000, score=score)
        self.assertEqual(sorted(calls), [1.0, 2.0])


class RtfTests(unittest.TestCase):
    """推論の RTF と peak VRAM（M5 ゴール 2）。

    **ゴール 2 は「特徴抽出と vocoder を含むか除くかを併記」を要求します。** 単一の RTF を
    出すと、どこまでを数えたかで 2 倍以上変わるため比較できません。**段ごとの内訳**を持たせ、
    含む / 除くの両方を計算します。

    段の実行そのものは引数で受け取ります（重いモデルを単体テストで動かさないため）。
    """

    def test_rtf_is_compute_time_over_audio_duration(self):
        from tools.rtf import rtf_from_stages
        r = rtf_from_stages({"flow": 1.0}, audio_seconds=2.0)
        self.assertAlmostEqual(r["rtf_total"], 0.5)

    def test_stages_are_summed(self):
        from tools.rtf import rtf_from_stages
        r = rtf_from_stages({"content": 0.5, "f0": 0.25, "flow": 0.25}, audio_seconds=2.0)
        self.assertAlmostEqual(r["rtf_total"], 0.5)

    def test_acoustic_only_excludes_features_and_vocoder(self):
        # 「1-step だから速い」を示すには acoustic だけの数字が要る。全体と混同しない。
        from tools.rtf import rtf_from_stages
        r = rtf_from_stages({"content": 1.0, "f0": 1.0, "flow": 0.5, "vocoder": 1.5},
                            audio_seconds=2.0)
        self.assertAlmostEqual(r["rtf_acoustic_only"], 0.25)
        self.assertAlmostEqual(r["rtf_total"], 2.0)

    def test_the_breakdown_keeps_every_stage(self):
        from tools.rtf import rtf_from_stages
        r = rtf_from_stages({"content": 1.0, "flow": 1.0}, audio_seconds=1.0)
        self.assertEqual(sorted(r["stage_seconds"]), ["content", "flow"])
        self.assertAlmostEqual(r["stage_rtf"]["content"], 1.0)

    def test_a_stage_that_is_not_a_known_category_is_rejected(self):
        # 未知の段を黙って「その他」に入れると、含む/除くの境界が曖昧になる。
        from tools.rtf import rtf_from_stages
        with self.assertRaises(ValueError):
            rtf_from_stages({"mystery": 1.0}, audio_seconds=1.0)

    def test_zero_length_audio_is_rejected(self):
        from tools.rtf import rtf_from_stages
        with self.assertRaises(ValueError):
            rtf_from_stages({"flow": 1.0}, audio_seconds=0.0)

    def test_realtime_claim_needs_end_to_end_not_acoustic_only(self):
        # doc の主張ルール: 「リアルタイム」は end-to-end 実測の後だけ。
        # acoustic だけが 1 を切っても total が超えていれば realtime_capable は False。
        from tools.rtf import rtf_from_stages
        r = rtf_from_stages({"content": 1.0, "flow": 0.1, "vocoder": 0.5}, audio_seconds=1.0)
        self.assertFalse(r["realtime_capable"])
        self.assertLess(r["rtf_acoustic_only"], 1.0)

    def test_measure_stage_records_wall_time_and_result(self):
        from tools.rtf import measure_stage
        out, sec = measure_stage(lambda: 42)
        self.assertEqual(out, 42)
        self.assertGreaterEqual(sec, 0.0)

    def test_repeats_take_the_median_not_the_first_run(self):
        # 初回は重みの読み込みや JIT を含む。それを RTF として報告しない。
        from tools.rtf import measure_stage
        calls = []

        def f():
            calls.append(1)
            return len(calls)
        out, sec = measure_stage(f, repeats=3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(out, 3)

