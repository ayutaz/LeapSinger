"""M5 ゴール 2 が要求する客観指標のうち、道具が無かった 4 つ。

**timing / CER / 信号品質 / 推論 RTF。** どれも「変換後の数値だけ」では読めないので、
このリポジトリの他の指標（`m3_verify.py` の content cos、`speaker_similarity.py` の回復率）
と同じく、**上限（GT mel をボコーダーに通した再合成）や source と並べて**読む形にします。

重いモデル（ASR / SQUIM）は**引数で受け取り**ます。単体テストはネットワークも GPU も使いません。
"""

import tempfile
import unittest
from pathlib import Path

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


class TestSetTests(unittest.TestCase):
    """M5 の test set を決定的に選ぶ（[実行計画](doc/svc-plan.md) M5「決定 4」）。

    **手で選ぶと偏ります。** M3 の測り直しで、`rglob` の並び順のまま 20 clip を取ったら
    **18 女性 / 2 男性**になりました。事前登録した構成（同性・異性の両方、日本語を含む、
    12 秒以上、held-out song と未知 source の両方）を**コードで満たさせます**。

    選択は seed から決定的に決まり、manifest に残せること。
    """

    def _pool(self, n_female=9, n_male=11, per_speaker=3):
        return [{"speaker": f"{g}{i}", "gender": g, "clip": f"c{k}", "seconds": 20.0,
                 "kind": "unseen"}
                for g, n in (("female", n_female), ("male", n_male))
                for i in range(1, n + 1) for k in range(per_speaker)]

    def test_every_speaker_contributes_exactly_one_clip(self):
        # VocalSet は女性 9 / 男性 11。全 20 名から 1 本ずつで n=20 になる。
        from tools.m5_testset import select_clips
        got = select_clips(self._pool(), n=20, seed=0)
        self.assertEqual(len({c["speaker"] for c in got}), 20)

    def test_both_genders_clear_the_floor(self):
        # 性別で回復率の伸びが倍違うので、片方が少なすぎると差を読めない。
        from tools.m5_testset import select_clips
        got = select_clips(self._pool(), n=20, seed=0)
        genders = [c["gender"] for c in got]
        for g in ("female", "male"):
            self.assertGreaterEqual(genders.count(g), 6, f"{g} が少なすぎる")

    def test_selection_is_deterministic_for_a_seed(self):
        from tools.m5_testset import select_clips
        pool = self._pool()
        a = [c["speaker"] + c["clip"] for c in select_clips(pool, n=20, seed=0)]
        b = [c["speaker"] + c["clip"] for c in select_clips(pool, n=20, seed=0)]
        self.assertEqual(a, b)

    def test_a_different_seed_gives_a_different_set(self):
        from tools.m5_testset import select_clips
        pool = self._pool()
        a = {c["speaker"] + c["clip"] for c in select_clips(pool, n=20, seed=0)}
        b = {c["speaker"] + c["clip"] for c in select_clips(pool, n=20, seed=1)}
        self.assertNotEqual(a, b)

    def test_clips_shorter_than_the_calibrated_length_are_dropped(self):
        # 話者類似度の較正は 12 秒以上でしか通らない。短い clip を入れると測れなくなる。
        from tools.m5_testset import select_clips
        pool = self._pool()
        for c in pool:
            c["seconds"] = 6.0
        with self.assertRaises(ValueError):
            select_clips(pool, n=20, seed=0)

    def test_no_speaker_appears_twice(self):
        # 1 人から複数取ると、その歌手の癖が test set を支配する。
        from tools.m5_testset import select_clips
        got = select_clips(self._pool(), n=20, seed=0)
        counts = {}
        for c in got:
            counts[c["speaker"]] = counts.get(c["speaker"], 0) + 1
        self.assertLessEqual(max(counts.values()), 1)

    def test_it_refuses_when_there_are_not_enough_speakers(self):
        # 足りないなら黙って 1 人から 2 本取らず、止める。
        from tools.m5_testset import select_clips
        with self.assertRaises(ValueError):
            select_clips(self._pool(n_female=5, n_male=5, per_speaker=4), n=20, seed=0)

    def test_it_refuses_a_set_that_is_lopsided_by_gender(self):
        # M3 では 18 女性 / 2 男性の set で「男性 source が暗い」を取り逃がした。
        # 素材が偏っているなら、黙って偏った set を返さずに止める。
        from tools.m5_testset import select_clips
        with self.assertRaises(ValueError):
            select_clips(self._pool(n_female=18, n_male=2, per_speaker=1), n=20, seed=0)

    def test_target_holdout_clips_are_kept_separate_from_unseen(self):
        # ゴール 1 は held-out song と未知 source singer の**両方**を要求する。
        from tools.m5_testset import build_testset
        unseen = self._pool()
        holdout = [{"speaker": "ritsu", "gender": "female", "clip": f"h{k}",
                    "seconds": 20.0, "kind": "holdout"} for k in range(8)]
        ts = build_testset(unseen, holdout, n_unseen=20, n_holdout=6, seed=0)
        self.assertEqual(len(ts["unseen"]), 20)
        self.assertEqual(len(ts["holdout"]), 6)
        self.assertTrue(all(c["kind"] == "holdout" for c in ts["holdout"]))

    def test_holdout_takes_several_segments_from_one_long_song(self):
        # hold-out 曲は 3 曲しかないが 1 曲 3〜5 分ある。曲単位 split を保ったまま
        # 区間を分けて 6 clip にする。**別の曲を混ぜて水増ししない。**
        from tools.m5_testset import segment_holdout
        songs = [{"speaker": "ritsu", "gender": "female", "song": "a", "path": "a.wav",
                  "full_seconds": 280.0, "kind": "holdout"},
                 {"speaker": "ritsu", "gender": "female", "song": "b", "path": "b.wav",
                  "full_seconds": 200.0, "kind": "holdout"}]
        got = segment_holdout(songs, n=4, seconds=20.0, seed=0)
        self.assertEqual(len(got), 4)
        self.assertEqual(sorted({c["song"] for c in got}), ["a", "b"])

    def test_segments_from_the_same_song_do_not_overlap(self):
        # 同じ区間を 2 回測ると n を水増ししただけになる。
        from tools.m5_testset import segment_holdout
        songs = [{"speaker": "ritsu", "gender": "female", "song": "a", "path": "a.wav",
                  "full_seconds": 100.0, "kind": "holdout"}]
        got = segment_holdout(songs, n=4, seconds=20.0, seed=0)
        spans = sorted((c["start"], c["start"] + c["seconds"]) for c in got)
        for (_s1, e1), (s2, _s2) in zip(spans, spans[1:], strict=False):
            self.assertLessEqual(e1, s2, f"区間が重なっている: {spans}")

    def test_segments_are_spread_over_each_song(self):
        # 曲頭に固まるとイントロばかりになる（無声率が高く、変換の検証に向かない）。
        from tools.m5_testset import segment_holdout
        songs = [{"speaker": "ritsu", "gender": "female", "song": "a", "path": "a.wav",
                  "full_seconds": 300.0, "kind": "holdout"}]
        got = segment_holdout(songs, n=3, seconds=20.0, seed=0)
        self.assertGreater(max(c["start"] for c in got), 100.0)

    def test_segments_are_chosen_where_the_singer_is_actually_singing(self):
        # **実際に踏んだ。** 曲を等分して窓の中でずらすと、イントロや間奏が選ばれる。
        # hold-out 6 本のうち 3 本が有声率 3.5〜34% になり、話者性を測れなくなった
        # （リツ本人の source が参照と 0.39 しか一致しない = 上限 0.67 から大きく外れる）。
        from tools.m5_testset import segment_holdout
        # 前半 100 秒は無声、後半 200 秒が有声、という曲を模す
        def voiced_ratio(path, start, seconds):
            return 0.05 if start < 100.0 else 0.9
        songs = [{"speaker": "ritsu", "gender": "female", "song": "a", "path": "a.wav",
                  "full_seconds": 300.0, "kind": "holdout"}]
        got = segment_holdout(songs, n=3, seconds=20.0, seed=0,
                              voiced_ratio=voiced_ratio, min_voiced=0.5)
        for c in got:
            self.assertGreaterEqual(c["start"], 100.0, f"無声区間が選ばれた: {c}")

    def test_the_voiced_ratio_is_recorded_for_each_segment(self):
        # 後から「なぜこの区間か」を見られるようにする。
        from tools.m5_testset import segment_holdout
        songs = [{"speaker": "ritsu", "gender": "female", "song": "a", "path": "a.wav",
                  "full_seconds": 300.0, "kind": "holdout"}]
        got = segment_holdout(songs, n=2, seconds=20.0, seed=0,
                              voiced_ratio=lambda p, s, d: 0.8, min_voiced=0.5)
        self.assertAlmostEqual(got[0]["voiced_ratio"], 0.8)

    def test_it_refuses_when_no_segment_is_voiced_enough(self):
        # 黙って無声区間を返さない。素材か閾値を見直させる。
        from tools.m5_testset import segment_holdout
        songs = [{"speaker": "ritsu", "gender": "female", "song": "a", "path": "a.wav",
                  "full_seconds": 300.0, "kind": "holdout"}]
        with self.assertRaises(ValueError):
            segment_holdout(songs, n=2, seconds=20.0, seed=0,
                            voiced_ratio=lambda p, s, d: 0.1, min_voiced=0.5)

    def test_without_a_voiced_check_the_old_behaviour_is_kept(self):
        from tools.m5_testset import segment_holdout
        songs = [{"speaker": "ritsu", "gender": "female", "song": "a", "path": "a.wav",
                  "full_seconds": 300.0, "kind": "holdout"}]
        got = segment_holdout(songs, n=3, seconds=20.0, seed=0)
        self.assertEqual(len(got), 3)

    def test_it_refuses_a_song_too_short_to_yield_its_share(self):
        from tools.m5_testset import segment_holdout
        songs = [{"speaker": "ritsu", "gender": "female", "song": "a", "path": "a.wav",
                  "full_seconds": 25.0, "kind": "holdout"}]
        with self.assertRaises(ValueError):
            segment_holdout(songs, n=4, seconds=20.0, seed=0)

    def test_the_manifest_records_the_seed_and_the_rule(self):
        # 後から「なぜこの 20 本か」を復元できないと、比較の土台が消える。
        from tools.m5_testset import build_testset
        unseen = self._pool()
        holdout = [{"speaker": "ritsu", "gender": "female", "clip": f"h{k}",
                    "seconds": 20.0, "kind": "holdout"} for k in range(8)]
        ts = build_testset(unseen, holdout, n_unseen=20, n_holdout=6, seed=3)
        self.assertEqual(ts["manifest"]["seed"], 3)
        self.assertEqual(ts["manifest"]["min_seconds"], 12.0)
        self.assertEqual(ts["manifest"]["n_unseen"], 20)

    def test_transpose_is_assigned_by_gender_not_left_to_the_operator(self):
        # 男性 source を移調し忘れると、モデルではなく音域差を測ることになる。
        from tools.m5_testset import build_testset
        unseen = self._pool()
        holdout = [{"speaker": "ritsu", "gender": "female", "clip": f"h{k}",
                    "seconds": 20.0, "kind": "holdout"} for k in range(8)]
        ts = build_testset(unseen, holdout, n_unseen=20, n_holdout=6, seed=0)
        for c in ts["unseen"]:
            self.assertEqual(c["transpose"], 12 if c["gender"] == "male" else 0)

    def test_holdout_clips_are_never_transposed(self):
        # target 自身の曲は音域が合っている。移調すると別の実験になる。
        from tools.m5_testset import build_testset
        unseen = self._pool()
        holdout = [{"speaker": "ritsu", "gender": "female", "clip": f"h{k}",
                    "seconds": 20.0, "kind": "holdout"} for k in range(8)]
        ts = build_testset(unseen, holdout, n_unseen=20, n_holdout=6, seed=0)
        self.assertTrue(all(c["transpose"] == 0 for c in ts["holdout"]))


class ConvertOutputNamingTests(unittest.TestCase):
    """変換結果のファイル名（M5 で同じ曲から複数区間を取るため）。

    **同じ WAV から区間を変えて 2 回変換すると、出力名が衝突します。** hold-out は 3 曲しか
    なく、1 曲から 2 区間取るのでこれが実際に起きます。**黙って上書きされると clip 数が
    減るだけで例外にならない**（GTSinger で 1,922 ファイルが 3 名に潰れたのと同じ形）。
    """

    def test_the_stem_defaults_to_the_input_filename(self):
        from tools.svc_convert import output_stem
        self.assertEqual(output_stem("download/ritsu/anywhere.wav", tag=None), "anywhere")

    def test_a_tag_disambiguates_two_segments_of_one_song(self):
        from tools.svc_convert import output_stem
        a = output_stem("a/anywhere.wav", tag="seg0")
        b = output_stem("a/anywhere.wav", tag="seg1")
        self.assertNotEqual(a, b)

    def test_the_tag_keeps_the_source_name_readable(self):
        # 出力を見て、どの曲のどの区間かが分かること。
        from tools.svc_convert import output_stem
        self.assertIn("anywhere", output_stem("a/anywhere.wav", tag="seg1"))

    def test_the_tag_is_sanitised_for_the_filesystem(self):
        from tools.svc_convert import output_stem
        got = output_stem("a/anywhere.wav", tag="seg/1 2")
        for bad in r"/\ ":
            self.assertNotIn(bad, got)


class BlindPreferenceTests(unittest.TestCase):
    """blind preference test の道具（M5 ゴール 3）。

    **N=1 でも blind の条件は要ります。** ラベルを隠し、順序を randomize し、
    どちらが A でどちらが B だったかを**後から復元できる**こと。復元できないと
    集計そのものが成り立ちません。

    **聴く前に答えが分かってはいけない**ので、割り当ては seed から決まりつつ、
    提示側のファイル名からは system が読めない形にします。
    """

    def _pairs(self, n=4):
        return [{"clip": f"c{i}", "a_system": "leapsvc", "b_system": "seedvc"}
                for i in range(n)]

    def test_each_pair_gets_a_random_side_assignment(self):
        from tools.blind_test import assign_sides
        got = assign_sides(["c0", "c1", "c2", "c3"], systems=("leapsvc", "seedvc"), seed=0)
        self.assertEqual(len(got), 4)
        for row in got:
            self.assertEqual(sorted([row["A"], row["B"]]), ["leapsvc", "seedvc"])

    def test_the_assignment_is_deterministic_for_a_seed(self):
        from tools.blind_test import assign_sides
        a = assign_sides(["c0", "c1"], systems=("x", "y"), seed=7)
        b = assign_sides(["c0", "c1"], systems=("x", "y"), seed=7)
        self.assertEqual(a, b)

    def test_both_systems_appear_on_each_side_across_the_set(self):
        # 常に A が LeapSVC だと、順序の癖が preference に化ける。
        from tools.blind_test import assign_sides
        got = assign_sides([f"c{i}" for i in range(20)], systems=("x", "y"), seed=0)
        a_sides = [r["A"] for r in got]
        self.assertGreater(a_sides.count("x"), 3)
        self.assertGreater(a_sides.count("y"), 3)

    def test_presentation_order_is_shuffled_not_the_clip_order(self):
        # clip の並びも混ぜる。曲順で聴くと後半に慣れが出る。
        from tools.blind_test import assign_sides
        clips = [f"c{i}" for i in range(20)]
        got = [r["clip"] for r in assign_sides(clips, systems=("x", "y"), seed=1)]
        self.assertNotEqual(got, clips)
        self.assertEqual(sorted(got), sorted(clips))

    def test_clip_tags_are_read_from_either_naming(self):
        # LeapSVC は `<name>__<tag>_converted.wav`、正規化した baseline は
        # `<tag>_converted.wav`。**どちらからも同じ tag が取れること。**
        from tools.blind_test import clip_tag
        self.assertEqual(clip_tag("m10_dona__unseen00_converted.wav"), "unseen00")
        self.assertEqual(clip_tag("unseen00_converted.wav"), "unseen00")

    def test_finding_clips_pairs_the_two_systems_by_tag(self):
        from tools.blind_test import find_clips
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a").mkdir()
            (root / "b").mkdir()
            (root / "a" / "song__unseen00_converted.wav").write_bytes(b"x")
            (root / "b" / "unseen00_converted.wav").write_bytes(b"x")
            a, b = find_clips(root / "a"), find_clips(root / "b")
            self.assertEqual(sorted(set(a) & set(b)), ["unseen00"])

    def test_source_and_ceiling_files_are_not_treated_as_clips(self):
        from tools.blind_test import find_clips
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for n in ("x__unseen00_converted.wav", "x__unseen00_source.wav",
                      "x__unseen00_vocoder_only.wav"):
                (root / n).write_bytes(b"x")
            self.assertEqual(sorted(find_clips(root)), ["unseen00"])

    def test_tally_counts_votes_per_system_not_per_side(self):
        from tools.blind_test import tally
        sheet = [{"clip": "c0", "A": "x", "B": "y", "vote": "A"},
                 {"clip": "c1", "A": "y", "B": "x", "vote": "A"},
                 {"clip": "c2", "A": "x", "B": "y", "vote": "B"}]
        got = tally(sheet)
        self.assertEqual(got["wins"]["x"], 1)
        self.assertEqual(got["wins"]["y"], 2)

    def test_ties_are_kept_not_dropped(self):
        # 引き分けを捨てると、差が無かったことが見えなくなる。
        from tools.blind_test import tally
        sheet = [{"clip": "c0", "A": "x", "B": "y", "vote": "tie"},
                 {"clip": "c1", "A": "x", "B": "y", "vote": "A"}]
        got = tally(sheet)
        self.assertEqual(got["ties"], 1)
        self.assertEqual(got["n_voted"], 2)

    def test_an_unvoted_row_is_reported_not_silently_skipped(self):
        from tools.blind_test import tally
        sheet = [{"clip": "c0", "A": "x", "B": "y", "vote": ""},
                 {"clip": "c1", "A": "x", "B": "y", "vote": "A"}]
        got = tally(sheet)
        self.assertEqual(got["n_missing"], 1)
        self.assertEqual(got["n_voted"], 1)

    def test_an_invalid_vote_is_rejected_rather_than_guessed(self):
        from tools.blind_test import tally
        with self.assertRaises(ValueError):
            tally([{"clip": "c0", "A": "x", "B": "y", "vote": "leapsvc"}])

    def test_the_sign_test_p_value_is_reported_with_n(self):
        # N=1 の評価者なので、統計は「参考」。それでも n と p を出して読み手に判断させる。
        from tools.blind_test import tally
        sheet = [{"clip": f"c{i}", "A": "x", "B": "y", "vote": "A"} for i in range(10)]
        got = tally(sheet)
        self.assertEqual(got["n_decisive"], 10)
        self.assertLess(got["p_two_sided"], 0.01)

    def test_a_split_result_is_not_significant(self):
        from tools.blind_test import tally
        sheet = ([{"clip": f"a{i}", "A": "x", "B": "y", "vote": "A"} for i in range(5)]
                 + [{"clip": f"b{i}", "A": "x", "B": "y", "vote": "B"} for i in range(5)])
        got = tally(sheet)
        self.assertGreater(got["p_two_sided"], 0.5)


class NormaliseOutputsTests(unittest.TestCase):
    """外部 baseline の出力を、測定ツールが読める形へ揃える（M5）。

    Seed-VC は `<tag>/vc_<tag>__<ref>_<...>.wav` に書き、LeapSVC は
    `<name>__<tag>_converted.wav` に書きます。**測定ツールは後者の形しか読みません。**

    **source をコピーで作らず、必ず同じ区間から取ること。** timing と CER は source を
    基準にするので、別の区間を source として置くと比較が壊れます。
    """

    def test_it_pairs_each_tag_with_its_converted_file(self):
        from tools.normalise_outputs import find_seedvc_outputs
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for tag in ("unseen00", "holdout01"):
                (root / tag).mkdir()
                (root / tag / f"vc_{tag}__ref_1.0_30_0.7.wav").write_bytes(b"x")
            got = find_seedvc_outputs(root)
            self.assertEqual(sorted(got), ["holdout01", "unseen00"])

    def test_it_ignores_the_segments_and_reference_files(self):
        from tools.normalise_outputs import find_seedvc_outputs
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "_segments").mkdir()
            (root / "_segments" / "unseen00.wav").write_bytes(b"x")
            (root / "_target_ref.wav").write_bytes(b"x")
            (root / "unseen01").mkdir()
            (root / "unseen01" / "vc_unseen01__ref.wav").write_bytes(b"x")
            self.assertEqual(sorted(find_seedvc_outputs(root)), ["unseen01"])

    def test_a_tag_with_two_outputs_is_rejected_rather_than_guessed(self):
        # 設定違いの出力が両方残っていると、どちらを測ったのか分からなくなる。
        from tools.normalise_outputs import find_seedvc_outputs
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "unseen00").mkdir()
            (root / "unseen00" / "vc_unseen00__ref_1.0_30_0.7.wav").write_bytes(b"x")
            (root / "unseen00" / "vc_unseen00__ref_1.0_50_0.7.wav").write_bytes(b"x")
            with self.assertRaises(ValueError):
                find_seedvc_outputs(root)

    def test_the_normalised_name_matches_what_the_metrics_expect(self):
        from tools.normalise_outputs import normalised_name
        self.assertEqual(normalised_name("unseen00", "converted"), "unseen00_converted.wav")
        self.assertEqual(normalised_name("unseen00", "source"), "unseen00_source.wav")


class GuardRailTests(unittest.TestCase):
    """事前登録した guard rail 判定（[実行計画](doc/svc-plan.md) M5「決定 1」）。

    **絶対閾値は置きません。** 「悪化」= **clip 間のばらつき（標準誤差）を超えて相手より低い**。
    手元のデータ量では任意の数字を発明することになるためです。

    **preference で勝っても guard rail を 1 つでも落としていたら「より良い」とは書きません。**
    その判断をコードに書いて、後から緩められないようにします。
    """

    def test_a_clear_drop_beyond_the_standard_error_is_flagged(self):
        from tools.guard_rail import compare_metric
        ours = [0.50] * 10
        theirs = [0.80] * 10
        r = compare_metric(ours, theirs, higher_is_better=True)
        self.assertTrue(r["worse"])

    def test_a_difference_inside_the_noise_is_not_flagged(self):
        from tools.guard_rail import compare_metric
        rng = np.random.default_rng(0)
        ours = list(rng.normal(0.80, 0.10, 20))
        theirs = list(rng.normal(0.81, 0.10, 20))
        r = compare_metric(ours, theirs, higher_is_better=True)
        self.assertFalse(r["worse"])

    def test_being_better_is_never_flagged_as_worse(self):
        from tools.guard_rail import compare_metric
        r = compare_metric([0.9] * 10, [0.5] * 10, higher_is_better=True)
        self.assertFalse(r["worse"])

    def test_lower_is_better_metrics_flip_the_direction(self):
        # CER やずれは小さいほうが良い。方向を取り違えると結論が反転する。
        from tools.guard_rail import compare_metric
        r = compare_metric([0.9] * 10, [0.2] * 10, higher_is_better=False)
        self.assertTrue(r["worse"])

    def test_the_margin_and_threshold_are_reported(self):
        from tools.guard_rail import compare_metric
        r = compare_metric([0.5] * 10, [0.8] * 10, higher_is_better=True)
        self.assertIn("diff", r)
        self.assertIn("threshold", r)
        self.assertLess(r["diff"], 0)

    def test_a_verdict_requires_every_rail_to_hold(self):
        from tools.guard_rail import verdict
        rails = {"content": {"worse": False}, "similarity": {"worse": True}}
        v = verdict(rails, preference_winner="ours", ours="ours")
        self.assertFalse(v["may_claim_better"])
        self.assertEqual(v["failed_rails"], ["similarity"])

    def test_winning_preference_with_all_rails_intact_allows_the_claim(self):
        from tools.guard_rail import verdict
        rails = {"content": {"worse": False}, "similarity": {"worse": False}}
        v = verdict(rails, preference_winner="ours", ours="ours")
        self.assertTrue(v["may_claim_better"])

    def test_losing_preference_never_allows_the_claim(self):
        from tools.guard_rail import verdict
        rails = {"content": {"worse": False}}
        v = verdict(rails, preference_winner="theirs", ours="ours")
        self.assertFalse(v["may_claim_better"])

    def test_a_tie_in_preference_does_not_allow_the_claim(self):
        from tools.guard_rail import verdict
        rails = {"content": {"worse": False}}
        v = verdict(rails, preference_winner=None, ours="ours")
        self.assertFalse(v["may_claim_better"])


class BatchConvertTests(unittest.TestCase):
    """複数 clip を 1 プロセスで変換する（M5 の 26 clip を現実的な時間で回すため）。

    **律速はモデルの読み込みです。** 実測で GPU 使用率は 5% しかなく、1 clip ごとに
    ContentVec と RMVPE を読み直すぶんが所要のほとんどでした。**モデルは 1 度だけ読み、
    全 clip に使い回します。**

    重いモデルは引数で受け取るので、このテストはネットワークも GPU も使いません。
    """

    def _job(self, tag, tp=0, start=0.0, seconds=1.0):
        return {"tag": tag, "path": f"{tag}.wav", "transpose": tp,
                "start": start, "seconds": seconds}

    def test_models_are_loaded_once_for_the_whole_batch(self):
        from tools.svc_batch import run_batch
        loads = []

        def load_models():
            loads.append(1)
            return {"model": "m"}
        run_batch([self._job("a"), self._job("b"), self._job("c")],
                  load_models=load_models, convert_one=lambda job, models: None)
        self.assertEqual(len(loads), 1)

    def test_each_job_is_converted_once(self):
        from tools.svc_batch import run_batch
        done = []
        run_batch([self._job("a"), self._job("b")],
                  load_models=lambda: {}, convert_one=lambda job, m: done.append(job["tag"]))
        self.assertEqual(done, ["a", "b"])

    def test_a_failing_clip_does_not_abort_the_rest(self):
        # 26 本の途中で 1 本落ちたときに、そこまでの結果を捨てない。
        from tools.svc_batch import run_batch
        done = []

        def convert(job, m):
            if job["tag"] == "b":
                raise RuntimeError("boom")
            done.append(job["tag"])
        rep = run_batch([self._job("a"), self._job("b"), self._job("c")],
                        load_models=lambda: {}, convert_one=convert)
        self.assertEqual(done, ["a", "c"])
        self.assertEqual(rep["failed"], ["b"])

    def test_the_failure_reason_is_kept(self):
        from tools.svc_batch import run_batch

        def convert(job, m):
            raise RuntimeError("特定の理由")
        rep = run_batch([self._job("a")], load_models=lambda: {}, convert_one=convert)
        self.assertIn("特定の理由", rep["errors"]["a"])

    def test_already_converted_clips_are_skipped(self):
        # 途中から再開できること。26 本を最初からやり直さない。
        from tools.svc_batch import run_batch
        done = []
        rep = run_batch([self._job("a"), self._job("b")],
                        load_models=lambda: {},
                        convert_one=lambda job, m: done.append(job["tag"]),
                        is_done=lambda job: job["tag"] == "a")
        self.assertEqual(done, ["b"])
        self.assertEqual(rep["skipped"], ["a"])

    def test_models_are_not_loaded_when_everything_is_already_done(self):
        # 全部済んでいるのに数百 MB を読み込まない。
        from tools.svc_batch import run_batch
        loads = []
        run_batch([self._job("a")], load_models=lambda: loads.append(1),
                  convert_one=lambda job, m: None, is_done=lambda job: True)
        self.assertEqual(loads, [])

    def test_a_float_transpose_does_not_break_the_progress_line(self):
        # **実際に踏んだ。** jobs.tsv は transpose を float で持つのに、進捗表示が :+3d
        # だったので、**書き出しの後**に例外が出て 16 本が「失敗」と記録された。
        # 出力は完全なのに失敗と報告されるのが最も危ない（再実行しても is_done で飛ばされる）。
        from tools.svc_batch import format_progress
        line = format_progress({"tag": "unseen19", "transpose": 12.0}, seconds=20.0,
                               elapsed=41.2)
        self.assertIn("unseen19", line)
        self.assertIn("+12", line)

    def test_an_int_transpose_formats_the_same_way(self):
        from tools.svc_batch import format_progress
        a = format_progress({"tag": "t", "transpose": 12}, seconds=1.0, elapsed=1.0)
        b = format_progress({"tag": "t", "transpose": 12.0}, seconds=1.0, elapsed=1.0)
        self.assertEqual(a, b)

    def test_a_negative_transpose_keeps_its_sign(self):
        from tools.svc_batch import format_progress
        self.assertIn("-7", format_progress({"tag": "t", "transpose": -7.0},
                                            seconds=1.0, elapsed=1.0))

    def test_the_report_counts_what_happened(self):
        from tools.svc_batch import run_batch
        rep = run_batch([self._job("a"), self._job("b")],
                        load_models=lambda: {}, convert_one=lambda job, m: None)
        self.assertEqual(rep["converted"], ["a", "b"])
        self.assertEqual(rep["n_total"], 2)


class ConversionConsistencyTests(unittest.TestCase):
    """同じ test set の中で変換条件が揃っていること（M5）。

    **実際に混ざりました。** 26 clip のうち 15 本を `svc_convert.py`（chunk 20 秒）、
    11 本を `svc_batch.py`（当時の既定 10 秒）で作ってしまい、**20 秒の clip では
    chunk 1 個と 2 個で境界処理が変わります**。比較の土台としては使えません。

    条件は各 clip の `*_convert.json` に残るので、**測る前に揃っているかを確かめます。**
    """

    def _rec(self, **kw):
        base = {"ckpt": "c.pt", "spk_id": 22, "num_steps": 1, "device": "cuda",
                "chunk_sec": 20.0}
        base.update(kw)
        return base

    def test_a_consistent_set_passes(self):
        from tools.svc_batch import check_consistent
        r = check_consistent([self._rec(), self._rec(), self._rec()])
        self.assertTrue(r["consistent"])

    def test_a_differing_chunk_size_is_caught(self):
        from tools.svc_batch import check_consistent
        r = check_consistent([self._rec(chunk_sec=20.0), self._rec(chunk_sec=10.0)])
        self.assertFalse(r["consistent"])
        self.assertIn("chunk_sec", r["differing"])

    def test_a_differing_checkpoint_is_caught(self):
        from tools.svc_batch import check_consistent
        r = check_consistent([self._rec(ckpt="a.pt"), self._rec(ckpt="b.pt")])
        self.assertIn("ckpt", r["differing"])

    def test_a_differing_device_is_caught(self):
        # CPU と GPU は bit 一致しない。同じ set に混ぜない。
        from tools.svc_batch import check_consistent
        r = check_consistent([self._rec(device="cpu"), self._rec(device="cuda")])
        self.assertIn("device", r["differing"])

    def test_per_clip_fields_are_not_compared(self):
        # transpose や start は clip ごとに違って当然。ここを条件違いと呼ばない。
        from tools.svc_batch import check_consistent
        r = check_consistent([self._rec(transpose=0, start=0.0),
                              self._rec(transpose=12, start=42.0)])
        self.assertTrue(r["consistent"])

    def test_a_missing_field_is_reported_not_ignored(self):
        # 古い形式の json（chunk_sec を持たない）が混ざったら気づけること。
        from tools.svc_batch import check_consistent
        rec = self._rec()
        del rec["chunk_sec"]
        r = check_consistent([self._rec(), rec])
        self.assertFalse(r["consistent"])
        self.assertIn("chunk_sec", r["differing"])


class PitchMetricsTests(unittest.TestCase):
    """F0 追従と V/UV（M5 ゴール 2）。

    **変換済みの WAV から測ります。** `m3_verify.py` は自分で変換してしまうので、
    既に作った test set（両システム分）には使えません。**同じ出力を両系で測る**必要が
    あります。

    F0 抽出器は引数で受け取るので、このテストは重いモデルを使いません。
    """

    def test_correlation_is_one_for_identical_pitch(self):
        from tools.pitch_metrics import pitch_report
        f0 = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float32)
        uv = np.ones(4, dtype=np.float32)
        r = pitch_report((f0, uv), (f0, uv))
        self.assertAlmostEqual(r["f0_corr"], 1.0, places=6)
        self.assertAlmostEqual(r["median_abs_semitones"], 0.0, places=6)

    def test_a_constant_octave_shift_shows_as_twelve_semitones(self):
        from tools.pitch_metrics import pitch_report
        f0 = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float32)
        uv = np.ones(4, dtype=np.float32)
        r = pitch_report((f0, uv), (f0 * 2, uv))
        self.assertAlmostEqual(r["median_abs_semitones"], 12.0, places=4)
        self.assertAlmostEqual(r["f0_corr"], 1.0, places=6)

    def test_an_expected_transpose_is_removed_before_comparing(self):
        # 男性 source は +12 半音で変換している。それを誤差として数えない。
        from tools.pitch_metrics import pitch_report
        f0 = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float32)
        uv = np.ones(4, dtype=np.float32)
        r = pitch_report((f0, uv), (f0 * 2, uv), transpose=12.0)
        self.assertAlmostEqual(r["median_abs_semitones"], 0.0, places=4)

    def test_unvoiced_frames_are_excluded_from_the_pitch_error(self):
        # 無声フレームの F0 は意味を持たない。混ぜると誤差が壊れる。
        from tools.pitch_metrics import pitch_report
        f0a = np.array([100.0, 0.0, 120.0], dtype=np.float32)
        f0b = np.array([100.0, 999.0, 120.0], dtype=np.float32)
        uv = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        r = pitch_report((f0a, uv), (f0b, uv))
        self.assertAlmostEqual(r["median_abs_semitones"], 0.0, places=6)

    def test_uv_agreement_counts_matching_frames(self):
        from tools.pitch_metrics import pitch_report
        f0 = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32)
        a = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        r = pitch_report((f0, a), (f0, b))
        self.assertAlmostEqual(r["uv_agree"], 0.75)

    def test_length_mismatch_is_trimmed_to_the_shorter(self):
        # 系ごとに端数フレームが違う。**黙って伸ばさず、短いほうへ揃える。**
        from tools.pitch_metrics import pitch_report
        f0a = np.array([100.0, 110.0, 120.0], dtype=np.float32)
        f0b = np.array([100.0, 110.0], dtype=np.float32)
        r = pitch_report((f0a, np.ones(3, np.float32)), (f0b, np.ones(2, np.float32)))
        self.assertEqual(r["n_frames"], 2)

    def test_no_voiced_frames_returns_none_rather_than_zero(self):
        # 無声だけの clip で 0 を返すと「完璧に一致」と読めてしまう。
        from tools.pitch_metrics import pitch_report
        z = np.zeros(4, dtype=np.float32)
        r = pitch_report((z, z), (z, z))
        self.assertIsNone(r["median_abs_semitones"])
        self.assertIsNone(r["f0_corr"])


class FailureTaxonomyTests(unittest.TestCase):
    """failure sample の分類（M5 ゴール 4、[評価計画](doc/svc-evaluation.md) 7 節）。

    **failure を除外せず、分類して残します。** 悪い clip を捨てると平均は良くなりますが、
    何が壊れるのかが分からなくなります。

    **機械的に判定できるものだけを扱います。** 「こもっている」「不自然」のような聴感は
    ここでは扱いません（blind test の担当）。判定に使う数値は既に測ってあるものです。
    """

    def _clip(self, **kw):
        base = {"tag": "unseen00", "uv_agree": 0.99, "median_abs_semitones": 0.1,
                "matched_ratio": 0.7, "cer_excess": 0.1, "centroid_ratio": 1.0,
                "peak": 0.5, "finite": True}
        base.update(kw)
        return base

    def test_a_clean_clip_has_no_categories(self):
        from tools.failure_taxonomy import classify
        self.assertEqual(classify(self._clip()), [])

    def test_a_silent_output_is_flagged(self):
        from tools.failure_taxonomy import classify
        self.assertIn("silence", classify(self._clip(peak=1e-5)))

    def test_a_non_finite_output_is_flagged(self):
        from tools.failure_taxonomy import classify
        self.assertIn("nonfinite", classify(self._clip(finite=False)))

    def test_an_octave_error_is_flagged_as_pitch(self):
        from tools.failure_taxonomy import classify
        self.assertIn("pitch", classify(self._clip(median_abs_semitones=11.8)))

    def test_a_small_pitch_deviation_is_not_flagged(self):
        from tools.failure_taxonomy import classify
        self.assertNotIn("pitch", classify(self._clip(median_abs_semitones=0.4)))

    def test_bad_voicing_agreement_is_flagged(self):
        from tools.failure_taxonomy import classify
        self.assertIn("voicing", classify(self._clip(uv_agree=0.70)))

    def test_many_lost_onsets_are_flagged_as_timing(self):
        from tools.failure_taxonomy import classify
        self.assertIn("timing", classify(self._clip(matched_ratio=0.35)))

    def test_a_large_cer_excess_is_flagged_as_content(self):
        from tools.failure_taxonomy import classify
        self.assertIn("content", classify(self._clip(cer_excess=0.6)))

    def test_a_dark_output_is_flagged_as_timbre(self):
        # 上限に対して明るさが大きく外れている（M3 で耳で分かった劣化）。
        from tools.failure_taxonomy import classify
        self.assertIn("timbre", classify(self._clip(centroid_ratio=0.5)))

    def test_a_bright_output_is_also_flagged(self):
        # 明るすぎるのも外れ。符号つきで見ると打ち消し合う。
        from tools.failure_taxonomy import classify
        self.assertIn("timbre", classify(self._clip(centroid_ratio=1.8)))

    def test_missing_values_do_not_produce_a_category(self):
        # 測れなかった軸を「失敗」と数えない（CER は 8 clip で判別不能だった）。
        from tools.failure_taxonomy import classify
        self.assertEqual(classify(self._clip(cer_excess=None,
                                             median_abs_semitones=None)), [])

    def test_several_categories_can_apply_at_once(self):
        from tools.failure_taxonomy import classify
        got = classify(self._clip(peak=1e-6, uv_agree=0.5))
        self.assertIn("silence", got)
        self.assertIn("voicing", got)

    def test_summarise_counts_clips_per_category(self):
        from tools.failure_taxonomy import summarise
        rows = [{"tag": "a", "categories": ["pitch"]},
                {"tag": "b", "categories": ["pitch", "timbre"]},
                {"tag": "c", "categories": []}]
        s = summarise(rows)
        self.assertEqual(s["counts"]["pitch"], 2)
        self.assertEqual(s["n_clean"], 1)
        self.assertEqual(s["n_total"], 3)

    def test_summarise_keeps_the_tags_for_each_category(self):
        # 「どの clip か」が分からないと後から聴き直せない。
        from tools.failure_taxonomy import summarise
        rows = [{"tag": "a", "categories": ["pitch"]}, {"tag": "b", "categories": ["pitch"]}]
        self.assertEqual(summarise(rows)["tags"]["pitch"], ["a", "b"])


class SvcDefaultStepsTests(unittest.TestCase):
    """SVC 推論の既定 step 数（2026-09-01 の掃引で決定）。

    **SVC 経路だけを変えます。** 未知 source 20 clip の掃引で、事前登録した規則
    （話者類似度を最大化、内容 cos の低下 0.02 以内、同点なら小さい step）が
    **16** を選びました。回復率 58.9% -> 68.6%、内容 cos −0.019、flow 時間は 2.1 倍。

    **SVS 経路（`infer_mel`）は触りません。** 今回の測定は SVC 経路のみで、SVS の
    1 step 品質は別途検証済みだからです。**測っていない経路の既定を変えないこと。**
    """

    def test_svc_inference_defaults_to_the_chosen_steps(self):
        import inspect

        from infer import infer_svc_mel
        from tools.svc_defaults import SVC_NUM_STEPS
        self.assertEqual(
            inspect.signature(infer_svc_mel).parameters["num_steps"].default, SVC_NUM_STEPS)

    def test_the_chosen_value_is_what_the_sweep_selected(self):
        from tools.svc_defaults import SVC_NUM_STEPS
        self.assertEqual(SVC_NUM_STEPS, 16)

    def test_the_svs_path_is_left_alone(self):
        # SVS の既定を巻き込むと、測っていない経路の挙動を変えてしまう。
        import inspect

        from infer import infer_mel
        self.assertEqual(inspect.signature(infer_mel).parameters["num_steps"].default, 10)

    def test_the_svc_clis_use_the_same_default(self):
        # ツールごとに既定が違うと、どの step で測ったのか分からなくなる。
        import re
        from pathlib import Path

        from tools.svc_defaults import SVC_NUM_STEPS
        for name in ("svc_convert.py", "svc_batch.py", "m3_verify.py"):
            src = (Path(__file__).parent / "tools" / name).read_text(encoding="utf-8")
            m = re.search(r'"--num-steps".*?default=(\w+)', src, re.S)
            self.assertIsNotNone(m, f"{name} に --num-steps が無い")
            self.assertEqual(m.group(1), "SVC_NUM_STEPS",
                             f"{name} が既定を直書きしている（{m.group(1)}）")
        self.assertEqual(SVC_NUM_STEPS, 16)


class SvcGanSmokeTests(unittest.TestCase):
    """SVC 経路の GAN 配線が smoke で踏まれること。

    **GAN 付き SVC は一度も動かしていない経路です。** smoke は SVS 側の GAN しか踏んで
    おらず、SVC は `gan.enabled: false` のままでした。**vast.ai で数時間を投じる前に、
    配線が通ることを手元で確かめます。**

    ここでは smoke の構成だけを検査します（実際の学習は smoke 本体が回す）。
    """

    def _smoke_src(self):
        from pathlib import Path
        return (Path(__file__).parent / "tools" / "smoke" / "run_smoke.py").read_text(
            encoding="utf-8")

    def test_a_gan_enabled_svc_config_is_written(self):
        # svc.yaml とは別に、GAN を有効にした config を書くこと。
        src = self._smoke_src()
        self.assertIn("svc_gan.yaml", src)

    def test_the_svc_gan_config_turns_gan_on(self):
        src = self._smoke_src()
        self.assertRegex(src, r'svc_gan\["gan"\]\.update\([^)]*enabled=True')

    def test_the_gan_starts_early_enough_for_a_short_smoke(self):
        # gan_start_step が既定の 2000 のままだと、20 step の smoke では GAN 経路を
        # 一度も通らずに「通った」ことになる。
        src = self._smoke_src()
        self.assertRegex(src, r'svc_gan\["gan"\]\.update\([^)]*gan_start_step=\d')

    def test_there_is_a_stage_running_svc_with_gan(self):
        src = self._smoke_src()
        self.assertIn("svc-train-gan", src)

    def test_the_stage_is_registered_in_the_stage_list(self):
        # 関数を書いても一覧に足さなければ走らない。
        src = self._smoke_src()
        self.assertRegex(src, r'\("svc-train-gan",\s*st_svc_train_gan')

