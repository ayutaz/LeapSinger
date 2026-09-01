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

