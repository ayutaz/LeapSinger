"""実モデル（ContentVec / RMVPE）を使う統合テスト。**既定では走りません。**

    LEAPSINGER_INTEGRATION=1 uv run python -m unittest test_svc_preprocess_integration -v

数百 MB のモデルをダウンロードし GPU を使うので、`test_svc_preprocess.py` の単体テストとは
分けています（[実行計画](doc/svc-plan.md) M1 ゴール 6）。単体テストは偽 encoder を注入して
契約だけを見ます。ここで見るのは「実物を差し込んでも同じ契約が成り立つか」です。
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

RUN = os.environ.get("LEAPSINGER_INTEGRATION") == "1"
DEVICE = os.environ.get("LEAPSINGER_INTEGRATION_DEVICE", "cpu")


@unittest.skipUnless(RUN, "LEAPSINGER_INTEGRATION=1 のときだけ走ります")
class RealEncoderTests(unittest.TestCase):
    """実物の ContentVec と RMVPE を差し込んだときの契約。"""

    @classmethod
    def setUpClass(cls):
        from leapsinger.config import MelSpec
        from preprocess.svc.encoders import ContentVecEncoder, RmvpeF0
        cls.mel = MelSpec()
        cls.encoder = ContentVecEncoder(device=DEVICE)
        cls.f0 = RmvpeF0(device=DEVICE)

    def _wav(self, seconds=3.0):
        t = np.arange(int(seconds * self.mel.sr)) / self.mel.sr
        f0 = 220.0 * 2 ** (0.5 * np.sin(2 * np.pi * 0.7 * t))
        phase = 2 * np.pi * np.cumsum(f0) / self.mel.sr
        w = sum((0.5 / k) * np.sin(k * phase) for k in range(1, 20))
        return (w / (np.abs(w).max() + 1e-9) * 0.7).astype(np.float32)

    def test_contentvec_returns_the_declared_width_at_50hz(self):
        out = self.encoder(np.zeros(16000, dtype=np.float32), 16000)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[1], self.encoder.manifest()["content_encoder_hidden"])
        # 16 kHz / stride 320 = 50 Hz。畳み込みの受容野ぶん 1 フレーム減る。
        self.assertAlmostEqual(out.shape[0], 50, delta=2)

    def test_contentvec_rejects_a_wrong_sample_rate(self):
        with self.assertRaises(ValueError):
            self.encoder(np.zeros(44100, dtype=np.float32), 44100)

    def test_rmvpe_returns_exactly_the_mel_frame_count(self):
        from leapsinger.mel import wav_to_mel_nhv
        for seconds in (1.0, 2.5, 3.7):
            with self.subTest(seconds=seconds):
                wav = self._wav(seconds)
                mel = wav_to_mel_nhv(wav, sr=self.mel.sr, n_fft=self.mel.n_fft,
                                     hop=self.mel.hop, win=self.mel.win,
                                     n_mels=self.mel.n_mels, fmin=self.mel.fmin,
                                     fmax=self.mel.fmax)
                f0, uv = self.f0(wav, self.mel.sr, self.mel.hop)
                self.assertEqual(f0.shape, (mel.shape[1],))
                self.assertEqual(uv.shape, (mel.shape[1],))

    def test_the_real_chain_produces_a_shard_the_loader_can_read(self):
        from preprocess.svc.extract import extract_phrase
        from preprocess.svc.shard import build_shard
        from svc_dataset import SVCFeatureDataset

        out = extract_phrase(self._wav(3.0), self.mel.sr, content_encoder=self.encoder,
                             f0_extract=self.f0, mel=self.mel)
        d = Path(tempfile.mkdtemp(prefix="int_"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        build_shard({"songA_0000": out, "songA_0001": out}, d,
                    n_dims=256, subset_seed=0, frame_rate=self.mel.frame_rate,
                    manifest_extra={**self.encoder.manifest(), **self.f0.manifest()})
        ds = SVCFeatureDataset([str(d)], split="train", eval_songs=0)
        item = ds[0]
        self.assertEqual(item["content"].shape[1], 256)
        self.assertEqual(item["content"].shape[0], item["target_mel"].shape[1])


if __name__ == "__main__":
    unittest.main()
