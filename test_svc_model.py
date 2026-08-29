import unittest
import json
import tempfile
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
