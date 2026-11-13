"""SR contracts, supervision consistency, official scan gradients and complete resume."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn
from model.student import EchoRenderSR
from model.card.mamba import SSMStage, selective_scan
from model.card.loss import RestorationLoss
from model.card.restoration import MaskedResidualUpsample
from tool.training_config import load_training_config, reserve_output_dir
from tool.temporal import FORMAT, synthesize, augmentation_ratio, tiled_forward
from tool.dataloader import build_dataloader
from tool.resume import read_checkpoint, capture_rng, restore_rng
from tool.trainer import learning_rate, fit
import preview


def tiny_model():
    cfg = {
        "input_channels": 6,
        "output_channels": 8,
        "input_layout": "BCHW",
        "motion": "feat",
        "gradient_checkpointing": False,
        "ssm": {
            "rotation": "clockwise",
            "directions": 2,
            "window": 4,
            "stride": 2,
            "tokens": 8,
            "prompt_rank": 4,
        },
        "backbone": [[-1, 1, "InputAdapter", [8]], [-1, 1, "SSMStage", [8, 2, 4, 2]]],
        "head": [],
        "reconstruction": {
            "scale": 4,
            "width": 4,
            "naf_depths": [1],
            "restormer_depths": [1],
            "middle": 1,
        },
    }
    return EchoRenderSR(cfg)


class RGBFixture(nn.Module):
    scale = 4

    def forward(self, x, previous=None):
        y = torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")
        return {key: y for key in ("sr", "deblur", "output")}


class MetricsFixture(nn.Module):
    def forward(self, p, t):
        error = (p - t).square().mean().clamp_min(1e-8)
        return [
            {"psnr": float(-10 * torch.log10(error)), "ssim": 0.8, "lpips": 0.2}
            for _ in p
        ]


class SRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_strict_current_config(self):
        cfg = load_training_config()
        self.assertNotIn("teacher_weights", cfg["model"])
        self.assertEqual(cfg["dataset"]["scale"], 4)
        with tempfile.TemporaryDirectory() as tmp:
            import yaml

            cfg["training"]["releas"] = True
            p = Path(tmp) / "bad.yaml"
            p.write_text(yaml.safe_dump(cfg))
            with self.assertRaises(ValueError):
                load_training_config(p)

    def test_schedule_endpoints(self):
        self.assertEqual(learning_rate(0, 13, 0.001, 0.000001), 0.001)
        self.assertEqual(learning_rate(12, 13, 0.001, 0.000001), 0.000001)
        a = load_training_config()["augmentation"]
        self.assertAlmostEqual(augmentation_ratio(1, 101, a), 0.1)
        self.assertAlmostEqual(augmentation_ratio(51, 101, a), 0.25)
        self.assertEqual(augmentation_ratio(76, 101, a), 0)
        self.assertAlmostEqual(augmentation_ratio(50, 100, a), 0.25)
        self.assertEqual(augmentation_ratio(75, 100, a), 0)

    def test_synthesis_targets_share_noise(self):
        cfg = load_training_config()["augmentation"]
        hr = torch.rand(2, 3, 64, 64)
        lr = torch.rand(2, 3, 16, 16)
        x, t = synthesize(lr, hr, 0.25, cfg, generator=torch.Generator().manual_seed(4))
        from tool.temporal import gaussian_blur

        torch.testing.assert_close(
            t["sr"] - gaussian_blur(hr, 2.0), t["deblur"] - hr, atol=1e-7, rtol=1e-5
        )
        self.assertEqual(x.shape, lr.shape)
        self.assertTrue(torch.equal(t["output"], hr))
        original, targets = synthesize(lr, hr, 0, cfg)
        torch.testing.assert_close(original, lr)
        for value in targets.values():
            torch.testing.assert_close(value, hr)

    def test_noise_is_observable_in_lr_and_shared_with_hr_targets(self):
        cfg = load_training_config()["augmentation"]
        lr = torch.full((1, 3, 16, 16), 0.5)
        hr = torch.full((1, 3, 64, 64), 0.5)
        x, target = synthesize(
            lr, hr, 0.25, cfg, mode="noise", generator=torch.Generator().manual_seed(17)
        )
        noise = (
            torch.randn(lr.shape, generator=torch.Generator().manual_seed(17)) * 0.05
        )
        torch.testing.assert_close(x, lr + noise)
        expected = torch.nn.functional.interpolate(
            noise, size=(64, 64), mode="bilinear", align_corners=False
        )
        torch.testing.assert_close(target["sr"] - hr, expected, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(target["deblur"], target["sr"])

    def test_three_heads_gradients_and_history(self):
        model = tiny_model()
        rgb = torch.rand(1, 3, 8, 9)
        previous = torch.rand_like(rgb)
        prediction = model(rgb, previous)
        targets = {k: torch.rand(1, 3, 32, 36) for k in prediction}
        loss, _ = RestorationLoss(load_training_config()["loss"])(prediction, targets)
        loss.backward()
        rgb_heads = [stage.rgb[-1] for stage in model.head.sr]
        for head in (*rgb_heads, model.head.deblur_rgb, model.head.denoise_rgb):
            self.assertIsNotNone(head.weight.grad)
            self.assertTrue(torch.isfinite(head.weight.grad).all())
        self.assertIsNotNone(model.motion.predictor[-1].weight.grad)
        scan = (
            next(m for m in model.modules() if isinstance(m, SSMStage)).blocks[0].mixer
        )
        for parameter in (
            scan.zero_map,
            scan.embedding_a,
            scan.parameters_proj.weight,
            scan.prompt.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_masked_upsample_preserves_anchors_and_rgb_base(self):
        stage = MaskedResidualUpsample(32)
        rgb = torch.rand(2, 3, 5, 7)
        features = torch.rand(2, 32, 5, 7)
        base = torch.nn.functional.interpolate(
            rgb, scale_factor=2, mode="bicubic", align_corners=False
        )
        up, image = stage(features, rgb)
        self.assertEqual(up.shape, (2, 32, 10, 14))
        torch.testing.assert_close(up[..., ::2, ::2], features, atol=0, rtol=0)
        torch.testing.assert_close(image, base, atol=0, rtol=0)
        with torch.no_grad():
            stage.rgb[-1].bias.fill_(0.125)
        _, image = stage(features, rgb)
        expected = torch.full_like(base, 0.125)
        expected[..., ::2, ::2] = 0
        torch.testing.assert_close(image, base + expected, atol=0, rtol=0)

    def test_shifted_window_uses_local_valid_anchors(self):
        stage = MaskedResidualUpsample(32)
        shifted = stage.shifted
        with torch.no_grad():
            for layer in (shifted.query, shifted.key, shifted.value, shifted.output):
                layer.weight.zero_()
                layer.bias.zero_()
            for layer in (shifted.value, shifted.output):
                layer.weight.copy_(torch.eye(32))
        original = torch.arange(35, dtype=torch.float32).reshape(1, 1, 5, 7)
        original = original.expand(1, 32, 5, 7)
        update = shifted(torch.zeros(1, 32, 10, 14), original, stage.tokens)
        self.assertTrue(torch.isfinite(update).all())
        # At the image edge, the shifted window only sees the upper-left 2×2 anchors.
        torch.testing.assert_close(
            update[..., 0, 1], original[..., :2, :2].mean((-2, -1))
        )
        torch.testing.assert_close(
            update[..., 0, 5], original[..., :2, 2:6].mean((-2, -1))
        )
        self.assertEqual(torch.count_nonzero(update[..., ::2, ::2]).item(), 0)

    def test_sr_supervision_reaches_both_attention_stages(self):
        model = tiny_model()
        features = torch.rand(1, 8, 5, 7, requires_grad=True)
        rgb = torch.rand(1, 3, 5, 7)
        # The zero-initialized RGB projection opens the attention path after one update.
        for stage in model.head.sr:
            nn.init.normal_(stage.rgb[-1].weight, std=0.02)
        prediction = model.head(features, rgb)["sr"]
        (prediction - torch.rand_like(prediction)).abs().mean().backward()
        for stage in model.head.sr:
            for parameter in (stage.tokens, stage.position.weight,
                              stage.standard.query.weight, stage.shifted.key.weight,
                              stage.shifted.value.weight, stage.shifted.relative_bias):
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_preview_lists_only_current_inference_checkpoints(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for name, row in (("old", {"epoch": 1}), ("new", {"format": FORMAT})):
                folder = output / name
                folder.mkdir()
                (folder / "best.pt").touch()
                (folder / "history.json").write_text(json.dumps([row]))
            with patch.object(preview, "OUTPUT", output):
                self.assertEqual(preview.checkpoints(), [output / "new" / "best.pt"])

    def test_sort_restore_deterministic_eval(self):
        model = tiny_model().eval()
        rgb = torch.rand(1, 3, 7, 9)
        with torch.no_grad():
            first = model(rgb)["output"]
            second = model(rgb)["output"]
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(first.shape, (1, 3, 28, 36))

    def test_official_scan_values_and_gradients(self):
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

        u = torch.randn(2, 5, 4, device=device, requires_grad=True)
        delta = torch.rand_like(u, requires_grad=True)
        A = -torch.rand(4, 3, device=device)
        B = torch.randn(2, 5, 3, device=device, requires_grad=True)
        C = torch.randn_like(B, requires_grad=True)
        D = torch.ones(4, device=device)
        actual = selective_scan(u, delta, A, B, C, D)
        expected = selective_scan_ref(
            u.transpose(1, 2).contiguous(),
            delta.transpose(1, 2).contiguous(),
            A,
            B.transpose(1, 2).contiguous(),
            C.transpose(1, 2).contiguous(),
            D,
        ).transpose(1, 2)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        gradients = torch.autograd.grad(
            actual.sum(), (u, delta, B, C), retain_graph=True
        )
        references = torch.autograd.grad(expected.sum(), (u, delta, B, C))
        for a, b in zip(gradients, references):
            torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)

    def test_tiling_preserves_geometry(self):
        rgb = torch.rand(1, 3, 23, 31)
        result = tiled_forward(RGBFixture(), rgb, None, tile_size=12, overlap=4)[
            "output"
        ]
        expected = RGBFixture()(rgb)["output"]
        torch.testing.assert_close(result, expected, atol=2e-6, rtol=1e-5)

    def test_clean_paired_patch_and_epoch_sampling(self):
        cfg = load_training_config()
        cfg["loader"]["num_workers"] = 0
        cfg["dataset"]["lr_patch_size"] = 16
        with tempfile.TemporaryDirectory() as tmp:
            cfg["dataset"]["root"] = tmp
            cfg["dataset"]["splits"]["train"] = {"source": "train", "scenes": ["000"]}
            for kind in ("train_sharp", "train_sharp_bicubic/X4"):
                (Path(tmp) / "train" / kind / "000").mkdir(parents=True)
            rng = np.random.default_rng(1)
            for i in range(3):
                array = rng.integers(0, 255, (24, 28, 3), dtype=np.uint8)
                Image.fromarray(array).save(
                    Path(tmp) / "train/train_sharp_bicubic/X4/000" / f"{i:08}.png"
                )
                Image.fromarray(array.repeat(4, 0).repeat(4, 1)).save(
                    Path(tmp) / "train/train_sharp/000" / f"{i:08}.png"
                )
            loader = build_dataloader(cfg)
            sample = loader.dataset[0]
            self.assertEqual(sample["lr"].shape[-2:], (16, 16))
            self.assertEqual(sample["hr"].shape[-2:], (64, 64))
            expected = sample["lr"].repeat_interleave(4, -2).repeat_interleave(4, -1)
            torch.testing.assert_close(expected, sample["hr"])

    def test_checkpoint_rng(self):
        state = capture_rng()
        expected = torch.rand(5)
        restore_rng(state)
        torch.testing.assert_close(torch.rand(5), expected)

    def test_preview_rejects_bad_controls(self):
        request = {
            "weights": "x",
            "input": "x",
            "device": "cpu",
            "tile_size": 32,
            "tile_overlap": 32,
            "max_frames": 1,
            "mode": "noise",
            "strength": 0.25,
            "blur_sigma": 2.0,
            "noise_std": 0.2,
        }
        with self.assertRaises(ValueError):
            preview.validate_request(request)
        request["tile_overlap"] = 0
        request["max_frames"] = 11
        with self.assertRaises(ValueError):
            preview.validate_request(request)

    def test_tensorboard_post_proxy(self):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from urllib.request import Request, urlopen

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                data = json.dumps({"received": body}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        service = preview.PreviewService()
        service.tensorboard_port = lambda: upstream.server_port
        server = ThreadingHTTPServer(("127.0.0.1", 0), preview.make_handler(service))
        workers = [
            threading.Thread(target=s.serve_forever, daemon=True)
            for s in (upstream, server)
        ]
        for worker in workers:
            worker.start()
        try:
            request = Request(
                f"http://127.0.0.1:{server.server_port}/tensorboard/data/plugin/timeseries/timeSeries",
                data=b'{"tag":"train/loss"}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request) as response:
                self.assertEqual(
                    json.load(response), {"received": {"tag": "train/loss"}}
                )
        finally:
            for instance in (server, upstream):
                instance.shutdown()
                instance.server_close()
            service.close()

    def test_output_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            reserve_output_dir(Path(tmp) / "run")
            second = reserve_output_dir(Path(tmp) / "run")
            self.assertEqual(second.name, "run_2")

    def test_fit_resume_matches_uninterrupted(self):
        class Loader:
            def __init__(self):
                self.dataset = type("Data", (), {"epoch": 0})()
                self.samples = [
                    {
                        "lr": torch.rand(1, 3, 16, 16),
                        "hr": torch.rand(1, 3, 64, 64),
                        "names": ["000/0"],
                    }
                ]

            def __len__(self):
                return 1

            def __iter__(self):
                return iter([self.samples])

        cfg = load_training_config()
        cfg["training"].update(
            epochs=2, max_steps=None, precision="float32", micro_batch_size=1
        )
        cfg["logging"].update(progress=False, tensorboard=False)
        cfg["reporting"]["enabled"] = False
        cfg["validation"].update(tile_size=32, tile_overlap=4)
        cfg["training"]["early_stopping"]["patience"] = 0
        loader = Loader()
        model = tiny_model()
        initial = deepcopy(model.state_dict())
        criterion = RestorationLoss(cfg["loss"])
        with tempfile.TemporaryDirectory() as tmp:
            cfg["runtime"]["output_dir"] = str(Path(tmp) / "full")
            Path(cfg["runtime"]["output_dir"]).mkdir()
            generator = torch.Generator().manual_seed(5)
            rng = capture_rng()
            fit(
                model,
                loader,
                loader,
                criterion,
                torch.optim.SGD(model.parameters(), lr=0.001),
                MetricsFixture(),
                cfg,
                torch.device("cpu"),
                generator,
            )
            expected = deepcopy(model.state_dict())
            model.load_state_dict(initial)
            restore_rng(rng)
            generator.manual_seed(5)
            cfg["runtime"]["output_dir"] = str(Path(tmp) / "resume")
            Path(cfg["runtime"]["output_dir"]).mkdir()
            from tool import trainer

            original = trainer.save_checkpoint

            def interrupt(path, payload):
                original(path, payload)
                if Path(path).name == "best.pt" and payload["epoch"] == 1:
                    raise KeyboardInterrupt

            with patch("tool.trainer.save_checkpoint", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    fit(
                        model,
                        loader,
                        loader,
                        criterion,
                        torch.optim.SGD(model.parameters(), lr=0.001),
                        MetricsFixture(),
                        cfg,
                        torch.device("cpu"),
                        generator,
                    )
            saved = read_checkpoint(Path(cfg["runtime"]["output_dir"]) / "last.pt")
            fit(
                model,
                loader,
                loader,
                criterion,
                torch.optim.SGD(model.parameters(), lr=0.001),
                MetricsFixture(),
                cfg,
                torch.device("cpu"),
                generator,
                saved,
            )
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
