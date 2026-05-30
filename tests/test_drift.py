import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if torch is not None:
    from drpo.drift import DRIFT_KERNELS, drift_loss, kernel_logits
else:
    drift_loss = None


@unittest.skipIf(torch is None, "requires torch")
class DriftLossTest(unittest.TestCase):
    def test_drift_loss_is_differentiable(self):
        generated = torch.randn(1, 3, 4, requires_grad=True)
        positive = torch.randn(1, 2, 4)
        negative = torch.randn(1, 2, 4)
        loss, info = drift_loss(
            generated,
            positive,
            negative,
            positive_weight=2.0,
            negative_weight=1.0,
            radii=(0.1,),
        )
        self.assertEqual(loss.shape, (1,))
        self.assertIn("feature_scale", info)
        loss.mean().backward()
        self.assertIsNotNone(generated.grad)

    def test_all_drift_kernels_are_differentiable(self):
        for kernel in DRIFT_KERNELS:
            with self.subTest(kernel=kernel):
                generated = torch.randn(1, 3, 4, requires_grad=True)
                positive = torch.randn(1, 2, 4)
                negative = torch.randn(1, 2, 4)
                loss, info = drift_loss(
                    generated,
                    positive,
                    negative,
                    positive_weight=2.0,
                    negative_weight=1.0,
                    radii=(0.1,),
                    kernel=kernel,
                )
                self.assertEqual(loss.shape, (1,))
                self.assertIn("drift_kernel", info)
                loss.mean().backward()
                self.assertIsNotNone(generated.grad)

    def test_kernel_logits_order_close_points_higher_for_distance_kernels(self):
        x = torch.tensor([[[0.0], [2.0]]])
        y = torch.tensor([[[0.0], [3.0]]])
        for kernel in ("laplacian", "exponential", "rbf"):
            with self.subTest(kernel=kernel):
                logits = kernel_logits(x, y, radius=1.0, kernel=kernel, scale=1.0)
                self.assertGreater(logits[0, 0, 0].item(), logits[0, 0, 1].item())

    def test_cosine_kernel_orders_aligned_vectors_higher(self):
        x = torch.tensor([[[1.0, 0.0]]])
        y = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
        logits = kernel_logits(x, y, radius=1.0, kernel="cosine")
        self.assertGreater(logits[0, 0, 0].item(), logits[0, 0, 1].item())


if __name__ == "__main__":
    unittest.main()
