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
    from drpo.utils.tensors import select_disjoint_pref_indices, select_top_fraction_indices
else:
    select_top_fraction_indices = None


@unittest.skipIf(torch is None, "requires torch")
class TensorHelperTest(unittest.TestCase):
    def test_select_top_fraction_indices(self):
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8])
        indices = select_top_fraction_indices(scores, 0.5)
        self.assertEqual(set(indices.tolist()), {1, 3})

    def test_full_feature_fraction_disables_filtering(self):
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8])
        best_idx, worst_idx, feature_idx = select_disjoint_pref_indices(
            scores,
            num_pos=1,
            num_neg=1,
            feature_top_fraction=1.0,
        )
        self.assertEqual(best_idx.tolist(), [1])
        self.assertEqual(worst_idx.tolist(), [0])
        self.assertEqual(set(feature_idx.tolist()), {0, 1, 2, 3})


if __name__ == "__main__":
    unittest.main()
