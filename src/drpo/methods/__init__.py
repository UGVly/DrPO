"""Training method implementations exposed by DrPO.

The public method namespace keeps the main DrPO method and comparison
baselines next to each other. Legacy top-level packages remain available while
scripts and imports migrate to this namespace.
"""

__all__ = [
    "drpo",
    "draft",
    "dpo",
    "grpo",
    "neighbor_grpo",
    "sdxl_draft",
    "sdxl_drpo",
    "sdxl_grpo",
    "spo",
    "vggflow",
]
