from __future__ import annotations

from drpo.methods._baseline_loader import load_baseline_module

pairwise_spo_loss = load_baseline_module("spo", "losses.py").pairwise_spo_loss
