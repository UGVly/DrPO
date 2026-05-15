from __future__ import annotations

from drpo.methods._baseline_loader import load_baseline_module

_losses = load_baseline_module("grpo", "losses.py")

construct_neighbor_noises = _losses.construct_neighbor_noises
latent_distances = _losses.latent_distances
neighbor_grpo_loss = _losses.neighbor_grpo_loss
quasi_norm_advantages = _losses.quasi_norm_advantages
softmax_distance_log_probs = _losses.softmax_distance_log_probs
validate_neighbor_sigma = _losses.validate_neighbor_sigma
