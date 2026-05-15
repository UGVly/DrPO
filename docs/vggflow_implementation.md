# VGG-Flow Implementation Notes

This document explains the current `vggflow` implementation in this repository, with formulas mapped directly to code.

Relevant files:

- `scripts/train/vggflow.sh`
- `baselines/vggflow/train_lora.py`
- `baselines/vggflow/reward_gradient.py`
- `src/drpo/sdturbo.py`

## 1. What This Implementation Is

This repository does not contain the original multi-step VGG-Flow training loop verbatim.

Instead, it adapts the core VGG-Flow idea to the one-step SD-Turbo setting:

\[
x_0^{\text{target}}
=
x_0^{\text{ref}}
+
\eta(t)\,\lambda\,\nabla_{x_0} R(\mathrm{decode}(x_0), p)
\]

Where:

- \(x_0\) is the clean latent predicted by the current one-step UNet
- \(x_0^{\text{ref}}\) is the clean latent predicted by the frozen reference UNet
- \(R(\cdot, p)\) is the reward for prompt \(p\), currently PickScore only
- \(\lambda\) is `reward_scale`
- \(\eta(t)\) is a timestep-dependent multiplier controlled by `eta_mode`

This high-level design is stated in [baselines/vggflow/train_lora.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:2).

The official multi-step value-network consistency branch is explicitly omitted here because SD-Turbo training in this repo is one-step only. See [trainer.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:10).

## 2. Training Entrypoint and Main Hyperparameters

The canonical launch script is [scripts/train/vggflow.sh](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/scripts/train/vggflow.sh:1).

Important runtime hyperparameters exposed there:

- `REWARD_SCALE`
- `ETA_MODE`
- `QUANTILE_CLIPPING`
- `RGRAD_CLIP_THRESHOLD`
- `RGRAD_QUANTILE`
- `RGRAD_JITTER_COUNT`
- `RGRAD_JITTER_STD`
- `REWARD_MASKING`
- `REWARD_MASK_THRESHOLD`
- `UNET_REG_SCALE`

These are passed through to [baselines/vggflow/train_lora.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:64).

## 3. One-Step SD-Turbo Prediction

The model is trained in latent space. For each prompt, the trainer samples noisy latents

\[
z_t \sim \mathcal N(0, I)
\]

at a fixed timestep \(t\), default `generation_timestep = 999`.

The current UNet predicts a one-step SD-Turbo output `model_pred`, then converts it into a clean latent:

\[
x_0^\theta = \Pi_{\text{turbo}}(z_t, \epsilon_\theta)
\]

In code:

- UNet forward: [trainer.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:378)
- latent projection: [trainer.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:379)

The actual SD-Turbo one-step projection is implemented in [src/drpo/sdturbo.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/src/drpo/sdturbo.py:30):

\[
x_0
=
\left(\frac{z_t - 0.9977\,\epsilon}{0.0683}\right) 0.9996 + 0.0292\,\epsilon
\]

This is the repository's clean-latent parameterization for one-step SD-Turbo.

## 4. Reference Branch

The trainer also keeps a frozen copy of the UNet, `ref_unet`, and runs it on the exact same noisy latents and text embeddings:

\[
x_0^{\text{ref}} = \Pi_{\text{turbo}}(z_t, \epsilon_{\text{ref}})
\]

Code:

- frozen reference forward: [trainer.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:381)
- projected reference clean latent: [trainer.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:383)

This reference output is the "base flow field" that the reward gradient perturbs.

## 5. Reward Model and Differentiable Reward

Reward gradient logic lives in [baselines/vggflow/reward_gradient.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/reward_gradient.py:1).

The reward model is `DifferentiablePickScoreReward`, which loads:

- a Hugging Face `AutoProcessor`
- a Hugging Face `AutoModel`

See [reward_gradient.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/reward_gradient.py:46).

Images are:

1. decoded from latent space
2. resized to the PickScore image size
3. normalized by the processor's image mean/std

See [reward_gradient.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/reward_gradient.py:90).

The reward itself is computed as a scaled cosine similarity:

\[
R(I, p)
=
\exp(s)\,
\left\langle
\frac{f_{\text{text}}(p)}{\|f_{\text{text}}(p)\|},
\frac{f_{\text{img}}(I)}{\|f_{\text{img}}(I)\|}
\right\rangle
\]

This corresponds to [reward_gradient.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/reward_gradient.py:117).

## 6. Reward Gradient in Clean-Latent Space

The core VGG-Flow adaptation is:

\[
g = \nabla_{x_0} R(\mathrm{decode}(x_0), p)
\]

Implementation steps:

1. Treat the current latent prediction as a differentiable leaf:

\[
x_0 \leftarrow \texttt{latents.detach().float().requires\_grad\_(True)}
\]

See [reward_gradient.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/reward_gradient.py:144).

2. Decode latent to image:

\[
I = \mathrm{decode}(x_0)
\]

See [reward_gradient.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/reward_gradient.py:163).

3. Compute reward:

\[
R = R(I, p)
\]

See [reward_gradient.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/reward_gradient.py:164).

4. Differentiate reward w.r.t. latent:

\[
g = \nabla_{x_0} R
\]

See [reward_gradient.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/reward_gradient.py:166).

## 7. Gradient Clipping and Running Threshold

The raw reward gradient norm is:

\[
\|g_i\|_2
\]

for each sample \(i\).

The clipped gradient is:

\[
\tilde g_i
=
g_i \cdot \min\left(1, \frac{\tau}{\|g_i\|_2}\right)
\]

where \(\tau\) is the current clipping threshold.

This is implemented in [clip_gradient_by_norm](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/reward_gradient.py:30).

The trainer updates \(\tau\) dynamically using the gathered reward-gradient norms from the current synchronized step:

\[
\tau \leftarrow \mathrm{Quantile}(\{\|g_i\|_2\}, q)
\]

with \(q =\) `rgrad_quantile`.

Code:

- gathering norms: [trainer.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:436)
- updating threshold: [trainer.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:437)

If `quantile_clipping` is disabled, the clipping function just returns the original gradient.

## 8. Jittered Reward Gradient

The implementation optionally smooths the reward gradient by evaluating several jittered latents:

\[
x_0^{(k)} = x_0 + \xi_k,\qquad \xi_k \sim \mathcal N(0, \sigma_{\text{jitter}}^2 I)
\]

and averaging reward over \(K\) jitter samples:

\[
R_{\text{smooth}}(x_0)
=
\frac{1}{K}\sum_{k=1}^{K} R(\mathrm{decode}(x_0^{(k)}), p)
\]

Then the gradient is taken with respect to the original \(x_0\).

Code:

- jitter construction: [reward_gradient.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/reward_gradient.py:145)
- reward averaging over jitter count: [reward_gradient.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/reward_gradient.py:165)

Controlled by:

- `rgrad_jitter_count`
- `rgrad_jitter_std`

## 9. Eta Scheduling

The multiplier \(\eta(t)\) is computed from `eta_mode` and

\[
\sigma = t / 1000
\]

in [trainer.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:340).

The supported modes in [reward_gradient.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/reward_gradient.py:19) are:

- `constant`

\[
\eta(t) = 1
\]

- `linear`

\[
\eta(t) = 1 - \sigma
\]

- `quad`

\[
\eta(t) = (1 - \sigma)^2
\]

The code then materializes one scalar `eta_value` used for the entire run, since the timestep is fixed in this one-step setup.

## 10. Target Construction

The final VGG-Flow target is:

\[
x_0^{\text{target}}
=
x_0^{\text{ref}} + \eta(t)\,\lambda\,\tilde g
\]

where:

- \(x_0^{\text{ref}}\) is the frozen reference clean latent
- \(\lambda\) is `reward_scale`
- \(\tilde g\) is the clipped reward gradient

This is implemented directly in [trainer.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:397).

## 11. Training Loss

The main regression loss is:

\[
\mathcal L_{\text{vgg}}
=
\frac{1}{K}\sum_{j=1}^{K}
\left\|
x_{0,j}^{\theta}
-
x_{0,j}^{\text{target}}
\right\|_2^2
\]

where \(K\) is `batchsize_gen`, the number of sampled candidates for one prompt.

Code:

- per-sample latent MSE: [trainer.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:399)
- average over generated samples: [trainer.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:403)

There is also an optional reference regularizer:

\[
\mathcal L_{\text{ref}}
=
\|x_0^\theta - x_0^{\text{ref}}\|_2^2
\]

Code: [trainer.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:404).

The total loss is:

\[
\mathcal L
=
\mathcal L_{\text{vgg}} + \alpha \mathcal L_{\text{ref}}
\]

where \(\alpha =\) `unet_reg_scale`.

Code: [trainer.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:405).

## 12. Optional Reward Masking

If `reward_masking` is enabled, only samples whose reward exceeds a threshold contribute to the VGG loss:

\[
m_i = \mathbf 1[R_i \ge \gamma]
\]

\[
\mathcal L_{\text{masked}}
=
\frac{\sum_i m_i \ell_i}{\sum_i m_i}
\]

where \(\ell_i\) is the per-sample latent regression loss.

Code:

- mask construction: [reward_gradient.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/reward_gradient.py:169)
- masked reduction: [trainer.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:400)

## 13. Batch Structure

The loop is organized as:

1. draw a minibatch of prompts from `PairsPromptDataset`
2. for each prompt:
   - replicate text embeddings `batchsize_gen` times
   - sample `batchsize_gen` noisy latents
   - produce `batchsize_gen` policy outputs
   - produce `batchsize_gen` reference outputs
   - compute reward gradient and target
   - compute prompt-level loss
3. average prompt losses inside the minibatch
4. backprop once

The important detail is that `batchsize_gen` is not a standard dataloader batch size. It is the number of generated candidates used to estimate the reward-gradient target for one prompt.

Code path: [trainer.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:343).

## 14. Optimization and Logging

Optimization is standard AdamW with optional LoRA on top of the UNet:

- LoRA wrapping: [trainer.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:229)
- trainable parameter collection: [trainer.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:243)

Each synchronized step logs:

- `train_loss`
- `vgg_loss_unaccumulated`
- `ref_l2_unaccumulated`
- `reward_mean_unaccumulated`
- `reward_std_unaccumulated`
- `rgrad_norm_unaccumulated`
- `rgrad_threshold`
- `reward_mask_fraction`
- `eta`
- `grad_norm_unaccumulated`

See [trainer.py](/datapool/jiangzhou/CODE/Text2ImageProject/DrPO/baselines/vggflow/train_lora.py:449).

## 15. What Is Missing Relative to Multi-Step VGG-Flow

This implementation intentionally omits several ingredients from the original multi-step setting:

- no multi-step flow integration
- no value-network consistency term
- no timestep-varying rollout trajectory inside one optimization step
- no reward model ensemble

The implementation is therefore best understood as:

\[
\text{one-step latent target matching against } x_0^{\text{ref}} + \text{reward gradient correction}
\]

not as a full reproduction of the original multi-step VGG-Flow training system.

## 16. Minimal Pseudocode

The per-prompt training logic is approximately:

```text
sample z_t
compute x0_policy from current UNet
compute x0_ref from frozen reference UNet
compute reward gradient g = grad_x0 R(decode(x0_policy), prompt)
clip g
build x0_target = x0_ref + eta * reward_scale * g
optimize ||x0_policy - x0_target||^2 + unet_reg_scale * ||x0_policy - x0_ref||^2
```

In the actual implementation this is repeated for `batchsize_gen` sampled candidates per prompt and then averaged.
