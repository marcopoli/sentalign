"""Mixture-of-perspectives head: a distribution that is not a rescaled softmax.

Every objective in this study outputs ``softmax(l)`` for a single logit vector ``l`` read
from the language-model head, and differs from the others only in the target it fits and
the anchor it applies. Measured across two tasks, two label spaces, K in {3,4} and M in
{5,100}, all of them are indistinguishable from temperature-calibrated SFT once a single
scalar is fitted on held-out data. That is not a coincidence: a fixed monotone rescaling
of ``l`` is exactly what a temperature reproduces, so no loss on that output can escape.

The escape has to change the output, not the loss. Human disagreement is systematic
rather than random, so this models it as a mixture over latent annotator perspectives:

    q(y | x) = sum_a  pi_a(x) * softmax(W_a h(x))

with ``A`` small heads and a gate, all reading the final hidden state ``h(x)``. Two
properties follow, and both are what the previous attempts lacked:

* A mixture of softmaxes is not a softmax, so no temperature reproduces it. The objective
  is structurally capable of separating from calibrated SFT.
* Per-item entropy arises from *disagreement between heads*, a quantity the model can
  actually represent and move, rather than from the magnitude of ``l``, which is fixed by
  the language-modelling scale and which LoRA could not shift. That was Polya's failure.

``A = 1`` with the heads initialised from the language-model head's verbalizer rows
reproduces ``sft_soft`` exactly, so the control that has beaten every proposal so far is
nested inside this one rather than standing beside it.
"""

from __future__ import annotations

import numpy as np


def mixture_log_probs(hidden, weights, gate_weights, bias=None, gate_bias=None):
    """Reference implementation of the mixture, in numpy.

    ``hidden``        (B, D)      final hidden state
    ``weights``       (A, K, D)   one readout per perspective
    ``gate_weights``  (A, D)      the gate over perspectives
    Returns           (B, K)      log q(y | x)
    """
    h = np.asarray(hidden, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    g = np.asarray(gate_weights, dtype=np.float64)
    if w.ndim != 3 or g.ndim != 2 or w.shape[0] != g.shape[0]:
        raise ValueError(f"expected (A,K,D) weights and (A,D) gate, got {w.shape} {g.shape}")

    logits = np.einsum("akd,bd->bak", w, h)                       # (B, A, K)
    if bias is not None:
        logits = logits + np.asarray(bias, dtype=np.float64)[None, :, :]
    log_component = logits - _logsumexp(logits, axis=-1, keepdims=True)

    gate = h @ g.T                                                # (B, A)
    if gate_bias is not None:
        gate = gate + np.asarray(gate_bias, dtype=np.float64)[None, :]
    log_pi = gate - _logsumexp(gate, axis=-1, keepdims=True)

    return _logsumexp(log_pi[:, :, None] + log_component, axis=1)


def _logsumexp(x, axis=-1, keepdims=False):
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def make_mixture_head(hidden_size: int, verbalizer_rows, n_perspectives: int = 4,
                      init_scale: float = 0.02, seed: int = 0):
    """A torch module whose ``A = 1`` limit is the model's own verbalizer readout.

    ``verbalizer_rows`` is the (K, D) slice of the unembedding for the label tokens. Head
    0 is initialised to exactly that, so before any training the mixture reproduces the
    base model's label distribution and the objective starts from its own control rather
    than from noise.
    """
    import torch
    from torch import nn

    rows = torch.as_tensor(np.asarray(verbalizer_rows), dtype=torch.float32)
    k, d = rows.shape
    if d != hidden_size:
        raise ValueError(f"verbalizer rows are {d}-dim, hidden state is {hidden_size}")

    class MixtureHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_perspectives = n_perspectives
            generator = torch.Generator().manual_seed(seed)
            weight = rows.unsqueeze(0).repeat(n_perspectives, 1, 1).clone()
            if n_perspectives > 1:
                # Identical heads have identical gradients and would never separate, so
                # the copies are perturbed. Head 0 is left exact.
                noise = torch.randn(weight[1:].shape, generator=generator) * init_scale
                weight[1:] = weight[1:] + noise * rows.abs().mean()
            self.weight = nn.Parameter(weight)                    # (A, K, D)
            self.bias = nn.Parameter(torch.zeros(n_perspectives, k))
            # Zero gate means a uniform mixture at initialisation.
            self.gate = nn.Parameter(torch.zeros(n_perspectives, d))
            self.gate_bias = nn.Parameter(torch.zeros(n_perspectives))

        def forward(self, hidden):
            logits = torch.einsum("akd,bd->bak", self.weight, hidden) + self.bias[None]
            log_component = torch.log_softmax(logits, dim=-1)
            gate = hidden @ self.gate.T + self.gate_bias[None]
            log_pi = torch.log_softmax(gate, dim=-1)
            return torch.logsumexp(log_pi[:, :, None] + log_component, dim=1)

    return MixtureHead()
