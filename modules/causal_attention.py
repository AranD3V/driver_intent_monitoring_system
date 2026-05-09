"""
Counterfactual Residual Encoding (CRE) -- causal cross-attention block.

Drop-in replacement for standard Multi-Head Cross-Attention when you want
to isolate the *causal* effect of scene affordances on driver gaze,
rather than any statistical co-occurrence.

Inspired by CaTFormer (Causal Transformer for driving-intention prediction)
and the counterfactual-reasoning literature (Pearl-style direct effect).

Pipeline
--------
    A_obs = Attn(Q = X_in,  K = X_out,       V = X_out)          # observed
    mu    = mean_t(X_out)                                        # baseline
    A_cf  = Attn(Q = X_in,  K = mu,          V = mu)             # counterfactual
    D     = A_obs - A_cf                                         # causal residual
    D     = D - (D . mu_hat) mu_hat            (optional, ortho projection)
    g     = sigmoid( Linear(D) )                                 # gate
    h_in  = X_in + g * D                                         # fused output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CounterfactualResidualEncoding(nn.Module):
    """
    Inputs
    ------
    x_in  : (B, T, D) queries  -- driver gaze / head-pose stream
    x_out : (B, T, D) keys/values -- scene / affordance stream
    key_padding_mask : optional (B, T) bool mask, True = pad (ignored in attn)

    Output
    ------
    h_in  : (B, T, D) causally-refined driver features
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_ortho_projection: bool = True,
        post_layernorm: bool = True,
        gate_mode: str = "elementwise",      # "elementwise" (D) or "scalar" (1)
    ):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_ortho = use_ortho_projection
        self.gate_mode = gate_mode

        # Shared attention -- critical: the SAME Q/K/V/out projections
        # are used for both the observed and counterfactual branches,
        # so their subtraction remains a meaningful contrast.
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        gate_out_dim = d_model if gate_mode == "elementwise" else 1
        self.gate = nn.Linear(d_model, gate_out_dim)
        # Initialize gate bias slightly negative so the fused path starts
        # near identity (h_in ~ x_in) and opens up as Delta proves informative.
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.0)

        self.norm = nn.LayerNorm(d_model) if post_layernorm else nn.Identity()

    def forward(
        self,
        x_in: torch.Tensor,
        x_out: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # 1. Observed dependency attention: real scene as context
        a_obs, _ = self.attn(
            query=x_in, key=x_out, value=x_out,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )                                                       # (B, T, D)

        # 2. Counterfactual attention: temporal-mean scene as neutral baseline.
        #    Mask-aware mean if padding provided.
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)   # (B, T, 1)
            mu = (x_out * valid).sum(dim=1, keepdim=True) \
                 / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            mu = x_out.mean(dim=1, keepdim=True)                # (B, 1, D)

        a_cf, _ = self.attn(
            query=x_in, key=mu, value=mu,
            need_weights=False,
        )                                                       # (B, T, D)

        # 3. Direct causal effect
        delta = a_obs - a_cf                                    # (B, T, D)

        if self.use_ortho:
            # Remove any residual component of `delta` that's collinear
            # with the baseline `mu`. Enforces strict orthogonality to the
            # baseline direction -- what remains is purely scene-variation-driven.
            mu_sq = (mu * mu).sum(dim=-1, keepdim=True).clamp(min=1e-8)  # (B,1,1)
            proj = (delta * mu).sum(dim=-1, keepdim=True) / mu_sq        # (B,T,1)
            delta = delta - proj * mu                                     # (B,T,D)

        # 4. Dynamic residual gating
        g = torch.sigmoid(self.gate(delta))                     # (B, T, D) or (B, T, 1)

        # 5. Residual fusion
        h_in = x_in + g * delta
        return self.norm(h_in)
