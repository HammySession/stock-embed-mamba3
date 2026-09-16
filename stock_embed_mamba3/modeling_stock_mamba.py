"""Self-contained model definition shipped inside the Hugging Face repo.

Depends only on torch, mamba_ssm (Mamba-3) and safetensors. Mirrors the training-time
MambaBlock and MambaEncoder and the pretraining wrapper, so the exported
state_dict loads with no dependency on the training repo. A CUDA GPU is required: the
Mamba-3 kernels have no CPU path and mamba_ssm fails at import on a CPU-only machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from mamba_ssm import Mamba3


class MambaBlock(nn.Module):
    """Pre-norm Mamba-3 block with residual: x + ssm(norm(x))."""

    def __init__(self, d_model: int, d_state: int, expand: int, dropout: float, mamba3_kwargs: dict):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = Mamba3(d_model=d_model, d_state=d_state, expand=expand, **mamba3_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.ssm(self.norm(x)))


class MambaEncoder(nn.Module):
    """Patch -> linear projection -> N MambaBlocks -> LayerNorm."""

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        n_layers: int,
        d_state: int,
        expand: int,
        dropout: float,
        patch_size_bars: int,
        mamba3_kwargs: dict | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size_bars = patch_size_bars
        self.input_projection = nn.Linear(n_channels * patch_size_bars, d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        kwargs = mamba3_kwargs or {}
        self.blocks = nn.ModuleList(MambaBlock(d_model, d_state, expand, dropout, kwargs) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(d_model)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, C) -> (B, L/P, P*C). L must be divisible by patch_size_bars."""
        b, length, c = x.shape
        p = self.patch_size_bars
        if length % p:
            raise ValueError(f"seq_len {length} not divisible by patch_size_bars {p}")
        return x.reshape(b, length // p, p * c)

    def forward_blocks(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return self.final_norm(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, C) float32 feature bars -> (B, L/P, d_model) token embeddings."""
        return self.forward_blocks(self.input_projection(self.patchify(x)))


class MaskedReconstructionModel(nn.Module):
    """Encoder plus the linear decoder used for masked-patch pretraining.

    Inputs are equal-length windows only. There is no padding mask, so batch windows of
    the same seq_len or call the model once per security.
    """

    def __init__(self, encoder: MambaEncoder, n_channels: int):
        super().__init__()
        self.encoder = encoder
        self.n_channels = n_channels
        self.decoder_head = nn.Linear(encoder.d_model, encoder.patch_size_bars * n_channels)

    @classmethod
    def from_config(cls, cfg: dict) -> MaskedReconstructionModel:
        enc = MambaEncoder(
            n_channels=len(cfg["input_channels"]),
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            d_state=cfg["d_state"],
            expand=cfg["expand_factor"],
            dropout=cfg["dropout"],
            patch_size_bars=cfg["patch_size_bars"],
            mamba3_kwargs=cfg.get("mamba3", {}),
        )
        return cls(enc, len(cfg["input_channels"]))

    @classmethod
    def from_pretrained(cls, repo_dir: str | Path, device: str = "cuda") -> MaskedReconstructionModel:
        """Load config.json + model.safetensors from a local directory (e.g. a snapshot_download)."""
        from safetensors.torch import load_file

        repo_dir = Path(repo_dir)
        cfg = json.loads((repo_dir / "config.json").read_text())
        model = cls.from_config(cfg)
        model.load_state_dict(load_file(repo_dir / "model.safetensors", device=device), strict=True)
        return model.to(device).eval()

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pooled embedding per window: (B, L, C) -> (B, d_model)."""
        return self.encoder(x).mean(dim=1)

    @torch.no_grad()
    def inpaint(self, x: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        """Reconstruct the masked patches of x.

        Args:
            x: (B, L, C) feature bars, L divisible by patch_size_bars.
            patch_mask: (B, L/P) bool, True where a patch is masked from the encoder.

        Returns:
            (B, L, C) tensor equal to x on visible bars and to the decoder output on
            masked bars. Only masked positions were trained; visible positions are
            passed through untouched.
        """
        if patch_mask.dtype != torch.bool:
            raise TypeError(f"patch_mask must be bool, got {patch_mask.dtype}")
        p = self.encoder.patch_size_bars
        tokens = self.encoder.input_projection(self.encoder.patchify(x))
        tokens = torch.where(patch_mask.unsqueeze(-1), self.encoder.mask_token.to(tokens.dtype), tokens)
        decoded = self.decoder_head(self.encoder.forward_blocks(tokens)).reshape_as(x)
        bar_mask = patch_mask.repeat_interleave(p, dim=1).unsqueeze(-1)
        return torch.where(bar_mask, decoded, x)
