"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

Based on: "TurboQuant: Online Vector Quantization with Near-optimal Distortion
Rate" (Zandieh et al., 2025) - arXiv:2504.19874

Two variants:
1. TurboQuant_MSE: Minimizes mean-squared error
   - Random rotation -> Scalar quantization with Lloyd-Max codebook
2. TurboQuant_Prod: Unbiased inner product estimation
   - TurboQuant_MSE with (b-1) bits -> QJL on residual (1 bit)

Algorithm 1 (MSE):
    Setup: Generate random rotation Pi, precompute codebook centroids
    Quant(x):  y = Pi * x; idx_j = argmin_k |y_j - c_k| for each j
    DeQuant(idx): y_tilde_j = c_{idx_j}; return Pi^T * y_tilde

Algorithm 2 (Inner Product):
    Setup: Instantiate TurboQuant_MSE with (b-1) bits, generate QJL matrix S
    Quant(x):
        idx = Quant_MSE(x)
        x_mse = DeQuant_MSE(idx)
        r = x - x_mse  (residual)
        gamma = ||r||
        qjl = sign(S * r / gamma)
    DeQuant(idx, qjl, gamma):
        x_tilde = DeQuant_MSE(idx) + gamma * Q_qjl^{-1}(qjl)
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

from .codebook import LloydMaxCodebook
from .qjl import QJL


class TurboQuantMSE(nn.Module):
    """TurboQuant optimized for MSE distortion.

    Applies random rotation to input vectors then quantizes each coordinate
    independently using a precomputed Lloyd-Max codebook.

    Achieves MSE <= (3*sqrt(pi)/2) * (1/4^b) for b-bit quantization.
    """

    def __init__(self, dim: int, bit_width: int = 3, device: str = "cuda"):
        super().__init__()
        self.dim = dim
        self.bit_width = bit_width
        self.device = device

        # Generate random rotation matrix via QR decomposition of Gaussian matrix
        # This is Pi in the paper
        random_matrix = torch.randn(dim, dim, device=device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(random_matrix)
        self.register_buffer("rotation", Q.contiguous())
        self.register_buffer("rotation_t", Q.t().contiguous())

        # Precomputed Lloyd-Max codebook
        self.codebook = LloydMaxCodebook(bit_width, dim, device=device)

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize vectors using random rotation + scalar quantization.

        Args:
            x: Input tensor (..., dim), should be unit-normalized

        Returns:
            (indices, norm): quantized indices (..., dim) and original norms (...)
        """
        orig_shape = x.shape
        # Store norms for reconstruction (paper assumes unit norm, we handle general case)
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_normalized = x / norms

        # Random rotation: y = Pi * x
        y = torch.matmul(x_normalized, self.rotation.t())

        # Scalar quantization per coordinate
        indices = self.codebook.quantize(y)

        return indices, norms.squeeze(-1)

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Dequantize indices back to vectors.

        Args:
            indices: Quantized indices (..., dim)
            norms: Original vector norms (...)

        Returns:
            Reconstructed vectors (..., dim)
        """
        # Map indices to centroids
        y_hat = self.codebook.dequantize(indices)

        # Rotate back: x_hat = Pi^T * y_hat
        x_hat = torch.matmul(y_hat, self.rotation)

        # Rescale
        return x_hat * norms.unsqueeze(-1)

    def quantize_dequantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full round-trip: quantize then dequantize.

        Returns:
            (reconstructed, indices, norms)
        """
        indices, norms = self.quantize(x)
        reconstructed = self.dequantize(indices, norms)
        return reconstructed, indices, norms


class TurboQuantProd(nn.Module):
    """TurboQuant optimized for unbiased inner product estimation.

    Two-stage approach:
    1. Apply TurboQuant_MSE with (b-1) bits to capture the main signal
    2. Apply QJL (1 bit) to the residual for unbiased correction

    The result is an unbiased inner product estimator:
        E[<y, x_tilde>] = <y, x>

    With inner product distortion:
        D_prod <= (3*sqrt(pi)/2) * (||y||^2 / d) * (1/4^b)
    """

    def __init__(self, dim: int, bit_width: int = 3, device: str = "cuda"):
        super().__init__()
        assert bit_width >= 2, "Inner product TurboQuant needs at least 2 bits (1 for MSE + 1 for QJL)"
        self.dim = dim
        self.bit_width = bit_width
        self.device = device

        # Stage 1: MSE quantizer with (b-1) bits
        self.mse_quantizer = TurboQuantMSE(dim, bit_width - 1, device=device)

        # Stage 2: QJL for 1-bit residual correction
        self.qjl = QJL(dim, device=device)

    def quantize(self, x: torch.Tensor) -> dict:
        """Quantize using two-stage TurboQuant_prod.

        Args:
            x: Input tensor (..., dim)

        Returns:
            Dictionary with quantization state:
                - mse_indices: Scalar quantizer indices
                - norms: Original vector norms
                - qjl_signs: QJL sign bits of residual
                - residual_norms: L2 norms of residuals (gamma)
        """
        # Stage 1: MSE quantization with (b-1) bits
        x_mse, mse_indices, norms = self.mse_quantizer.quantize_dequantize(x)

        # Compute residual
        residual = x - x_mse
        residual_norms = residual.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        residual_normalized = residual / residual_norms

        # Stage 2: QJL on normalized residual
        qjl_signs = self.qjl.quantize(residual_normalized)

        return {
            "mse_indices": mse_indices,
            "norms": norms,
            "qjl_signs": qjl_signs,
            "residual_norms": residual_norms.squeeze(-1),
        }

    def dequantize(self, quant_state: dict) -> torch.Tensor:
        """Dequantize from two-stage representation.

        Args:
            quant_state: Dictionary from quantize()

        Returns:
            Reconstructed vectors (..., dim)
        """
        # Reconstruct MSE component
        x_mse = self.mse_quantizer.dequantize(
            quant_state["mse_indices"], quant_state["norms"]
        )

        # Reconstruct QJL residual component
        residual_hat = self.qjl.dequantize(quant_state["qjl_signs"])
        residual_hat = residual_hat * quant_state["residual_norms"].unsqueeze(-1)

        return x_mse + residual_hat

    def estimate_attention_score(
        self,
        query: torch.Tensor,
        quant_state: dict,
    ) -> torch.Tensor:
        """Efficiently estimate attention scores <query, key> for quantized keys.

        This avoids full dequantization by computing the inner product directly:
            <y, x_tilde> = <y, x_mse> + gamma * QJL_estimate(y, r/||r||)

        Args:
            query: Query vectors (batch, heads, seq_q, dim)
            quant_state: Quantized key state from quantize()

        Returns:
            Attention scores (batch, heads, seq_q, seq_k)
        """
        # MSE component: reconstruct key and compute dot product
        keys_mse = self.mse_quantizer.dequantize(
            quant_state["mse_indices"], quant_state["norms"]
        )

        # <query, keys_mse> via standard matmul
        scores_mse = torch.matmul(query, keys_mse.transpose(-2, -1))

        # QJL component: estimate <query, residual>
        # Project query through QJL random matrix
        qjl_signs = quant_state["qjl_signs"]
        residual_norms = quant_state["residual_norms"]

        # For each query position, estimate inner product with each key's residual
        # query_proj: (..., seq_q, num_proj)
        query_proj = torch.matmul(query.float(), self.qjl.S.t())
        # scores_qjl: (..., seq_q, seq_k)
        scores_qjl = self.qjl.dequant_scale * torch.matmul(
            query_proj, qjl_signs.float().transpose(-2, -1)
        )
        # Scale by residual norms
        scores_qjl = scores_qjl * residual_norms.unsqueeze(-2)

        return scores_mse + scores_qjl


class TurboQuantPerChannel(nn.Module):
    """TurboQuant with outlier-aware per-channel quantization.

    Splits channels into outlier and non-outlier groups, applying
    higher bit precision to outlier channels. This matches the paper's
    experimental setup (Section 4.3):
        - 2.5-bit: 32 outlier channels at 3 bits + 96 channels at 2 bits
        - 3.5-bit: higher ratio of outlier channels at 4 bits
    """

    def __init__(
        self,
        dim: int,
        bit_width_regular: int = 2,
        bit_width_outlier: int = 3,
        num_outlier_channels: int = 32,
        device: str = "cuda",
    ):
        super().__init__()
        self.dim = dim
        self.num_outlier = min(num_outlier_channels, dim)
        self.num_regular = dim - self.num_outlier
        self.device = device

        self.effective_bits = (
            self.num_outlier * bit_width_outlier + self.num_regular * bit_width_regular
        ) / dim

        # Separate quantizers for outlier and regular channels
        if self.num_regular > 0:
            self.regular_quant = TurboQuantProd(self.num_regular, bit_width_regular, device)
        if self.num_outlier > 0:
            self.outlier_quant = TurboQuantProd(self.num_outlier, bit_width_outlier, device)

        # Outlier indices will be determined per-tensor based on channel norms
        self._outlier_indices = None

    def _find_outliers(self, x: torch.Tensor) -> torch.Tensor:
        """Identify outlier channels based on L2 norm across the sequence."""
        # x shape: (..., seq, dim)
        channel_norms = x.float().norm(dim=-2)  # (..., dim)
        _, indices = channel_norms.topk(self.num_outlier, dim=-1)
        return indices.sort(dim=-1).values

    def quantize(self, x: torch.Tensor) -> dict:
        """Quantize with outlier-aware channel splitting."""
        outlier_idx = self._find_outliers(x)
        self._outlier_indices = outlier_idx

        # Build masks
        batch_dims = x.shape[:-1]
        all_idx = torch.arange(self.dim, device=x.device)

        # Gather outlier and regular channels
        outlier_mask = torch.zeros(*x.shape[:-2], self.dim, dtype=torch.bool, device=x.device)
        outlier_mask.scatter_(-1, outlier_idx, True)

        x_outlier = x[..., :][..., outlier_mask.unsqueeze(-2).expand_as(x)].view(
            *x.shape[:-1], self.num_outlier
        ) if self.num_outlier > 0 else None

        x_regular = x[..., :][..., (~outlier_mask).unsqueeze(-2).expand_as(x)].view(
            *x.shape[:-1], self.num_regular
        ) if self.num_regular > 0 else None

        result = {"outlier_mask": outlier_mask}
        if x_outlier is not None:
            result["outlier"] = self.outlier_quant.quantize(x_outlier)
        if x_regular is not None:
            result["regular"] = self.regular_quant.quantize(x_regular)

        return result

    def dequantize(self, quant_state: dict) -> torch.Tensor:
        """Dequantize with channel reassembly."""
        outlier_mask = quant_state["outlier_mask"]

        x_outlier = self.outlier_quant.dequantize(quant_state["outlier"]) if "outlier" in quant_state else None
        x_regular = self.regular_quant.dequantize(quant_state["regular"]) if "regular" in quant_state else None

        # Determine output shape from whichever part we have
        ref = x_outlier if x_outlier is not None else x_regular
        out = torch.zeros(*ref.shape[:-1], self.dim, device=ref.device, dtype=ref.dtype)

        if x_outlier is not None:
            out[..., :][..., outlier_mask.unsqueeze(-2).expand_as(out)] = x_outlier.reshape(-1)
        if x_regular is not None:
            out[..., :][..., (~outlier_mask).unsqueeze(-2).expand_as(out)] = x_regular.reshape(-1)

        return out
