"""
QJL: Quantized Johnson-Lindenstrauss Transform

1-bit quantization with zero memory overhead for inner product estimation.
Based on: "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization
with Zero Overhead" (Zandieh et al., 2024) - arXiv:2406.03482

The QJL map Q_qjl: R^d -> {-1, +1}^d is defined as:
    Q_qjl(x) = sign(S * x)
where S is a random d x d matrix with i.i.d. N(0,1) entries.

Dequantization: Q_qjl^{-1}(z) = (pi/2) / d * S^T * z

This provides an unbiased inner product estimator with variance <= (pi / 2d) * ||y||^2.
"""

import math
import torch
import torch.nn as nn


class QJL(nn.Module):
    """Quantized Johnson-Lindenstrauss 1-bit quantizer.

    Provides zero-overhead 1-bit quantization for inner product estimation.
    Uses a random Gaussian projection followed by sign-bit quantization.
    """

    def __init__(self, dim: int, num_projections: int = None, device: str = "cuda"):
        super().__init__()
        self.dim = dim
        self.num_projections = num_projections or dim
        self.device = device

        # Random projection matrix S with i.i.d. N(0,1) entries
        S = torch.randn(self.num_projections, dim, device=device, dtype=torch.float32)
        self.register_buffer("S", S)

        # Precompute S^T for dequantization
        self.register_buffer("S_T", S.t().contiguous())

        # Scaling factor for dequantization: sqrt(pi/2) / num_projections
        self.dequant_scale = math.sqrt(math.pi / 2) / self.num_projections

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize vectors to sign bits.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Sign bits tensor of shape (..., num_projections) with values {-1, +1}
        """
        # Project: S * x
        projected = torch.matmul(x.float(), self.S.t())
        # Sign quantization
        return projected.sign()

    def dequantize(self, z: torch.Tensor) -> torch.Tensor:
        """Dequantize sign bits back to vectors.

        Args:
            z: Sign bits tensor of shape (..., num_projections) with values {-1, +1}

        Returns:
            Reconstructed tensor of shape (..., dim)
        """
        # Q_qjl^{-1}(z) = (pi/2) / d * S^T * z
        return self.dequant_scale * torch.matmul(z.float(), self.S)

    def estimate_inner_product(self, query: torch.Tensor, quantized_key: torch.Tensor) -> torch.Tensor:
        """Estimate inner product <query, key> using quantized key.

        Uses the asymmetric estimator: apply full JL transform to query,
        and use quantized (sign-bit) key for efficient computation.

        Args:
            query: Full-precision query tensor (..., dim)
            quantized_key: Sign-bit quantized key (..., num_projections)

        Returns:
            Estimated inner product values
        """
        # Project query through S (full precision)
        query_projected = torch.matmul(query.float(), self.S.t())
        # Inner product estimate: (pi/2) / d * sum_i (S_i^T * y) * sign(S_i^T * x)
        return self.dequant_scale * (query_projected * quantized_key.float()).sum(dim=-1)


class QJLWithOrthogonal(QJL):
    """QJL variant using orthogonalized random projections for improved performance.

    Instead of raw Gaussian matrix, uses QR decomposition to get orthogonal
    projection directions, which reduces variance of the estimator.
    """

    def __init__(self, dim: int, num_projections: int = None, device: str = "cuda"):
        # Initialize parent but we'll override the projection matrix
        nn.Module.__init__(self)
        self.dim = dim
        self.num_projections = num_projections or dim
        self.device = device

        # Generate orthogonalized projection matrix via QR decomposition
        # of random Gaussian matrix (as done in the QJL reference implementation)
        raw = torch.randn(self.num_projections, dim, device=device, dtype=torch.float32)

        num_chunks = (self.num_projections + dim - 1) // dim
        ortho_blocks = []
        for i in range(num_chunks):
            start = i * dim
            end = min((i + 1) * dim, self.num_projections)
            block = raw[start:end, :]
            if block.shape[0] <= block.shape[1]:
                q, _ = torch.linalg.qr(block.t(), mode="reduced")
                ortho_blocks.append(q.t() * math.sqrt(dim))
            else:
                ortho_blocks.append(block)

        S = torch.cat(ortho_blocks, dim=0)[:self.num_projections]
        self.register_buffer("S", S.contiguous())
        self.register_buffer("S_T", S.t().contiguous())
        self.dequant_scale = math.sqrt(math.pi / 2) / self.num_projections


class QJLBitPacked(nn.Module):
    """Memory-efficient QJL that stores sign bits packed into uint8.

    Packs 8 sign bits per byte, reducing storage by 8x compared to
    storing signs as float/int tensors.
    """

    def __init__(self, dim: int, num_projections: int = None, device: str = "cuda"):
        super().__init__()
        self.dim = dim
        self.num_projections = num_projections or dim
        self.device = device

        S = torch.randn(self.num_projections, dim, device=device, dtype=torch.float32)
        self.register_buffer("S", S)
        self.register_buffer("S_T", S.t().contiguous())
        self.dequant_scale = math.sqrt(math.pi / 2) / self.num_projections

        # Encoding vector for bit packing
        enc = 2 ** torch.arange(8, dtype=torch.uint8, device=device)
        self.register_buffer("enc_vec", enc)

    def quantize_packed(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and pack sign bits into uint8.

        Args:
            x: Input tensor (..., dim)

        Returns:
            Packed uint8 tensor (..., ceil(num_projections / 8))
        """
        projected = torch.matmul(x.float(), self.S.t())
        signs = projected > 0  # bool tensor

        # Reshape last dim to groups of 8 for bit packing
        pad_len = (8 - self.num_projections % 8) % 8
        if pad_len > 0:
            signs = torch.nn.functional.pad(signs, (0, pad_len), value=False)

        signs = signs.view(*signs.shape[:-1], -1, 8)
        packed = (signs * self.enc_vec).sum(dim=-1, dtype=torch.uint8)
        return packed

    def unpack_signs(self, packed: torch.Tensor) -> torch.Tensor:
        """Unpack uint8 back to sign tensor {-1, +1}.

        Args:
            packed: uint8 tensor (..., ceil(num_projections / 8))

        Returns:
            Sign tensor (..., num_projections)
        """
        unpacked = packed.unsqueeze(-1).bitwise_and(self.enc_vec.unsqueeze(0)) > 0
        unpacked = unpacked.view(*packed.shape[:-1], -1)
        unpacked = unpacked[..., :self.num_projections]
        return unpacked.float() * 2 - 1  # Convert {0,1} -> {-1,+1}

    def estimate_inner_product_packed(self, query: torch.Tensor, packed_key: torch.Tensor) -> torch.Tensor:
        """Estimate inner product with bit-packed keys.

        Args:
            query: Full-precision query (..., dim)
            packed_key: Bit-packed quantized key (..., ceil(num_projections / 8))

        Returns:
            Inner product estimates
        """
        signs = self.unpack_signs(packed_key)
        query_projected = torch.matmul(query.float(), self.S.t())
        return self.dequant_scale * (query_projected * signs).sum(dim=-1)
