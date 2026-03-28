"""
Lloyd-Max Scalar Quantizer Codebook

Precomputes optimal scalar quantizer centroids for the Beta distribution
induced by random rotation onto the unit hypersphere.

From the paper (Section 3.1):
    After random rotation, each coordinate of Pi*x follows:
        f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1-x^2)^((d-3)/2)
    In high dimensions this converges to N(0, 1/d).

    The optimal quantizer is found by solving the continuous 1-D k-means
    (Lloyd-Max algorithm) for this distribution.

Standard Gaussian Lloyd-Max centroids (well-known values):
    b=1: {-0.7979, 0.7979}
    b=2: {-1.5104, -0.4528, 0.4528, 1.5104}
    b=3: 8 centroids
    b=4: 16 centroids

For the Beta distribution on the hypersphere, centroids are scaled by 1/sqrt(d).
"""

import math
import torch
import numpy as np
from scipy import integrate
from scipy.special import gamma as gamma_func


# Precomputed Lloyd-Max centroids for standard Gaussian N(0,1)
# These are optimal quantizer centroids from the literature.
GAUSSIAN_LLOYD_MAX_CENTROIDS = {
    1: [-0.7978845608, 0.7978845608],
    2: [-1.5104176087, -0.4527800398, 0.4527800398, 1.5104176087],
    3: [
        -2.1519740547, -1.3439092613, -0.7560052548, -0.2451209526,
        0.2451209526, 0.7560052548, 1.3439092613, 2.1519740547,
    ],
    4: [
        -2.7326368950, -2.0690715233, -1.6180334930, -1.2562297699,
        -0.9423402690, -0.6568029240, -0.3880170717, -0.1284185687,
        0.1284185687, 0.3880170717, 0.6568029240, 0.9423402690,
        1.2562297699, 1.6180334930, 2.0690715233, 2.7326368950,
    ],
}

# Precomputed Lloyd-Max boundaries for standard Gaussian
GAUSSIAN_LLOYD_MAX_BOUNDARIES = {
    1: [0.0],
    2: [-0.9815515893, 0.0, 0.9815515893],
    3: [
        -1.7479320818, -1.0500070587, -0.5005532340, 0.0,
        0.5005532340, 1.0500070587, 1.7479320818,
    ],
    4: [
        -2.4008243530, -1.8435286460, -1.4370787917, -1.0993383102,
        -0.7996084837, -0.5224397472, -0.2582204052, 0.0,
        0.2582204052, 0.5224397472, 0.7996084837, 1.0993383102,
        1.4370787917, 1.8435286460, 2.4008243530,
    ],
}


class LloydMaxCodebook:
    """Precomputed Lloyd-Max codebook for TurboQuant scalar quantization.

    Stores centroids and boundaries for optimal scalar quantization of
    random rotation coordinates which follow approximately N(0, 1/d).
    """

    def __init__(self, bit_width: int, dim: int, device: str = "cuda"):
        assert 1 <= bit_width <= 4, f"Supported bit widths: 1-4, got {bit_width}"
        self.bit_width = bit_width
        self.dim = dim
        self.num_levels = 2 ** bit_width
        self.device = device

        # Scale standard Gaussian centroids by 1/sqrt(d) for hypersphere coordinates
        scale = 1.0 / math.sqrt(dim)

        centroids_np = np.array(GAUSSIAN_LLOYD_MAX_CENTROIDS[bit_width]) * scale
        self.centroids = torch.tensor(centroids_np, dtype=torch.float32, device=device)

        boundaries_np = np.array(GAUSSIAN_LLOYD_MAX_BOUNDARIES[bit_width]) * scale
        # Add -inf and +inf as outer boundaries
        full_boundaries = np.concatenate([[-np.inf], boundaries_np, [np.inf]])
        self.boundaries = torch.tensor(full_boundaries, dtype=torch.float32, device=device)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Map each value to the index of its nearest centroid.

        Args:
            x: Input tensor of any shape (values should be ~N(0, 1/d) distributed)

        Returns:
            Index tensor of same shape with values in [0, num_levels-1]
        """
        # Compute distances to all centroids and pick nearest
        # x: (...), centroids: (num_levels,)
        dists = (x.unsqueeze(-1) - self.centroids).abs()
        return dists.argmin(dim=-1)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Map indices back to centroid values.

        Args:
            indices: Index tensor with values in [0, num_levels-1]

        Returns:
            Reconstructed values (centroid values)
        """
        return self.centroids[indices]

    def quantize_and_dequantize(self, x: torch.Tensor):
        """Quantize then immediately dequantize (for computing residuals).

        Returns:
            (reconstructed, indices) tuple
        """
        indices = self.quantize(x)
        reconstructed = self.dequantize(indices)
        return reconstructed, indices


def compute_lloyd_max_codebook(bit_width: int, dim: int, max_iter: int = 200):
    """Numerically compute Lloyd-Max centroids for the exact Beta distribution.

    For high dimensions (d > 50), the result is very close to the precomputed
    Gaussian approximation. This function is provided for completeness and
    low-dimensional cases.

    Args:
        bit_width: Number of bits (1-4)
        dim: Vector dimension
        max_iter: Maximum Lloyd iterations

    Returns:
        (centroids, boundaries) numpy arrays
    """
    num_levels = 2 ** bit_width

    # Beta distribution PDF for coordinate of random point on S^{d-1}
    def pdf(x):
        if abs(x) >= 1:
            return 0.0
        coeff = gamma_func(dim / 2) / (math.sqrt(math.pi) * gamma_func((dim - 1) / 2))
        return coeff * (1 - x**2) ** ((dim - 3) / 2)

    # Initialize centroids uniformly in [-1, 1]
    centroids = np.linspace(-0.9, 0.9, num_levels)

    for _ in range(max_iter):
        # Compute boundaries (midpoints between consecutive centroids)
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        full_bounds = np.concatenate([[-1.0], boundaries, [1.0]])

        # Update centroids: centroid_i = E[X | X in bucket_i]
        new_centroids = np.zeros(num_levels)
        for i in range(num_levels):
            lo, hi = full_bounds[i], full_bounds[i + 1]
            numerator, _ = integrate.quad(lambda x: x * pdf(x), lo, hi)
            denominator, _ = integrate.quad(pdf, lo, hi)
            if denominator > 1e-15:
                new_centroids[i] = numerator / denominator
            else:
                new_centroids[i] = (lo + hi) / 2

        if np.max(np.abs(new_centroids - centroids)) < 1e-10:
            break
        centroids = new_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2
    return centroids, boundaries
