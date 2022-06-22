import torch
from torch import nn


def gaussian_smearing(distances, offset, widths, centered=False):
    if not centered:
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 / torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components
        diff = distances[..., None] - offset
    else:
        # if Gaussian functions are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances[..., None]
    # compute smear distance values
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    return gauss


class GaussianBasis(nn.Module):
    def __init__(
            self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False
    ):
        super(GaussianBasis, self).__init__()
        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered

    def forward(self, distances):
        """Compute smeared-gaussian distance values.

        Args:
            distances (torch.Tensor): interatomic distance values of
                (N_b x N_at x N_nbh) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh x N_g) shape.

        """
        return gaussian_smearing(
            distances, self.offsets, self.width, centered=self.centered
        )