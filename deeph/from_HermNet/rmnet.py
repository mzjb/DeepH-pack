import math

import torch
from torch import nn, Tensor
import numpy as np


_eps = 1e-3

r"""Tricks: Introducing the parameter `_eps` is to avoid NaN.
In HVNet and HTNet, a subgraph will be extracted to calculate angles. 
And with all the nodes still be included in the subgraph, 
each hidden state in such a subgraph will contain 0 value.
In `painn`, the calculation w.r.t $r / \parallel r \parallel$ will be taken.
If just alternate $r / \parallel r \parallel$ with $r / (\parallel r \parallel + _eps)$, 
NaN will still occur in during the training.
Considering the following example,
$$
(\frac{x}{r+_eps})^\prime = \frac{r+b-\frac{x^2}{r}}{(r+b)^2}
$$
where $r = \sqrt{x^2+y^2+z^2}$. It is obvious that NaN will occur.
Thus the solution is change the norm $r$ as $r^\prime = \sqrt(x^2+y^2+z^2+_eps)$.
Since $r$ is rotational invariant, $r^2$ is rotational invariant.
Obviously, $\sqrt(r^2 + _eps)$ is rotational invariant.
"""
class RBF(nn.Module):
    r"""Radial basis function.
    A modified version of feature engineering in `DimeNet`,
    which is used in `PAINN`.

    Parameters
    ----------
    rc      : float
        Cutoff radius
    l       : int
        Parameter in feature engineering in DimeNet
    """
    def __init__(self, rc: float, l: int):
        super(RBF, self).__init__()
        self.rc = rc
        self.l = l

    def forward(self, x: Tensor):
        ls = torch.arange(1, self.l + 1).float().to(x.device)
        norm = torch.sqrt((x ** 2).sum(dim=-1) + _eps).unsqueeze(-1)
        return torch.sin(math.pi / self.rc * norm@ls.unsqueeze(0)) / norm


class cosine_cutoff(nn.Module):
    r"""Cutoff function in https://aip.scitation.org/doi/pdf/10.1063/1.3553717.

    Parameters
    ----------
    rc      : float
        Cutoff radius
    """
    def __init__(self, rc: float):
        super(cosine_cutoff, self).__init__()
        self.rc = rc

    def forward(self, x: Tensor):
        norm = torch.norm(x, dim=-1, keepdim=True) + _eps
        return 0.5 * (torch.cos(math.pi * norm / self.rc) + 1)

class ShiftedSoftplus(nn.Module):
    r"""

    Description
    -----------
    Applies the element-wise function:

    .. math::
        \text{SSP}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x)) - \log(\text{shift})

    Attributes
    ----------
    beta : int
        :math:`\beta` value for the mathematical formulation. Default to 1.
    shift : int
        :math:`\text{shift}` value for the mathematical formulation. Default to 2.
    """
    def __init__(self, beta=1, shift=2, threshold=20):
        super(ShiftedSoftplus, self).__init__()

        self.shift = shift
        self.softplus = nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, inputs):
        """

        Description
        -----------
        Applies the activation function.

        Parameters
        ----------
        inputs : float32 tensor of shape (N, *)
            * denotes any number of additional dimensions.

        Returns
        -------
        float32 tensor of shape (N, *)
            Result of applying the activation function to the input.
        """
        return self.softplus(inputs) - np.log(float(self.shift))
