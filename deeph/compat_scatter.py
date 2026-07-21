"""
Drop-in replacement for torch_scatter.scatter and torch_scatter.scatter_add
using native PyTorch operations. This avoids the need to install torch_scatter.
"""
import torch


def broadcast_index(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    """Broadcast index to match the dimensions of src for scatter operations."""
    if index.dim() == src.dim():
        return index
    # index is 1D, src has more dimensions
    # Expand index to match src shape
    for _ in range(src.dim() - 1):
        index = index.unsqueeze(-1)
    # Expand to match src shape
    shape = list(src.shape)
    shape[dim] = -1
    return index.expand(shape)


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                out: torch.Tensor = None, dim_size: int = None) -> torch.Tensor:
    """
    Drop-in replacement for torch_scatter.scatter_add.
    """
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    if out is None:
        size = list(src.shape)
        size[dim] = dim_size
        out = torch.zeros(size, dtype=src.dtype, device=src.device)

    index = broadcast_index(index, src, dim)
    out.scatter_add_(dim, index, src)
    return out


def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
            out: torch.Tensor = None, dim_size: int = None,
            reduce: str = 'sum') -> torch.Tensor:
    """
    Drop-in replacement for torch_scatter.scatter.
    Supports reduce modes: 'sum', 'add', 'mean', 'min', 'max', 'mul'.
    """
    if reduce == 'add':
        reduce = 'sum'

    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    if out is None:
        size = list(src.shape)
        size[dim] = dim_size
        out = torch.zeros(size, dtype=src.dtype, device=src.device)

    if reduce == 'sum':
        index = broadcast_index(index, src, dim)
        out.scatter_add_(dim, index, src)
    elif reduce == 'mean':
        index = broadcast_index(index, src, dim)
        out.scatter_add_(dim, index, src)
        # Count elements per index for averaging
        count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        ones = torch.ones(index.shape[0], dtype=src.dtype, device=src.device)
        count.scatter_add_(0, index[:, 0], ones)
        count = count.clamp(min=1)
        # Reshape count for broadcasting
        for _ in range(out.dim() - 1):
            count = count.unsqueeze(-1)
        out = out / count
    elif reduce == 'min':
        index = broadcast_index(index, src, dim)
        out.fill_(float('inf'))
        # Use scatter_reduce if available (PyTorch >= 1.12), otherwise manual
        if hasattr(out, 'scatter_reduce_'):
            out.scatter_reduce_(dim, index, src, reduce='amin')
        else:
            # Fallback: use a loop (slow but correct)
            for i in range(index.shape[0]):
                idx = index[i, 0].item()
                out[idx] = torch.min(out[idx], src[i])
    elif reduce == 'max':
        index = broadcast_index(index, src, dim)
        out.fill_(float('-inf'))
        if hasattr(out, 'scatter_reduce_'):
            out.scatter_reduce_(dim, index, src, reduce='amax')
        else:
            for i in range(index.shape[0]):
                idx = index[i, 0].item()
                out[idx] = torch.max(out[idx], src[i])
    elif reduce == 'mul':
        index = broadcast_index(index, src, dim)
        out.fill_(1.0)
        for i in range(index.shape[0]):
            idx = index[i, 0].item()
            out[idx] = out[idx] * src[i]
    else:
        raise ValueError(f"Unknown reduce mode: {reduce}")

    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                 out: torch.Tensor = None, dim_size: int = None) -> torch.Tensor:
    """Drop-in replacement for torch_scatter.scatter_mean."""
    return scatter(src, index, dim=dim, out=out, dim_size=dim_size, reduce='mean')
