import torch


def tv_loss_2d(y: torch.Tensor, wmask: torch.Tensor = None) -> torch.Tensor:
    """
    Spatial total variation loss (anisotropic):
      y: (B, 1, H, W)
      wmask: optional (1,1,H,W) or (H,W) with 0/1 weights; applied to gradients.

    Returns: scalar tensor
    """
    # finite differences
    dy = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
    dx = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])

    if wmask is not None:
        if wmask.dim() == 2:
            wmask = wmask.view(1, 1, *wmask.shape)
        # match shapes for dy/dx
        wy = wmask[:, :, 1:, :]
        wx = wmask[:, :, :, 1:]
        dy = dy * wy
        dx = dx * wx
        denom = (wy.sum() + wx.sum()).clamp_min(1.0)
        return (dy.sum() + dx.sum()) / denom

    denom = float(dy.numel() + dx.numel())
    return (dy.sum() + dx.sum()) / denom


def temporal_smooth_loss_single_step(pred: torch.Tensor, x_last: torch.Tensor, wmask: torch.Tensor = None) -> torch.Tensor:
    """
    Temporal smoothness for lead_time=1:
      pred:  (B,1,H,W)  prediction at t+1
      x_last:(B,1,H,W)  last observed SIC at t
    Uses L1 smoothness: |pred - x_last| averaged over mask.
    """
    diff = torch.abs(pred - x_last)
    if wmask is not None:
        if wmask.dim() == 2:
            wmask = wmask.view(1, 1, *wmask.shape)
        diff = diff * wmask
        denom = wmask.sum().clamp_min(1.0)
        return diff.sum() / denom

    return diff.mean()
