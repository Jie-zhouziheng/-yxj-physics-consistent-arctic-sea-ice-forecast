import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Minimal ConvLSTM cell."""
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMBaseline(nn.Module):
    """
    Phase-1 ConvLSTM baseline:
      input:  (B, T, 1, H, W)
      output: (B, 1, H, W)
    """
    def __init__(self, in_channels: int = 1, hidden_channels: int = 16, kernel_size: int = 3):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size=kernel_size)
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        device = x.device
        h = torch.zeros(B, self.cell.hidden_channels, H, W, device=device)
        c = torch.zeros_like(h)

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)

        y = self.head(h)
        return y
