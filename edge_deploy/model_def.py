"""Model definition shared by the notebook, compile_trt.py and infer.py (copied from AISE4010_Project_Group8.ipynb)."""
import torch
import torch.nn as nn

FS = 50               # DeviceMotion sampling rate (Hz)
WINDOW = 100          # 2.0 s of history at 50 Hz
CLASSES = ["dws", "jog", "sit", "std", "ups", "wlk"]
CHANNELS = (["attitude.roll", "attitude.pitch", "attitude.yaw"]
            + [f"{sensor}.{axis}" for sensor in ("gravity", "rotationRate", "userAcceleration") for axis in "xyz"])
N_CLASSES, N_CHANNELS = len(CLASSES), len(CHANNELS)

# Best model of the notebook (untuned TCN: pseudo-test accuracy 0.848, macro-F1 0.823) = the default TCN() settings.
BEST_CONFIG = {"channels": 32, "kernel": 5, "dropout": 0.1}


class CausalBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation                   # left padding only -> causal
        self.net = nn.Sequential(
            nn.ConstantPad1d((pad, 0), 0.0), nn.Conv1d(c_in, c_out, kernel, dilation=dilation), nn.ReLU(), nn.Dropout(dropout),
            nn.ConstantPad1d((pad, 0), 0.0), nn.Conv1d(c_out, c_out, kernel, dilation=dilation), nn.ReLU(), nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return torch.relu(self.net(x) + self.residual(x))


def tcn_receptive_field(kernel, levels):
    return 1 + 2 * (kernel - 1) * (2 ** levels - 1)


class TCN(nn.Module):
    def __init__(self, channels=32, kernel=5, levels=None, dropout=0.1):
        super().__init__()
        if levels is None:                              # smallest depth whose receptive field covers the window
            levels = next(L for L in range(1, 12) if tcn_receptive_field(kernel, L) >= WINDOW)
        self.receptive_field = tcn_receptive_field(kernel, levels)
        self.blocks = nn.Sequential(*[
            CausalBlock(N_CHANNELS if i == 0 else channels, channels, kernel, 2 ** i, dropout) for i in range(levels)
        ])
        self.head = nn.Linear(channels, N_CLASSES)

    def forward(self, x):          # x: (batch, N_CHANNELS, WINDOW), already standardised
        return self.head(self.blocks(x)[:, :, -1])      # features at the last (current) time step


def load_checkpoint(path, device="cpu"):
    """-> (eval-mode model, mu (C,1), sigma (C,1) as float32 numpy) from the file written by the notebook."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TCN(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["mu"], ckpt["sigma"]
