"""
Shared spatiotemporal backbone + multi-task heads.

Architecture:
  - Per-timestep 2D conv encoder (extracts moisture/instability/lift features)
  - A ConvGRU over the time axis (captures rapid IWV accumulation / CTT drop trend)
  - Shared dense trunk
  - Three task heads: thunderstorm, cloudburst, flash-flood-seed
  - Dropout kept active at inference (MC-Dropout) to get calibrated
    uncertainty bands per head -> reduces false-alarm rate vs a single
    point estimate (this + DEM flow-routing in dem_utils.py are the
    two novelty pieces beyond a standard nowcasting CNN).
"""
import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, k=3):
        super().__init__()
        pad = k // 2
        self.hidden_ch = hidden_ch
        self.gate_conv = nn.Conv2d(in_ch + hidden_ch, 2 * hidden_ch, k, padding=pad)
        self.cand_conv = nn.Conv2d(in_ch + hidden_ch, hidden_ch, k, padding=pad)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=1)
        gates = self.gate_conv(combined)
        z, r = torch.chunk(torch.sigmoid(gates), 2, dim=1)
        combined_r = torch.cat([x, r * h], dim=1)
        cand = torch.tanh(self.cand_conv(combined_r))
        h_new = (1 - z) * h + z * cand
        return h_new


class NowcastNet(nn.Module):
    def __init__(self, in_features=9, enc_ch=24, hidden_ch=32, dropout=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_features, enc_ch, 3, padding=1),
            nn.BatchNorm2d(enc_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_ch, enc_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gru = ConvGRUCell(enc_ch, hidden_ch)
        self.hidden_ch = hidden_ch

        self.trunk = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(hidden_ch * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.head_thunder = nn.Linear(64, 1)
        self.head_cloudburst = nn.Linear(64, 1)
        self.head_flashflood = nn.Linear(64, 1)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.hidden_ch, H, W, device=x.device, dtype=x.dtype)
        for t in range(T):
            feat = self.encoder(x[:, t])
            h = self.gru(feat, h)
        z = self.trunk(h)
        thunder = torch.sigmoid(self.head_thunder(z)).squeeze(-1)
        cloudburst = torch.sigmoid(self.head_cloudburst(z)).squeeze(-1)
        flashflood = torch.sigmoid(self.head_flashflood(z)).squeeze(-1)
        return thunder, cloudburst, flashflood, h

    def mc_predict(self, x, n_samples=12):
        """Monte-Carlo Dropout inference: keep dropout active, sample n times,
        return (mean, std) per head -> calibrated uncertainty for the dashboard."""
        self.train()  # keeps dropout active
        outs = {"thunder": [], "cloudburst": [], "flashflood": []}
        last_h = None
        with torch.no_grad():
            for _ in range(n_samples):
                t, c, f, h = self.forward(x)
                outs["thunder"].append(t)
                outs["cloudburst"].append(c)
                outs["flashflood"].append(f)
                last_h = h
        self.eval()
        result = {}
        for k, v in outs.items():
            stacked = torch.stack(v, dim=0)
            result[k] = (stacked.mean(0), stacked.std(0))
        return result, last_h
