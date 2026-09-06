"""
Data simulator.

Real deployment would pull:
  - IMDAA reanalysis (temperature, humidity, geopotential, U/V wind)
  - INSAT-3D/3DR WV & TIR channels via MOSDAC (-> IWV, CTT drop rate)
  - Satellite QPE (rainfall intensity)
  - CartoDEM / SRTM (elevation, slope, drainage)

Those feeds need ISRO/MOSDAC/NCMRWF credentials this environment doesn't
have, so this module generates physically-plausible synthetic fields with
the SAME shape/semantics the real model consumes. Swap `generate_frame()`
for real API connectors without touching the model or backend.

Grid convention: each "frame" is a stack of 2D fields over an NxN patch
around a station, at time t. A sample is a short sequence of frames
(spatiotemporal) ending at "now", used to predict risk 2-6h ahead.
"""
import numpy as np

GRID = 16          # NxN spatial patch per station
SEQ_LEN = 6         # number of past timesteps (e.g. 30-min cadence -> 3h history)
FEATURES = [
    "iwv",              # integrated water vapor (kg/m^2)
    "iwv_rate",         # d(IWV)/dt  -- rapid moisture accumulation
    "cape",             # J/kg
    "cin",               # J/kg (negative = more inhibition)
    "conv_low",         # low-level wind convergence (1/s)
    "shear_06km",       # 0-6km bulk wind shear (m/s)
    "ctt",               # cloud top temperature (K)
    "ctt_drop_rate",     # K per 15 min (negative = rapid cooling)
    "qpe",               # satellite rainfall estimate (mm/hr)
]
N_FEATURES = len(FEATURES)


def _smooth_field(rng, base, scale, sigma=2.0):
    """Cheap Gaussian-smoothed random field without scipy dependency."""
    raw = rng.normal(base, scale, size=(GRID, GRID))
    # separable box-blur approx of Gaussian, a few passes
    k = max(1, int(sigma))
    out = raw.copy()
    for _ in range(3):
        out = (
            np.roll(out, 1, 0) + np.roll(out, -1, 0) +
            np.roll(out, 1, 1) + np.roll(out, -1, 1) + out
        ) / 5.0
    return out


def generate_sequence(rng: np.random.Generator, storm_injection: bool = None):
    """
    Returns:
        seq: np.ndarray [SEQ_LEN, N_FEATURES, GRID, GRID]
        labels: dict with thunderstorm/cloudburst/flash_flood probability in [0,1]
                (ground truth used only for supervised training)
    """
    if storm_injection is None:
        storm_injection = rng.random() < 0.35  # ~35% of samples are storm-onset cases

    seq = np.zeros((SEQ_LEN, N_FEATURES, GRID, GRID), dtype=np.float32)

    # baseline quiet-atmosphere fields
    iwv = _smooth_field(rng, base=35, scale=4)
    cape = _smooth_field(rng, base=500, scale=150)
    cin = _smooth_field(rng, base=-40, scale=15)
    conv = _smooth_field(rng, base=0.0, scale=1e-5)
    shear = _smooth_field(rng, base=8, scale=2)
    ctt = _smooth_field(rng, base=270, scale=5)
    qpe = np.clip(_smooth_field(rng, base=0.5, scale=1.0), 0, None)

    onset_step = rng.integers(SEQ_LEN - 2, SEQ_LEN) if storm_injection else None

    for t in range(SEQ_LEN):
        if storm_injection and t >= onset_step - 2 and onset_step is not None:
            # inject a growing moisture pocket + destabilization + convergence + cooling top
            progress = max(0, (t - (onset_step - 2)) / 2.0)
            cy, cx = GRID // 2 + rng.integers(-2, 3), GRID // 2 + rng.integers(-2, 3)
            yy, xx = np.mgrid[0:GRID, 0:GRID]
            blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (2.5 ** 2))))

            iwv = iwv + blob * (18 * progress) + rng.normal(0, 0.5, (GRID, GRID))
            cape = cape + blob * (900 * progress)
            cin = cin + blob * (30 * progress)  # CIN erodes toward 0
            conv = conv - blob * (4e-5 * progress)  # negative = convergence
            shear = shear + blob * (6 * progress)
            ctt = ctt - blob * (35 * progress)
            qpe = qpe + blob * (25 * progress)

        iwv_rate = np.zeros((GRID, GRID)) if t == 0 else (iwv - seq[t - 1, 0])
        ctt_prev = ctt if t == 0 else (ctt + seq[t - 1, 6])  # rough
        ctt_drop = np.zeros((GRID, GRID)) if t == 0 else (ctt - seq[t - 1, 6])

        seq[t, 0] = iwv
        seq[t, 1] = iwv_rate
        seq[t, 2] = cape
        seq[t, 3] = cin
        seq[t, 4] = conv
        seq[t, 5] = shear
        seq[t, 6] = ctt
        seq[t, 7] = ctt_drop
        seq[t, 8] = qpe

    # ground-truth labels derived from final-frame physical intensity
    peak_iwv_rate = float(np.max(seq[-1, 1]))
    peak_cape = float(np.max(seq[-1, 2]))
    peak_cin = float(np.max(seq[-1, 3]))
    peak_conv = float(-np.min(seq[-1, 4]))
    peak_ctt_drop = float(-np.min(seq[-1, 7]))
    peak_qpe = float(np.max(seq[-1, 8]))

    thunder = _sigmoid(0.006 * peak_cape + 40 * peak_conv + 0.15 * peak_ctt_drop - 3.0)
    cloudburst = _sigmoid(0.35 * peak_iwv_rate + 0.02 * peak_qpe + 0.08 * peak_ctt_drop - 3.5)
    flash_flood_seed = _sigmoid(0.9 * cloudburst + 0.015 * peak_qpe - 2.0)

    return seq, {
        "thunderstorm": float(np.clip(thunder, 0, 1)),
        "cloudburst": float(np.clip(cloudburst, 0, 1)),
        "flash_flood_seed": float(np.clip(flash_flood_seed, 0, 1)),
    }


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def batch(rng: np.random.Generator, n: int):
    seqs, y_t, y_c, y_f = [], [], [], []
    for _ in range(n):
        s, lab = generate_sequence(rng)
        seqs.append(s)
        y_t.append(lab["thunderstorm"])
        y_c.append(lab["cloudburst"])
        y_f.append(lab["flash_flood_seed"])
    return (
        np.stack(seqs).astype(np.float32),
        np.array(y_t, dtype=np.float32),
        np.array(y_c, dtype=np.float32),
        np.array(y_f, dtype=np.float32),
    )
