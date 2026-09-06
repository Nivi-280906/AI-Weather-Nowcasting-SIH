"""
Builds real training samples in EXACTLY the tensor contract
data_simulator.py already established, so model.py, train.py, and the
inference/serving code need zero changes:

    seq:    np.ndarray [SEQ_LEN, N_FEATURES, GRID, GRID]
    labels: {"thunderstorm": p, "cloudburst": p, "flash_flood_seed": p}

Feature order matches data_simulator.FEATURES exactly:
    iwv, iwv_rate, cape, cin, conv_low, shear_06km, ctt, ctt_drop_rate, qpe

Data source per feature (per the problem statement's own architecture):
    iwv / iwv_rate     <- INSAT WV channel proxy
    cape, cin          <- IMDAA (via metpy parcel calc)
    conv_low           <- IMDAA (850 hPa wind convergence)
    shear_06km         <- IMDAA (1000 hPa vs 500 hPa bulk wind shear)
    ctt / ctt_drop_rate<- INSAT TIR-1 channel
    qpe                <- GPM IMERG

This module does NOT hit the network or need any government login -- it
only reads files you've already downloaded and points it at.
"""
from __future__ import annotations

import numpy as np

from . import imdaa as imdaa_mod
from . import insat as insat_mod
from . import qpe as qpe_mod
from . import labels as labels_mod

GRID = 16
SEQ_LEN = 6
FEATURE_ORDER = [
    "iwv", "iwv_rate", "cape", "cin", "conv_low", "shear_06km",
    "ctt", "ctt_drop_rate", "qpe",
]


def align_common_times(imdaa_ds, insat_ds, qpe_ds, tolerance_minutes: int = 45) -> list[dict]:
    """IMDAA is 3-hourly; INSAT/IMERG are higher-frequency. Walks IMDAA's
    timestamps (the coarsest/reference cadence) and finds the nearest
    INSAT and QPE timestamp within `tolerance_minutes`, skipping any
    IMDAA time that has no match in either source.

    Returns a list of {"imdaa": i, "insat": j, "qpe": k} index dicts, one
    per aligned reference timestamp, in chronological order."""
    imdaa_times = imdaa_ds["time"].values
    insat_times = insat_ds["time"].values
    qpe_times = qpe_ds["time"].values
    tol = np.timedelta64(tolerance_minutes, "m")

    aligned = []
    for i, t in enumerate(imdaa_times):
        j_diffs = np.abs(insat_times - t)
        k_diffs = np.abs(qpe_times - t)
        j = int(np.argmin(j_diffs))
        k = int(np.argmin(k_diffs))
        if j_diffs[j] <= tol and k_diffs[k] <= tol:
            aligned.append({"imdaa": i, "insat": j, "qpe": k})
    return aligned


def sliding_windows(aligned: list[dict], seq_len: int = SEQ_LEN, stride: int = 1) -> list[list[dict]]:
    """Turns the flat aligned-timestamp list into overlapping SEQ_LEN
    windows, e.g. for 8 aligned times and seq_len=6: windows [0:6], [1:7],
    [2:8] (stride=1) -- more samples from the same download, at the cost
    of temporal overlap between samples."""
    windows = []
    for start in range(0, len(aligned) - seq_len + 1, stride):
        windows.append(aligned[start:start + seq_len])
    return windows



def build_station_sequence(
    imdaa_ds, insat_ds, qpe_ds, lat: float, lon: float, window: list[dict]
) -> tuple[np.ndarray, dict]:
    """window: SEQ_LEN dicts of {"imdaa": i, "insat": j, "qpe": k} aligned
    time indices, as produced by align_common_times() + sliding_windows()."""
    assert len(window) == SEQ_LEN

    imdaa_patch = imdaa_mod.extract_patch(imdaa_ds, lat, lon, GRID)
    insat_patch = insat_mod.extract_patch(insat_ds, lat, lon, GRID)
    qpe_patch = qpe_mod.extract_patch(qpe_ds, lat, lon, GRID)

    seq = np.zeros((SEQ_LEN, len(FEATURE_ORDER), GRID, GRID), dtype=np.float32)
    prev_iwv = None
    prev_ctt = None

    for t, idx in enumerate(window):
        cape, cin = imdaa_mod.compute_cape_cin(imdaa_patch, idx["imdaa"])
        shear = imdaa_mod.compute_shear_06km(imdaa_patch, idx["imdaa"])
        conv = imdaa_mod.compute_convergence_low(imdaa_patch, idx["imdaa"])

        wv_bt = insat_patch["wv_bt"].isel(time=idx["insat"]).values
        tir_bt = insat_patch["tir1_bt"].isel(time=idx["insat"]).values
        iwv = insat_mod.iwv_proxy_from_wv(wv_bt)
        ctt = insat_mod.ctt_from_tir1(tir_bt)

        qpe_vals = qpe_patch["qpe"].isel(time=idx["qpe"]).values.astype(np.float32)

        iwv_rate = np.zeros_like(iwv) if prev_iwv is None else (iwv - prev_iwv)
        ctt_drop = np.zeros_like(ctt) if prev_ctt is None else (ctt - prev_ctt)
        prev_iwv, prev_ctt = iwv, ctt

        seq[t, 0] = iwv
        seq[t, 1] = iwv_rate
        seq[t, 2] = cape
        seq[t, 3] = cin
        seq[t, 4] = conv
        seq[t, 5] = shear
        seq[t, 6] = ctt
        seq[t, 7] = ctt_drop
        seq[t, 8] = qpe_vals

    final_frame = {name: seq[-1, i] for i, name in enumerate(FEATURE_ORDER)}
    label = labels_mod.derive_labels(final_frame)

    return seq, label


def build_batch(
    imdaa_ds, insat_ds, qpe_ds, stations: list[dict], time_windows: list[list[dict]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """stations: list of {"lat":..., "lon":...}
    time_windows: list of SEQ_LEN-length aligned-index-dict windows, as
    produced by align_common_times() + sliding_windows()."""
    seqs, y_t, y_c, y_f = [], [], [], []
    for station in stations:
        for window in time_windows:
            seq, lab = build_station_sequence(
                imdaa_ds, insat_ds, qpe_ds, station["lat"], station["lon"], window
            )
            seqs.append(seq)
            y_t.append(lab["thunderstorm"])
            y_c.append(lab["cloudburst"])
            y_f.append(lab["flash_flood_seed"])

    return (
        np.stack(seqs).astype(np.float32),
        np.array(y_t, dtype=np.float32),
        np.array(y_c, dtype=np.float32),
        np.array(y_f, dtype=np.float32),
    )


def build_full_dataset(
    imdaa_ds, insat_ds, qpe_ds, stations: list[dict], stride: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One-call convenience: aligns timestamps, builds sliding windows,
    and returns the full (X, y_thunder, y_cloudburst, y_flashflood) arrays
    ready for train.py. `stride` > 1 trades sample count for less
    temporal overlap between consecutive training samples."""
    aligned = align_common_times(imdaa_ds, insat_ds, qpe_ds)
    windows = sliding_windows(aligned, seq_len=SEQ_LEN, stride=stride)
    if not windows:
        raise ValueError(
            "No aligned time windows found -- check that your IMDAA/INSAT/QPE "
            "downloads actually overlap in date range, and that each file's "
            "time coordinate parsed correctly (inspect with xr.open_dataset(...).time)."
        )
    return build_batch(imdaa_ds, insat_ds, qpe_ds, stations, windows)
