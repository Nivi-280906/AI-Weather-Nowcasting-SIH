"""
Trains NowcastNet on the physics-plausible synthetic data stream by
default, or on REAL IMDAA + INSAT + GPM-IMERG data when --real-data-dir
is given (see ml/real_data/README.md for the expected folder layout).

Run (synthetic, current default):
    python -m ml.train --epochs 8 --steps-per-epoch 200 --batch-size 32

Run (real data):
    python -m ml.train --real-data-dir /path/to/downloaded/data --epochs 8
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn

from ml.model import NowcastNet
from ml import data_simulator as ds


def _load_real_dataset(real_data_dir: str, grid_stride: int = 2):
    """Expects subfolders: imdaa/, insat/, qpe/ (see
    ml/real_data/README.md). Uses the station list already defined in
    seed.py so station locations stay consistent with the live dashboard."""
    from ml.real_data import build_dataset as rd
    from ml.real_data import imdaa as imdaa_mod
    from ml.real_data import insat as insat_mod
    from ml.real_data import qpe as qpe_mod
    import seed

    imdaa_files = sorted(glob.glob(os.path.join(real_data_dir, "imdaa", "*.nc")))
    insat_files = sorted(
        glob.glob(os.path.join(real_data_dir, "insat", "*.nc"))
        + glob.glob(os.path.join(real_data_dir, "insat", "*.h5"))
        + glob.glob(os.path.join(real_data_dir, "insat", "*.hdf5"))
    )
    qpe_files = sorted(
        glob.glob(os.path.join(real_data_dir, "qpe", "*.nc4"))
        + glob.glob(os.path.join(real_data_dir, "qpe", "*.nc"))
        + glob.glob(os.path.join(real_data_dir, "qpe", "*.HDF5"))
    )
    if not (imdaa_files and insat_files and qpe_files):
        raise FileNotFoundError(
            f"Expected files under {real_data_dir}/imdaa, /insat, /qpe -- "
            f"found {len(imdaa_files)} imdaa, {len(insat_files)} insat, "
            f"{len(qpe_files)} qpe files. See ml/real_data/README.md."
        )

    cache_path = os.path.join(real_data_dir, "_built_dataset_cache.npz")
    if os.path.exists(cache_path):
        print(f"Found cached built dataset at {cache_path} -- loading instead of "
              f"recomputing CAPE/CIN (delete this file to force a rebuild after "
              f"adding new downloads).")
        cached = np.load(cache_path)
        return cached["X"], cached["y_t"], cached["y_c"], cached["y_f"]

    imdaa_ds = imdaa_mod.load_imdaa(imdaa_files)
    insat_ds = insat_mod.load_insat(insat_files)
    qpe_ds = qpe_mod.load_imerg(qpe_files)

    stations = [{"lat": s[3], "lon": s[4]} for s in seed.STATIONS]
    X, y_t, y_c, y_f = rd.build_full_dataset(imdaa_ds, insat_ds, qpe_ds, stations, stride=grid_stride)

    np.savez_compressed(cache_path, X=X, y_t=y_t, y_c=y_c, y_f=y_f)
    print(f"Cached built dataset -> {cache_path} ({X.shape[0]} samples). "
          f"Future runs will load this instantly instead of recomputing.")
    return X, y_t, y_c, y_f


def run(
    epochs=8,
    steps_per_epoch=200,
    batch_size=32,
    lr=1e-3,
    out_path="ml/weights/nowcastnet.pt",
    real_data_dir=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NowcastNet(in_features=ds.N_FEATURES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    rng = np.random.default_rng(42)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if real_data_dir:
        print(f"Loading REAL data from {real_data_dir} (this can take a while -- "
              f"CAPE/CIN is computed per grid cell per timestep) ...")
        X, y_t_full, y_c_full, y_f_full = _load_real_dataset(real_data_dir)
        n = X.shape[0]
        print(f"Built {n} real training samples.")
        if n < batch_size:
            batch_size = max(1, n)

        for epoch in range(1, epochs + 1):
            model.train()
            perm = rng.permutation(n)
            running, n_batches = 0.0, 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                x = torch.from_numpy(X[idx]).to(device)
                yt = torch.from_numpy(y_t_full[idx]).to(device)
                yc = torch.from_numpy(y_c_full[idx]).to(device)
                yf = torch.from_numpy(y_f_full[idx]).to(device)

                opt.zero_grad()
                pt, pc, pf, _ = model(x)
                loss = loss_fn(pt, yt) + loss_fn(pc, yc) + loss_fn(pf, yf)
                loss.backward()
                opt.step()
                running += loss.item()
                n_batches += 1

            avg = running / max(1, n_batches)
            print(f"epoch {epoch}/{epochs}  avg_loss={avg:.4f}  (real data, n={n})")
    else:
        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0
            for step in range(steps_per_epoch):
                seq, y_t, y_c, y_f = ds.batch(rng, batch_size)
                x = torch.from_numpy(seq).to(device)          # [B,T,C,H,W]
                yt = torch.from_numpy(y_t).to(device)
                yc = torch.from_numpy(y_c).to(device)
                yf = torch.from_numpy(y_f).to(device)

                opt.zero_grad()
                pt, pc, pf, _ = model(x)
                loss = loss_fn(pt, yt) + loss_fn(pc, yc) + loss_fn(pf, yf)
                loss.backward()
                opt.step()
                running += loss.item()

            avg = running / steps_per_epoch
            print(f"epoch {epoch}/{epochs}  avg_loss={avg:.4f}  (synthetic data)")

    torch.save(model.state_dict(), out_path)
    print(f"saved weights -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--real-data-dir", type=str, default=None,
                    help="If given, trains on real IMDAA+INSAT+IMERG data "
                         "in this directory instead of the synthetic simulator.")
    args = p.parse_args()
    run(args.epochs, args.steps_per_epoch, args.batch_size, args.lr, real_data_dir=args.real_data_dir)
