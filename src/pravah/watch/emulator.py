"""Rain→discharge emulator (GRU) for the Bhote Koshi–Trishuli main stem, trained on GB10.

Target: daily GloFAS reanalysis discharge (m³/s) at the main-stem gauge cells. Inputs: 90
days of daily ERA5 precipitation and mean temperature at catchment points plus day-of-year.
The last ``horizon`` input days are the target days themselves (their rainfall is "known"
in training = perfect-forecast assumption; live runs use Open-Meteo forecast rainfall, a
replay uses only rainfall up to the replay clock and zero after, flagged).

Splits by calendar year, never shuffled across time: train ≤ 2022, validation 2023–2024,
test 2025–2026 (both Rasuwa floods are in test only). Emulating GloFAS means it inherits
GloFAS limits: glacial-lake outbursts and landslide-dam breaches are NOT in the target.
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np

SEQ_DAYS = 90
HORIZON = 3
TRAIN_END, VAL_END = 2022, 2024


def features(precip: np.ndarray, temp: np.ndarray, days: list[date]) -> np.ndarray:
    """(T, F): log1p precipitation per point, temperature per point, sin/cos day-of-year."""
    doy = np.array([d.timetuple().tm_yday for d in days], float)
    return np.concatenate([np.log1p(np.clip(precip, 0, None)), temp / 10.0,
                           np.sin(2 * math.pi * doy / 365.25)[:, None], np.cos(2 * math.pi * doy / 365.25)[:, None]],
                          axis=1).astype("float32")


def windows(X: np.ndarray, Y: np.ndarray, end_index: np.ndarray):
    """Sample i ends at day e (inclusive); targets are Y[e-HORIZON+1 .. e]."""
    xs = np.stack([X[e - SEQ_DAYS + 1:e + 1] for e in end_index])
    ys = np.stack([Y[e - HORIZON + 1:e + 1] for e in end_index])
    return xs, ys


def build_model(n_in: int, n_out: int, hidden: int = 128):
    from torch import nn

    class GRUEmulator(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_in, hidden, num_layers=2, batch_first=True, dropout=0.1)
            self.head = nn.Linear(hidden, HORIZON * n_out)
            self.n_out = n_out

        def forward(self, x):
            h, _ = self.gru(x)
            return self.head(h[:, -1]).view(-1, HORIZON, self.n_out)
    return GRUEmulator()


def nse(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    d = ((obs[ok] - obs[ok].mean()) ** 2).sum()
    return float(1 - ((obs[ok] - sim[ok]) ** 2).sum() / d) if d > 0 else None


def kge(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    if ok.sum() < 3 or obs[ok].std() == 0 or obs[ok].mean() == 0:
        return None
    r = np.corrcoef(obs[ok], sim[ok])[0, 1]
    return float(1 - math.sqrt((r - 1) ** 2 + (sim[ok].std() / obs[ok].std() - 1) ** 2 + (sim[ok].mean() / obs[ok].mean() - 1) ** 2))


def train(table: dict, out_dir: Path, epochs: int = 40, device: str | None = None, seed: int = 12,
          batch_size: int = 1024) -> dict:
    """``table``: {"days": [iso], "precip": (T,K), "temp": (T,K), "discharge": (T,G), "stations": [...]}."""
    import time

    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    days = [date.fromisoformat(d) for d in table["days"]]
    X = features(np.asarray(table["precip"], float), np.asarray(table["temp"], float), days)
    Q = np.asarray(table["discharge"], float)
    Ylog = np.log1p(np.clip(Q, 0, None))
    mu, sd = np.nanmean(Ylog[[d.year <= TRAIN_END for d in days]], 0), np.nanstd(Ylog[[d.year <= TRAIN_END for d in days]], 0) + 1e-6
    Yn = (Ylog - mu) / sd
    years = np.array([d.year for d in days])
    ends = np.arange(SEQ_DAYS - 1, len(days))
    complete = np.array([np.isfinite(Yn[e - HORIZON + 1:e + 1]).all() and np.isfinite(X[e - SEQ_DAYS + 1:e + 1]).all() for e in ends])
    split = {"train": ends[complete & (years[ends] <= TRAIN_END)],
             "val": ends[complete & (years[ends] > TRAIN_END) & (years[ends] <= VAL_END)],
             "test": ends[complete & (years[ends] > VAL_END)]}
    if len(split["train"]) < 1000 or len(split["val"]) < 100:
        raise ValueError("Not enough history for a time-split emulator")
    model = build_model(X.shape[1], Q.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    amp = dev.type == "cuda"  # bf16 autocast on GB10; weights/optimizer stay fp32
    if amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt = torch.as_tensor(X, device=dev)
    Yt = torch.as_tensor(Yn.astype("float32"), device=dev)
    offs = torch.arange(-SEQ_DAYS + 1, 1, device=dev)
    hoffs = torch.arange(-HORIZON + 1, 1, device=dev)

    def batch(idx):
        e = torch.as_tensor(idx, device=dev)
        return Xt[e[:, None] + offs], Yt[e[:, None] + hoffs]

    def evaluate(idx):
        model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(idx), 8192):
                xb, _ = batch(idx[s:s + 8192])
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    preds.append(model(xb).float().cpu().numpy())
        model.train()
        return np.concatenate(preds)

    best, best_state, log = math.inf, None, []
    t0 = time.perf_counter()
    for ep in range(epochs):
        perm = np.random.permutation(split["train"])
        losses = []
        for s in range(0, len(perm), batch_size):
            xb, yb = batch(perm[s:s + batch_size])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = torch.nn.functional.mse_loss(model(xb).float(), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        pv = evaluate(split["val"])
        _, yv = windows(X, Yn, split["val"])
        vloss = float(np.mean((pv - yv) ** 2))
        log.append({"epoch": ep + 1, "train_mse": float(np.mean(losses)), "val_mse": vloss})
        print(f"[emulator seed {seed}] epoch {ep + 1}/{epochs} train {np.mean(losses):.4f} val {vloss:.4f}", flush=True)
        if vloss < best:
            best, best_state = vloss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if dev.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    model.load_state_dict(best_state)
    metrics = {}
    for name in ("val", "test"):
        p = evaluate(split[name])[:, -1]           # same-day (nowcast) skill per station
        y = Q[split[name]]
        sim = np.expm1(p * sd + mu)
        metrics[name] = {st: {"nse": nse(y[:, i], sim[:, i]), "kge": kge(y[:, i], sim[:, i]), "days": len(y)}
                         for i, st in enumerate(table["stations"])}
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "n_in": X.shape[1], "n_out": Q.shape[1], "mu": mu.tolist(), "sd": sd.tolist(),
                "stations": table["stations"], "points": table.get("points"), "seq_days": SEQ_DAYS, "horizon": HORIZON},
               out_dir / "emulator.pt")
    summary = {"schema": "pravah-flow-emulator-v1", "target": "GloFAS reanalysis daily discharge (m³/s)",
               "inputs": "ERA5 daily precipitation + mean temperature at catchment points (Open-Meteo archive)",
               "splits": {"train": f"≤{TRAIN_END}", "val": f"{TRAIN_END + 1}–{VAL_END}", "test": f"≥{VAL_END + 1}"},
               "samples": {k: len(v) for k, v in split.items()}, "epochs": epochs, "seed": seed,
               "batch_size": batch_size, "precision": "bf16 autocast" if amp else "fp32", "best_val_mse": best,
               "metrics_same_day": metrics, "train_seconds_measured": round(train_s, 2), "device": str(dev),
               "device_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
               "log": log, "limitation": __doc__.split("\n\n")[-1].strip()}
    (out_dir / "emulator_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _member(args):
    table_path, out_dir, seed, epochs, device = args
    table = json.loads(Path(table_path).read_text())
    return train(table, Path(out_dir), epochs=epochs, seed=seed, device=device)


def train_ensemble(table_path: Path, out_dir: Path, members: int = 5, workers: int = 5, epochs: int = 40,
                   devices: list[str] | None = None) -> dict:
    """Train ``members`` seeds concurrently in ``workers`` spawned processes (small GRUs leave a GB10
    mostly idle, so several share it). Member spread is reported as ensemble uncertainty."""
    import multiprocessing as mp
    import time
    devices = devices or ["cuda" if __import__("torch").cuda.is_available() else "cpu"]
    jobs = [(str(table_path), str(out_dir / f"member_{i}"), 12 + i, epochs, devices[i % len(devices)]) for i in range(members)]
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(workers) as pool:
        results = pool.map(_member, jobs)
    wall = time.perf_counter() - t0
    stations = results[0]["metrics_same_day"]["test"].keys()
    test = {st: {"nse_members": [r["metrics_same_day"]["test"][st]["nse"] for r in results],
                 "kge_members": [r["metrics_same_day"]["test"][st]["kge"] for r in results]} for st in stations}
    summary = {"schema": "pravah-flow-emulator-ensemble-v1", "members": members, "concurrent_workers": workers,
               "wall_seconds_measured": round(wall, 2), "member_train_seconds": [r["train_seconds_measured"] for r in results],
               "devices": sorted({r["device"] for r in results}), "device_names": sorted({str(r["device_name"]) for r in results}),
               "test_metrics_same_day": test, "member_dirs": [Path(j[1]).name for j in jobs]}
    (out_dir / "ensemble_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


class Emulator:
    """Loads one member (``emulator.pt``) or every ``member_*/emulator.pt`` under a directory."""

    def __init__(self, path: Path, device: str | None = None):
        import torch
        self.torch = torch
        path = Path(path)
        files = sorted(path.glob("member_*/emulator.pt")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"No emulator checkpoints under {path}")
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.members = []
        for f in files:
            ck = torch.load(f, map_location="cpu", weights_only=False)
            m = build_model(ck["n_in"], ck["n_out"]).to(self.dev)
            m.load_state_dict(ck["state_dict"])
            m.eval()
            self.members.append((m, np.asarray(ck["mu"]), np.asarray(ck["sd"])))
        self.meta = ck

    def predict(self, precip: np.ndarray, temp: np.ndarray, days: list[date]) -> dict:
        """Inputs are the last SEQ_DAYS days ending on the final target day. Returns (HORIZON, G) m³/s."""
        X = self.torch.as_tensor(features(precip, temp, days)[-SEQ_DAYS:][None], device=self.dev)
        qs = []
        with self.torch.no_grad():
            for m, mu, sd in self.members:
                qs.append(np.expm1(m(X).float().cpu().numpy()[0] * sd + mu))
        q = np.stack(qs)
        return {"days": [d.isoformat() for d in days[-HORIZON:]], "stations": self.meta["stations"],
                "discharge_m3s": np.median(q, 0).tolist(), "member_min_m3s": q.min(0).tolist(),
                "member_max_m3s": q.max(0).tolist(), "members": len(qs)}
