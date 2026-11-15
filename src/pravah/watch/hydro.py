"""Downstream river response: DEM cross-sections, Manning normal depth, routing, exposure.

Method (documented assumptions, every one carried into outputs):
* Main stem = shortest connected path along mapped OSM Bhote Koshi/Trishuli centrelines
  (EPSG:32645). Small gaps between mapped segments are bridged by straight lines (flagged).
* Cross-sections every 500 m, ±1.5 km, sampled every 30 m from Copernicus DEM GLO-30. The
  DEM records the water surface at acquisition, so depths are *above that surface*, not
  above the riverbed. 30 m sampling cannot resolve narrow gorges; widths there are coarse.
* Steady uniform flow (Manning): Q = (1/n) A R^(2/3) S^(1/2), solved for the water-surface
  elevation by bisection on the GPU for every section × ensemble member at once. Only the
  connected wet region around the thalweg counts (no filling of disconnected side valleys).
* Arrival time: kinematic-wave celerity c = 5/3·V integrated downstream from the source.
* GLOF scenario: lake volume from area (Huggins et al. 2011: V = 0.104 A^1.42) and peak
  discharge from two published empirical relations (Clague & Mathews 1973; Popov 1991),
  attenuated with distance by an assumed e-folding length. Order-of-magnitude uncertainty.
* Empirical width: at-a-station hydraulic geometry W = a Q^b fitted per reach from historical
  Sentinel-2 widths vs GloFAS discharge (training years only).
This is a screening model, not a calibrated 2-D hydrodynamic simulation.
"""
from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass

import numpy as np

from pravah.watch.config import (
    GLOF_ATTENUATION_KM,
    MAIN_STEM_NAMES,
    METRIC_CRS,
    SECTION_HALF_WIDTH_M,
    SECTION_SPACING_M,
    SECTION_STEP_M,
)


# ----------------------------------------------------------------------------------- main stem
def main_stem(rivers_geojson: dict, source_hint, sink_hint, names=MAIN_STEM_NAMES, snap_m=50.0,
              max_gap_m=12000.0):
    """Shortest connected path along named river lines from source to sink (metric LineString)."""
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    feats = [f for f in rivers_geojson["features"]
             if f["properties"].get("name") in names or f["properties"].get("name") is None]
    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326").to_crs(METRIC_CRS)
    lines = []
    for g in gdf.geometry:
        lines += list(g.geoms) if g.geom_type == "MultiLineString" else [g]
    nodes: list[tuple[float, float]] = []

    def node(xy):
        for i, n in enumerate(nodes):
            if math.hypot(n[0] - xy[0], n[1] - xy[1]) <= snap_m:
                return i
        nodes.append(xy)
        return len(nodes) - 1

    edges: dict[int, list[tuple[int, float, list, bool]]] = {}
    for ln in lines:
        cs = list(ln.coords)
        a, b = node(cs[0][:2]), node(cs[-1][:2])
        edges.setdefault(a, []).append((b, ln.length, cs, False))
        edges.setdefault(b, []).append((a, ln.length, cs[::-1], False))
    # bridge gaps: connect each node to its nearest node in another component within max_gap_m
    comp = _components(len(nodes), edges)
    for i, p in enumerate(nodes):
        best = None
        for j, q in enumerate(nodes):
            if comp[i] != comp[j]:
                d = math.hypot(p[0] - q[0], p[1] - q[1])
                if d <= max_gap_m and (best is None or d < best[1]):
                    best = (j, d)
        if best:
            j, d = best
            edges.setdefault(i, []).append((j, d * 1.5, [p, nodes[j]], True))  # penalise bridges
            edges.setdefault(j, []).append((i, d * 1.5, [nodes[j], p], True))
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
    src_xy, snk_xy = t.transform(*source_hint), t.transform(*sink_hint)
    src = min(range(len(nodes)), key=lambda i: math.hypot(nodes[i][0] - src_xy[0], nodes[i][1] - src_xy[1]))
    snk = min(range(len(nodes)), key=lambda i: math.hypot(nodes[i][0] - snk_xy[0], nodes[i][1] - snk_xy[1]))
    dist, prev = {src: 0.0}, {}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == snk:
            break
        if d > dist.get(u, math.inf):
            continue
        for v, w, cs, bridged in edges.get(u, []):
            if d + w < dist.get(v, math.inf):
                dist[v], prev[v] = d + w, (u, cs, bridged)
                heapq.heappush(pq, (d + w, v))
    if snk not in prev and snk != src:
        raise ValueError("Mapped main stem is not connected from source to sink")
    parts, bridges, cur = [], 0, snk
    while cur != src:
        u, cs, bridged = prev[cur]
        parts.append(cs)
        bridges += bridged
        cur = u
    coords: list = []
    for cs in reversed(parts):
        coords += cs if not coords else cs[1:]
    line = LineString([c[:2] for c in coords])
    return line, {"bridged_gaps": int(bridges), "length_km": round(line.length / 1000, 2),
                  "source_snap_m": round(Point(src_xy).distance(Point(nodes[src])), 1),
                  "sink_snap_m": round(Point(snk_xy).distance(Point(nodes[snk])), 1)}


def _components(n, edges):
    comp, c = [-1] * n, 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        stack = [s]
        comp[s] = c
        while stack:
            u = stack.pop()
            for v, *_ in edges.get(u, []):
                if comp[v] < 0:
                    comp[v] = c
                    stack.append(v)
        c += 1
    return comp


# ---------------------------------------------------------------------------- cross-sections
@dataclass
class Sections:
    chainage_m: np.ndarray      # (S,) distance downstream from source
    cx: np.ndarray              # (S,) centre x (EPSG:32645)
    cy: np.ndarray
    nx: np.ndarray              # (S,) unit normal (left bank direction)
    ny: np.ndarray
    offsets_m: np.ndarray       # (P,) lateral offsets, negative = right bank
    z: np.ndarray               # (S, P) ground/water-surface elevation (m, EGM2008)
    bed_m: np.ndarray           # (S,) monotone thalweg elevation
    slope: np.ndarray           # (S,) energy-slope proxy
    thalweg_idx: np.ndarray     # (S,) index of thalweg along offsets

    def to_json(self) -> dict:
        return {k: getattr(self, k).tolist() for k in self.__dataclass_fields__}

    @classmethod
    def from_json(cls, d: dict) -> Sections:
        return cls(**{k: np.asarray(v) for k, v in d.items()})


def build_sections(line, sample_dem, spacing=SECTION_SPACING_M, half_width=SECTION_HALF_WIDTH_M,
                   step=SECTION_STEP_M, thalweg_search_m=150.0, slope_window_m=2000.0) -> Sections:
    """``sample_dem(x, y) -> z`` takes EPSG:32645 arrays and returns elevations (NaN outside)."""
    L = line.length
    ch = np.arange(0, L, spacing)
    pts = [line.interpolate(c) for c in ch]
    ahead = [line.interpolate(min(c + 50, L)) for c in ch]
    behind = [line.interpolate(max(c - 50, 0)) for c in ch]
    tx = np.array([a.x - b.x for a, b in zip(ahead, behind)])
    ty = np.array([a.y - b.y for a, b in zip(ahead, behind)])
    norm = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / norm, tx / norm
    cx, cy = np.array([p.x for p in pts]), np.array([p.y for p in pts])
    off = np.arange(-half_width, half_width + step / 2, step)
    X = cx[:, None] + nx[:, None] * off[None, :]
    Y = cy[:, None] + ny[:, None] * off[None, :]
    z = sample_dem(X.ravel(), Y.ravel()).reshape(X.shape).astype("float64")
    near = np.abs(off) <= thalweg_search_m
    zn = np.where(near[None, :] & np.isfinite(z), z, np.inf)
    thal = zn.argmin(1)
    tz = zn[np.arange(len(ch)), thal]
    tz = np.where(np.isfinite(tz), tz, np.nan)
    # Fill gaps and enforce a non-increasing bed downstream (DEM noise creates false rises).
    idx = np.arange(len(tz))
    good = np.isfinite(tz)
    if good.sum() < 2:
        raise ValueError("DEM does not cover the river corridor")
    tz = np.interp(idx, idx[good], tz[good])
    bed = np.minimum.accumulate(tz)
    w = max(1, round(slope_window_m / spacing))
    up = bed[np.clip(idx - w, 0, len(bed) - 1)]
    dn = bed[np.clip(idx + w, 0, len(bed) - 1)]
    span = (np.clip(idx + w, 0, len(bed) - 1) - np.clip(idx - w, 0, len(bed) - 1)) * spacing
    slope = np.clip((up - dn) / np.maximum(span, spacing), 5e-4, 0.08)
    z = np.where(np.isfinite(z), z, np.nanmax(z) if np.isfinite(z).any() else 9999.0)
    # The thalweg column carries the (monotone) bed so every section has a defined low point.
    z[np.arange(len(ch)), thal] = np.minimum(z[np.arange(len(ch)), thal], bed)
    return Sections(ch, cx, cy, nx, ny, off, z, bed, slope, thal)


# ---------------------------------------------------------------------------- Manning solver
def normal_depth(sec: Sections, Q: np.ndarray, n: np.ndarray, device: str | None = None, iters: int = 48,
                 max_depth_m: float = 80.0) -> dict:
    """Solve WSE for discharge ``Q`` (S, M) and Manning ``n`` (M,) — batched on GPU when available.

    Returns depth, top width, area, velocity, wet left/right offsets, all (S, M)."""
    import torch
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    z = torch.as_tensor(sec.z, dtype=torch.float64, device=dev)[:, None, :]             # S,1,P
    bed = torch.as_tensor(sec.bed_m, dtype=torch.float64, device=dev)[:, None]          # S,1
    S = torch.as_tensor(sec.slope, dtype=torch.float64, device=dev)[:, None]
    Qt = torch.as_tensor(np.asarray(Q, dtype="float64"), device=dev)                     # S,M
    nt = torch.as_tensor(np.asarray(n, dtype="float64"), device=dev)[None, :]           # 1,M
    dx = float(sec.offsets_m[1] - sec.offsets_m[0])
    P = z.shape[-1]
    th = torch.as_tensor(sec.thalweg_idx, device=dev)
    pos = torch.arange(P, device=dev)[None, :]
    left_side = (pos >= th[:, None])[:, None, :]    # offsets ≥ thalweg (left bank side)
    dz = torch.diff(z, dim=-1)
    seg = torch.sqrt(dx * dx + dz * dz)                                                  # S,1,P-1

    def hydraulics(wse):                      # wse: S,M
        below = (wse[..., None] > z)                                                     # S,M,P
        # Connected run from the thalweg outward: left side = running product thalweg→right end,
        # right side = running product from each point up to the thalweg.
        ones = torch.ones_like(below)
        Lr = torch.cumprod(torch.where(left_side, below, ones).long(), -1)
        Rl = torch.flip(torch.cumprod(torch.flip(torch.where(~left_side, below, ones).long(), [-1]), -1), [-1])
        wet = torch.where(left_side, Lr, Rl).bool() & below
        depth_pts = torch.where(wet, wse[..., None] - z, torch.zeros((), dtype=z.dtype, device=dev))
        area = depth_pts.sum(-1) * dx
        wseg = wet[..., 1:] & wet[..., :-1]
        perim = (seg.expand_as(wseg) * wseg).sum(-1) + dx
        width = wet.sum(-1) * dx
        R = area / perim.clamp_min(1e-6)
        q = area * R.clamp_min(0) ** (2 / 3) * S.sqrt() / nt
        return q, area, width, wet

    lo = bed.expand_as(Qt).clone()
    hi = lo + max_depth_m
    for _ in range(iters):
        mid = (lo + hi) / 2
        q, *_ = hydraulics(mid)
        too_low = q < Qt
        lo = torch.where(too_low, mid, lo)
        hi = torch.where(too_low, hi, mid)
    wse = (lo + hi) / 2
    q, area, width, wet = hydraulics(wse)
    offs = torch.as_tensor(sec.offsets_m, dtype=torch.float64, device=dev)
    big = torch.full_like(wet, 1e9, dtype=torch.float64)
    left = torch.where(wet, offs.expand_as(big), -big).amax(-1)
    right = torch.where(wet, offs.expand_as(big), big).amin(-1)
    vel = Qt / area.clamp_min(1e-6)
    saturated = (hi - bed) >= max_depth_m - 1e-3
    cpu = lambda t: t.detach().cpu().numpy()
    return {"wse_m": cpu(wse), "depth_m": cpu(wse - bed), "width_m": cpu(width), "area_m2": cpu(area),
            "velocity_ms": cpu(vel), "left_m": cpu(left + dx / 2), "right_m": cpu(right - dx / 2),
            "section_full": cpu(saturated | (width >= (P - 1) * dx)), "device": str(dev)}


# ------------------------------------------------------------------- empirical width (W = aQ^b)
def fit_hydraulic_geometry(widths_m, discharges_m3s, min_points: int = 6) -> dict | None:
    w, q = np.asarray(widths_m, float), np.asarray(discharges_m3s, float)
    ok = np.isfinite(w) & np.isfinite(q) & (w > 0) & (q > 0)
    if ok.sum() < min_points:
        return None
    X, Y = np.log(q[ok]), np.log(w[ok])
    b, loga = np.polyfit(X, Y, 1)
    pred = loga + b * X
    ss = ((Y - Y.mean()) ** 2).sum()
    r2 = 1 - ((Y - pred) ** 2).sum() / ss if ss > 0 else 0.0
    return {"a": float(math.exp(loga)), "b": float(b), "r2": float(r2), "n": int(ok.sum()),
            "q_range_m3s": [float(q[ok].min()), float(q[ok].max())],
            "residual_sd_log": float(np.std(Y - pred))}


def empirical_width(fit: dict | None, Q):
    if not fit:
        return None
    return fit["a"] * np.asarray(Q, float) ** fit["b"]


# ------------------------------------------------------------------------------ routing / GLOF
def arrival_hours(sec: Sections, velocity_ms: np.ndarray, origin_chainage_m: float = 0.0) -> np.ndarray:
    """Kinematic-wave travel time (h) from ``origin`` to each section (S, M); NaN upstream of origin."""
    c = np.clip(5.0 / 3.0 * velocity_ms, 0.3, 15.0)
    ds = np.diff(sec.chainage_m, prepend=sec.chainage_m[0])[:, None]
    after = (sec.chainage_m >= origin_chainage_m)[:, None]
    t = np.cumsum(np.where(after, ds / c, 0.0), axis=0) / 3600.0
    return np.where(after, t, np.nan)


def glof_peak(area_m2: float) -> dict:
    """Empirical outburst volume and peak discharge envelope from lake area (no bathymetry)."""
    V = 0.104 * area_m2 ** 1.42                           # Huggins et al. 2011, m³
    vols = np.array([V / 2, V, V * 2])                    # area–volume scatter is ~×2
    clague = 75.0 * (vols / 1e6) ** 0.67                  # Clague & Mathews 1973, m³/s
    popov = 0.0048 * vols ** 0.896                        # Popov 1991, m³/s
    peaks = np.sort(np.concatenate([clague, popov]))
    return {"lake_area_m2": float(area_m2), "volume_m3": [float(v) for v in vols],
            "peak_m3s_low": float(peaks[1]), "peak_m3s_mid": float(np.median(peaks)), "peak_m3s_high": float(peaks[-2]),
            "relations": ["V=0.104·A^1.42 (Huggins et al. 2011)", "Qp=75·(V/1e6)^0.67 (Clague & Mathews 1973)",
                          "Qp=0.0048·V^0.896 (Popov 1991)"],
            "semantics": "Empirical envelope for a full drainage; not a breach model or a probability"}


def attenuate(peak_m3s: float, base_q: np.ndarray, distance_km: np.ndarray, lengths_km=GLOF_ATTENUATION_KM):
    """Outburst peak added to base flow, decaying with distance; one column per length (S, len)."""
    d = np.clip(np.asarray(distance_km, float), 0, None)[:, None]
    L = np.asarray(lengths_km, float)[None, :]
    return np.asarray(base_q, float)[:, None] + peak_m3s * np.exp(-d / L)


# ------------------------------------------------------------------------------ exposure index
def index_assets(sec: Sections, xs: np.ndarray, ys: np.ndarray, ground_z: np.ndarray | None = None,
                 max_offset_m: float = SECTION_HALF_WIDTH_M) -> dict:
    """Nearest section and signed lateral offset of each asset point (EPSG:32645)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    k = np.empty(len(xs), int)
    for s in range(0, len(xs), 20000):     # chunked brute force: a few hundred sections
        d = (xs[s:s + 20000, None] - sec.cx[None]) ** 2 + (ys[s:s + 20000, None] - sec.cy[None]) ** 2
        k[s:s + 20000] = d.argmin(1)
    off = (xs - sec.cx[k]) * sec.nx[k] + (ys - sec.cy[k]) * sec.ny[k]
    along = np.abs((xs - sec.cx[k]) * sec.ny[k] - (ys - sec.cy[k]) * sec.nx[k])
    inside = (np.abs(off) <= max_offset_m) & (along <= SECTION_SPACING_M)
    return {"section": k.astype(int), "offset_m": off, "inside": inside,
            "ground_z": None if ground_z is None else np.asarray(ground_z, float)}


def flooded(asset_index: dict, result: dict) -> np.ndarray:
    """(N, M) boolean: asset within the connected wet extent (and below WSE when ground z known)."""
    k, off = asset_index["section"], asset_index["offset_m"][:, None]
    hit = asset_index["inside"][:, None] & (off <= result["left_m"][k]) & (off >= result["right_m"][k])
    if asset_index.get("ground_z") is not None:
        gz = asset_index["ground_z"][:, None]
        hit &= ~np.isfinite(gz) | (gz <= result["wse_m"][k])
    return hit


def extent_polygon(sec: Sections, left_m: np.ndarray, right_m: np.ndarray):
    """Polygon (EPSG:4326 GeoJSON geometry) joining wet extents of consecutive sections."""
    import geopandas as gpd
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    lx, ly = sec.cx + sec.nx * left_m, sec.cy + sec.ny * left_m
    rx, ry = sec.cx + sec.nx * right_m, sec.cy + sec.ny * right_m
    quads = []
    for i in range(len(sec.cx) - 1):
        q = Polygon([(lx[i], ly[i]), (lx[i + 1], ly[i + 1]), (rx[i + 1], ry[i + 1]), (rx[i], ry[i])])
        if q.is_valid and q.area > 0:
            quads.append(q)
        elif q.area > 0:
            quads.append(q.buffer(0))
    if not quads:
        return None
    geom = gpd.GeoSeries([unary_union(quads).simplify(10)], crs=METRIC_CRS).to_crs("EPSG:4326").iloc[0]
    return json.loads(gpd.GeoSeries([geom]).to_json())["features"][0]["geometry"]
