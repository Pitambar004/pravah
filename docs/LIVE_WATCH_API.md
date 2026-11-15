# PRAVAH Live Watch API

**Audience:** anyone on the event network who wants Rasuwa watch data in a script, notebook or another app.
**What it is:** a 24/7 service running on a Team 12 GB10 inside Fuzzball (workflow `10_watch_serve`).
It watches rainfall, earthquakes, DHM river gauges, glacial lakes, glacier motion and satellite water for
the Bhote Koshi–Trishuli river (Rasuwa, Nepal). It combines them into one watch level and predicts how the
river downstream would respond.

> **Not an official warning service.** Watch levels come from analyst rules for human review, and the
> rules have not been validated. Nepal's Department of Hydrology and Meteorology (DHM) issues flood
> warnings. Read `limitations` before you act on any number.

## 1. Base URL and interactive docs

The base URL is the Fuzzball endpoint named `api` of the running `pravah-watch-serve-*` workflow.
Each redeploy creates a new workflow id, so always look up the current URLs:

```bash
fuzzball workflow list | grep pravah-watch-serve          # newest Started one
fuzzball workflow endpoints <serve-workflow-id>           # prints the api and dashboard URLs
```

The event cluster's endpoints use a TLS certificate from the event's private certificate authority. On a
machine that does not trust that CA, browsers show a certificate warning and `curl`/`httpx` refuse to connect.
Install the organisers' CA certificate. Do not turn off certificate verification.
From a laptop with the Fuzzball CLI you can also use
`fuzzball workflow port-forward <serve-workflow-id> api 18000:8000` and the base URL `http://127.0.0.1:18000`.

| Page | Path |
|---|---|
| Interactive docs (try every route in the browser) | `<base>/docs` |
| OpenAPI schema (for client generators) | `<base>/openapi.json` |
| Liveness | `<base>/health` |
| Dashboard (separate endpoint `dashboard`) | Live Watch 24/7 page |

## 2. Access

- **Reading is open.** The endpoint scope is `public`, so anyone with the URL can call `GET` routes. CORS allows browser `GET` from any origin.
- **Changing shared state needs a token.** These routes require header `X-Pravah-Token`: `POST /api/v1/watch/refresh` and the existing inference write routes. Ask the Team 12 operator for the token. Never put it in code you share.
- Please poll no more often than **every 30 s**. Live data only changes every 5 minutes (rain, gauges, quakes) or every 30 minutes (satellite passes, discharge).

## 3. Routes

| Method | Path | Returns |
|---|---|---|
| GET | `/api/v1/watch/state` | Latest live frame: watch level, every indicator, evidence, downstream prediction |
| GET | `/api/v1/watch/timeline?limit=288` | History of watch levels (one row per engine tick, newest last) |
| GET | `/api/v1/watch/geography` | Main river line, gauge stations (with distance along the river), 10 km reaches |
| GET | `/api/v1/watch/model` | Model card: emulator test scores, width-fit quality, lake/glacier baselines, measured build timings |
| GET | `/api/v1/watch/status` | Health of components (river model, emulator, satellite observer), source errors, GPU name, replay progress |
| GET | `/api/v1/watch/replay` | Rasuwa 26 Aug 2026 replay summary: frame count, level per frame, provenance |
| GET | `/api/v1/watch/replay/frames/{i}` | One hourly replay frame (same shape as `state`) |
| POST | `/api/v1/watch/refresh` | Force a live tick now (token required) |

Until the first live tick finishes (a few minutes after start), `/state` returns **503**. Until the GPU has
computed the replay frames, `/replay` returns `status: waiting_for_build | building` with progress.

## 4. The `state` object

```jsonc
{
  "mode": "live",
  "updated_at_utc": "<ISO-8601 UTC>",
  "assessment": {
    "level": "ok | watch | warning | severe",
    "families_elevated": ["rain", "cryosphere"],      // independent hazard families at watch or above
    "indicators": [
      {
        "id": "lake_change_s2",                          // see table below
        "family": "cryosphere",
        "label": "Glacial-lake area change (optical)",
        "status": "ok | watch | warning | severe | unknown",   // unknown = no evidence, NOT safe
        "value": <number|null>, "units": "% vs baseline",
        "threshold": {"watch": 0.10, "warning": 0.25},
        "observed_at_utc": "<ISO-8601 UTC of the evidence>",
        "source": "Sentinel-2 L2A",
        "detail": "lake <GLO id>: <current km²> vs <baseline km²>"
      }
    ]
  },
  "downstream": {
    "member_count": <int>,                                  // ensemble: discharge source × Manning n (+ GLOF members)
    "reaches": [
      {
        "reach_id": "R03", "label": "<nearest mapped settlement>", "start_km": 30, "end_km": 40,
        "discharge_m3s": [min, median, max],
        "hydraulic_width_m": [p10, median, p90],         // Manning normal depth on DEM cross-sections
        "empirical_width_m": <number|null>,                       // W = a·Q^b fitted on 2018–2024 Sentinel-2 widths (null if no fit)
        "width_methods_agree": true,                     // false = the two width methods differ by more than ×2
        "depth_m": [p10, median, p90],                   // above the DEM water surface, not the riverbed
        "arrival_h": [min, median, max],                 // kinematic travel time from the flood origin
        "velocity_ms": <number>
      }
    ],
    "exposure": {                                        // NEWLY exposed: wet in the member, dry at this month's normal flow
      "buildings": {"median": <n>, "max": <n>, "in_wet_extent_max": <n>, "per_reach_max": {"R03": <n>},
                    "semantics": "newly exposed vs monthly normal flow"},
      "roads": {"median": <km>, "max": <km>, "units": "km"},
      "bridges": {...}, "facilities": {...}, "settlements": {...}
    },
    "normal_flow_reference": {"month": <1-12>, "station_q_m3s": {"<station id>": <m³/s>}},  // GloFAS 2016–2025 monthly median
    "glof": null,                                        // or the outburst scenario when a lake indicator fires
    "extent_median": {GeoJSON geometry}, "extent_max": {GeoJSON geometry},
    "exposed_points_max": {GeoJSON FeatureCollection},
    "device": "cuda", "hydraulics_seconds_measured": <measured s>
  },
  "evidence": {"gauges": [...], "rain": {...}, "seismic": [...], "lakes": {"s2": [...], "s1": [...]},
               "glacier": [...], "widths": [...], "discharge": [...], "latest_passes": [...]},
  "source_errors": {"<source>": "<error>"}          // last good evidence is kept and shown as stale
}
```

Assets already "wet" at normal flow sit within one 30 m DEM cell of the channel. That is a resolution
artefact, so the headline numbers count only *newly* exposed assets. `in_wet_extent_max` keeps the raw count.
Any value that could not be computed is `null`; the paths are listed in `non_finite_fields`.

All geometry is GeoJSON in EPSG:4326 (lon, lat). Lengths and areas were computed in EPSG:32645 (UTM 45N, metres).

### Indicators

| id | Family | Fires when (analyst rule) |
|---|---|---|
| `rain_72h` | rain | Catchment mean ≥ 60 mm in 72 h (watch), ≥ 120 mm (warning) |
| `rain_24h_forecast` | rain | Any catchment point forecast ≥ 50 mm in 24 h (watch), ≥ 100 mm (warning) |
| `seismic` | seismic | Earthquake ≤ 100 km from Rasuwa in the last 72 h: M ≥ 4.5 (watch), ≥ 5.5 (warning) |
| `gauge_level` | gauge | Main-stem gauge at or above DHM warning level (warning) or danger level (severe) |
| `gauge_rise` | gauge | A gauge rose ≥ 0.5 m within 1 h |
| `gauge_silent` | gauge | Rasuwagadhi/upstream gauge reported within 6 h, then went silent for > 1 h. An outage and flood damage look the same from here |
| `lake_change_s2` / `lake_change_s1` | cryosphere | Lake area +10 % (watch) or +25 % (warning) vs its baseline, or −30 % (possible drainage), and the change is ≥ 0.02 km². Optical and radar are never mixed |
| `glacier_speed` | cryosphere | Glacier-zone median surface speed ≥ 2× (watch) or 3× (warning) its baseline |
| `width_anomaly` | river | Observed river width ≥ 1.5× (watch) or 2× (warning) the width expected at current flow |
| `discharge` | river | GloFAS or emulator discharge above the monthly p90 (watch) or p99 (warning) |
| `sar_new_water` | river | ≥ 0.05 km² new radar-dark water in the corridor between two same-orbit Sentinel-1 passes |

**Combined level** = the worst indicator status. It is raised one step when three or more independent families
are at watch or above at the same time (compound hazard).

## 5. Examples

```bash
BASE=https://<fuzzball-endpoint-url>
curl -s $BASE/api/v1/watch/state | python3 -m json.tool | head -40
curl -s "$BASE/api/v1/watch/timeline?limit=12"
curl -s $BASE/api/v1/watch/model | python3 -c "import json,sys; print(json.load(sys.stdin)['emulator'])"
curl -s -X POST -H "X-Pravah-Token: $TOKEN" $BASE/api/v1/watch/refresh
```

```python
import httpx
base = "https://<fuzzball-endpoint-url>"
s = httpx.get(f"{base}/api/v1/watch/state", timeout=20).json()
print(s["assessment"]["level"], s["assessment"]["families_elevated"])
for r in s["downstream"]["reaches"]:
    print(r["reach_id"], r["label"], "width", r["hydraulic_width_m"][1], "m, arrival", r["arrival_h"])

# Rasuwa replay, frame by frame
meta = httpx.get(f"{base}/api/v1/watch/replay").json()
first_warning = next((i for i, lv in enumerate(meta["levels"]) if lv in ("warning", "severe")), None)
frame = httpx.get(f"{base}/api/v1/watch/replay/frames/{first_warning or 0}").json()
```

## 6. Models and data behind the numbers

| Component | Trained / fitted on | Where it ran |
|---|---|---|
| Rain → discharge emulator (5-member GRU ensemble) | Input: ERA5 daily rainfall and temperature at 8 catchment points (Open-Meteo archive). Target: GloFAS reanalysis daily discharge at the main-stem gauge cells, 1984→now. Train ≤ 2022, validate 2023–24, **test 2025–26** (both Rasuwa floods are only in test) | GB10, Fuzzball |
| Hydraulic width/depth/arrival | Copernicus DEM GLO-30 cross-sections every 500 m along the OSM main stem; Manning n ∈ {0.035, 0.05, 0.07} | GB10 (GPU solver) |
| Empirical width W = a·Q^b | Sentinel-2 L2A river widths 2018–2024 paired with same-day GloFAS discharge | GB10 |
| Lake and glacier baselines | Sentinel-2 L2A / Sentinel-1 RTC, Sep–Dec of 2023–2024 | GB10 |
| GLOF scenario | Published empirical relations (Huggins et al. 2011; Clague & Mathews 1973; Popov 1991), no lake bathymetry | GB10 |
| Exposure | OpenStreetMap Nepal extract (ODbL) | GB10 |

Measured scores and timings are served live at `/api/v1/watch/model`. This document deliberately quotes none.

## 7. Limitations

- The emulator learns GloFAS, so it cannot see glacial-lake outbursts or landslide-dam bursts. Those come only from the lake/glacier indicators and the GLOF scenario.
- Hydraulics are steady uniform flow on a 30 m DEM. Narrow gorges are coarse, and depths are above the DEM water surface, not the riverbed. This is screening, not a 2-D flood simulation.
- DHM serves only about 2.5 weeks of gauge history, so the Aug-2026 replay has no gauge readings and those indicators show `unknown`.
- In the replay, satellite acquisition times are real but provider availability times were not verified. The replay is a reconstruction, not a forecast that could have been issued at the time.
- Live satellite cadence is about 5 days (Sentinel-2) and about 12 days (Sentinel-1). Monsoon cloud often leaves optical indicators `unknown`.
