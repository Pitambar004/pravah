# Nepal monitor and Rasuwa demo handoff

The Nepal dashboard combines current public source watches with a separately
recorded retrospective Rasuwa replay and optional exploratory river-corridor
exposure. It does not contain a calibrated downstream forecast or a measured
Nepal forecast accuracy score. The historical case remains 26 August 2026;
the NDRRMA July 2025 report is separate event context.

## Start locally

Use the existing `.venv` with `.[full]` dependencies and the prepared real
checkpoint/config. If the geography has not been staged:

```sh
.venv/bin/python scripts/nepal_replay/prepare_monitor.py all
.venv/bin/python scripts/nepal_replay/prepare_lakes.py
.venv/bin/python scripts/nepal_replay/prepare_national_rivers.py
.venv/bin/python scripts/nepal_replay/prepare_images.py
```

The geography command downloads the current Nepal PBF from Geofabrik, verifies
provider MD5 and local SHA256, then exports bounded mapped river and asset
geometry. The PBF is excluded from the transfer package and Git. All assets
remain current mapped context, not verified pre-event infrastructure.

```sh
.venv/bin/python scripts/serve_nepal_replay.py --api-port 8006 --ui-port 8506 --device cpu
```

Open `http://127.0.0.1:8506` (Nepal default), or `/simulate`. Existing services
on ports 8000/8502 can remain running. Do not stop another person's service.
The launcher stops only its own child processes on shutdown.
The dashboard synchronizes changed project imports during a full rerun, so
older Streamlit processes do not retain an incompatible map-component signature.
For existing local benchmark dashboards targeting port 8000, Nepal panels use
the prepared monitor on 8006. Set `PRAVAH_NEPAL_API_URL` to override that routing;
hosted API addresses retain their explicit configuration.
The private CARTO basemap key is loaded at render time from
`PRAVAH_CARTO_BASEMAP_API_KEY` or the ignored `.streamlit/secrets.toml`.
Authenticated raster styles use `dark_all` and `light_all`; both actual tile
requests succeeded. The key is excluded from source control and the transfer
bundle. Configure it through the hosted environment/secret facility after
Team 12 v4.3.1 secret handling is verified. Without it, the Esri basemap remains
available; geographic evidence layers do not depend on the basemap provider.

## Demo sequence

For the short judge presentation, open `/simulate` and click
**Start 2-minute judges demo**. It prepares an immutable 26 August 08:30 NPT
snapshot and a separately saved exploratory scenario, then starts the clock.
The first 32 seconds visit all 15 pre-event dates using catalog metadata.
At 32 seconds, glacier review opens actual 22/24 August Sentinel-2 previews;
at 44 seconds, lake review opens the same real pair at a mapped inventory
location. Red highlights RGB appearance differences, including cloud changes,
and cannot measure glacier displacement, lake filling or depth. Historical
availability remains unverified, so these are retrospective inspection images.
The regional disturbance uses an explicitly assumed focus, not a claimed
earthquake epicentre. The Syabrubesi data-gap stage flies to its real station
location, pulses the marker and briefly locks map navigation for six seconds.
It explicitly reports that a raw historical last-reading value is unavailable.
Lake-change, seismic-disturbance, unavailable-gauge and high-water signals are
explicitly assumed workflow alerts. At 88 seconds the exploratory office
warning shows the red assumed corridor and mapped asset intersections; at
100 seconds the map fits the downstream corridor through Chitwan. At 110
seconds it reveals the separately stored later reports and opens the actual
24/27 August pre/post image pair. At 120 seconds it stops and offers the
presentation record, an Operations scenario handoff, and functioning satellite
comparison or reported-gauge-chronology evaluation. Comparison results are
saved separately and downloadable; forecast accuracy stays unavailable.
Older browser sessions load the new image fields safely after a source update.
The actual elapsed duration is recorded rather than fabricated. These alerts
demonstrate what an office could do; they are not verified alerts actually
received before this historical source failure. Level/depth/flow and forecast
accuracy remain unavailable. **Return to manual replay** restores the detailed
controls. The longer manual demonstration is below.

1. Open **Rasuwa Operations**, inspect actual source status and the Nepal default map.
   Focus the Rasuwa downstream corridor. Blue geometry represents mapped river
   centrelines and available riverbank polygons, not measured normal width.
   The national layer contains 455 mapped named major-river segments, simplified
   by 30 m for country display. Switch to Rasuwa office area or downstream
   through Chitwan for detail. PRAVAH is not an official government service.
   The map sits above monitoring tabs, with compact metric cards and a closed
   layer panel by default. Country labels and a grouped legend keep rivers
   readable; national stations are an optional layer, with corridor stations
   visible initially. Satellite previews use a bounded display size.
2. Inspect DHM sample timestamps, quality and each station's own water-level
   datum. Missing readings mean unavailable evidence, not confirmed failure.
   Weather is a gridded model estimate; the USGS search includes surrounding
   countries within its displayed rectangular region.
   The five functional tabs are Gauge readings, Water levels, Satellite images,
   Weather & seismic, and Alerts. Water-level history stores actual received
   timestamps, deduplicated and capped at 288 samples per station. No missing
   samples are filled. Charts require two real samples.
   The latest obtainable Sentinel-2 L1C provider thumbnail remains visible
   until a newer preview succeeds. Its actual acquisition timestamp determines
   its age; a newly listed scene with an unavailable preview is shown separately.
   The current verified preview was acquired on 12 November 2026, not six hours
   before the demo. It is a regional tile overview, not full model-ready imagery.
3. Browser notifications require an explicit click. They work while the page
   remains open on HTTPS or localhost. Dashboard dismissal persists in the
   session, and notifications deduplicate IDs in browser storage. No external
   messaging or background push service is implemented.
4. Select **Simulate Rasuwa flood**. Its 15-day slider starts on 11 August
   and ends on 26 August. Choose a historical cutoff and optional
   **Exploratory scenario** assumptions. Run the replay to seal eligible inputs
   and preserve the scenario separately. Changing controls requires a new run.
5. Blue assumed channel and red additional corridor use uniform buffers in
   EPSG:32645. They are not hydraulic overflow predictions. OSM intersections
   count mapped building footprints, road length, bridge-tagged features and
   selected facilities/settlement points; they do not establish household
   counts, residents, damage, safe areas or complete coverage.
   Use **Open exploratory watch in Operations** to review these intersections
   on the main map. Clear the watch to remove them. Assumptions stay in session
   and the separately saved scenario, never in the live-source observations.
   Normal monitoring omits infrastructure from the map entirely. Fresh regional
   threshold/rainfall/received-stage-rise watches reveal mapped corridor assets
   for review. A recent observed transition to unavailable readings can prompt
   review; initial missing records and provider failures do not diagnose a gauge
   failure. Stale/error caches cannot activate threshold watches. These signals
   have no calibrated flood footprint, so the context is not labelled affected.
   Houses, roads and bridges come from OSM; an image-based infrastructure
   detector has not been trained. Water-change evidence has separate model labels.
6. Reveal post-event evidence. The current reviewed USGS origin is 08:37:10 NPT
   and is classified as a landslide. The hydrology report gives the last
   Syabrubesi reading at 08:40 and an outage at 08:50, after source failure.
   Neither is established as an earlier earthquake/outage precursor.
7. Download the frozen evidence, exploratory scenario and separate evaluation.
   Forecast output and forecast accuracy remain null/unavailable. Five real
   visual previews are downloaded; archived catalog entries still do not
   establish complete spectral rasters or historical availability.

The mapped corridor extends through Chitwan: the verified extract contains
65 river lines, 100 riverbank polygons, 25,324 building footprints, 3,252 road
features, 155 bridge-tagged features, 94 selected facilities and 58 settlement
points within 350 m of selected rivers. This is an incomplete current OSM
inventory. The default 25 m half-width plus 100 m additional margin scenario
intersects 3,750 mapped building footprints, 226.84 km of roads, 102
bridge-tagged features and 21 facilities. These are scenario intersections,
not affected/damaged counts or a Chitwan flood forecast.

**Glacier & Lake Watch** uses verified GLO baselines: 214 optical and 130 SAR
regional inventory polygons, kept separate. They represent merged maximum
annual 2017–2024 outlines, not lake surfaces observed in August 2026. The
source's annual expansion rates and ALOS DSM elevations keep their original
units; neither is a 15-day filling curve, current water level or water depth.
The lake surface comparison API enforces identical metric grids, classes
0/1/255, matching optical model/bands/threshold and verified cloud-validity.
It reports water area on commonly valid pixels; level, depth and volume stay
null without the necessary elevation/bathymetric measurements.

ICIMOD's 13 October synthesis is cited only in later evaluation. It reports
pre-collapse InSAR deformation and identifies a rock/ice collapse rather than
a conventional GLOF; no single trigger is confirmed. Its direct page download
failed, so the local source receipt records the actual report metadata from
the library API. No raw waveform or daily displacement series was obtained.
Unavailable government data remains unavailable, and no earthquake-triggered
glacier failure or imminent lake breach is asserted. Those optional controls
remain explicit exploratory assumptions.

## Live source rules

The API polls every 300 seconds; the dashboard updates watch/status panels
every 30 seconds and the map every 300 seconds. Requests have bounded timeouts
and response sizes. A failed refresh retains the prior response and flags its
status. Retrieval now establishes availability now, not at a historical cutoff.
Responses are hashed and stored outside Git. Satellite asset URLs and public
DHM session/CSRF values are excluded from the portable cache.

Watch rules are provider warning/danger levels, an analyst-selected stage rise
of at least 0.5 m between received current samples, an analyst-selected 50 mm
next-24h rainfall watch, and newly listed USGS events/Sentinel-2 L1C items.
Initial catalog history does not generate new-event alerts. Event type is
preserved, including landslide. Implausible values, unknown/future sample times
and stale readings cannot generate water-level threshold alerts.

Automatic optical comparisons watch up to 500 completed run metadata files.
They require non-synthetic Nepal observations, Sentinel-2 L1C, the exact six
bands, identical model/threshold, explicit verified cloud-validity metadata,
ordered acquisitions and identical raster grids. Only pixels valid on both
dates are compared. Newly detected water appears in red and triggers an
analyst 0.01 km² watch. Current benchmark outputs do not satisfy these Nepal
requirements. Downloading/processing daily Nepal imagery still requires real
compatible assets and verified masks; catalog polling alone cannot do that.

## API and transfer

Added routes:

- `GET /api/v1/nepal/live`
- `POST /api/v1/nepal/live/refresh`
- `GET /api/v1/nepal/geography`
- `GET /api/v1/nepal/rivers`
- `GET /api/v1/nepal/imagery`
- `POST /api/v1/nepal/scenarios`
- `GET /api/v1/nepal/lakes`
- `POST /api/v1/nepal/lakes/change`

POST routes honor `X-Pravah-Token` when configured. Scenario request fields
are `assumed_half_width_m` (5–100) and `extra_margin_m` (0–250). Set
`PRAVAH_LIVE_ENABLED=0` to disable automatic polling. Data roots are
`PRAVAH_NEPAL_GEOGRAPHY` and `PRAVAH_LIVE_ROOT`; the bundle launcher configures
both to portable paths. Public-source egress on Fuzzball remains unverified.
`PRAVAH_LAKE_ROOT` points to the verified GLO inventory and its later-reference
receipt; the bundle launcher also configures that root.
`PRAVAH_NATIONAL_ROOT` points to the hash-verified national river layer. The
bundle includes the national layer, lake baselines and the current hashed
provider preview/cache without the large raw PBF or lake-source archives.

```sh
.venv/bin/python scripts/package_nepal_replay.py --with-training
.venv/bin/python scripts/run_nepal_bundle.py --bundle PATH_FROM_LATEST_RECEIPT --service verify
```

Use `outputs/nepal_replay/packages/latest_prepared.json` as the authoritative
receipt. It includes actual file hashes, current source cache, Nepal geometry,
the completed MiniUNet checkpoint and water-segmentation training preparation.
It does not include the large raw PBF. See `NEPAL_REPLAY_HANDOFF.md` and
`FUZZBALL_MORNING.md` for v4.3.1 deployment gates. Container build, registry,
Team 12 allocation, GPU training and endpoint hosting still need actual access
and validation; locally runnable code does not prove those steps.

## Morning continuation prompt

> Continue PRAVAH from `docs/NEPAL_MONITOR_HANDOFF.md` and the latest package
> receipt. Preserve scientific cutoff separation and the working local demo.
> Inspect our authorized Team 12 Fuzzball v4.3.1 access, registry/builder,
> allocation and volume. Validate the image and staging workflow, transfer and
> verify the bundle, run prepared water-segmentation training and CUDA inference,
> capture actual hashes/job IDs/GPU identities/timings, and host the API and Nepal
> dashboard on approved endpoints. Test public-source egress and the replay in
> the hosted browser. Continue forecast/Prithvi work only with compatible real
> inputs, time-eligible evidence and independently evaluable targets. Request
> P2/P3/P4 review before merging.

Peer review is required before merge: P3 geometry/exposure and source semantics,
P2 model/comparison eligibility, P4 portable package and Fuzzball validation.

## Validation record

Commands and outputs are recorded under `outputs/nepal_replay/monitor_*`.
The required CI lint scope is `ruff check app/ tests/ src/`, plus checks of the
modified integration scripts. The broader `ruff check .` also detects existing
style/executable-bit errors in other owners' unchanged starter scripts; those
are outside this P5 change. Final measured results are recorded after real
geography, browser and relocated-package verification.

Service dependency resolution passed against public PyPI for Linux ARM64 and
CPython 3.12 using binary wheels. The updated 54-package resolver report includes
python-dotenv; the candidate
hash lock are under `outputs/nepal_replay/monitor_arm64_resolution.json` and
`requirements-linux-arm64-py312.candidate.txt`. This is dependency resolution,
not a container installation, verified base-image Python version or CUDA run.

Final local checks before transfer:

```sh
.venv/bin/pytest -q
# 176 passed, 3 existing warnings (final output recorded in monitor_final_pytest.log)
.venv/bin/ruff check app/ tests/ src/ scripts/nepal_replay/prepare_lakes.py \
  scripts/nepal_replay/prepare_monitor.py scripts/nepal_replay/prepare_national_rivers.py \
  scripts/serve_nepal_replay.py \
  scripts/package_nepal_replay.py scripts/run_nepal_bundle.py
# All checks passed
.venv/bin/python -u outputs/nepal_replay/verify_nepal_monitor.py
# Real source cache, OSM scenario API, browser freeze/reveal/download: passed
.venv/bin/python -u outputs/nepal_replay/verify_judge_demo.py
# Actual two-minute browser rehearsal; measured time in judge_demo_verification.json
.venv/bin/python -u outputs/nepal_replay/verify_portable.py
# Extracted package: hashes, actual CPU model inference, API and browser checks
```

Browser-only synthetic notification fixtures verified opt-in, denial,
deduplication and disabling. These did not enter the event/source cache.
Actual Chrome clicks verified notification enable/disable and station details.
The lake inventory and 15-day controls are separately browser-checked. Transfer
verification records remain in `outputs/nepal_replay/` and the latest package
receipt; remote Fuzzball execution remains pending.

## Image/animation repair verification — 15 November

The actual browser rehearsal completed in **120.60876308300067 seconds** and
visited all ten phases. It verified three automatically opened real-image
dialogs, the original/highlight toggle, five animated map focuses, gauge
lock/unlock, downstream scenario layers, the final comparison button and
unchanged frozen snapshot. All 484 recorded CARTO tile responses returned
200; there were no browser errors. Evidence is in
`outputs/nepal_replay/judge_demo_verification.json` and
`simulation_fix_browser.log`. An earlier rehearsal hit a browser-check race
during a fragment update; the final check waits for the complete rendered stage.

Executed commands:

```sh
.venv/bin/python -u scripts/nepal_replay/prepare_images.py
# Five genuine same-tile/orbit previews, three image pairs; image hashes verified.
.venv/bin/pytest -q
# 179 passed, 3 warnings in 19.03s; simulation_fix_full_pytest.log
.venv/bin/ruff check app/ tests/ src/ scripts/nepal_replay/prepare_images.py \
  scripts/package_nepal_replay.py scripts/run_nepal_bundle.py scripts/serve_nepal_replay.py --fix
# All checks passed; simulation_fix_lint.log
.venv/bin/python -u outputs/nepal_replay/verify_judge_demo.py
# Passed; actual imagery and interactions checked in Chrome.
```

The transfer includes the verified preview/overlay files and `.env.example`.
Private `.env` and Streamlit credentials remain excluded. P1/P2/P3/P4 review
is requested before merge; no remote training, GPU hosting or downstream
forecast accuracy is established by these local results.
