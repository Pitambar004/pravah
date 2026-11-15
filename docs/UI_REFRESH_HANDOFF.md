# Dashboard UI refresh — 15 November 2026

Implemented in the existing feature checkout, without switching branches,
resetting files, committing other work, restarting shared services or touching
forecasting, training, source collection, Fuzzball or infrastructure modules.
Claude's concurrent changes remain outside this UI scope. Peer review requested
before merge; no merge or deployment was performed.

## Presentation changes

- Sidebar groups: Monitor, Explore and Model workspace. Existing routes and
  controls remain accessible; user-facing labels describe their purpose.
- More legible secondary text, consistent metric cards, restrained callouts,
  clearer typography and two-column phone metric layouts.
- One visible persistence caveat instead of repeated warning blocks. All source
  and interpretation limitations remain in Method, sources & limitations.
- Friendly evaluation labels; comparison and error maps in separate tabs.
  Only the active map renders, avoiding Leaflet initialization in hidden tabs.
- Comparison-map headers no longer overlap zoom/basemap controls. Maps stack on
  phones. Keyboard focus styling and provider attribution remain present.
- Empty-state CLI commands and artifact filenames live in Setup details;
  missing data no longer gets a generic Awaiting processing badge.
- UI reload dependencies include the modified presentation components.

Files: `app/Home.py`, `app/theme.py`, `app/context.py`, `app/hot_reload.py`,
`app/components/{layout,empty_state,compare_map}.py`, `app/sections/validation.py`.
The validation page also includes earlier baseline work; review that diff in context.

## Checks

```text
$ .venv/bin/python -m pytest -q tests/test_ui_components.py tests/test_ui_apptest.py tests/test_ui_new_pages.py tests/test_operations.py
59 passed in 14.43s
$ git diff --check
(no output; exit 0)
```

Browser review used the existing local dashboard on port 8506 and installed
Chrome via Playwright. The capture helper and screenshots are preserved locally
in ignored `outputs/ui_review/`. The exact final browser command was:

```text
$ .venv/bin/python /private/tmp/pravah-ui-review/capture.py --label verified
validation exceptions: 0 overflow: False JS errors: []
Map tabs switch successfully
monitor exceptions: 0 overflow: False JS errors: []
mobile exceptions: 0 overflow: False JS errors: []
```

Desktop viewport: 1440×1000. Phone viewport: 390×844. Inspected the monitoring
page, prediction comparison, error-map tab and phone comparison layout. Earlier
browser inspection caught a hidden-tab zoom problem, resolved by lazy tab rendering.
No broad cross-browser or remote-deployment validation is claimed. Screenshots
show local application state at capture time, not new scientific measurements.
