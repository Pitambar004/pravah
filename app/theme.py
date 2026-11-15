"""PRAVAH design system — tokens and global stylesheet.

All colours, radii, spacing and type sizes used anywhere in the app come from
``TOKENS``. Python components reference ``TOKENS`` (for inline SVG / map styles);
CSS references the same values through custom properties generated below.
"""
from pravah.configuration import configured_url

TOKENS: dict[str, str] = {
    # Surfaces
    "bg_app": "#090E18",
    "bg_sidebar": "#0D1420",
    "surface": "#111B2A",
    "surface_raised": "#172335",
    "surface_hover": "#1D2B3E",
    "border": "#263448",
    "border_subtle": "#1C293A",
    # Text
    "text": "#F3F6FA",
    "text_secondary": "#A4B2C5",
    "text_muted": "#718198",
    "text_disabled": "#536176",
    # Semantic accents
    "accent": "#38BDF8",
    "accent_hover": "#7DD3FC",
    "success": "#34D399",
    "warning": "#FBBF24",
    "danger": "#FB7185",
    "scenario": "#A78BFA",
    # Map semantics
    "map_water": "#38BDF8",
    "map_exposure": "#FBBF24",
    "map_scenario": "#A78BFA",
    "map_reference": "#F59E0B",
    "map_unavailable": "#536176",
}

FONT_SANS = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def _css_vars() -> str:
    return "\n".join(f"  --pv-{k.replace('_', '-')}: {v};" for k, v in TOKENS.items())


DESIGN_SYSTEM_CSS = """
<style>
@import url('%FONT_CSS%');
@import url('%ICONS_CSS%');

:root {
%VARS%
  --pv-font: %FONT_SANS%;
  --pv-mono: %FONT_MONO%;
  --pv-accent-soft: rgba(56, 189, 248, 0.10);
  --pv-success-soft: rgba(52, 211, 153, 0.10);
  --pv-warning-soft: rgba(251, 191, 36, 0.10);
  --pv-danger-soft: rgba(251, 113, 133, 0.10);
  --pv-scenario-soft: rgba(167, 139, 250, 0.10);
  --pv-neutral-soft: rgba(164, 178, 197, 0.08);
  --pv-r-sm: 8px;
  --pv-r-md: 12px;
  --pv-r-lg: 16px;
  --pv-s1: 4px; --pv-s2: 8px; --pv-s3: 12px; --pv-s4: 16px;
  --pv-s5: 24px; --pv-s6: 32px; --pv-s7: 40px; --pv-s8: 48px;
  --pv-ease: cubic-bezier(0.2, 0, 0, 1);
  --pv-dur: 160ms;
}

/* ---------- Base ---------- */
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--pv-bg-app) !important;
  color: var(--pv-text);
  font-family: var(--pv-font);
  -webkit-font-smoothing: antialiased;
  font-feature-settings: "cv11", "ss01";
}
.stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp textarea { font-family: var(--pv-font); }
.pv-icon {
  font-family: 'Material Symbols Rounded' !important; font-weight: normal; font-style: normal;
  line-height: 1; letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap;
  direction: ltr; font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased;
  font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 20; vertical-align: middle;
  overflow: hidden; flex: none; user-select: none;
}
code, pre, .pv-mono { font-family: var(--pv-mono) !important; }
.stApp code {
  background: var(--pv-surface-raised); color: var(--pv-text-secondary);
  border: 1px solid var(--pv-border-subtle); border-radius: 6px;
  padding: 1px 6px; font-size: 12px;
}
header[data-testid="stHeader"] { background: transparent !important; pointer-events: none; }
header[data-testid="stHeader"] button, header[data-testid="stHeader"] a { pointer-events: auto; }
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
[data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
[data-testid="stToolbar"] { display: flex !important; background: transparent !important; }
[data-testid="stExpandSidebarButton"] { color: var(--pv-text-secondary) !important; }
[data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] {
  top: 12px; left: 12px; background: var(--pv-surface); border: 1px solid var(--pv-border);
  border-radius: var(--pv-r-sm); z-index: 1000;
}

*:focus-visible {
  outline: 2px solid var(--pv-accent) !important;
  outline-offset: 2px !important;
  border-radius: 6px;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important; }
}

/* ---------- Page container ---------- */
.block-container, [data-testid="stMainBlockContainer"] {
  max-width: 1440px !important;
  padding: 32px 32px 48px !important;
}
@media (max-width: 1024px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding: 56px 24px 40px !important; }
}
@media (max-width: 640px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding: 56px 16px 32px !important; }
}
[data-testid="stVerticalBlock"] { gap: 16px; }
/* Tablet: metric rows become 2-up; map + side panel rows stack */
@media (max-width: 1180px) and (min-width: 641px) {
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .pv-metric) { flex-wrap: wrap !important; }
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .pv-metric) > [data-testid="stColumn"] {
    flex: 1 1 calc(50% - 8px) !important; min-width: calc(50% - 8px) !important; }
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] iframe),
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stButtonGroup"]) { flex-direction: column !important; align-items: stretch !important; }
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stButtonGroup"]) > [data-testid="stColumn"] { width: 100% !important; }
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] iframe) > [data-testid="stColumn"] { width: 100% !important; flex: 1 1 auto !important; }
}
[data-testid="stHorizontalBlock"] { gap: 16px; }
.stMarkdown { margin-bottom: 0 !important; }
[data-testid="stMarkdownContainer"]:has(> .pv-root) { margin-bottom: 0 !important; }
[data-testid="stMarkdownContainer"] p:empty { margin: 0; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
  background: var(--pv-bg-sidebar) !important;
  border-right: 1px solid var(--pv-border-subtle) !important;
}
section[data-testid="stSidebar"][data-collapsed="false"] { min-width: 248px !important; max-width: 248px !important; }
@media (min-width: 641px) {
  section[data-testid="stSidebar"][data-collapsed="true"] { min-width: 0 !important; max-width: 0 !important; width: 0 !important; overflow: hidden; border: 0 !important; }
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 0; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding: 20px 16px 24px; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 2px; }
[data-testid="stSidebarHeader"] { padding: 12px 16px 0; height: auto; }

.pv-brand { display: flex; align-items: center; gap: 10px; padding: 4px 8px 20px; }
.pv-brand svg { flex: none; }
.pv-brand-name { font-size: 15px; font-weight: 600; letter-spacing: 0.08em; color: var(--pv-text); line-height: 18px; }
.pv-brand-sub { font-size: 11px; color: var(--pv-text-muted); line-height: 16px; }
.pv-brand-deva { color: var(--pv-text-disabled); font-weight: 400; letter-spacing: 0; margin-left: 4px; }

.pv-nav-label {
  font-size: 11px; line-height: 16px; font-weight: 500; color: var(--pv-text-disabled);
  padding: 12px 10px 8px;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a,
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
  position: relative;
  border-radius: var(--pv-r-sm); padding: 7px 10px !important; margin: 0;
  color: var(--pv-text-secondary) !important; text-decoration: none;
  transition: background var(--pv-dur) var(--pv-ease), color var(--pv-dur) var(--pv-ease);
  min-height: 36px;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a span,
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] span {
  color: inherit !important; font-size: 13.5px !important; font-weight: 500;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover,
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {
  background: var(--pv-surface) !important; color: var(--pv-text) !important;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"],
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {
  background: var(--pv-surface-raised) !important; color: var(--pv-text) !important;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"]::before,
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"]::before {
  content: ""; position: absolute; left: 0; top: 9px; bottom: 9px; width: 2px;
  border-radius: 0 2px 2px 0; background: var(--pv-accent);
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] [data-testid="stIconMaterial"],
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] [data-testid="stIconMaterial"] {
  color: var(--pv-accent) !important;
}
section[data-testid="stSidebar"] [data-testid="stIconMaterial"] { font-size: 18px !important; color: var(--pv-text-muted); }

.pv-side-divider { height: 1px; background: var(--pv-border-subtle); margin: 16px 0 4px; }
.pv-side-meta { display: flex; justify-content: space-between; align-items: center; gap: 8px;
  font-size: 12px; line-height: 18px; color: var(--pv-text-muted); padding: 4px 10px; }
.pv-side-meta .pv-mono { font-size: 11px; color: var(--pv-text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 130px; }
.pv-side-footer { font-size: 11px; line-height: 16px; color: var(--pv-text-disabled); padding: 20px 10px 0; }
.pv-side-footer strong { color: var(--pv-text-muted); font-weight: 500; }
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  font-size: 11px !important; color: var(--pv-text-disabled) !important; font-weight: 500;
}
section[data-testid="stSidebar"] .stSelectbox, section[data-testid="stSidebar"] .stTextInput { padding: 4px 10px 0; }
section[data-testid="stSidebar"] .stCheckbox, section[data-testid="stSidebar"] [data-testid="stToggle"] { padding: 8px 10px 0; }

/* ---------- Topbar / page header ---------- */
.pv-topbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
  font-size: 12px; line-height: 18px; color: var(--pv-text-muted); margin-bottom: 20px; }
.pv-crumbs { display: flex; align-items: center; gap: 6px; }
.pv-crumbs span.sep { color: var(--pv-text-disabled); }
.pv-crumbs span.cur { color: var(--pv-text-secondary); }
.pv-topbar-meta { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }

.pv-page-header { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap;
  padding-bottom: 24px; margin-bottom: 8px; border-bottom: 1px solid var(--pv-border-subtle); }
.stApp .pv-page-title { font-family: var(--pv-font) !important; font-size: 28px !important; line-height: 36px !important; font-weight: 600 !important; letter-spacing: -0.02em; color: var(--pv-text); margin: 0 !important; padding: 0 !important; }
.stApp .pv-page-subtitle { font-size: 14px; line-height: 22px; color: var(--pv-text-secondary); margin: 6px 0 0 !important; max-width: 760px; }
@media (max-width: 640px) { .stApp .pv-page-title { font-size: 22px !important; line-height: 30px !important; } }

.pv-section-header { display: flex; justify-content: space-between; align-items: flex-end; gap: 12px; margin: 16px 0 0; }
.stApp .pv-section-title { font-family: var(--pv-font) !important; font-size: 16px !important; line-height: 24px !important; font-weight: 600 !important; color: var(--pv-text); margin: 0 !important; padding: 0 !important; letter-spacing: -0.005em; }
.stApp .pv-page-header a, .stApp .pv-section-header a, [data-testid="stHeaderActionElements"] { display: none !important; }
.stApp .pv-section-desc { font-size: 13px; line-height: 20px; color: var(--pv-text-muted); margin: 2px 0 0 !important; }

/* ---------- Cards ---------- */
.pv-card { background: var(--pv-surface); border: 1px solid var(--pv-border-subtle); border-radius: var(--pv-r-md); padding: 20px; }
.pv-card + .pv-card { margin-top: 16px; }
.pv-card-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 12px; }
.pv-card-title { font-size: 14px; line-height: 22px; font-weight: 600; color: var(--pv-text); }
.pv-card-body { font-size: 13px; line-height: 20px; color: var(--pv-text-secondary); }
.pv-card-body p { margin: 0 0 8px; }

/* Metric card */
.pv-metric { background: var(--pv-surface); border: 1px solid var(--pv-border-subtle); border-radius: var(--pv-r-md);
  padding: 18px 20px; display: flex; flex-direction: column; gap: 6px; min-height: 132px; height: 100%;
  transition: border-color var(--pv-dur) var(--pv-ease); }
.pv-metric:hover { border-color: var(--pv-border); }
.pv-metric-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.pv-metric-label { display: flex; align-items: center; gap: 8px; font-size: 13px; line-height: 20px; color: var(--pv-text-secondary); font-weight: 500; }
.pv-metric-label .pv-icon { color: var(--pv-text-muted); }
.pv-metric-value-row { display: flex; align-items: baseline; gap: 6px; margin-top: 4px; }
.pv-metric-value { font-size: 28px; line-height: 34px; font-weight: 600; letter-spacing: -0.02em; color: var(--pv-text); font-variant-numeric: tabular-nums; }
.pv-metric-value.is-empty { font-size: 15px; line-height: 34px; font-weight: 500; color: var(--pv-text-muted); letter-spacing: 0; }
.pv-metric-unit { font-size: 13px; color: var(--pv-text-muted); font-weight: 500; }
.pv-metric-foot { margin-top: auto; font-size: 12px; line-height: 18px; color: var(--pv-text-muted); display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }

/* ---------- Badges ---------- */
.pv-badge { display: inline-flex; align-items: center; gap: 6px; height: 22px; padding: 0 8px; border-radius: 6px;
  font-size: 11.5px; font-weight: 500; line-height: 1; white-space: nowrap; border: 1px solid transparent; }
.pv-badge .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex: none; }
.pv-badge-neutral { background: var(--pv-neutral-soft); color: var(--pv-text-secondary); border-color: var(--pv-border-subtle); }
.pv-badge-accent { background: var(--pv-accent-soft); color: var(--pv-accent); border-color: rgba(56,189,248,.22); }
.pv-badge-success { background: var(--pv-success-soft); color: var(--pv-success); border-color: rgba(52,211,153,.22); }
.pv-badge-warning { background: var(--pv-warning-soft); color: var(--pv-warning); border-color: rgba(251,191,36,.22); }
.pv-badge-danger, .pv-badge-error { background: var(--pv-danger-soft); color: var(--pv-danger); border-color: rgba(251,113,133,.22); }
.pv-badge-scenario { background: var(--pv-scenario-soft); color: var(--pv-scenario); border-color: rgba(167,139,250,.22); }
.pv-badge-outline { background: transparent; color: var(--pv-text-muted); border-color: var(--pv-border); }

/* ---------- Callouts ---------- */
.pv-callout { display: flex; gap: 12px; align-items: flex-start; background: var(--pv-surface); border: 1px solid var(--pv-border-subtle);
  border-radius: var(--pv-r-md); padding: 12px 16px; font-size: 13px; line-height: 20px; color: var(--pv-text-secondary); }
.pv-callout > .pv-icon { margin-top: 1px; }
.pv-callout strong { color: var(--pv-text); font-weight: 600; }
.pv-callout-info > .pv-icon { color: var(--pv-accent); }
.pv-callout-warning { border-color: rgba(251,191,36,.25); background: linear-gradient(0deg, rgba(251,191,36,.04), rgba(251,191,36,.04)), var(--pv-surface); }
.pv-callout-warning > .pv-icon { color: var(--pv-warning); }
.pv-callout-danger, .pv-callout-error { border-color: rgba(251,113,133,.3); }
.pv-callout-danger > .pv-icon, .pv-callout-error > .pv-icon { color: var(--pv-danger); }
.pv-callout-success > .pv-icon { color: var(--pv-success); }
.pv-callout-scenario > .pv-icon { color: var(--pv-scenario); }

/* ---------- Empty state ---------- */
.pv-empty { background: var(--pv-surface); border: 1px dashed var(--pv-border); border-radius: var(--pv-r-md);
  padding: 40px 24px; text-align: center; display: flex; flex-direction: column; align-items: center; gap: 8px; }
.pv-empty-icon { width: 48px; height: 48px; border-radius: 12px; background: var(--pv-surface-raised); border: 1px solid var(--pv-border-subtle);
  display: grid; place-items: center; color: var(--pv-text-muted); margin-bottom: 4px; }
.pv-empty-title { font-size: 15px; line-height: 22px; font-weight: 600; color: var(--pv-text); }
.pv-empty-desc { font-size: 13px; line-height: 20px; color: var(--pv-text-secondary); max-width: 520px; }
.pv-empty-cmd { margin-top: 8px; font-family: var(--pv-mono); font-size: 12px; color: var(--pv-text-secondary);
  background: var(--pv-bg-app); border: 1px solid var(--pv-border-subtle); border-radius: var(--pv-r-sm); padding: 8px 12px;
  max-width: 100%; overflow-x: auto; white-space: nowrap; text-align: left; }
.pv-empty-files { margin-top: 4px; font-size: 12px; color: var(--pv-text-muted); display: flex; gap: 6px; flex-wrap: wrap; justify-content: center; }
.pv-empty.compact { padding: 24px 20px; }

/* ---------- Key/value list ---------- */
.pv-kv { display: grid; grid-template-columns: minmax(120px, 38%) 1fr; font-size: 13px; line-height: 20px; }
.pv-kv > div { padding: 8px 0; border-bottom: 1px solid var(--pv-border-subtle); }
.pv-kv > div:nth-last-child(-n+2) { border-bottom: 0; }
.pv-kv .k { color: var(--pv-text-muted); padding-right: 12px; }
.pv-kv .v { color: var(--pv-text); word-break: break-word; }
.pv-kv .v.pv-mono { font-size: 12px; color: var(--pv-text-secondary); }

/* ---------- Status list ---------- */
.pv-status-list { display: flex; flex-direction: column; }
.pv-status-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 0;
  border-bottom: 1px solid var(--pv-border-subtle); font-size: 13px; line-height: 20px; }
.pv-status-row:last-child { border-bottom: 0; }
.pv-status-name { display: flex; align-items: center; gap: 10px; color: var(--pv-text); min-width: 0; }
.pv-status-name > div { min-width: 0; overflow: hidden; }
.pv-status-row > .pv-badge { flex: none; }
.pv-status-name small { display: block; font-size: 12px; color: var(--pv-text-muted); font-family: var(--pv-mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pv-status-ico { width: 20px; height: 20px; border-radius: 50%; display: grid; place-items: center; flex: none; }
.pv-status-ico.ok { color: var(--pv-success); background: var(--pv-success-soft); }
.pv-status-ico.wait { color: var(--pv-text-muted); background: var(--pv-neutral-soft); }
.pv-status-ico.warn { color: var(--pv-warning); background: var(--pv-warning-soft); }

/* ---------- Evidence cards ---------- */
.pv-evidence-list { display: flex; flex-direction: column; gap: 8px; }
.pv-evidence { display: grid; grid-template-columns: 28px 1fr auto; gap: 12px; align-items: start;
  background: var(--pv-surface); border: 1px solid var(--pv-border-subtle); border-radius: var(--pv-r-md); padding: 14px 16px; }
.pv-evidence-mark { width: 28px; height: 28px; border-radius: 8px; display: grid; place-items: center; }
.pv-evidence-title { font-size: 13.5px; font-weight: 600; color: var(--pv-text); line-height: 20px; }
.pv-evidence-desc { font-size: 13px; color: var(--pv-text-secondary); line-height: 20px; margin-top: 2px; }
.pv-evidence-meta { display: flex; flex-wrap: wrap; gap: 4px 16px; margin-top: 8px; font-size: 12px; color: var(--pv-text-muted); line-height: 18px; }
.pv-evidence-meta b { font-weight: 500; color: var(--pv-text-secondary); font-variant-numeric: tabular-nums; }
.pv-evidence-id { font-family: var(--pv-mono); font-size: 11px; color: var(--pv-text-disabled); margin-top: 6px; word-break: break-all; }
@media (max-width: 640px) { .pv-evidence { grid-template-columns: 28px 1fr; } .pv-evidence > .pv-badge { grid-column: 2; justify-self: start; } }

/* ---------- Timeline ---------- */
.pv-timeline { background: var(--pv-surface); border: 1px solid var(--pv-border-subtle); border-radius: var(--pv-r-md); padding: 16px 20px 12px; }
.pv-timeline-scroll { overflow-x: auto; }
.pv-timeline-scroll > svg { display: block; min-width: 640px; width: 100%; height: auto; }
.pv-legend svg { flex: none; }
.pv-legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12px; color: var(--pv-text-muted); margin-top: 8px; }
.pv-legend span { display: inline-flex; align-items: center; gap: 6px; }

/* ---------- Docs ---------- */
.pv-doc { max-width: 760px; font-size: 14px; line-height: 22px; color: var(--pv-text-secondary); }
.pv-doc h3 { font-size: 15px; line-height: 22px; color: var(--pv-text); font-weight: 600; margin: 20px 0 6px; }
.pv-doc p { margin: 0 0 10px; }
.pv-doc ol, .pv-doc ul { padding-left: 20px; margin: 0 0 10px; }
.pv-doc li { margin-bottom: 6px; }
.pv-doc strong { color: var(--pv-text); font-weight: 600; }

/* ---------- Streamlit widget alignment ---------- */
[data-testid="stWidgetLabel"] p { font-size: 12px !important; color: var(--pv-text-muted) !important; font-weight: 500 !important; }
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {
  background: var(--pv-surface) !important; border-color: var(--pv-border) !important;
  border-radius: var(--pv-r-sm) !important; min-height: 36px; font-size: 13px !important;
}
div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover { border-color: #33455E !important; }
div[data-baseweb="input"]:focus-within > div, div[data-baseweb="select"]:focus-within > div { border-color: var(--pv-accent) !important; box-shadow: 0 0 0 1px var(--pv-accent) !important; }
div[data-baseweb="popover"] ul { background: var(--pv-surface-raised) !important; }

/* Buttons */
.stButton > button, .stDownloadButton > button, [data-testid="stPageLink"] a.pv-btn,
.stLinkButton a {
  border-radius: var(--pv-r-sm) !important; min-height: 36px; padding: 0 14px !important;
  font-size: 13px !important; font-weight: 500 !important;
  background: var(--pv-surface-raised) !important; color: var(--pv-text) !important;
  border: 1px solid var(--pv-border) !important;
  transition: background var(--pv-dur) var(--pv-ease), border-color var(--pv-dur) var(--pv-ease);
}
.stButton > button:hover, .stDownloadButton > button:hover { background: var(--pv-surface-hover) !important; border-color: #33455E !important; }
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(0.5px); }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background: var(--pv-accent) !important; color: #04121D !important; border-color: var(--pv-accent) !important;
}
.stButton > button[kind="primary"]:hover { background: var(--pv-accent-hover) !important; }
.stButton > button:disabled { opacity: .5; cursor: not-allowed; }

/* Main-area page link used as CTA */
.main [data-testid="stPageLink"] a, [data-testid="stMain"] [data-testid="stPageLink"] a {
  display: inline-flex; border: 1px solid var(--pv-border); background: var(--pv-surface-raised);
  border-radius: var(--pv-r-sm); padding: 6px 12px; min-height: 36px; align-items: center;
  transition: background var(--pv-dur) var(--pv-ease);
}
[data-testid="stMain"] [data-testid="stPageLink"] a:hover { background: var(--pv-surface-hover); }
[data-testid="stMain"] [data-testid="stPageLink"] a span { color: var(--pv-text) !important; font-size: 13px; font-weight: 500; }
[data-testid="stMain"] [data-testid="stPageLink"] [data-testid="stIconMaterial"] { color: var(--pv-accent); font-size: 16px !important; }

/* Segmented control / pills */
[data-testid="stButtonGroup"] button {
  min-height: 32px; font-size: 13px !important; border-radius: var(--pv-r-sm) !important;
  background: var(--pv-surface) !important; border-color: var(--pv-border) !important; color: var(--pv-text-secondary) !important;
}
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[kind*="Active"] {
  background: var(--pv-accent-soft) !important; color: var(--pv-accent) !important; border-color: rgba(56,189,248,.4) !important;
}

/* Tabs */
div[data-baseweb="tab-list"] { gap: 4px !important; border-bottom: 1px solid var(--pv-border-subtle) !important; }
button[data-baseweb="tab"] { background: transparent !important; padding: 8px 12px !important; color: var(--pv-text-muted) !important; }
button[data-baseweb="tab"] p { font-size: 13px !important; font-weight: 500 !important; }
button[data-baseweb="tab"]:hover { color: var(--pv-text) !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--pv-text) !important; }
div[data-baseweb="tab-highlight"] { background: var(--pv-accent) !important; height: 2px !important; }
div[data-baseweb="tab-border"] { display: none; }

/* Dataframes */
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--pv-border-subtle) !important; border-radius: var(--pv-r-md) !important; overflow: hidden;
}

/* Expanders */
[data-testid="stExpander"] details { background: var(--pv-surface) !important; border: 1px solid var(--pv-border-subtle) !important; border-radius: var(--pv-r-md) !important; }
[data-testid="stExpander"] summary { padding: 12px 16px !important; }
[data-testid="stExpander"] summary p { font-size: 13px !important; font-weight: 500 !important; color: var(--pv-text-secondary) !important; }
[data-testid="stExpander"] summary:hover p { color: var(--pv-text) !important; }

/* Tooltip icon */
[data-testid="stTooltipIcon"] svg { color: var(--pv-text-disabled) !important; }

/* Map iframe */
[data-testid="stMetric"] { min-height: 108px; }
[data-testid="stMetricValue"] { font-size: 26px; letter-spacing: -0.02em; }
[data-testid="stDataFrame"] { width: 100% !important; }
iframe[title="st.iframe"], [data-testid="stIFrame"], .stCustomComponentV1 {
  border: 1px solid var(--pv-border-subtle) !important; border-radius: var(--pv-r-md); background: var(--pv-surface);
}

/* Vertical alignment helper for header CTA columns */
.pv-spacer { height: 8px; }

hr { border: 0 !important; border-top: 1px solid var(--pv-border-subtle) !important; margin: 24px 0 !important; }

/* Toasts */
[data-testid="stToast"] { background: var(--pv-surface-raised) !important; border: 1px solid var(--pv-border) !important; border-radius: var(--pv-r-md) !important; }

/* Footer */
.pv-footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--pv-border-subtle);
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; font-size: 12px; line-height: 18px; color: var(--pv-text-disabled); }
</style>
""".replace("%VARS%", _css_vars()).replace("%FONT_SANS%", FONT_SANS).replace("%FONT_MONO%", FONT_MONO).replace("%FONT_CSS%", configured_url("PRAVAH_FONT_CSS_URL")).replace("%ICONS_CSS%", configured_url("PRAVAH_ICONS_CSS_URL"))


def apply_theme() -> None:
    """Inject the PRAVAH stylesheet into the current Streamlit run."""
    import streamlit as st

    st.markdown(DESIGN_SYSTEM_CSS, unsafe_allow_html=True)
