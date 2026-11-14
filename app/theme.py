"""PRAVAH Unified Design System — Design Tokens & Global CSS.

Enterprise-grade SaaS aesthetic inspired by Linear, Vercel, and Stripe Dashboard,
tailored for scientific geospatial intelligence.
"""

DESIGN_SYSTEM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-app: #0B1120;
    --bg-surface: #111827;
    --bg-card: #151F30;
    --bg-card-hover: #1A263C;
    --border-color: #263244;
    --border-hover: #334155;
    
    --text-primary: #F1F5F9;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    
    --accent: #38BDF8;
    --accent-subtle: rgba(56, 189, 248, 0.12);
    --success: #34D399;
    --success-subtle: rgba(52, 211, 153, 0.12);
    --warning: #FBBF24;
    --warning-subtle: rgba(251, 191, 36, 0.12);
    --error: #FB7185;
    --error-subtle: rgba(251, 113, 133, 0.12);
    
    --radius-sm: 4px;
    --radius-md: 6px;
    --radius-lg: 8px;
    --radius-full: 9999px;
}

/* Base resets & typography */
html, body, [class*="css"], .stApp {
    background-color: var(--bg-app) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

code, pre, .mono-text {
    font-family: 'JetBrains Mono', ui-monospace, Menlo, Consolas, monospace !important;
}

/* App layout constraint */
.block-container {
    max-width: 1280px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--border-color) !important;
    width: 280px !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem !important;
    padding-left: 1.25rem !important;
    padding-right: 1.25rem !important;
}

/* Sidebar Navigation Items */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    gap: 4px !important;
}

section[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius-md) !important;
    padding: 0.5rem 0.75rem !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background-color: rgba(255, 255, 255, 0.04) !important;
    color: var(--text-primary) !important;
}

section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio div[aria-checked="true"] + label {
    background-color: var(--bg-card) !important;
    border-color: var(--border-color) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
}

/* Standard Card Primitive */
.pv-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    position: relative;
}

.pv-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-color);
}

.pv-card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.01em;
}

/* Page Header Component */
.pv-page-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding-bottom: 1.25rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border-color);
}

.pv-page-title {
    font-size: 1.75rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin: 0 0 0.35rem 0;
}

.pv-page-subtitle {
    font-size: 0.875rem;
    color: var(--text-secondary);
    line-height: 1.5;
    margin: 0;
    max-width: 800px;
}

/* Metric Cards */
.pv-metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 1rem 1.25rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 104px;
    transition: border-color 0.15s ease;
}

.pv-metric-card:hover {
    border-color: var(--border-hover);
}

.pv-metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 0.35rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.pv-metric-value-row {
    display: flex;
    align-items: baseline;
    gap: 0.35rem;
}

.pv-metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    line-height: 1.1;
}

.pv-metric-unit {
    font-size: 0.825rem;
    font-weight: 500;
    color: var(--text-secondary);
}

.pv-metric-subtext {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 0.35rem;
}

/* Status Badges */
.pv-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.2rem 0.55rem;
    border-radius: var(--radius-sm);
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    line-height: 1;
}

.pv-badge-accent {
    background: var(--accent-subtle);
    color: var(--accent);
    border: 1px solid rgba(56, 189, 248, 0.25);
}

.pv-badge-success {
    background: var(--success-subtle);
    color: var(--success);
    border: 1px solid rgba(52, 211, 153, 0.25);
}

.pv-badge-warning {
    background: var(--warning-subtle);
    color: var(--warning);
    border: 1px solid rgba(251, 191, 36, 0.25);
}

.pv-badge-error {
    background: var(--error-subtle);
    color: var(--error);
    border: 1px solid rgba(251, 113, 133, 0.25);
}

.pv-badge-neutral {
    background: rgba(148, 163, 184, 0.08);
    color: var(--text-secondary);
    border: 1px solid var(--border-color);
}

/* Callout / Information Banner */
.pv-callout {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-md);
    padding: 0.75rem 1rem;
    margin-bottom: 1.25rem;
    font-size: 0.825rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

.pv-callout-warning {
    border-left-color: var(--warning);
    background: rgba(251, 191, 36, 0.04);
}

.pv-callout-warning strong {
    color: var(--warning);
}

.pv-callout-info {
    border-left-color: var(--accent);
}

.pv-callout-info strong {
    color: var(--accent);
}

/* Empty & Awaiting State */
.pv-empty-state {
    background: var(--bg-card);
    border: 1px dashed var(--border-color);
    border-radius: var(--radius-lg);
    padding: 2.25rem 2rem;
    text-align: center;
    margin: 1rem 0;
}

.pv-empty-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0.5rem 0 0.35rem 0;
}

.pv-empty-desc {
    font-size: 0.825rem;
    color: var(--text-secondary);
    max-width: 520px;
    margin: 0 auto 0.75rem auto;
    line-height: 1.5;
}

.pv-empty-code {
    display: inline-block;
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    padding: 0.35rem 0.65rem;
    font-size: 0.775rem;
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.5rem;
}

/* Native Streamlit Element Overrides */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.25rem !important;
}

div[data-testid="stMetric"] label {
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: var(--text-muted) !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
}

/* Tabs */
div[data-baseweb="tab-list"] {
    gap: 0.5rem !important;
    border-bottom: 1px solid var(--border-color) !important;
    padding-bottom: 0 !important;
}

button[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 0.85rem !important;
    transition: all 0.15s ease !important;
}

button[data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom-color: var(--accent) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
}

/* Tables / DataFrames */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    background: var(--bg-card) !important;
}

/* Inputs & Form Controls */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: var(--bg-card) !important;
    border-color: var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-size: 0.85rem !important;
}

div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

/* Horizontal Dividers */
hr {
    border: none !important;
    border-top: 1px solid var(--border-color) !important;
    margin: 1.5rem 0 !important;
}

/* Expanders */
details[data-testid="stExpander"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-lg) !important;
}

details[data-testid="stExpander"] summary {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
}

details[data-testid="stExpander"] summary:hover {
    color: var(--text-primary) !important;
}

/* Map frame styling */
.pv-map-container {
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    overflow: hidden;
    background: var(--bg-card);
}
</style>
"""


def apply_theme():
    """Inject PRAVAH Design System styles into active Streamlit session."""
    import streamlit as st
    st.markdown(DESIGN_SYSTEM_CSS, unsafe_allow_html=True)
