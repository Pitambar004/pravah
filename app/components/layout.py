"""Layout and typography primitives for the PRAVAH design system.

Every helper returns or renders HTML built from classes defined in
``app/theme.py``; no page should hardcode colours or spacing.
"""
from __future__ import annotations

from collections.abc import Iterable
from html import escape

import streamlit as st

BADGE_VARIANTS = {"neutral", "accent", "success", "warning", "danger", "error", "scenario", "outline"}
CALLOUT_ICONS = {
    "info": "info",
    "warning": "warning",
    "danger": "error",
    "error": "error",
    "success": "check_circle",
    "scenario": "science",
}


def html(markup: str) -> None:
    """Render trusted component markup (single line, so Markdown never re-parses it)."""
    st.markdown(f'<div class="pv-root">{_squash(markup)}</div>', unsafe_allow_html=True)


def _squash(markup: str) -> str:
    return "".join(line.strip() for line in markup.splitlines())


def icon(name: str, size: int = 16, color: str | None = None) -> str:
    """Material Symbols Rounded icon (the single icon set used app-wide)."""
    style = f"font-size:{size}px;width:{size}px;height:{size}px"
    if color:
        style += f";color:{color}"
    return f'<span class="pv-icon" style="{style}" aria-hidden="true">{escape(name)}</span>'


def render_badge(text: str, variant: str = "neutral", dot: bool = False, title: str | None = None) -> str:
    """Return an inline status badge."""
    variant = variant if variant in BADGE_VARIANTS else "neutral"
    dot_html = '<span class="dot"></span>' if dot else ""
    title_attr = f' title="{escape(title)}"' if title else ""
    return f'<span class="pv-badge pv-badge-{variant}"{title_attr}>{dot_html}{escape(text)}</span>'


def render_topbar(crumbs: Iterable[str], meta_badges: Iterable[str] = ()) -> None:
    """Global header strip: breadcrumb + region/data-status badges."""
    parts = list(crumbs)
    crumb_html = '<span class="sep">/</span>'.join(
        f'<span class="{"cur" if i == len(parts) - 1 else ""}">{escape(c)}</span>' for i, c in enumerate(parts)
    )
    html(
        f'<div class="pv-topbar" role="navigation" aria-label="Breadcrumb">'
        f'<div class="pv-crumbs">{crumb_html}</div>'
        f'<div class="pv-topbar-meta">{"".join(meta_badges)}</div></div>'
    )


def render_page_header(
    title: str,
    subtitle: str | None = None,
    badge: str | None = None,
    badge_variant: str = "neutral",
) -> None:
    """Standard page title block used by every page."""
    sub = f'<p class="pv-page-subtitle">{escape(subtitle)}</p>' if subtitle else ""
    badge_html = render_badge(badge, badge_variant) if badge else ""
    html(
        f'<div class="pv-page-header"><div><h1 class="pv-page-title">{escape(title)}</h1>{sub}</div>'
        f"<div>{badge_html}</div></div>"
    )


def render_section_header(title: str, description: str | None = None, trailing: str = "") -> None:
    """Section heading within a page."""
    desc = f'<p class="pv-section-desc">{escape(description)}</p>' if description else ""
    html(
        f'<div class="pv-section-header"><div><h2 class="pv-section-title">{escape(title)}</h2>{desc}</div>'
        f"<div>{trailing}</div></div>"
    )


def render_callout(message: str, title: str | None = None, variant: str = "info", as_html: bool = False) -> str | None:
    """Inline note / scientific caveat. ``message`` may contain trusted inline HTML."""
    ico = icon(CALLOUT_ICONS.get(variant, "info"), 18)
    title_html = f"<strong>{escape(title)}.</strong> " if title else ""
    markup = f'<div class="pv-callout pv-callout-{variant}" role="note">{ico}<div>{title_html}{message}</div></div>'
    if as_html:
        return markup
    html(markup)
    return None


def card(title: str | None, body: str, trailing: str = "", extra_class: str = "") -> str:
    """Return a neutral card; ``body`` is trusted HTML."""
    head = (
        f'<div class="pv-card-head"><span class="pv-card-title">{escape(title)}</span>{trailing}</div>'
        if title
        else ""
    )
    return f'<div class="pv-card {extra_class}">{head}<div class="pv-card-body">{body}</div></div>'


def kv_list(rows: Iterable[tuple[str, str]], mono_keys: Iterable[str] = ()) -> str:
    """Return a compact key/value grid. Values are escaped."""
    mono = set(mono_keys)
    cells = []
    for k, v in rows:
        cls = "v pv-mono" if k in mono else "v"
        cells.append(f'<div class="k">{escape(k)}</div><div class="{cls}">{escape(str(v))}</div>')
    return f'<div class="pv-kv">{"".join(cells)}</div>'


def status_list(rows: Iterable[tuple[str, str | None, str, str]]) -> str:
    """Return an artifact/status list.

    Each row: (name, detail, state, badge_html) where state is ``ok``, ``wait`` or ``warn``.
    """
    glyph = {"ok": "check", "wait": "schedule", "warn": "priority_high"}
    items = []
    for name, detail, state, badge_html in rows:
        detail_html = f"<small>{escape(detail)}</small>" if detail else ""
        items.append(
            f'<div class="pv-status-row"><div class="pv-status-name">'
            f'<span class="pv-status-ico {state}">{icon(glyph.get(state, "schedule"), 14)}</span>'
            f"<div>{escape(name)}{detail_html}</div></div>{badge_html}</div>"
        )
    return f'<div class="pv-status-list">{"".join(items)}</div>'


def render_footer() -> None:
    html(
        '<div class="pv-footer"><div>PRAVAH <span>· Team 12</span></div>'
        "<div>Research prototype · not an operational warning system</div></div>"
    )
