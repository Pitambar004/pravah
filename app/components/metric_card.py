"""KPI metric card used on every page that shows numbers."""
from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from app.components.layout import html, icon, render_badge

NOT_AVAILABLE = "Not available"


def format_value(value: Any) -> str | None:
    """Format numbers with tabular-friendly separators; ``None`` means unknown."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.1f}" if abs(value) >= 10 else f"{value:,.2f}"
    return str(value)


def render_metric_card(
    label: str,
    value: Any,
    unit: str | None = None,
    subtext: str | None = None,
    badge: str | None = None,
    badge_variant: str = "neutral",
    icon_name: str | None = None,
    empty_label: str = NOT_AVAILABLE,
    help_text: str | None = None,
    as_html: bool = False,
) -> str | None:
    """Render a metric card.

    ``value=None`` renders ``empty_label`` (never zero) so unknown states are explicit.
    ``subtext`` should name the source or timestamp of the value.
    """
    shown = format_value(value)
    ico = icon(icon_name, 18) if icon_name else ""
    title_attr = f' title="{escape(help_text)}"' if help_text else ""
    badge_html = render_badge(badge, badge_variant) if badge else ""
    if shown is None:
        value_html = f'<span class="pv-metric-value is-empty">{escape(empty_label)}</span>'
    else:
        unit_html = f'<span class="pv-metric-unit">{escape(unit)}</span>' if unit else ""
        value_html = f'<span class="pv-metric-value">{escape(shown)}</span>{unit_html}'
    foot = f'<div class="pv-metric-foot">{escape(subtext)}</div>' if subtext else ""
    markup = (
        f'<div class="pv-metric pv-metric-card"{title_attr}>'
        f'<div class="pv-metric-top"><span class="pv-metric-label">{ico}{escape(label)}</span>{badge_html}</div>'
        f'<div class="pv-metric-value-row">{value_html}</div>{foot}</div>'
    )
    if as_html:
        return markup
    html(markup)
    return None


def render_metric_row(cards: list[dict[str, Any]]) -> None:
    """Render metric cards in equal columns (stack automatically on narrow screens)."""
    cols = st.columns(len(cards))
    for col, spec in zip(cols, cards, strict=True):
        with col:
            render_metric_card(**spec)
