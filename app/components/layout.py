"""Reusable layout and visual hierarchy primitives for PRAVAH Design System."""

import streamlit as st


def render_page_header(
    title: str,
    subtitle: str,
    badge: str | None = None,
    badge_variant: str = "accent",
):
    """Render a standardized page header with consistent typography and alignment."""
    badge_html = ""
    if badge:
        badge_html = f'<span class="pv-badge pv-badge-{badge_variant}">{badge}</span>'

    html = f"""
    <div class="pv-page-header">
        <div>
            <h1 class="pv-page-title">{title}</h1>
            <p class="pv-page-subtitle">{subtitle}</p>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.25rem;">
            {badge_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_section_header(title: str, description: str | None = None):
    """Render a consistent section header within a page."""
    desc_html = f'<p style="font-size: 0.825rem; color: #94A3B8; margin: 0.2rem 0 0.85rem 0;">{description}</p>' if description else ""
    html = f"""
    <div style="margin: 1.25rem 0 0.6rem 0;">
        <h3 style="font-size: 1.05rem; font-weight: 600; color: #F1F5F9; margin: 0; letter-spacing: -0.01em;">{title}</h3>
        {desc_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_callout(
    message: str,
    title: str | None = None,
    variant: str = "info",  # info, warning, error, success
):
    """Render a structured alert or scientific callout."""
    title_html = f"<strong>{title}:</strong> " if title else ""
    html = f"""
    <div class="pv-callout pv-callout-{variant}">
        {title_html}{message}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_badge(text: str, variant: str = "neutral") -> str:
    """Return an inline HTML badge string."""
    return f'<span class="pv-badge pv-badge-{variant}">{text}</span>'
