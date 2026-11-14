"""Standard empty / awaiting state."""
from __future__ import annotations

from html import escape

from app.components.layout import html, icon, render_badge


def render_empty_state(
    title: str = "Awaiting artifacts",
    description: str = "No computed outputs are present in the selected run folder.",
    action_command: str | None = None,
    expected_files: list[str] | None = None,
    badge: str | None = "Awaiting processing",
    icon_name: str = "hourglass_empty",
    compact: bool = False,
    as_html: bool = False,
) -> str | None:
    """Explain what is missing and how it will be produced. Never shows placeholder data."""
    badge_html = render_badge(badge, "neutral") if badge else ""
    cmd = f'<div class="pv-empty-cmd">$ {escape(action_command)}</div>' if action_command else ""
    files = ""
    if expected_files:
        files = '<div class="pv-empty-files">' + "".join(
            f"<code>{escape(f)}</code>" for f in expected_files
        ) + "</div>"
    markup = (
        f'<div class="pv-empty{" compact" if compact else ""}" role="status">'
        f'<div class="pv-empty-icon">{icon(icon_name, 24)}</div>'
        f"{badge_html}"
        f'<div class="pv-empty-title">{escape(title)}</div>'
        f'<div class="pv-empty-desc">{escape(description)}</div>{cmd}{files}</div>'
    )
    if as_html:
        return markup
    html(markup)
    return None
