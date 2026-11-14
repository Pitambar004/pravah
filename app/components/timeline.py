"""Evidence timeline (SVG) and evidence cards for historical replay.

Status vocabulary comes from ``pravah.replay.eligibility.classify_asset``:
``known_before_cutoff``, ``retrospective_reveal``, ``excluded_unknown_availability``.
Markers differ by shape as well as colour so status is never colour-only.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from html import escape
from typing import Any

from app.components.layout import icon, render_badge
from app.theme import TOKENS
from pravah.replay.eligibility import parse_time

NPT = timezone(timedelta(hours=5, minutes=45), "NPT")
EVENT_DAY_START = datetime(2026, 8, 26, 0, 0, tzinfo=NPT)
EVENT_DAY_END = EVENT_DAY_START + timedelta(days=1)

STATUS_META = {
    "known_before_cutoff": {
        "label": "Available at cutoff", "variant": "success", "color": TOKENS["success"], "icon": "check_circle",
    },
    "retrospective_reveal": {
        "label": "Later evidence", "variant": "warning", "color": TOKENS["warning"], "icon": "update",
    },
    "excluded_unknown_availability": {
        "label": "Excluded", "variant": "danger", "color": TOKENS["danger"], "icon": "block",
    },
}


def fmt_time(value: str | datetime | None, tz: timezone = NPT) -> str:
    try:
        dt = parse_time(value)
    except (ValueError, TypeError):
        return str(value)
    if dt is None:
        return "Unknown"
    local = dt.astimezone(tz)
    return local.strftime("%d %b %Y, %H:%M ") + (tz.tzname(None) or "")


def _lane(asset: dict[str, Any]) -> str:
    platform = f"{asset.get('platform', '')} {asset.get('product_type', '')}".lower()
    if "sentinel-2" in platform or "msi" in platform:
        return "Optical · Sentinel-2"
    if "sentinel-1" in platform or "sar" in platform:
        return "SAR · Sentinel-1"
    return "Reference & vector"


LANES = ["Optical · Sentinel-2", "SAR · Sentinel-1", "Reference & vector"]


def build_timeline_svg(
    assets: list[dict[str, Any]],
    cutoff: str | datetime,
    strict: bool,
) -> str:
    """Return the timeline as an inline SVG string.

    In strict mode, anything acquired or made available after the cutoff is not
    drawn at all; the region after the cutoff is hatched and labelled as hidden.
    """
    c = parse_time(cutoff)
    rows: list[tuple[dict[str, Any], datetime | None, datetime | None]] = []
    for a in assets:
        try:
            acq = parse_time(a.get("acquired_at_utc"))
            av = parse_time(a.get("available_at_utc"))
        except (ValueError, TypeError):
            continue
        status = a.get("_status")
        if strict:
            if status == "retrospective_reveal":
                continue
            if status == "excluded_unknown_availability" and (acq is None or acq > c):
                continue
        rows.append((a, acq, av))

    # Domain: focus on the event window; earlier items are summarised at the left edge.
    focus_start = EVENT_DAY_START - timedelta(days=12)
    times = [t for _, acq, av in rows for t in (acq, av) if t is not None] + [c, EVENT_DAY_END]
    start = max(min(times), focus_start) - timedelta(hours=12)
    end = max(times) + timedelta(hours=18)
    if strict:
        end = max(end, c + timedelta(days=3))
    span = (end - start).total_seconds()

    width, left, right, top, lane_h = 1000, 150, 24, 56, 44
    height = top + lane_h * len(LANES) + 34
    plot_w = width - left - right

    def x(t: datetime) -> float:
        return left + max(0.0, min(1.0, (t - start).total_seconds() / span)) * plot_w

    t = TOKENS
    parts = [
        (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Evidence timeline" '
         'xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif">'),
        ('<defs><pattern id="pvHatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
         f'<line x1="0" y1="0" x2="0" y2="8" stroke="{t["border"]}" stroke-width="2"/></pattern></defs>'),
    ]
    # Event day band
    ex0, ex1 = x(EVENT_DAY_START), x(EVENT_DAY_END)
    parts.append(f'<rect x="{ex0:.1f}" y="{top - 8}" width="{max(ex1 - ex0, 2):.1f}" height="{lane_h * len(LANES) + 8}" '
                 f'fill="{t["danger"]}" fill-opacity="0.06"/>')
    parts.append(f'<text x="{ex0 - 6:.1f}" y="{top - 14}" text-anchor="end" font-size="11" fill="{t["danger"]}">Event day · 26 Aug →</text>')
    # Hidden region in strict mode
    cx = x(c)
    if strict and cx < left + plot_w:
        parts.append(f'<rect x="{cx:.1f}" y="{top - 8}" width="{left + plot_w - cx:.1f}" height="{lane_h * len(LANES) + 8}" '
                     f'fill="url(#pvHatch)" opacity="0.55"/>')
        parts.append(f'<text x="{left + plot_w - 8:.1f}" y="{top + lane_h * len(LANES) - 10:.1f}" text-anchor="end" '
                     f'font-size="11.5" fill="{t["text_muted"]}">Hidden in strict replay</text>')
    # Lanes
    for i, lane in enumerate(LANES):
        y = top + i * lane_h + lane_h / 2
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="{t["border_subtle"]}"/>')
        parts.append(f'<text x="0" y="{y + 4}" font-size="12" fill="{t["text_secondary"]}">{escape(lane)}</text>')
    # Day ticks
    day = datetime(start.year, start.month, start.day, tzinfo=NPT) + timedelta(days=1)
    n_days = int(span // 86400) + 1
    step = 1 if n_days <= 12 else 2 if n_days <= 24 else 7
    axis_y = top + lane_h * len(LANES) + 6
    k = 0
    while day < end:
        if k % step == 0:
            dx = x(day)
            parts.append(f'<line x1="{dx:.1f}" y1="{top - 4}" x2="{dx:.1f}" y2="{axis_y}" stroke="{t["border_subtle"]}" stroke-dasharray="2 4"/>')
            parts.append(f'<text x="{dx:.1f}" y="{axis_y + 16}" text-anchor="middle" font-size="11" fill="{t["text_muted"]}">{day.strftime("%d %b")}</text>')
        day += timedelta(days=1)
        k += 1
    # Earlier-than-window summary
    earlier = [r for r in rows if (r[2] or r[1]) is not None and (r[2] or r[1]) < start]
    # Markers
    for a, acq, av in rows:
        status = a.get("_status", "")
        meta = STATUS_META.get(status, STATUS_META["excluded_unknown_availability"])
        lane_i = LANES.index(_lane(a))
        y = top + lane_i * lane_h + lane_h / 2
        when = av or acq
        if when is None:
            continue
        mx = x(when)
        title = (f"{a.get('asset_id', '')}\n{meta['label']}\nAcquired: {fmt_time(acq)}\n"
                 f"Available: {fmt_time(av) if av else 'Unknown'}")
        parts.append(f"<g><title>{escape(title)}</title>")
        if acq and av and x(acq) < mx - 1:
            parts.append(f'<line x1="{x(acq):.1f}" y1="{y}" x2="{mx:.1f}" y2="{y}" stroke="{meta["color"]}" stroke-width="2" stroke-opacity="0.45"/>')
            parts.append(f'<circle cx="{x(acq):.1f}" cy="{y}" r="2.5" fill="{meta["color"]}" fill-opacity="0.6"/>')
        if status == "known_before_cutoff":
            parts.append(f'<circle cx="{mx:.1f}" cy="{y}" r="6" fill="{meta["color"]}" stroke="{t["surface"]}" stroke-width="2"/>')
        elif status == "retrospective_reveal":
            parts.append(f'<rect x="{mx - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" transform="rotate(45 {mx:.1f} {y})" '
                         f'fill="{t["surface"]}" stroke="{meta["color"]}" stroke-width="2"/>')
        else:
            parts.append(f'<path d="M{mx - 5:.1f} {y - 5} L{mx + 5:.1f} {y + 5} M{mx + 5:.1f} {y - 5} L{mx - 5:.1f} {y + 5}" '
                         f'stroke="{meta["color"]}" stroke-width="2.2" stroke-linecap="round"/>')
        parts.append("</g>")
    if earlier:
        parts.append(f'<text x="{left + 4}" y="{top - 14}" font-size="11" fill="{t["text_muted"]}">← {len(earlier)} earlier item(s) pinned to edge</text>')
    # Cutoff line
    parts.append(f'<line x1="{cx:.1f}" y1="{top - 36}" x2="{cx:.1f}" y2="{axis_y}" stroke="{t["accent"]}" stroke-width="2"/>')
    label = "Cutoff " + c.astimezone(NPT).strftime("%d %b %H:%M NPT")
    anchor = "end" if cx > left + plot_w * 0.8 else "start"
    off = -6 if anchor == "end" else 6
    parts.append(f'<text x="{cx + off:.1f}" y="{top - 40}" text-anchor="{anchor}" font-size="11.5" font-weight="600" fill="{t["accent"]}">{escape(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def timeline_legend(strict: bool) -> str:
    t = TOKENS
    items = [
        f'<span><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="{t["success"]}"/></svg>Available at cutoff</span>',
        f'<span><svg width="12" height="12"><path d="M2 2 L10 10 M10 2 L2 10" stroke="{t["danger"]}" stroke-width="2"/></svg>Excluded (availability unverified)</span>',
    ]
    if not strict:
        items.insert(1, f'<span><svg width="12" height="12"><rect x="3" y="3" width="6" height="6" transform="rotate(45 6 6)" fill="none" stroke="{t["warning"]}" stroke-width="1.6"/></svg>Later evidence</span>')
    items.append(f'<span><svg width="14" height="12"><line x1="7" y1="0" x2="7" y2="12" stroke="{t["accent"]}" stroke-width="2"/></svg>Selected cutoff</span>')
    items.append('<span>Line: acquisition → verified availability</span>')
    return f'<div class="pv-legend">{"".join(items)}</div>'


def evidence_card(asset: dict[str, Any]) -> str:
    """Return one evidence card. All asset strings are escaped."""
    status = asset.get("_status", "")
    meta = STATUS_META.get(status, STATUS_META["excluded_unknown_availability"])
    soft = {"success": "var(--pv-success-soft)", "warning": "var(--pv-warning-soft)", "danger": "var(--pv-danger-soft)"}[meta["variant"]]
    cloud = asset.get("cloud_cover_percent")
    meta_items = [
        ("Platform", asset.get("platform") or "—"),
        ("Product", asset.get("product_type") or "—"),
        ("Acquired", fmt_time(asset.get("acquired_at_utc"))),
        ("Available", fmt_time(asset.get("available_at_utc")) if asset.get("available_at_utc") else "Unverified"),
    ]
    if cloud is not None:
        meta_items.append(("Cloud", f"{cloud:.1f}%"))
    if asset.get("license"):
        meta_items.append(("License (as recorded)", asset["license"]))
    meta_html = "".join(f"<span>{escape(k)} <b>{escape(str(v))}</b></span>" for k, v in meta_items)
    return (
        f'<div class="pv-evidence"><div class="pv-evidence-mark" style="background:{soft};color:{meta["color"]}">'
        f'{icon(meta["icon"], 16)}</div><div>'
        f'<div class="pv-evidence-title">{escape(asset.get("description") or asset.get("asset_id") or "Unnamed asset")}</div>'
        f'<div class="pv-evidence-desc">{escape(asset.get("provider") or "")}</div>'
        f'<div class="pv-evidence-meta">{meta_html}</div>'
        f'<div class="pv-evidence-id">{escape(asset.get("asset_id") or "")}</div></div>'
        f'{render_badge(meta["label"], meta["variant"])}</div>'
    )

