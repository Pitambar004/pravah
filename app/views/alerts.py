"""Route: alerts."""
from app.context import get_context
from app.sections.alerts import render_alerts_section

render_alerts_section(get_context())
