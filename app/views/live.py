"""Route: live."""
from app.context import get_context
from app.sections.live import render_live_section

render_live_section(get_context())
