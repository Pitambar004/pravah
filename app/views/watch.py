"""Route: Live Watch 24/7."""
from app.context import get_context
from app.sections.watch import render_watch_section

render_watch_section(get_context())
