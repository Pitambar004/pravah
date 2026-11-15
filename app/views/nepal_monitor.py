"""Route: Nepal live-source monitoring."""

from app.context import get_context
from app.sections.nepal_monitor import render_nepal_monitor

render_nepal_monitor(get_context())
