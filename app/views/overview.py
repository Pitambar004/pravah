"""Route: overview."""
from app.context import get_context
from app.sections.overview import render_overview_section

render_overview_section(get_context())
