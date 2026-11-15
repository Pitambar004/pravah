"""Route: river outlook."""
from app.context import get_context
from app.sections.outlook import render_outlook_section

render_outlook_section(get_context())
