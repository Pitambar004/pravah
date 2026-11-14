"""Route: methodology."""
from app.context import get_context
from app.sections.methodology import render_methodology_section

render_methodology_section(get_context())
