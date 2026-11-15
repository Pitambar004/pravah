"""Route: validation."""
from app.context import get_context
from app.sections.validation import render_validation_section

render_validation_section(get_context())
