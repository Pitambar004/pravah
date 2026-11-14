"""Route: exposure."""
from app.context import get_context
from app.sections.exposure import render_exposure_section

render_exposure_section(get_context())
