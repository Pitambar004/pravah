"""Route: training."""
from app.context import get_context
from app.sections.training import render_training_section

render_training_section(get_context())
