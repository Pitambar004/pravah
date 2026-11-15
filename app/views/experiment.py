"""Route: Nepal research experiment."""

from app.context import get_context
from app.sections.experiment import render_experiment

render_experiment(get_context())
