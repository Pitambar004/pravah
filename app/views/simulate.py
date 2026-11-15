"""Route: Rasuwa replay and exploratory downstream exposure."""

from app.context import get_context
from app.sections.simulate import render_simulate

render_simulate(get_context())
