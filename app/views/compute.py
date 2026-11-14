"""Route: compute."""
from app.context import get_context
from app.sections.gpu_benchmarks import render_compute_section

render_compute_section(get_context())
