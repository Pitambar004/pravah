"""Route: glacier/lake inventory and surface-change watch."""

from app.context import get_context
from app.sections.lakes import render_lakes

render_lakes(get_context())
