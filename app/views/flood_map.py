"""Route: flood_map."""
from app.context import get_context
from app.sections.flood_map import render_flood_map_section

render_flood_map_section(get_context())
