"""Route: gallery."""
from app.context import get_context
from app.sections.gallery import render_gallery_section

render_gallery_section(get_context())
