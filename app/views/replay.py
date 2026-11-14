"""Route: replay."""
from app.context import get_context
from app.sections.replay import render_replay_section

render_replay_section(get_context())
