"""Keep dashboard dependencies in sync across Streamlit source reruns."""

import importlib
import sys
from pathlib import Path

_VERSIONS = globals().get("_VERSIONS", {})
_MODULES = [
    "pravah.replay.scenario", "pravah.replay.lakes", "pravah.replay.cascade",
    "pravah.replay.imagery", "pravah.replay.image_pairs", "pravah.replay.live", "pravah.replay.presentation",
    "app.api_client", "app.theme", "app.components.layout", "app.components.empty_state",
    "app.components.metric_card", "app.context", "app.components.map_view", "app.components.compare_map", "app.components.notifications",
    "app.nepal_data", "app.image_data", "app.sections.operations_data", "app.sections.lakes",
    "app.sections.nepal_monitor", "app.sections.judge_demo", "app.sections.simulate", "app.sections.validation",
]


def synchronize_modules():
    """Reload owned dependencies together when files changed or are untracked.

    Streamlit reruns Home.py but can retain imported modules from an older
    build. Reload leaves data caches and third-party modules under their own
    lifecycle; dependent page functions are rebound in dependency order.
    """
    root = Path(__file__).resolve().parents[1]
    versions = {}
    for name in _MODULES:
        base = root / "src" if name.startswith("pravah.") else root
        path = base / (name.replace(".", "/") + ".py")
        if path.exists():
            stat = path.stat()
            versions[name] = (stat.st_mtime_ns, stat.st_size)
    if versions == _VERSIONS:
        return
    importlib.invalidate_caches()
    for name in _MODULES:
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve().is_relative_to(root):
            importlib.reload(module)
    _VERSIONS.clear()
    _VERSIONS.update(versions)
