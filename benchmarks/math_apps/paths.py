from pathlib import Path
import os
from evomem.common import DATA
ASSETS = Path(os.environ.get("EVOMEMORY_BENCHMARK_DATA", str(DATA / "math_apps"))).expanduser()
