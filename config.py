import platform
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
REPORT_FIGS = BASE_DIR / "reports" / "figures"

DEVICE = "cuda" if platform.system() == "Windows" else "cpu"

for folder in [DATA_DIR, EXPERIMENTS_DIR, REPORT_FIGS]:
    folder.mkdir(parents=True, exist_ok=True)