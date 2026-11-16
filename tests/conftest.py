import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent

# Pozwala importować main/load_prepare_data z katalogu głównego oraz generator z tests/.
for p in (REPO_ROOT, TESTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from synthetic_urls import generate_dataset  # noqa: E402

SEED = 42


@pytest.fixture(scope="session")
def synthetic_df():
    """4000 syntetycznych URL-i (50% phishing), deterministycznie dla SEED."""
    return generate_dataset(n=4000, seed=SEED)
