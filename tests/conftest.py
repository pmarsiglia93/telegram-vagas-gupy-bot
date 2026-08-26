import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

import pytest  # noqa: E402

from jobmatch.domain.profile import load_profile  # noqa: E402


@pytest.fixture(scope="session")
def profile():
    return load_profile(os.path.join(RAIZ, "profile.yaml"))
