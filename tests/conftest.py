import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

import pytest  # noqa: E402

from dataclasses import replace  # noqa: E402

from jobmatch.domain.profile import Focus, load_profile  # noqa: E402


@pytest.fixture(scope="session")
def profile():
    return load_profile(os.path.join(RAIZ, "profile.yaml"))


@pytest.fixture(scope="session")
def profile_amplo(profile):
    """O perfil real sem a seção `focus` — as regras-base, sem as escolhas do candidato.

    Localização, senioridade neutra e gap não-eliminatório são garantias do
    modo amplo; testá-las contra o perfil com foco misturaria as duas coisas.
    """
    return replace(profile, focus=Focus())
