"""Observabilidade dos coletores: erro HTTP precisa aparecer no log, não só no contador."""

from unittest.mock import MagicMock

from jobmatch.collectors.base import BaseCollector


class ColetorFake(BaseCollector):
    name = "fake"


def _resp(status, texto="", headers=None):
    r = MagicMock()
    r.status_code = status
    r.text = texto
    r.headers = headers or {}
    return r


def test_loga_bloqueio_cloudflare(profile, capsys):
    c = ColetorFake(profile)
    c._log_http_error("termo-x", _resp(403, "Checking your browser before accessing..."))
    saida = capsys.readouterr().out
    assert "fake" in saida and "403" in saida
    assert "Cloudflare" in saida or "anti-bot" in saida


def test_loga_403_generico(profile, capsys):
    c = ColetorFake(profile)
    c._log_http_error("termo-x", _resp(403, "Access Denied"))
    assert "bloqueio por IP" in capsys.readouterr().out


def test_loga_429(profile, capsys):
    c = ColetorFake(profile)
    c._log_http_error("termo-x", _resp(429))
    assert "rate limit" in capsys.readouterr().out


def test_nao_inunda_o_log(profile, capsys):
    """Uma fonte bloqueada falha do mesmo jeito em toda página — loga só as N primeiras."""
    c = ColetorFake(profile)
    for i in range(10):
        c._log_http_error(f"termo-{i}", _resp(403))
    linhas = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert len(linhas) == c.MAX_AVISOS_HTTP


# --------------------------------------------------------------------------
# LinkedIn: o endpoint guest ignora f_WT, então a busca "REMOTO" não prova nada
# --------------------------------------------------------------------------

def _card_linkedin(titulo, local):
    from bs4 import BeautifulSoup

    html = f"""
    <div class="base-card">
      <a class="base-card__full-link" href="https://br.linkedin.com/jobs/view/dev-4471571571?trk=x"></a>
      <h3 class="base-search-card__title">{titulo}</h3>
      <h4 class="base-search-card__subtitle">Empresa X</h4>
      <span class="job-search-card__location">{local}</span>
      <time datetime="2026-09-29"></time>
    </div>"""
    return BeautifulSoup(html, "html.parser").find("div", class_="base-card")


def test_linkedin_nao_marca_remoto_pela_busca(profile):
    """Regressão: vaga presencial em Blumenau chegava rotulada como remota."""
    from jobmatch.collectors.linkedin import LinkedInCollector
    from jobmatch.domain.job import WorkModel

    coletor = LinkedInCollector(profile)
    job = coletor._to_job(_card_linkedin("Desenvolvedor(a) Full Stack", "Blumenau, SC"),
                          "FULL STACK · REMOTO")
    assert job.work_model is WorkModel.UNKNOWN

    remota = coletor._to_job(_card_linkedin("Desenvolvedor Frontend - Trabalho Remoto", "Recife"),
                             "FRONT END · REMOTO")
    assert remota.work_model is WorkModel.REMOTE
