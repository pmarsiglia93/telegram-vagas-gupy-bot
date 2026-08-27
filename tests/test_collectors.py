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
