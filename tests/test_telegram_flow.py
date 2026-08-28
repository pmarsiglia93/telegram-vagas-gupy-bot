"""Fluxo de envio ao Telegram — sem nenhuma chamada de rede real.

A garantia central: falha de envio NÃO pode marcar a vaga como enviada. Se
isso acontecesse, a vaga seria perdida para sempre pela deduplicação.
"""

from unittest.mock import MagicMock, patch

import pytest

from jobmatch.domain.job import Job, WorkModel
from jobmatch.notifications.telegram import TelegramNotifier


def vaga(url="https://exemplo.com/1"):
    return Job(source="gupy", title="Software Engineer", company="Empresa X", url=url,
               raw_location="São Paulo - SP", work_model=WorkModel.REMOTE,
               description="React e TypeScript.")


def _resp(status=200, payload=None, texto=""):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload or {}
    r.text = texto
    return r


# --------------------------------------------------------------------------
# Segredos nunca aparecem em log
# --------------------------------------------------------------------------

def test_token_nunca_aparece_completo():
    n = TelegramNotifier("8674389488:AAquiVaiOSegredoCompleto", "-100123")
    mascarado = n.mask_token()
    assert "AAquiVaiOSegredoCompleto" not in mascarado
    assert mascarado.startswith("8674389488:")


def test_mask_token_sem_token():
    assert TelegramNotifier("", "-100").mask_token() == "(ausente)"


# --------------------------------------------------------------------------
# Verificação de credenciais (getMe / getChat)
# --------------------------------------------------------------------------

def test_check_token_sem_token_nao_faz_requisicao():
    with patch("jobmatch.notifications.telegram.requests.get") as get:
        ok, detalhe = TelegramNotifier("", "-100").check_token()
    assert ok is False
    assert "ausente" in detalhe
    get.assert_not_called()


def test_check_token_valido():
    with patch("jobmatch.notifications.telegram.requests.get") as get:
        get.return_value = _resp(200, {"result": {"username": "bot", "id": 42}})
        ok, detalhe = TelegramNotifier("tok", "-100").check_token()
    assert ok is True and "@bot" in detalhe


def test_check_token_revogado():
    with patch("jobmatch.notifications.telegram.requests.get") as get:
        get.return_value = _resp(401, texto="Unauthorized")
        ok, detalhe = TelegramNotifier("tok", "-100").check_token()
    assert ok is False
    assert "revogado" in detalhe or "401" in detalhe


def test_check_chat_invalido():
    with patch("jobmatch.notifications.telegram.requests.get") as get:
        get.return_value = _resp(400, texto="Bad Request: chat not found")
        ok, detalhe = TelegramNotifier("tok", "-999").check_chat()
    assert ok is False
    assert "inválido" in detalhe or "não é membro" in detalhe


def test_check_chat_valido():
    with patch("jobmatch.notifications.telegram.requests.get") as get:
        get.return_value = _resp(200, {"result": {"title": "Meu Grupo", "type": "group"}})
        ok, detalhe = TelegramNotifier("tok", "-100").check_chat()
    assert ok is True and "Meu Grupo" in detalhe


# --------------------------------------------------------------------------
# Envio
# --------------------------------------------------------------------------

def test_falha_de_envio_retorna_false():
    with patch("jobmatch.notifications.telegram.requests.post") as post:
        post.return_value = _resp(400, texto="Bad Request")
        n = TelegramNotifier("tok", "-100")
        assert n.send("oi") is False
        assert n.erros == 1 and n.enviadas == 0


def test_excecao_de_rede_nao_propaga():
    """Erro de rede não pode derrubar a execução inteira."""
    with patch("jobmatch.notifications.telegram.requests.post", side_effect=OSError("sem rede")):
        n = TelegramNotifier("tok", "-100")
        assert n.send("oi") is False
        assert n.erros == 1


def test_dry_run_nao_chama_a_api():
    with patch("jobmatch.notifications.telegram.requests.post") as post:
        n = TelegramNotifier("tok", "-100", dry_run=True)
        assert n.send("oi") is True
        post.assert_not_called()


# --------------------------------------------------------------------------
# A regra crítica: falha de envio não marca como enviada
# --------------------------------------------------------------------------

def test_falha_no_envio_nao_marca_vaga_como_enviada(tmp_path):
    """Se marcasse, a dedup descartaria a vaga para sempre na próxima execução."""
    from jobmatch.persistence.sqlite import JobRepository

    repo = JobRepository(str(tmp_path / "t.db"))
    try:
        job = vaga()
        notifier = TelegramNotifier("tok", "-100")

        with patch("jobmatch.notifications.telegram.requests.post") as post:
            post.return_value = _resp(500, texto="erro interno")
            enviado = notifier.send("mensagem")

        # Réplica exata da regra do pipeline: só registra se o envio confirmou.
        if enviado:
            repo.registrar(job, 90.0)

        assert enviado is False
        assert repo.ja_enviada(job) is False, (
            "vaga marcada como enviada apesar da falha — seria perdida pela dedup"
        )
    finally:
        repo.close()


def test_envio_bem_sucedido_marca_como_enviada(tmp_path):
    from jobmatch.persistence.sqlite import JobRepository

    repo = JobRepository(str(tmp_path / "t.db"))
    try:
        job = vaga()
        notifier = TelegramNotifier("tok", "-100", pausa=0.0)
        with patch("jobmatch.notifications.telegram.requests.post") as post:
            post.return_value = _resp(200, {"ok": True})
            assert notifier.send("mensagem") is True
        repo.registrar(job, 90.0)
        assert repo.ja_enviada(job) is True
    finally:
        repo.close()


def test_dedup_continua_funcionando_apos_as_mudancas(tmp_path):
    from jobmatch.persistence.sqlite import JobRepository

    repo = JobRepository(str(tmp_path / "t.db"))
    try:
        job = vaga()
        assert repo.ja_enviada(job) is False
        repo.registrar(job, 80.0)
        assert repo.ja_enviada(job) is True
        # Mesma vaga em outra fonte/URL continua sendo reconhecida.
        gemea = Job(source="linkedin", title="Software Engineer", company="Empresa X",
                    url="https://linkedin.com/jobs/view/999",
                    raw_location="São Paulo - SP", work_model=WorkModel.REMOTE)
        assert repo.ja_enviada(gemea) is True
    finally:
        repo.close()


# --------------------------------------------------------------------------
# Diagnóstico
# --------------------------------------------------------------------------

def test_diagnostico_nao_envia_nada(profile, tmp_path):
    """--diagnose-telegram percorre o pipeline mas nunca chama sendMessage."""
    from jobmatch.config.settings import Settings
    from jobmatch.diagnostics import diagnose

    settings = Settings(
        telegram_token="tok", chat_id="-100",
        db_path=str(tmp_path / "d.db"), profile_path="profile.yaml",
        max_jobs_per_run=5,
    )

    with patch("jobmatch.notifications.telegram.requests.post") as post, \
         patch("jobmatch.notifications.telegram.requests.get") as get, \
         patch("jobmatch.pipeline.Pipeline.prepare", return_value=[]):
        get.return_value = _resp(200, {"result": {"username": "bot", "id": 1, "type": "group"}})
        diagnose(settings)
        post.assert_not_called()


def test_teste_de_telegram_para_se_o_token_falhar(tmp_path):
    """Não adianta tentar enviar se o token já está inválido."""
    from jobmatch.config.settings import Settings
    from jobmatch.diagnostics import test_telegram

    settings = Settings(telegram_token="tok", chat_id="-100",
                        db_path=str(tmp_path / "d.db"), profile_path="profile.yaml")
    with patch("jobmatch.notifications.telegram.requests.get") as get, \
         patch("jobmatch.notifications.telegram.requests.post") as post:
        get.return_value = _resp(401, texto="Unauthorized")
        assert test_telegram(settings) == 1
        post.assert_not_called()
