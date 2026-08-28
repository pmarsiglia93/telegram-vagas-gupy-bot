"""Leitura do agendamento e cálculo da próxima execução.

Regressão protegida: em 27/08 o cron foi alterado duas vezes e o GitHub parou
de criar execuções `schedule`. Estes testes garantem que o agendamento
declarado no workflow continua sendo lido e interpretado corretamente — e que
o diagnóstico consegue dizer quando é a próxima execução.
"""

import datetime

import pytest

from jobmatch.scheduler_info import (
    TZ_CRON,
    describe_schedule,
    next_runs,
    read_cron_expressions,
)


def test_workflow_declara_agendamento():
    """Sem isto o bot depende exclusivamente de disparo manual."""
    crons = read_cron_expressions()
    assert crons, "o workflow precisa declarar pelo menos um cron"


def test_agendamento_cobre_dias_uteis_e_fim_de_semana():
    campos_dow = [c.split()[4] for c in read_cron_expressions()]
    assert any("1-5" in d or "1,2,3,4,5" in d for d in campos_dow), "faltou dia útil"
    assert any("0" in d and "6" in d for d in campos_dow), "faltou fim de semana"


def test_calcula_proxima_execucao_em_dia_util():
    # 28/08/2026 é uma sexta-feira, 21:30 UTC.
    agora = datetime.datetime(2026, 8, 28, 21, 30, tzinfo=TZ_CRON)
    proximas = next_runs(["0 11,13,15,17,19,21,23 * * 1-5"], a_partir_de=agora, quantos=1)
    assert proximas == [datetime.datetime(2026, 8, 28, 23, 0, tzinfo=TZ_CRON)]


def test_pula_o_fim_de_semana_quando_o_cron_e_de_dia_util():
    # Sábado 29/08/2026: o cron 1-5 só volta na segunda 31/08.
    agora = datetime.datetime(2026, 8, 29, 12, 0, tzinfo=TZ_CRON)
    proxima = next_runs(["0 11 * * 1-5"], a_partir_de=agora, quantos=1)[0]
    assert proxima == datetime.datetime(2026, 8, 31, 11, 0, tzinfo=TZ_CRON)


def test_combina_multiplas_expressoes_em_ordem():
    agora = datetime.datetime(2026, 8, 28, 10, 0, tzinfo=TZ_CRON)
    proximas = next_runs(["0 11 * * 1-5", "0 13 * * 1-5"], a_partir_de=agora, quantos=2)
    assert [d.hour for d in proximas] == [11, 13]


def test_minuto_deslocado_e_respeitado():
    """O cron ':07' que testamos precisa ser interpretado corretamente."""
    agora = datetime.datetime(2026, 8, 28, 11, 0, tzinfo=TZ_CRON)
    proxima = next_runs(["7 12 * * 1-5"], a_partir_de=agora, quantos=1)[0]
    assert (proxima.hour, proxima.minute) == (12, 7)


@pytest.mark.parametrize("campo,esperado", [
    ("*", set(range(0, 7))),
    ("1-5", {1, 2, 3, 4, 5}),
    ("0,6", {0, 6}),
])
def test_expansao_de_campos(campo, esperado):
    from jobmatch.scheduler_info import _expandir
    assert _expandir(campo, 0, 6) == esperado


def test_describe_schedule_nao_quebra_sem_workflow(tmp_path):
    linhas = describe_schedule(str(tmp_path / "inexistente.yml"))
    assert linhas == ["nenhum agendamento encontrado no workflow"]


def test_cron_invalido_e_ignorado_sem_estourar():
    assert next_runs(["expressão inválida"], quantos=1) == []
