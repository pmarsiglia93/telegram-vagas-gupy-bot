"""Leitura do agendamento configurado no workflow do GitHub Actions.

O bot não roda um scheduler próprio: quem dispara é o `schedule:` do GitHub
Actions. Este módulo lê esse agendamento do próprio arquivo do workflow para
que o diagnóstico possa responder "qual é a próxima execução prevista?" sem
depender de rede nem de dependência extra.

Suporta o subconjunto de cron usado no projeto: minuto e hora fixos ou em
lista, dia-do-mês/mês curinga, dia-da-semana em lista ou faixa.
"""

from __future__ import annotations

import datetime
import os
import re

WORKFLOW_PADRAO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".github", "workflows", "vagas.yml",
)

# O GitHub Actions interpreta cron sempre em UTC.
TZ_CRON = datetime.timezone.utc
TZ_LOCAL = datetime.timezone(datetime.timedelta(hours=-3))  # BRT

_CRON_RE = re.compile(r"^\s*-\s*cron:\s*['\"]([^'\"]+)['\"]", re.MULTILINE)


def read_cron_expressions(caminho: str = WORKFLOW_PADRAO) -> list[str]:
    """Expressões cron declaradas no workflow. Lista vazia se não houver."""
    if not os.path.exists(caminho):
        return []
    return _CRON_RE.findall(open(caminho, encoding="utf-8").read())


def _expandir(campo: str, minimo: int, maximo: int) -> set[int]:
    """Expande '*', '1-5', '0,6' e '*/2' para o conjunto de valores."""
    if campo == "*":
        return set(range(minimo, maximo + 1))
    valores: set[int] = set()
    for parte in campo.split(","):
        if parte.startswith("*/"):
            passo = int(parte[2:])
            valores |= set(range(minimo, maximo + 1, passo))
        elif "-" in parte:
            ini, fim = (int(x) for x in parte.split("-", 1))
            valores |= set(range(ini, fim + 1))
        else:
            valores.add(int(parte))
    return valores


def next_runs(expressoes: list[str], a_partir_de: datetime.datetime | None = None,
              quantos: int = 3) -> list[datetime.datetime]:
    """Próximas execuções previstas, em UTC, ordenadas."""
    agora = a_partir_de or datetime.datetime.now(TZ_CRON)
    agora = agora.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)

    regras = []
    for expr in expressoes:
        campos = expr.split()
        if len(campos) != 5:
            continue
        minuto, hora, _dia_mes, _mes, dow = campos
        regras.append((
            _expandir(minuto, 0, 59),
            _expandir(hora, 0, 23),
            _expandir(dow, 0, 6),
        ))
    if not regras:
        return []

    achados: list[datetime.datetime] = []
    candidato = agora
    # Uma semana à frente cobre qualquer agendamento semanal do projeto.
    limite = agora + datetime.timedelta(days=8)
    while candidato < limite and len(achados) < quantos:
        dow_cron = candidato.isoweekday() % 7  # cron: 0=domingo
        for minutos, horas, dows in regras:
            if candidato.minute in minutos and candidato.hour in horas and dow_cron in dows:
                achados.append(candidato)
                break
        candidato += datetime.timedelta(minutes=1)
    return achados


def describe_schedule(caminho: str = WORKFLOW_PADRAO) -> list[str]:
    """Linhas prontas para o diagnóstico."""
    expressoes = read_cron_expressions(caminho)
    if not expressoes:
        return ["nenhum agendamento encontrado no workflow"]

    linhas = [f"cron: {e}" for e in expressoes]
    proximas = next_runs(expressoes)
    if not proximas:
        linhas.append("não foi possível calcular a próxima execução")
        return linhas
    for dt in proximas:
        local = dt.astimezone(TZ_LOCAL)
        linhas.append(f"próxima: {local:%d/%m %H:%M} BRT  ({dt:%H:%M} UTC)")
    return linhas
