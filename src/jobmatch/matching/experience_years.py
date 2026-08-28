"""Detecção de exigência de TEMPO de experiência ("3+ anos de Java").

Por que isto existe separado da cobertura técnica:

    "Java obrigatório"        -> o perfil TEM evidência profissional de Java.
                                 Não é incompatibilidade.
    "3+ anos de Java"         -> o perfil tem evidência profissional, mas NÃO
                                 declara tempo de experiência em lugar nenhum.
                                 Isso é uma lacuna PARCIAL, não um gap.

Ter usado uma tecnologia profissionalmente não prova uma quantidade específica
de anos. O sistema nunca converte uma coisa na outra — ele registra a exigência,
aplica uma penalidade pequena e limitada, e diz isso na análise.
"""

from __future__ import annotations

import re

from ..domain.text import normalize

# "3 anos", "3+ anos", "mínimo de 5 anos", "entre 3 e 5 anos", "3 years",
# "at least 4 years". Captura o menor número quando há faixa.
_PADROES = (
    re.compile(r"(?<![0-9])(\d{1,2})\s*\+?\s*(?:a|ate|até)?\s*\d{0,2}\s*anos?\b"),
    re.compile(r"(?<![0-9])(\d{1,2})\s*\+?\s*years?\b"),
)

# Abaixo deste patamar a exigência é essencialmente de nível de entrada e não
# diferencia candidatos — não penaliza.
LIMIAR_ANOS = 3
# Pontos de score descontados por ano acima do limiar.
PENALIDADE_POR_ANO = 0.8
# Teto absoluto. Exigir muitos anos reduz a aderência, mas nunca inviabiliza
# a vaga: o objetivo é ordenar melhor, não descartar (§3).
PENALIDADE_MAXIMA = 6.0

# Anos citados acima disto quase sempre são ruído ("empresa com 30 anos de
# mercado", "produto usado há 20 anos"), não requisito de candidato.
ANOS_MAXIMO_PLAUSIVEL = 15


def detect_year_requirements(texto: str) -> list[int]:
    """Anos de experiência exigidos, em ordem decrescente de exigência."""
    if not texto:
        return []
    alvo = normalize(texto)
    achados: set[int] = set()
    for padrao in _PADROES:
        for m in padrao.finditer(alvo):
            anos = int(m.group(1))
            if 0 < anos <= ANOS_MAXIMO_PLAUSIVEL:
                achados.add(anos)
    return sorted(achados, reverse=True)


def years_penalty(anos_exigidos: list[int]) -> float:
    """Penalidade em pontos de score (0..PENALIDADE_MAXIMA).

    Aplicada apenas porque o perfil não declara tempo de experiência. Se um dia
    o perfil passar a declarar anos por skill, esta função deve consultá-los em
    vez de assumir a lacuna.
    """
    if not anos_exigidos:
        return 0.0
    maior = max(anos_exigidos)
    if maior < LIMIAR_ANOS:
        return 0.0
    excedente = maior - LIMIAR_ANOS + 1
    return min(PENALIDADE_MAXIMA, excedente * PENALIDADE_POR_ANO)


def describe(anos_exigidos: list[int]) -> list[str]:
    """Texto curto para a mensagem: o que a vaga pede em tempo."""
    return [f"{a}+ anos de experiência" for a in anos_exigidos[:2]]
