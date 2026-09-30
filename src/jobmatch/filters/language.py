"""Exigência de idioma: a vaga pede um inglês acima do nível do candidato?

Duas evidências, nesta ordem:
  1. exigência explícita ("inglês fluente", "advanced English") fora do bloco
     de diferenciais — "inglês avançado será um diferencial" não exige nada;
  2. descrição escrita em inglês — na prática, rotina e entrevista em inglês.
"""

from __future__ import annotations

import re

from ..domain.job import Job
from ..domain.text import contains_phrase, normalize

# Marcadores de que a linha fala de diferencial, não de requisito.
_NICE_MARKERS = (
    "diferencial", "diferenciais", "desejavel", "desejaveis", "sera um plus", "e um plus",
    "nice to have", "bonus", "is a plus", "a plus", "preferred", "nao obrigatorio",
)

_EN_STOPWORDS = frozenset(
    "the and with you will our to of for are we your in on is be an this that".split()
)
_PT_STOPWORDS = frozenset(
    "de e com para voce que em da do dos das os uma um nossa nosso ser sua seu".split()
)
_WORD_RE = re.compile(r"[a-z]+")

# Abaixo disso a contagem de palavras não diz nada sobre o idioma.
_MIN_STOPWORDS = 30


def required_english(job: Job, termos: tuple[str, ...], termos_titulo: tuple[str, ...] = ()) -> str:
    """Termo de exigência de inglês encontrado como requisito, ou ''.

    `termos_titulo` são mais amplos: "Inglês" sozinho no título já é exigência
    ("Front-end Angular | Pleno | Inglês").
    """
    titulo_n = normalize(job.title)
    for termo in (*termos_titulo, *termos):
        if contains_phrase(titulo_n, termo):
            return termo
    if not termos:
        return ""

    texto = "\n".join(
        v for k, v in job.sections.items() if k != "nice_to_have"
    ) or job.description
    for linha in texto.splitlines():
        linha_n = normalize(linha)
        if not linha_n:
            continue
        termo = next((t for t in termos if contains_phrase(linha_n, t)), "")
        if termo and not any(contains_phrase(linha_n, m) for m in _NICE_MARKERS):
            return termo
    return ""


def is_english_posting(job: Job) -> bool:
    """Descrição predominantemente em inglês. Título sozinho não conta:
    "Fullstack Developer | Pleno" é título comum em vaga brasileira."""
    if not job.has_description:
        return False
    palavras = _WORD_RE.findall(normalize(job.description))
    en = sum(1 for p in palavras if p in _EN_STOPWORDS)
    pt = sum(1 for p in palavras if p in _PT_STOPWORDS)
    if en + pt < _MIN_STOPWORDS:
        return False
    return en > 2 * pt
