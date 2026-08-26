"""Normalização de texto compartilhada por todo o pipeline.

Regra geral: comparações sempre acontecem sobre texto *normalizado*
(minúsculo, sem acento, pontuação virando espaço). Isso evita o bug clássico
de `"sp" in "jaspion"` porque a busca passa a ser por token, não substring.
"""

from __future__ import annotations

import html as _html
import re
import unicodedata

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^a-z0-9+#.]+")


def strip_html(texto: str | None) -> str:
    """Remove tags HTML e resolve entidades. A Gupy devolve descrição com HTML."""
    if not texto:
        return ""
    # <br>, </p>, </li> viram quebra de linha para preservar a estrutura de seções.
    texto = re.sub(r"(?i)<\s*(br|/p|/li|/div|/h[1-6])\s*/?>", "\n", texto)
    texto = _TAG_RE.sub(" ", texto)
    texto = _html.unescape(texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def deaccent(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize(texto: str | None) -> str:
    """Minúsculo, sem acento, espaços colapsados. Mantém pontuação."""
    if not texto:
        return ""
    return _WS_RE.sub(" ", deaccent(str(texto)).lower()).strip()


def tokenize(texto: str | None) -> list[str]:
    """Tokens normalizados. Preserva `+`, `#` e `.` para casar c#, node.js, c++.

    Pontuação de FIM de palavra é removida: sem isso "traffic." e "traffic"
    viravam tokens distintos, e toda palavra em fim de frase deixava de casar
    nos embeddings. Pontos internos (node.js) e iniciais (.net) sobrevivem.
    """
    if not texto:
        return []
    saida = []
    for bruto in _NON_WORD_RE.split(normalize(texto)):
        token = bruto.rstrip(".")
        if token:
            saida.append(token)
    return saida


def token_set(texto: str | None) -> set[str]:
    return set(tokenize(texto))


def contains_phrase(texto_normalizado: str, frase: str) -> bool:
    """Casamento por limite de palavra, não por substring.

    `contains_phrase("vaga em jaspion", "sp")` -> False
    `contains_phrase("vaga em sao paulo - sp", "sp")` -> True
    """
    frase_n = normalize(frase)
    if not frase_n:
        return False
    padrao = r"(?<![a-z0-9])" + re.escape(frase_n).replace(r"\ ", r"[\s\-]+") + r"(?![a-z0-9])"
    return re.search(padrao, texto_normalizado) is not None


def truncate(texto: str, limite: int, sufixo: str = "…") -> str:
    texto = texto.strip()
    if len(texto) <= limite:
        return texto
    corte = texto[:limite].rsplit(" ", 1)[0]
    return corte + sufixo
