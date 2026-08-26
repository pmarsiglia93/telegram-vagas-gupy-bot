"""Camada de embeddings — trocável por variável de ambiente.

Providers:
  hashing : local, determinístico, sem chave e sem rede. É similaridade
            LEXICAL (hashing de tokens + bigramas), não semântica de verdade.
            Existe para o bot nunca depender de API para funcionar.
  openai  : text-embedding-3-small por padrão.
  gemini  : text-embedding-004 por padrão.

Nenhuma chave é lida fora de `Settings`, que por sua vez só lê `.env`/ambiente.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

import requests

from ..domain.text import tokenize


class EmbeddingProvider(Protocol):
    name: str
    dimension: int
    calls: int

    def embed(self, textos: list[str]) -> list[list[float]]: ...


def _l2(vetor: list[float]) -> list[float]:
    norma = math.sqrt(sum(v * v for v in vetor))
    return [v / norma for v in vetor] if norma else vetor


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)


# Palavras funcionais de PT/EN. O provider local não tem IDF, então sem este
# filtro a similaridade é dominada por "de", "com", "para" — dois textos sobre
# assuntos diferentes ficam parecidos só por compartilhar preposições.
STOPWORDS: frozenset[str] = frozenset("""
a as o os um uma uns umas de do da dos das em no na nos nas por para pelo pela
com sem sob sobre entre ate apos e ou mas que se como quando onde qual quais
ao aos à às isso isto aquilo este esta esse essa aquele aquela seu sua seus suas
nosso nossa meu minha ser estar ter haver foi sao eh e' era serao tem tinha
mais menos muito pouco todo toda todos todas outro outra outros outras
voce vocemesmo nao sim ja tambem entao assim porque pois cada qualquer
bloco competencias experiencia profissional projeto proprio producao estudo
cursos concluido em pratica projetos
the of and or to in on for with without at by from as is are was were be been
you your we our they their this that these those it its will would can could
a an if then than so such use using used
""".split())


class HashingEmbeddingProvider:
    """Fallback local. Zero dependências, zero custo, zero rede."""

    name = "hashing"

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension
        self.calls = 0

    def _hash(self, token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % self.dimension

    def embed(self, textos: list[str]) -> list[list[float]]:
        self.calls += 1
        saida: list[list[float]] = []
        for texto in textos:
            # Bigramas são formados ANTES do filtro, para preservar expressões
            # como "vector database", e filtrados depois só se ambos os lados
            # forem palavra funcional.
            brutos = tokenize(texto)
            bigramas = [
                f"{a}_{b}" for a, b in zip(brutos, brutos[1:])
                if not (a in STOPWORDS and b in STOPWORDS)
            ]
            tokens = [t for t in brutos if t not in STOPWORDS and len(t) > 1]
            grams = tokens + bigramas
            contagem: dict[int, float] = {}
            for g in grams:
                idx = self._hash(g)
                contagem[idx] = contagem.get(idx, 0.0) + 1.0
            vetor = [0.0] * self.dimension
            for idx, freq in contagem.items():
                # tf sublinear: evita que uma palavra repetida domine o vetor.
                vetor[idx] = 1.0 + math.log(freq)
            saida.append(_l2(vetor))
        return saida


class OpenAIEmbeddingProvider:
    name = "openai"

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", timeout: int = 30) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.dimension = 1536
        self.calls = 0

    def embed(self, textos: list[str]) -> list[list[float]]:
        self.calls += 1
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "input": textos},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        dados = sorted(resp.json()["data"], key=lambda d: d["index"])
        vetores = [d["embedding"] for d in dados]
        if vetores:
            self.dimension = len(vetores[0])
        return vetores


class GeminiEmbeddingProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str = "text-embedding-004", timeout: int = 30) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.dimension = 768
        self.calls = 0

    def embed(self, textos: list[str]) -> list[list[float]]:
        self.calls += 1
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:batchEmbedContents"
            f"?key={self.api_key}"
        )
        payload = {
            "requests": [
                {"model": f"models/{self.model}", "content": {"parts": [{"text": t}]}}
                for t in textos
            ]
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        vetores = [e["values"] for e in resp.json().get("embeddings", [])]
        if vetores:
            self.dimension = len(vetores[0])
        return vetores


class ResilientEmbeddingProvider:
    """Envelopa um provider remoto e cai para o local quando ele falha (§22)."""

    def __init__(self, primario: EmbeddingProvider, fallback: EmbeddingProvider) -> None:
        self.primario = primario
        self.fallback = fallback
        self.name = primario.name
        self.dimension = primario.dimension
        self.calls = 0
        self.degraded = False
        self.last_error = ""

    def embed(self, textos: list[str]) -> list[list[float]]:
        self.calls += 1
        if not self.degraded:
            try:
                vetores = self.primario.embed(textos)
                self.dimension = self.primario.dimension
                return vetores
            except Exception as exc:  # rede, quota, chave inválida...
                self.degraded = True
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.name = f"{self.primario.name}→{self.fallback.name}"
        self.dimension = self.fallback.dimension
        return self.fallback.embed(textos)


def create_embedding_provider(
    provider: str, api_key: str = "", model: str = "", timeout: int = 30
) -> EmbeddingProvider:
    """Fábrica. Um provider remoto sem chave cai silenciosamente para `hashing`."""
    local = HashingEmbeddingProvider()
    escolha = (provider or "hashing").strip().lower()

    if escolha in ("", "none", "hashing", "local"):
        return local
    if not api_key:
        return local

    if escolha == "openai":
        remoto: EmbeddingProvider = OpenAIEmbeddingProvider(api_key, model or "text-embedding-3-small", timeout)
    elif escolha in ("gemini", "google"):
        remoto = GeminiEmbeddingProvider(api_key, model or "text-embedding-004", timeout)
    else:
        return local

    return ResilientEmbeddingProvider(remoto, local)
