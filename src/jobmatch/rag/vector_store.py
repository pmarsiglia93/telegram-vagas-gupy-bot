"""Abstração de banco vetorial.

O resto do projeto só conhece `VectorStore`. Trocar ChromaDB por pgvector,
Qdrant ou Pinecone significa adicionar uma classe aqui — nada mais.

Default = `memory`. O índice do perfil tem ~25 chunks; instalar chromadb
(+onnxruntime, ~250 MB) a cada run do GitHub Actions seria infra desnecessária
(§23). `VECTOR_STORE=chroma` liga a persistência quando fizer sentido.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .embeddings import cosine


@dataclass
class VectorRecord:
    id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)


@dataclass
class SearchHit:
    record: VectorRecord
    score: float  # similaridade 0..1 (maior = mais parecido)


class VectorStore(Protocol):
    name: str

    def reset(self) -> None: ...
    def add(self, records: list[VectorRecord]) -> None: ...
    def query(self, embedding: list[float], k: int = 5) -> list[SearchHit]: ...
    def count(self) -> int: ...


class InMemoryVectorStore:
    """Busca exata por cosseno. Para dezenas de chunks é instantâneo."""

    name = "memory"

    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def reset(self) -> None:
        self._records = []

    def add(self, records: list[VectorRecord]) -> None:
        self._records.extend(records)

    def query(self, embedding: list[float], k: int = 5) -> list[SearchHit]:
        hits = [SearchHit(r, cosine(embedding, r.embedding)) for r in self._records]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def count(self) -> int:
        return len(self._records)


class ChromaVectorStore:
    """ChromaDB persistente ou efêmero.

    Recebe os embeddings prontos (`embeddings=`) em vez de deixar o Chroma
    calcular: o EmbeddingProvider do projeto é a única fonte de vetores, para
    perfil e vaga usarem sempre o mesmo espaço vetorial.
    """

    name = "chroma"

    def __init__(self, path: str = "", collection: str = "profile_chunks") -> None:
        import chromadb  # import tardio: dependência opcional

        if path:
            self._client = chromadb.PersistentClient(path=path)
        else:
            self._client = chromadb.EphemeralClient()
        self._collection_name = collection
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        self._collection.add(
            ids=[r.id for r in records],
            documents=[r.text for r in records],
            embeddings=[r.embedding for r in records],
            metadatas=[{k: str(v) for k, v in r.metadata.items()} or {"_": ""} for r in records],
        )

    def query(self, embedding: list[float], k: int = 5) -> list[SearchHit]:
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, max(1, self.count())),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        for i, doc_id in enumerate(res.get("ids", [[]])[0]):
            distancia = res["distances"][0][i]
            hits.append(SearchHit(
                VectorRecord(
                    id=doc_id,
                    text=res["documents"][0][i],
                    metadata=res["metadatas"][0][i] or {},
                ),
                # Chroma devolve distância cosseno: similaridade = 1 - distância.
                score=max(0.0, 1.0 - float(distancia)),
            ))
        return hits

    def count(self) -> int:
        return self._collection.count()


def create_vector_store(kind: str, path: str = "") -> tuple[VectorStore, str]:
    """Retorna (store, aviso). Chroma indisponível cai para memória (§22)."""
    escolha = (kind or "memory").strip().lower()
    if escolha in ("chroma", "chromadb"):
        try:
            return ChromaVectorStore(path=path), ""
        except Exception as exc:
            return InMemoryVectorStore(), f"ChromaDB indisponível ({type(exc).__name__}); usando memória"
    return InMemoryVectorStore(), ""
