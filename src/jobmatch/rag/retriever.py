"""Retriever: dada a descrição de uma vaga, quais partes do perfil importam.

Este é o problema que o RAG resolve aqui — não é ChromaDB decorativo. O LLM
recebe só os 4-6 chunks relevantes (com o nível de evidência de cada um) em
vez do perfil inteiro, o que reduz token, reduz custo e reduz alucinação de
experiência inexistente.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.profile import Profile
from .chunker import build_chunks
from .embeddings import EmbeddingProvider
from .vector_store import SearchHit, VectorStore

# Calibração da similaridade bruta para a escala 0..1 do score.
# Espaços vetoriais diferentes têm distribuições de cosseno diferentes; sem isto
# o componente semântico puxaria todo score para baixo com o provider `hashing`.
SIMILARITY_SCALE: dict[str, float] = {
    "hashing": 0.30,
    "openai": 0.55,
    "gemini": 0.68,
}
DEFAULT_SCALE = 0.55


@dataclass
class RetrievalResult:
    hits: list[SearchHit]
    similarity: float  # 0..1, já calibrado

    @property
    def titles(self) -> list[str]:
        return [h.record.metadata.get("title", h.record.id) for h in self.hits]


class ProfileRetriever:
    """Indexa o perfil e recupera contexto por similaridade semântica.

    O índice é reconstruído a cada execução: são ~25 chunks, o custo é
    desprezível e isso remove qualquer dependência de estado persistente entre
    runs do GitHub Actions (§23).
    """

    def __init__(self, profile: Profile, embedder: EmbeddingProvider, store: VectorStore) -> None:
        self.profile = profile
        self.embedder = embedder
        self.store = store
        self.ready = False
        self.error = ""
        self.chunks_indexed = 0

    def build(self) -> bool:
        try:
            chunks = build_chunks(self.profile)
            if not chunks:
                self.error = "perfil sem chunks"
                return False
            vetores = self.embedder.embed([c.text for c in chunks])
            if len(vetores) != len(chunks):
                self.error = "embeddings incompletos"
                return False
            for chunk, vetor in zip(chunks, vetores):
                chunk.embedding = vetor
            self.store.reset()
            self.store.add(chunks)
            self.chunks_indexed = len(chunks)
            self.ready = True
            return True
        except Exception as exc:  # RAG nunca derruba o bot (§22)
            self.error = f"{type(exc).__name__}: {exc}"
            self.ready = False
            return False

    def retrieve(self, texto: str, k: int = 5) -> RetrievalResult | None:
        if not self.ready or not texto.strip():
            return None
        try:
            vetor = self.embedder.embed([texto])[0]
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return None

        hits = self.store.query(vetor, k=k)
        if not hits:
            return None

        escala = SIMILARITY_SCALE.get(getattr(self.embedder, "name", ""), DEFAULT_SCALE)
        top = hits[: min(3, len(hits))]
        media = sum(h.score for h in top) / len(top)
        similaridade = max(0.0, min(1.0, media / escala))
        return RetrievalResult(hits=hits, similarity=similaridade)
