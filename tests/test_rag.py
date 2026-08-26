"""Chunking semântico, recuperação e a resiliência da camada de IA."""

from jobmatch.domain.job import Job, WorkModel
from jobmatch.rag.chunker import build_chunks
from jobmatch.rag.embeddings import HashingEmbeddingProvider, cosine, create_embedding_provider
from jobmatch.rag.retriever import ProfileRetriever
from jobmatch.rag.vector_store import InMemoryVectorStore, create_vector_store


def test_chunks_preservam_o_nivel_de_evidencia(profile):
    chunks = build_chunks(profile)
    assert chunks
    niveis = {c.metadata["experience_level"] for c in chunks}
    assert "study" in niveis, "os estudos de IA precisam continuar rotulados como estudo"
    for chunk in chunks:
        assert chunk.metadata.get("type")
        assert chunk.metadata.get("title")
        assert chunk.text.strip()


def test_chunks_sao_semanticos_nao_fatias_de_n_caracteres(profile):
    ids = {c.id for c in build_chunks(profile)}
    assert any(i.startswith("skill_") for i in ids)
    assert any(i.startswith("study_") for i in ids)
    assert any(i.startswith("proj_") for i in ids)


def test_embedding_local_e_deterministico():
    provider = HashingEmbeddingProvider()
    a = provider.embed(["react typescript node"])[0]
    b = provider.embed(["react typescript node"])[0]
    assert a == b
    assert cosine(a, b) > 0.99


def test_embedding_local_separa_textos_diferentes():
    provider = HashingEmbeddingProvider()
    frontend, backend, distante = provider.embed([
        "react vue angular interfaces web spa",
        "react vue angular componentes de interface",
        "contabilidade fiscal tributário balanço patrimonial",
    ])
    assert cosine(frontend, backend) > cosine(frontend, distante)


def test_provider_remoto_sem_chave_cai_para_local():
    provider = create_embedding_provider("openai", api_key="")
    assert provider.name == "hashing"


def test_retriever_recupera_o_chunk_relevante(profile):
    retriever = ProfileRetriever(profile, HashingEmbeddingProvider(), InMemoryVectorStore())
    assert retriever.build()

    resultado = retriever.retrieve(
        "Buscamos pessoa desenvolvedora para construir interfaces SPA com React, "
        "Vue e TypeScript, consumindo APIs REST.",
        k=3,
    )
    assert resultado is not None
    assert 0.0 <= resultado.similarity <= 1.0
    # Com experiências profissionais no perfil, o chunk mais relevante para uma
    # vaga de frontend passou a ser a experiência de frontend — e não mais o
    # bloco genérico de skills. Aceita qualquer um dos dois.
    contexto = " ".join(
        f"{h.record.metadata.get('family', '')} {h.record.metadata.get('domains', '')}"
        for h in resultado.hits
    ).lower()
    assert "frontend" in contexto or "language_web" in contexto


def test_retriever_falho_nao_derruba_nada(profile):
    """§22: RAG quebrado degrada para None, nunca levanta exceção."""

    class EmbeddingQuebrado:
        name = "quebrado"
        dimension = 8
        calls = 0

        def embed(self, textos):
            raise RuntimeError("provider fora do ar")

    retriever = ProfileRetriever(profile, EmbeddingQuebrado(), InMemoryVectorStore())
    assert retriever.build() is False
    assert retriever.retrieve("qualquer coisa") is None
    assert retriever.error


def test_vector_store_default_e_memoria():
    store, aviso = create_vector_store("memory")
    assert store.name == "memory"
    assert aviso == ""


def test_chroma_indisponivel_cai_para_memoria():
    store, aviso = create_vector_store("chroma")
    # Com chromadb instalado usa chroma; sem ele, degrada com aviso.
    assert store.name in ("chroma", "memory")
    if store.name == "memory":
        assert "ChromaDB indisponível" in aviso


def test_similaridade_semantica_entra_no_score(profile):
    from jobmatch.matching.heuristic import HeuristicMatcher

    matcher = HeuristicMatcher(profile)
    job = Job(
        source="teste",
        title="Software Engineer",
        company="X",
        url="https://exemplo.com/1",
        raw_location="São Paulo - SP",
        work_model=WorkModel.REMOTE,
        description="Requisitos e qualificações\nReact\nTypeScript\n",
    )
    sem_semantica = matcher.match(job, semantic_similarity=None)
    com_semantica = matcher.match(job, semantic_similarity=0.9)
    assert com_semantica.engine == "semantic"
    assert sem_semantica.engine == "heuristic"
    assert com_semantica.score > sem_semantica.score
