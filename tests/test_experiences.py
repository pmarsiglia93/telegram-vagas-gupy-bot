"""Experiências profissionais reais: retrieval, proveniência e complementaridade (§19).

O que estes testes protegem: a experiência profissional entrou no perfil e
chega ao RAG com contexto útil, SEM promover projeto/estudo a emprego formal.
"""

import pytest

from jobmatch.domain.evidence import CATEGORY_PRACTICAL, CATEGORY_PROFESSIONAL
from jobmatch.domain.job import Job, WorkModel
from jobmatch.matching.heuristic import HeuristicMatcher
from jobmatch.rag.chunker import build_chunks
from jobmatch.rag.embeddings import HashingEmbeddingProvider
from jobmatch.rag.retriever import ProfileRetriever
from jobmatch.rag.vector_store import InMemoryVectorStore

EXPERIENCIAS = ["exp_corebiz", "exp_stalse", "exp_quality_digital", "exp_eac_barkeley"]


@pytest.fixture(scope="module")
def retriever(profile):
    r = ProfileRetriever(profile, HashingEmbeddingProvider(), InMemoryVectorStore())
    assert r.build()
    return r


def vaga(titulo, requisitos, descricao_extra=""):
    return Job(
        source="teste", title=titulo, company="Empresa X",
        url=f"https://exemplo.com/{abs(hash(titulo)) % 10**8}",
        raw_location="São Paulo - SP", work_model=WorkModel.REMOTE,
        description="Requisitos e qualificações\n" + "\n".join(requisitos) + "\n" + descricao_extra,
    )


def titulos(resultado) -> str:
    return " | ".join(h.record.metadata.get("title", "") for h in resultado.hits)


# --------------------------------------------------------------------------
# Chunks
# --------------------------------------------------------------------------

def test_cada_experiencia_vira_um_chunk_proprio(profile):
    ids = {c.id for c in build_chunks(profile)}
    for exp in EXPERIENCIAS:
        assert exp in ids, f"{exp} não virou chunk"


def test_chunk_de_experiencia_carrega_contexto_e_nao_so_tecnologias(profile):
    chunk = next(c for c in build_chunks(profile) if c.id == "exp_corebiz")
    texto = chunk.text.lower()
    # tecnologias + responsabilidades + domínio + contexto de produto
    assert "react" in texto and "vtex" in texto
    assert "e-commerce" in texto
    assert "corebiz" in texto
    assert "domínios:" in texto
    assert "experiência profissional" in texto


def test_metadata_do_chunk_permite_auditoria(profile):
    chunk = next(c for c in build_chunks(profile) if c.id == "exp_stalse")
    md = chunk.metadata
    assert md["type"] == "professional_experience"
    assert md["company"] == "Stalse Analytics"
    assert md["evidence_type"] == "professional"
    assert "frontend" in md["domains"] and "backend" in md["domains"]


# --------------------------------------------------------------------------
# §12 — retrieval por caso
# --------------------------------------------------------------------------

@pytest.mark.parametrize("nome,consulta,esperado", [
    ("A frontend/e-commerce",
     "Frontend Engineer. Requisitos: React, TypeScript, GraphQL, e-commerce, REST APIs.",
     "CoreBiz"),
    ("B Vue/PHP",
     "Full Stack Developer. Requisitos: Vue, PHP, Laravel, MySQL, REST APIs.",
     "EAC"),
    ("C Python/IA",
     "Software Engineer - AI. Requisitos: Python, Django, OpenAI API, LLM integrations, PostgreSQL.",
     "Stalse"),
    ("D e-commerce moderno",
     "Frontend Developer. Requisitos: TypeScript, Preact, Tailwind, e-commerce, high traffic.",
     "Quality Digital"),
])
def test_retrieval_traz_a_experiencia_certa(retriever, nome, consulta, esperado):
    resultado = retriever.retrieve(consulta, k=4)
    assert resultado is not None
    assert esperado in titulos(resultado), f"{nome}: esperava {esperado}, veio {titulos(resultado)}"


def test_complementaridade_experiencia_mais_projeto(retriever):
    """§17: Stalse (profissional) e JobMatch AI (projeto) devem coexistir."""
    resultado = retriever.retrieve(
        "AI Engineer. Requisitos: Python, OpenAI API, RAG, Embeddings, LLMs, vector database.",
        k=5,
    )
    achados = titulos(resultado)
    assert "Stalse" in achados, f"faltou a experiência profissional: {achados}"
    assert "JobMatch AI" in achados, f"faltou o projeto de RAG: {achados}"

    # As duas fontes precisam continuar rotuladas de forma diferente.
    niveis = {
        h.record.metadata.get("title", ""): h.record.metadata.get("evidence_type", "")
        for h in resultado.hits
    }
    stalse = next(v for k, v in niveis.items() if "Stalse" in k)
    jobmatch = next(v for k, v in niveis.items() if "JobMatch AI" in k)
    assert stalse == "professional"
    assert jobmatch == "production_project"


# --------------------------------------------------------------------------
# §8 — proveniência: o que NÃO pode virar profissional
# --------------------------------------------------------------------------

@pytest.mark.parametrize("nome", [
    "RAG", "ChromaDB", "Embeddings", "Vector databases", "Busca semântica",
    "Prompt Engineering", "Structured Outputs", "n8n", "Make", "AI Agents",
    "Anthropic / Gemini APIs",
])
def test_competencia_de_projeto_nao_vira_profissional(profile, nome):
    """A experiência com OpenAI na Stalse não pode arrastar RAG/ChromaDB junto."""
    skill = next(s for s in profile.skills if s.name == nome)
    assert skill.category != CATEGORY_PROFESSIONAL, (
        f"'{nome}' foi promovida a experiência profissional: {skill.evidence_summary()}"
    )
    assert not any(e.type in ("professional", "freelance") for e in skill.evidence)


@pytest.mark.parametrize("nome,fonte", [
    ("OpenAI API", "exp_stalse"),
    ("LLMs", "exp_stalse"),
    ("React", "exp_corebiz"),
    ("Vue", "exp_eac_barkeley"),
    ("Django", "exp_stalse"),
    ("VTEX IO", "exp_corebiz"),
    ("Preact", "exp_quality_digital"),
])
def test_evidencia_profissional_aponta_para_a_experiencia(profile, nome, fonte):
    """Toda evidência profissional precisa de proveniência rastreável."""
    skill = next(s for s in profile.skills if s.name == nome)
    profissionais = [e for e in skill.evidence if e.type == "professional"]
    assert profissionais, f"'{nome}' deveria ter evidência profissional"
    assert any(e.source == fonte for e in profissionais), (
        f"'{nome}' não aponta para {fonte}: {[e.source for e in profissionais]}"
    )


def test_rag_continua_pratico_mas_com_peso_real(profile):
    rag = next(s for s in profile.skills if s.name == "RAG")
    assert rag.category == CATEGORY_PRACTICAL
    assert rag.weight >= 0.75, "evidência de projeto precisa valer crédito real"


# --------------------------------------------------------------------------
# §16 — sem contagem dupla
# --------------------------------------------------------------------------

def test_react_conta_uma_vez_so(profile):
    """CoreBiz + skill React + grupo frontend não podem virar 3 evidências."""
    matcher = HeuristicMatcher(profile)
    resultado = matcher.match(vaga("Frontend Developer", ["React", "TypeScript", "CSS"]))
    ocorrencias = [h for h in resultado.hits if h.required == "React"]
    assert len(ocorrencias) == 1, f"React contou {len(ocorrencias)} vezes"
    assert ocorrencias[0].coverage <= 1.0


def test_evidencias_repetidas_nao_estouram_o_teto(profile):
    """React tem 2 evidências profissionais (CoreBiz e Stalse): peso continua 1.0."""
    react = next(s for s in profile.skills if s.name == "React")
    assert len([e for e in react.evidence if e.type == "professional"]) >= 2
    assert react.weight == pytest.approx(1.0)


def test_score_permanece_na_faixa_valida(profile):
    matcher = HeuristicMatcher(profile)
    resultado = matcher.match(vaga(
        "Full Stack Developer",
        ["React", "Vue", "TypeScript", "JavaScript", "PHP", "Laravel", "MySQL",
         "PostgreSQL", "REST APIs", "Git", "GraphQL", "VTEX IO"],
    ))
    assert 0.0 <= resultado.score <= 100.0
    assert resultado.required_coverage <= 1.0


# --------------------------------------------------------------------------
# §13 — IA aplicada sobe; ML/MLOps não
# --------------------------------------------------------------------------

def test_vaga_de_ia_aplicada_reconhece_as_duas_fontes(profile):
    resultado = HeuristicMatcher(profile).match(vaga(
        "AI Software Engineer",
        ["Python", "Django", "OpenAI API", "LLMs", "RAG", "Embeddings", "REST APIs"],
    ))
    fortes = " ".join(resultado.strengths).lower()
    praticos = " ".join(resultado.practical_experience).lower()
    assert "openai" in fortes and "python" in fortes, "faltou a evidência profissional"
    assert "rag" in praticos and "embeddings" in praticos, "faltou a evidência de projeto"
    assert resultado.score >= 70


def test_vaga_de_ml_nao_sobe_por_causa_de_openai(profile):
    """§13: treinamento de modelo é outra competência — continua gap."""
    matcher = HeuristicMatcher(profile)
    ml = matcher.match(vaga(
        "Machine Learning Engineer",
        ["PyTorch", "TensorFlow", "MLOps", "SageMaker", "Kubeflow",
         "feature engineering", "model training", "deep learning"],
    ))
    ia_aplicada = matcher.match(vaga(
        "AI Software Engineer",
        ["Python", "OpenAI API", "LLMs", "RAG", "Embeddings", "REST APIs"],
    ))
    assert ml.score < 60, f"vaga de ML inflada: {ml.score} — {ml.reason}"
    assert ml.score < ia_aplicada.score - 15


# --------------------------------------------------------------------------
# §19 — regras já aprovadas continuam valendo
# --------------------------------------------------------------------------

def test_senioridade_continua_neutra(profile):
    matcher = HeuristicMatcher(profile)
    reqs = ["React", "TypeScript", "REST APIs", "Git", "PostgreSQL"]
    base = matcher.match(vaga("Software Engineer", reqs))
    senior = matcher.match(vaga("Senior Software Engineer", reqs))
    assert senior.score == pytest.approx(base.score)


@pytest.mark.parametrize("local,modelo,valido", [
    ("Paraná", WorkModel.REMOTE, True),
    ("São Paulo - SP", WorkModel.HYBRID, True),
    ("São Paulo - SP", WorkModel.ONSITE, True),
    ("Curitiba - PR", WorkModel.HYBRID, False),
])
def test_localizacao_continua_valendo(profile, local, modelo, valido):
    from jobmatch.filters.eligibility import check_eligibility

    job = Job(source="teste", title="Frontend Developer", company="X",
              url="https://exemplo.com/1", raw_location=local, work_model=modelo,
              description="Requisitos e qualificações\nReact\nTypeScript\n")
    assert check_eligibility(job, profile).eligible is valido
