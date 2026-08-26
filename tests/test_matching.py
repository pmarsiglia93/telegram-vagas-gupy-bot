"""Scoring: descrição acima do título, níveis de evidência, gaps nomeados."""

import pytest

from jobmatch.domain.job import Job, WorkModel, split_sections
from jobmatch.domain.match import classify
from jobmatch.matching.heuristic import HeuristicMatcher

DESCRICAO_COMPLETA = """
Sobre a vaga
Buscamos uma pessoa para atuar no time de produto.

Responsabilidades e atribuições
Desenvolver e manter aplicações web.
Integrar APIs REST.

Requisitos e qualificações
React
TypeScript
Node.js
PostgreSQL
Docker

Diferenciais
AWS
Kubernetes
"""


def vaga(titulo="Software Engineer", descricao=DESCRICAO_COMPLETA, modelo=WorkModel.REMOTE):
    return Job(
        source="teste",
        title=titulo,
        company="Empresa X",
        url="https://exemplo.com/1",
        raw_location="São Paulo - SP",
        work_model=modelo,
        description=descricao,
    )


@pytest.fixture
def matcher(profile):
    return HeuristicMatcher(profile)


# --------------------------------------------------------------------------
# §25 — matching pela descrição, não pelo título
# --------------------------------------------------------------------------

def test_titulo_generico_com_stack_compativel_pontua(matcher):
    """'Software Engineer' não cita tecnologia nenhuma no título."""
    resultado = matcher.match(vaga())
    assert resultado.score >= 60, f"score baixo demais: {resultado.score} ({resultado.reason})"
    nomes = " ".join(resultado.strengths).lower()
    assert "react" in nomes and "typescript" in nomes


def test_descricao_supera_o_titulo(matcher):
    """Título idêntico, descrições diferentes → scores diferentes."""
    compativel = matcher.match(vaga(descricao=DESCRICAO_COMPLETA))
    incompativel = matcher.match(vaga(descricao="""
Requisitos e qualificações
Salesforce
ABAP
SAP
COBOL
"""))
    assert compativel.score > incompativel.score


def test_gaps_sao_nomeados_nao_eliminatorios(matcher):
    resultado = matcher.match(vaga())
    assert "AWS" in resultado.gaps or "AWS" in " ".join(resultado.partial_matches)
    assert resultado.score > 0


def test_tecnologia_transferivel_da_familia(matcher):
    """A vaga pede Vue; o perfil tem React — crédito parcial, não gap."""
    resultado = matcher.match(vaga(descricao="""
Requisitos e qualificações
Svelte
"""))
    # Svelte não está no vocabulário: vira ausência silenciosa, não erro.
    assert resultado.score >= 0

    resultado_vue = matcher.match(vaga(descricao="""
Requisitos e qualificações
Vue.js
Nuxt
"""))
    assert resultado_vue.strengths or resultado_vue.partial_matches


def test_projeto_nao_vira_experiencia_profissional(matcher):
    """RAG/LLM/ChromaDB têm evidência de PROJETO — contam, mas não como emprego."""
    resultado = matcher.match(vaga(titulo="AI Engineer", descricao="""
Requisitos e qualificações
RAG
LLM
Embeddings
ChromaDB
"""))
    praticos = " ".join(resultado.practical_experience).lower()
    assert "rag" in praticos, "evidência de projeto precisa aparecer na análise"
    assert "chromadb" in praticos

    # A regra que não pode regredir: projeto nunca é apresentado como
    # experiência profissional.
    fortes = " ".join(resultado.strengths).lower()
    for tech in ("rag", "chromadb", "embeddings"):
        assert tech not in fortes, f"'{tech}' é evidência de projeto, não experiência profissional"


def test_curso_aparece_como_conhecimento_relacionado(matcher):
    """Skill sustentada só por curso vai para `related_knowledge`, com o rótulo."""
    resultado = matcher.match(vaga(titulo="Analista de Segurança", descricao="""
Requisitos e qualificações
Cibersegurança
OWASP
"""))
    relacionados = " ".join(resultado.related_knowledge).lower()
    assert "ciberseguranca" in relacionados or "cibersegurança" in relacionados
    assert "curso" in relacionados, "o rótulo da evidência precisa acompanhar a skill"
    assert not any("seguran" in s.lower() for s in resultado.strengths)


def test_sem_descricao_nao_zera_o_score(matcher):
    """Falta de dado não é sinal negativo (§9)."""
    resultado = matcher.match(vaga(descricao=""))
    assert resultado.score > 0
    assert "descricao_indisponivel" in resultado.notes


# --------------------------------------------------------------------------
# Seções da descrição
# --------------------------------------------------------------------------

def test_split_sections_separa_requisitos_de_diferenciais():
    secoes = split_sections(DESCRICAO_COMPLETA)
    assert "React" in secoes["requirements"]
    assert "AWS" in secoes["nice_to_have"]
    assert "React" not in secoes.get("nice_to_have", "")


def test_split_sections_com_cabecalho_grudado():
    """A Gupy concatena cabeçalho e texto sem separador."""
    secoes = split_sections("Sobre a empresaSomos X.Requisitos e qualificaçõesReact e TypeScript.")
    assert "requirements" in secoes
    assert "React" in secoes["requirements"]


# --------------------------------------------------------------------------
# Classificação
# --------------------------------------------------------------------------

@pytest.mark.parametrize("score,esperado", [
    (95, "Excelente compatibilidade"),
    (85, "Alta compatibilidade"),
    (75, "Boa compatibilidade"),
    (65, "Compatibilidade razoável"),
    (55, "Possível oportunidade"),
    (30, "Baixa compatibilidade"),
])
def test_faixas_de_classificacao(score, esperado):
    assert classify(score)[0] == esperado
