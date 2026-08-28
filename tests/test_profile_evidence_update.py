"""Reconhecimento das tecnologias informadas pelo usuário e regra dos anos.

Duas garantias que não podem regredir:
  1. As tecnologias abaixo têm evidência profissional/prática reconhecida.
  2. "usou profissionalmente" NUNCA vira "tem N anos de experiência".
"""

import pytest

from jobmatch.domain.evidence import CATEGORY_EDUCATION, CATEGORY_PROFESSIONAL
from jobmatch.domain.job import Job, WorkModel
from jobmatch.matching.experience_years import (
    detect_year_requirements,
    describe,
    years_penalty,
)
from jobmatch.matching.heuristic import HeuristicMatcher


def vaga(titulo, requisitos):
    return Job(
        source="teste", title=titulo, company="Empresa X",
        url=f"https://exemplo.com/{abs(hash(titulo + str(requisitos))) % 10**8}",
        raw_location="São Paulo - SP", work_model=WorkModel.REMOTE,
        description="Requisitos e qualificações\n" + "\n".join(requisitos) + "\n",
    )


# --------------------------------------------------------------------------
# Tecnologias com evidência profissional
# --------------------------------------------------------------------------

@pytest.mark.parametrize("nome", ["Java", "Node.js", "GCP", "MongoDB", "Angular"])
def test_tecnologia_tem_evidencia_profissional(profile, nome):
    skill = next(s for s in profile.skills if s.name == nome)
    assert skill.category == CATEGORY_PROFESSIONAL, (
        f"{nome} deveria ter evidência profissional, tem: {skill.evidence_summary()}"
    )
    assert any(e.type == "professional" for e in skill.evidence)


@pytest.mark.parametrize("nome", ["Java", "Node.js", "GCP", "MongoDB", "Angular"])
def test_tecnologia_aparece_como_ponto_forte(profile, nome):
    """Não basta estar no perfil: precisa chegar ao bloco certo da mensagem."""
    resultado = HeuristicMatcher(profile).match(vaga("Software Engineer", [nome, "REST APIs"]))
    assert nome not in resultado.gaps, f"{nome} não pode ser gap"
    assert nome in resultado.strengths, (
        f"{nome} deveria estar em experiência profissional; ficou em "
        f"praticas={resultado.practical_experience} formacao={resultado.education}"
    )


def test_angular_tem_as_tres_fontes_de_evidencia(profile):
    """CoreBiz, projetos próprios e bootcamp da Generation."""
    angular = next(s for s in profile.skills if s.name == "Angular")
    tipos = {e.type for e in angular.evidence}
    assert tipos == {"professional", "project", "bootcamp"}
    fontes = {e.source for e in angular.evidence if e.source}
    assert "exp_corebiz" in fontes
    assert "course_generation" in fontes


def test_vaga_java_spring_mongo_gcp_angular_tem_fit_alto(profile):
    """O caso concreto do enunciado: essa combinação não pode dar fit baixo."""
    resultado = HeuristicMatcher(profile).match(vaga(
        "Desenvolvedor Full Stack",
        ["Angular", "Java", "Spring Boot", "MongoDB", "GCP"],
    ))
    assert resultado.score >= 75, (
        f"fit baixo demais: {resultado.score:.1f} — {resultado.reason}"
    )
    assert not resultado.gaps, f"nada aí deveria ser gap: {resultado.gaps}"


def test_bootcamp_e_categoria_propria(profile):
    """Formação concluída é evidência distinta de 'estou estudando'."""
    spring = next(s for s in profile.skills if s.name == "Spring Boot")
    assert any(e.type == "bootcamp" for e in spring.evidence)
    assert spring.category != CATEGORY_PROFESSIONAL
    # Spring Boot: projeto + bootcamp -> prático, mas com formação declarada.
    assert "formação/bootcamp" in spring.evidence_summary()


def test_formacao_nao_vira_experiencia_profissional(profile):
    """Bootcamp reforça a skill, mas sozinho nunca a torna profissional."""
    from jobmatch.domain.evidence import Evidence, category_of

    assert category_of([Evidence("bootcamp")]) == CATEGORY_EDUCATION
    assert category_of([Evidence("bootcamp"), Evidence("course")]) == CATEGORY_EDUCATION


# --------------------------------------------------------------------------
# §3 — experiência profissional não vira anos inventados
# --------------------------------------------------------------------------

@pytest.mark.parametrize("texto,esperado", [
    ("Requisitos: 3+ anos de experiência com Java", [3]),
    ("Mínimo de 5 anos em backend", [5]),
    ("at least 4 years of experience", [4]),
    ("Java obrigatório, React desejável", []),
    ("2 anos de experiência", [2]),
])
def test_deteccao_de_exigencia_de_anos(texto, esperado):
    assert detect_year_requirements(texto) == esperado


def test_anos_de_mercado_da_empresa_nao_conta_como_requisito():
    """Ruído comum em descrição: 'empresa com 30 anos de mercado'."""
    assert detect_year_requirements("Somos uma empresa com 30 anos de mercado") == []


def test_requisito_sem_anos_nao_penaliza():
    """'Java obrigatório' com evidência profissional não é incompatibilidade."""
    assert years_penalty(detect_year_requirements("Java obrigatório")) == 0.0


def test_exigencia_de_anos_e_lacuna_parcial_nao_gap(profile):
    matcher = HeuristicMatcher(profile)
    sem_anos = matcher.match(vaga("Backend", ["Java", "Spring Boot", "MongoDB"]))
    com_anos = matcher.match(Job(
        source="teste", title="Backend", company="X", url="https://e.com/anos",
        raw_location="São Paulo - SP", work_model=WorkModel.REMOTE,
        description="Requisitos e qualificações\nJava\nSpring Boot\nMongoDB\n"
                    "5+ anos de experiência com Java\n",
    ))
    # A exigência reduz o score, mas Java continua sendo ponto forte.
    assert com_anos.score < sem_anos.score
    assert "Java" in com_anos.strengths
    assert "Java" not in com_anos.gaps
    assert com_anos.year_requirements, "a exigência de tempo precisa ficar visível"
    assert com_anos.years_penalty > 0


def test_penalidade_de_anos_e_limitada():
    """Exigir muitos anos reduz aderência, mas nunca inviabiliza a vaga."""
    from jobmatch.matching.experience_years import PENALIDADE_MAXIMA

    assert years_penalty([15]) <= PENALIDADE_MAXIMA
    assert years_penalty([3]) < years_penalty([8]) <= PENALIDADE_MAXIMA


def test_perfil_nao_declara_anos_de_experiencia(profile):
    """O sistema não pode inventar tempo: nenhuma nota afirma quantidade de anos."""
    import re

    padrao = re.compile(r"\b\d+\s*\+?\s*anos?\b", re.IGNORECASE)
    for skill in profile.skills:
        for ev in skill.evidence:
            assert not padrao.search(ev.note or ""), (
                f"{skill.name} declara tempo de experiência na nota: {ev.note!r}"
            )


def test_descricao_da_exigencia_e_legivel():
    assert describe([5, 3]) == ["5+ anos de experiência", "3+ anos de experiência"]
