"""Regras críticas: senioridade nunca elimina, localização por modelo de trabalho."""

import pytest

from jobmatch.domain.job import Job, WorkModel, detect_seniority, detect_work_model
from jobmatch.filters.eligibility import check_eligibility


def vaga(titulo, local="", modelo=WorkModel.UNKNOWN, descricao="", **kw):
    return Job(
        source="teste",
        title=titulo,
        company="Empresa X",
        url=f"https://exemplo.com/{abs(hash(titulo)) % 10**8}",
        raw_location=local,
        work_model=modelo,
        description=descricao or "Vaga de desenvolvimento de software com React e TypeScript.",
        **kw,
    )


# --------------------------------------------------------------------------
# §2 — senioridade NUNCA elimina
# --------------------------------------------------------------------------

TITULOS_SENIORIDADE = [
    "Senior React Developer",
    "Desenvolvedor Front-end Sênior",
    "Junior React Developer",
    "Desenvolvedor Júnior",
    "Desenvolvedor Full Stack Jr",
    "Desenvolvedor Pleno",
    "Mid-Level Software Engineer",
    "Software Engineer Specialist",
    "Especialista em Desenvolvimento",
    "Staff Software Engineer",
    "Principal Engineer",
    "Tech Lead Frontend",
]


@pytest.mark.parametrize("titulo", TITULOS_SENIORIDADE)
def test_senioridade_nunca_elimina(profile, titulo):
    resultado = check_eligibility(vaga(titulo, "São Paulo - SP", WorkModel.REMOTE), profile)
    assert resultado.eligible, f"'{titulo}' foi eliminada por senioridade: {resultado.reason}"


# Sufixos de senioridade aplicados a um MESMO cargo. Comparar
# "Desenvolvedor" com "Principal Engineer" mediria cobertura de cargo, não
# senioridade — aqui só o nível varia.
SUFIXOS_SENIORIDADE = [
    "Júnior", "Jr", "Pleno", "Sênior", "Senior", "Sr", "Especialista",
    "Staff", "Principal", "Lead", "I", "II", "III",
]


@pytest.mark.parametrize("sufixo", SUFIXOS_SENIORIDADE)
def test_senioridade_nao_muda_o_score(profile, sufixo):
    """Mesmo cargo, só o nível muda → o score não pode cair."""
    from jobmatch.matching.heuristic import HeuristicMatcher

    matcher = HeuristicMatcher(profile)
    descricao = (
        "Requisitos e qualificações\n"
        "React\nTypeScript\nREST APIs\nGit\nPostgreSQL\n"
    )
    base = matcher.match(vaga("Desenvolvedor Front-end", "São Paulo - SP", WorkModel.REMOTE, descricao))
    atual = matcher.match(
        vaga(f"Desenvolvedor Front-end {sufixo}", "São Paulo - SP", WorkModel.REMOTE, descricao)
    )
    assert atual.score == pytest.approx(base.score), (
        f"'{sufixo}' alterou o score: {base.score:.2f} -> {atual.score:.2f}"
    )


def test_senioridade_e_apenas_informativa():
    assert "senior" in detect_seniority("Senior Frontend Developer")
    assert "junior" in detect_seniority("Desenvolvedor Júnior")
    assert detect_seniority("Desenvolvedor Front-end") == "nao informado"


def test_gap_tecnologico_nao_elimina(profile):
    """Antes, '.NET' ou 'Kubernetes' no título descartavam a vaga inteira."""
    for titulo in ["Desenvolvedor .NET / C#", "Engenheiro Kafka e Kubernetes", "Dev Flutter"]:
        assert check_eligibility(vaga(titulo, "São Paulo - SP", WorkModel.REMOTE), profile).eligible


# --------------------------------------------------------------------------
# §4 — localização
# --------------------------------------------------------------------------

@pytest.mark.parametrize("local", [
    "Brasil", "Rio de Janeiro - RJ", "Minas Gerais", "Curitiba - PR",
    "Porto Alegre - RS", "São Paulo - SP", "Florianópolis - SC",
])
def test_remoto_vale_para_o_brasil_inteiro(profile, local):
    resultado = check_eligibility(vaga("Frontend Developer", local, WorkModel.REMOTE), profile)
    assert resultado.eligible, f"Remoto em {local} deveria ser elegível: {resultado.reason}"


@pytest.mark.parametrize("local", [
    "São Paulo - SP", "Santo André - SP", "Osasco - SP", "Barueri - SP",
    "Guarulhos - SP", "Grande São Paulo", "São Bernardo do Campo",
])
def test_hibrido_e_presencial_na_grande_sp(profile, local):
    for modelo in (WorkModel.HYBRID, WorkModel.ONSITE):
        resultado = check_eligibility(vaga("Frontend Developer", local, modelo), profile)
        assert resultado.eligible, f"{modelo.label} em {local}: {resultado.reason}"
        assert resultado.location_confirmed


@pytest.mark.parametrize("local", [
    "Rio de Janeiro - RJ", "Curitiba - PR", "Belo Horizonte - MG",
    "Porto Alegre - RS", "Brasília - DF", "Recife - PE", "Campinas - SP",
])
def test_hibrido_e_presencial_fora_da_grande_sp(profile, local):
    for modelo in (WorkModel.HYBRID, WorkModel.ONSITE):
        resultado = check_eligibility(vaga("Frontend Developer", local, modelo), profile)
        assert not resultado.eligible, f"{modelo.label} em {local} não deveria passar"
        assert resultado.reason == "fora_da_grande_sp"


def test_localizacao_ausente_mantem_a_vaga(profile):
    """§4: na dúvida, manter e marcar como não confirmada — nunca descartar."""
    resultado = check_eligibility(vaga("Frontend Developer", "", WorkModel.HYBRID), profile)
    assert resultado.eligible
    assert resultado.location_confirmed is False


def test_estrangeiro_e_descartado(profile):
    for local in ["United States", "Remote - Canada", "Buenos Aires, Argentina", "Lisboa, Portugal"]:
        resultado = check_eligibility(vaga("Frontend Developer", local, WorkModel.REMOTE), profile)
        assert not resultado.eligible
        assert resultado.reason == "estrangeiro"


def test_sp_nao_casa_por_substring(profile):
    """Regressão: `"sp" in "jaspion"` era True na versão anterior."""
    resultado = check_eligibility(vaga("Frontend Developer", "Jaspion - RJ", WorkModel.ONSITE), profile)
    assert not resultado.eligible


# --------------------------------------------------------------------------
# §3 — modelo de trabalho
# --------------------------------------------------------------------------

def test_detecta_modelo_de_trabalho():
    assert detect_work_model("100% Remoto") is WorkModel.REMOTE
    assert detect_work_model("Híbrido - São Paulo") is WorkModel.HYBRID
    assert detect_work_model("Presencial") is WorkModel.ONSITE
    assert detect_work_model("") is WorkModel.UNKNOWN
    # Híbrido vence remoto quando os dois aparecem.
    assert detect_work_model("Trabalho híbrido, 2x remoto por semana") is WorkModel.HYBRID


def test_presencial_em_sp_nao_e_penalizado(profile):
    """Presencial em SP continua válido e não leva penalidade forte (§10)."""
    from jobmatch.matching.heuristic import HeuristicMatcher

    matcher = HeuristicMatcher(profile)
    descricao = "Requisitos: React, TypeScript, Node.js, PostgreSQL."
    remoto = matcher.match(vaga("Software Engineer", "São Paulo - SP", WorkModel.REMOTE, descricao))
    presencial = matcher.match(vaga("Software Engineer", "São Paulo - SP", WorkModel.ONSITE, descricao))
    assert presencial.score > 0
    # A diferença é só o bônus de preferência, nunca uma penalidade.
    assert remoto.score - presencial.score <= profile.work_model_bonus + 0.01
