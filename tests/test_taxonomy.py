"""Taxonomia de infraestrutura e teto de evidência não profissional.

Duas propriedades que não podem regredir:
  1. Git NÃO significa observabilidade.
  2. Evidência não profissional forte continua forte, mas não fica
     numericamente idêntica a experiência profissional.
"""

import pytest

from jobmatch.domain.evidence import (
    FREELANCE_CAP,
    NON_PROFESSIONAL_CAP,
    PROFESSIONAL_CAP,
    Evidence,
    combine,
)
from jobmatch.domain.job import Job, WorkModel
from jobmatch.domain.profile import _build_profile
from jobmatch.matching.heuristic import HeuristicMatcher


def vaga(titulo, requisitos):
    return Job(
        source="teste", title=titulo, company="Empresa X",
        url=f"https://exemplo.com/{abs(hash(titulo + str(requisitos))) % 10**8}",
        raw_location="São Paulo - SP", work_model=WorkModel.REMOTE,
        description="Requisitos e qualificações\n" + "\n".join(requisitos) + "\n",
    )


def hit(resultado, nome):
    return next((h for h in resultado.hits if h.required == nome), None)


# --------------------------------------------------------------------------
# §4 — taxonomia de infraestrutura
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tech", ["Datadog", "Grafana", "Prometheus", "OpenTelemetry"])
def test_git_nao_implica_observabilidade(profile, tech):
    """Regressão: Git (profissional) dava 55% de crédito para observabilidade."""
    resultado = HeuristicMatcher(profile).match(vaga(
        "Site Reliability Engineer", [tech, "Grafana", "Prometheus", "Datadog", "alertas"],
    ))
    obs = hit(resultado, "Observabilidade")
    assert obs is not None, "observabilidade deveria ser detectada"
    assert obs.coverage == 0.0, (
        f"observabilidade recebeu {obs.coverage:.2f} via '{obs.matched}' "
        f"(grupo {obs.group}) — deveria ser gap"
    )
    assert "Observabilidade" in resultado.gaps
    assert not any("observabilidade" in p.lower() for p in resultado.partial_matches)


def test_git_nao_implica_iac_nem_nuvem(profile):
    resultado = HeuristicMatcher(profile).match(vaga(
        "Platform Engineer", ["Terraform", "AWS", "CloudFormation", "Ansible", "Pulumi"],
    ))
    iac = hit(resultado, "Terraform / IaC")
    assert iac is not None and iac.coverage == 0.0, "IaC não pode herdar peso de Git"
    assert not any(h.matched in ("Git", "GitHub") for h in resultado.hits if h.coverage > 0)


def test_observabilidade_transfere_dentro_do_grupo():
    """Quem tem Grafana recebe crédito relevante para Datadog."""
    perfil = _build_profile({
        "identity": {"name": "Teste"},
        "skills": [{
            "name": "Grafana", "family": "observability", "aliases": ["grafana"],
            "evidence": [{"type": "professional"}],
        }],
        "roles": {"sre": {"label": "SRE", "keywords": ["site reliability"]}},
        "tech_signals": ["engineer", "devops"],
        "skill_groups": {
            "observability": {
                "transfer_factor": 0.60,
                "members": ["grafana", "datadog", "prometheus", "observabilidade"],
            }
        },
        "preferences": {"work_models": {"remote": 3}, "work_model_bonus": 4, "emergent_bonus": 0},
    })
    resultado = HeuristicMatcher(perfil).match(vaga("Site Reliability Engineer", ["Datadog"]))
    datadog = hit(resultado, "Datadog") or hit(resultado, "Observabilidade")
    assert datadog is not None
    assert datadog.coverage > 0.4, "Grafana → Datadog deveria transferir com força"
    assert datadog.transferable


def test_docker_vira_kubernetes_fraco_nunca_ponto_forte(profile):
    resultado = HeuristicMatcher(profile).match(vaga(
        "DevOps Engineer", ["Kubernetes", "Helm", "Docker", "CI/CD", "Linux"],
    ))
    k8s = hit(resultado, "Kubernetes")
    assert k8s is not None
    assert k8s.transferable, "Kubernetes não pode ser match direto"
    assert 0.0 < k8s.coverage <= 0.30, f"transferência forte demais: {k8s.coverage:.2f}"
    assert k8s.group == "orchestration"
    assert "Kubernetes" not in resultado.strengths


def test_github_actions_transfere_para_ci_cd(profile):
    resultado = HeuristicMatcher(profile).match(vaga(
        "DevOps Engineer", ["Jenkins", "pipelines", "deploy automatizado"],
    ))
    jenkins = hit(resultado, "Jenkins")
    assert jenkins is not None, f"Jenkins não detectado: {[h.required for h in resultado.hits]}"
    assert jenkins.transferable
    assert 0.3 <= jenkins.coverage <= 0.7, f"transferência incoerente: {jenkins.coverage:.2f}"
    assert jenkins.group == "cicd"


def test_familias_de_infra_estao_separadas(profile):
    """Nenhuma família pode misturar controle de versão com observabilidade."""
    familias = {s.family for s in profile.skills}
    assert {"version_control", "containers", "cicd"} <= familias
    for skill in profile.skills_da_familia("version_control"):
        assert skill.name in ("Git", "GitHub")
    assert profile.melhor_do_grupo("observability") is None
    assert profile.melhor_do_grupo("iac") is None
    # "cloud" TEM skill própria (AWS/Azure em estudo, GCP hands-on) — o ponto
    # que importa aqui é que nenhuma delas veio de Git/GitHub.
    melhor_cloud = profile.melhor_do_grupo("cloud")
    assert melhor_cloud is not None
    assert melhor_cloud.name not in ("Git", "GitHub")


# --------------------------------------------------------------------------
# §9 — teto de evidência não profissional
# --------------------------------------------------------------------------

def test_professional_chega_a_um():
    assert combine([Evidence("professional")]) == pytest.approx(PROFESSIONAL_CAP)
    assert combine([
        Evidence("professional"), Evidence("project"), Evidence("study"),
    ]) == pytest.approx(1.00)


def test_nao_profissional_respeita_o_teto():
    acumulado = combine([Evidence("production_project"), Evidence("course"), Evidence("study")])
    assert acumulado < 1.00
    assert acumulado <= NON_PROFESSIONAL_CAP + 1e-9
    assert acumulado == pytest.approx(NON_PROFESSIONAL_CAP)


def test_freelance_tem_teto_proprio():
    assert combine([Evidence("freelance"), Evidence("project")]) == pytest.approx(FREELANCE_CAP)


def test_projeto_forte_continua_competitivo():
    """O teto não pode transformar projeto sério em evidência fraca."""
    forte = combine([
        Evidence("production_project"), Evidence("hands_on"),
        Evidence("course"), Evidence("study"),
    ])
    assert forte >= 0.88, f"projeto forte foi desvalorizado: {forte}"
    assert forte > combine([Evidence("study")]) + 0.4


def test_ordenacao_completa_das_evidencias():
    profissional = combine([Evidence("professional")])
    projeto_forte = combine([
        Evidence("production_project"), Evidence("hands_on"),
        Evidence("course"), Evidence("study"),
    ])
    projeto = combine([Evidence("project")])
    curso = combine([Evidence("course")])
    nenhuma = combine([])
    assert profissional > projeto_forte > projeto > curso > nenhuma


def test_rag_fica_no_teto_nao_profissional(profile):
    """RAG tem projeto em produção + curso + estudo: forte, mas não profissional."""
    rag = next(s for s in profile.skills if s.name == "RAG")
    assert rag.weight == pytest.approx(NON_PROFESSIONAL_CAP)
    assert rag.category == "practical"

    react = next(s for s in profile.skills if s.name == "React")
    assert react.weight > rag.weight, "profissional precisa pesar mais que projeto"


def test_vaga_de_genai_continua_fortemente_compativel(profile):
    """O teto não pode derrubar vagas de GenAI — é o ponto forte do perfil."""
    resultado = HeuristicMatcher(profile).match(vaga(
        "AI Engineer",
        ["Python", "RAG", "LLMs", "Embeddings", "Vector Database", "OpenAI API", "REST APIs"],
    ))
    assert resultado.score >= 75, f"vaga de GenAI caiu demais: {resultado.score}"
    assert "RAG" in resultado.practical_experience
