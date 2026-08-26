"""Calibração do modelo de evidência (§13 e §14).

Regra que sustenta tudo: conhecimento real conta, sem nunca ser apresentado
como experiência profissional formal.
"""

import pytest

from jobmatch.domain.evidence import Evidence, combine
from jobmatch.domain.job import Job, WorkModel
from jobmatch.domain.profile import Skill, _build_profile
from jobmatch.matching.heuristic import HeuristicMatcher


def vaga(titulo, requisitos, modelo=WorkModel.REMOTE, diferenciais=""):
    descricao = "Requisitos e qualificações\n" + "\n".join(requisitos)
    if diferenciais:
        descricao += "\n\nDiferenciais\n" + diferenciais
    return Job(
        source="teste",
        title=titulo,
        company="Empresa X",
        url=f"https://exemplo.com/{abs(hash(titulo + descricao)) % 10**8}",
        raw_location="São Paulo - SP",
        work_model=modelo,
        description=descricao,
    )


# --------------------------------------------------------------------------
# §13 — hierarquia de evidência
# --------------------------------------------------------------------------

# Cinco skills para que a vaga de teste atinja confiança total na medição —
# assim o que varia entre os perfis é só o tipo de evidência.
SKILLS_TESTE = ["RAG", "Embeddings", "LLMs", "Vector databases", "Python"]


def _perfil_com_evidencia(tipos: list[str] | None):
    """Perfil mínimo com as mesmas skills, variando só a evidência."""
    skills = []
    if tipos:
        skills = [
            {
                "name": nome,
                "family": "ai",
                "aliases": [nome.lower()],
                "evidence": [{"type": t} for t in tipos],
            }
            for nome in SKILLS_TESTE
        ]
    return _build_profile({
        "identity": {"name": "Teste"},
        "skills": skills,
        "roles": {"ai": {"label": "AI Engineer", "keywords": ["ai engineer"]}},
        "tech_signals": ["desenvolvedor", "engineer"],
        "skill_groups": {"rag": {"transfer_factor": 0.75, "members": ["rag", "embeddings"]}},
        "preferences": {"work_models": {"remote": 3}, "work_model_bonus": 4, "emergent_bonus": 0},
    })


def test_hierarquia_de_evidencia_A_maior_B_maior_C_maior_D():
    """A(professional) > B(project) > C(study) > D(nenhuma) — todos com crédito real."""
    job = vaga("AI Engineer", SKILLS_TESTE)
    scores = {}
    for rotulo, tipos in [
        ("A", ["professional"]),
        ("B", ["project"]),
        ("C", ["study"]),
        ("D", None),
    ]:
        scores[rotulo] = HeuristicMatcher(_perfil_com_evidencia(tipos)).match(job).score

    assert scores["A"] > scores["B"] > scores["C"] > scores["D"], scores

    # B e C precisam de crédito REAL, não simbólico: a distância entre estudo e
    # nenhuma evidência tem de ser comparável à distância entre estudo e projeto.
    assert scores["C"] - scores["D"] >= 8, f"estudo recebeu crédito irrelevante: {scores}"
    assert scores["B"] - scores["D"] >= 25, f"projeto recebeu crédito irrelevante: {scores}"


def test_multiplas_evidencias_somam_sem_alcancar_experiencia_profissional():
    job = vaga("AI Engineer", SKILLS_TESTE)
    so_estudo = HeuristicMatcher(_perfil_com_evidencia(["study"])).match(job).score
    acumulado = HeuristicMatcher(
        _perfil_com_evidencia(["study", "project", "hands_on", "course"])
    ).match(job).score
    profissional = HeuristicMatcher(_perfil_com_evidencia(["professional"])).match(job).score

    assert acumulado > so_estudo, "acumular evidências precisa valer alguma coisa"
    assert acumulado < profissional, "acumular evidências não pode simular emprego formal"


def test_peso_combinado_respeita_o_teto():
    assert combine([Evidence("study")]) == pytest.approx(0.40)
    assert combine([Evidence("professional")]) == pytest.approx(1.00)
    # Bônus por múltiplas evidências é limitado.
    muitas = combine([Evidence("project"), Evidence("study"), Evidence("hands_on"), Evidence("course")])
    assert muitas <= 0.75 + 0.15 + 1e-9


def test_skill_sem_evidencia_valida_nao_quebra():
    skill = Skill(name="X", family="ai", aliases=("x",), evidence=())
    assert skill.weight == 0.0
    assert skill.level == "study"


# --------------------------------------------------------------------------
# §14 — casos obrigatórios
# --------------------------------------------------------------------------

def test_caso_1_ai_engineer_com_rag(profile):
    """Vaga de AI Engineer coberta por evidência de projeto deve pontuar alto."""
    resultado = HeuristicMatcher(profile).match(vaga(
        "AI Engineer",
        ["Python", "RAG", "LLMs", "Embeddings", "Vector Database", "APIs REST"],
    ))
    assert resultado.score >= 70, f"falso negativo em AI Engineer: {resultado.score} — {resultado.reason}"
    praticos = " ".join(resultado.practical_experience).lower()
    assert "rag" in praticos and "embeddings" in praticos
    assert resultado.emergent_bonus > 0, "competência emergente coberta deveria bonificar"


def test_caso_2_automation_engineer_com_n8n(profile):
    """n8n via curso + hands-on não pode virar baixa compatibilidade."""
    resultado = HeuristicMatcher(profile).match(vaga(
        "Automation Engineer",
        ["n8n", "APIs REST", "LLMs", "Integração de APIs", "Automação"],
    ))
    assert resultado.score >= 60, f"falso negativo em automação: {resultado.score} — {resultado.reason}"
    citados = " ".join(
        resultado.related_knowledge + resultado.practical_experience + resultado.strengths
    ).lower()
    assert "n8n" in citados
    assert "n8n" not in " ".join(resultado.strengths).lower(), \
        "n8n é curso + hands-on, nunca experiência profissional"


def test_caso_3_zapier_e_transferivel(profile):
    """Zapier não está no perfil, mas Make e n8n estão no mesmo grupo."""
    resultado = HeuristicMatcher(profile).match(vaga(
        "Automation Engineer", ["Zapier", "REST APIs", "Automação"],
    ))
    transferiveis = " ".join(resultado.partial_matches).lower()
    assert "zapier" in transferiveis, f"Zapier deveria ser transferível: {resultado.gaps}"
    assert "zapier" not in " ".join(resultado.gaps).lower()


def test_caso_4_pinecone_nao_e_gap_completo(profile):
    """Vaga pede Pinecone; o perfil tem ChromaDB — compatibilidade parcial."""
    resultado = HeuristicMatcher(profile).match(vaga(
        "AI Engineer", ["Pinecone", "RAG", "Embeddings"],
    ))
    transferiveis = " ".join(resultado.partial_matches).lower()
    assert "pinecone" in transferiveis, f"Pinecone virou gap: {resultado.gaps}"
    assert "pinecone" not in " ".join(resultado.gaps).lower()
    hit = next(h for h in resultado.hits if h.required == "Pinecone")
    assert hit.group == "rag"
    assert hit.coverage > 0.4, "transferência dentro do grupo rag deveria valer bastante"


def test_caso_5_react_profissional_pesa_mais_que_academico(profile):
    """Experiência profissional continua sendo a evidência mais forte."""
    matcher = HeuristicMatcher(profile)
    job = vaga("Frontend Developer", ["React", "TypeScript", "CSS", "HTML", "REST APIs"])
    resultado = matcher.match(job)

    assert "React" in resultado.strengths
    hit_react = next(h for h in resultado.hits if h.required == "React")
    hit_java = next(
        (h for h in matcher.match(vaga("Backend", ["Java", "Spring Boot"])).hits
         if h.required == "Java"), None,
    )
    assert hit_java is not None
    assert hit_react.coverage > hit_java.coverage, \
        "React (profissional) tem de pesar mais que Java (bootcamp + projeto)"
    assert hit_react.category == "professional"
    assert hit_java.category != "professional"


def test_fastapi_e_transferivel_de_python(profile):
    """§12: FastAPI com Python/Django no perfil não é gap completo."""
    resultado = HeuristicMatcher(profile).match(vaga(
        "Backend Engineer", ["FastAPI", "Python", "PostgreSQL"],
    ))
    assert "FastAPI" not in resultado.gaps
    assert any("fastapi" in p.lower() for p in resultado.partial_matches)


# --------------------------------------------------------------------------
# §18 — não superestimar
# --------------------------------------------------------------------------

def test_vaga_de_sre_nao_infla_por_causa_de_docker(profile):
    """Docker é evidência real, mas não pode fazer uma vaga de SRE pontuar alto."""
    matcher = HeuristicMatcher(profile)
    sre = matcher.match(vaga(
        "Site Reliability Engineer",
        ["Kubernetes", "Terraform", "AWS", "Helm", "ArgoCD", "Observabilidade"],
    ))
    frontend = matcher.match(vaga(
        "Frontend Developer", ["React", "TypeScript", "CSS", "HTML", "REST APIs"],
    ))
    assert sre.score < 55, f"vaga de SRE inflada: {sre.score} — {sre.reason}"
    assert sre.score < frontend.score - 20


def test_stack_totalmente_alheia_continua_baixa(profile):
    resultado = HeuristicMatcher(profile).match(vaga(
        "Consultor SAP", ["SAP", "ABAP", "Salesforce", "COBOL", "Power BI"],
    ))
    assert resultado.score < 50, f"stack alheia não deveria pontuar: {resultado.score}"


def test_poucos_requisitos_reduzem_a_confianca(profile):
    """Descrição magra não pode produzir a mesma certeza que uma detalhada."""
    matcher = HeuristicMatcher(profile)
    magra = matcher.match(vaga("Frontend Developer", ["React"]))
    detalhada = matcher.match(vaga(
        "Frontend Developer",
        ["React", "TypeScript", "CSS", "HTML", "REST APIs", "Git"],
    ))
    assert detalhada.score > magra.score
    assert any(n.startswith("poucos_requisitos") for n in magra.notes)


def test_bonus_emergente_exige_cobertura_real(profile):
    """Sem tecnologia emergente na vaga, não há bônus."""
    resultado = HeuristicMatcher(profile).match(vaga(
        "Frontend Developer", ["React", "CSS", "HTML"],
    ))
    assert resultado.emergent_bonus == 0.0
