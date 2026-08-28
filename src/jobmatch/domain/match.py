"""Resultado da análise de aderência de uma vaga."""

from __future__ import annotations

from dataclasses import dataclass, field

# Faixas de classificação (§11). Score é ordenação/priorização/explicação —
# não é filtro de envio nesta versão.
_FAIXAS: list[tuple[int, str, str]] = [
    (90, "Excelente compatibilidade", "🔥"),
    (80, "Alta compatibilidade", "🟢"),
    (70, "Boa compatibilidade", "🟢"),
    (60, "Compatibilidade razoável", "🟡"),
    (50, "Possível oportunidade", "🟡"),
    (0, "Baixa compatibilidade", "⚪"),
]


def classify(score: float) -> tuple[str, str]:
    """Retorna (classificação, emoji) para um score 0-100."""
    valor = max(0, min(100, int(round(score))))
    for minimo, rotulo, emoji in _FAIXAS:
        if valor >= minimo:
            return rotulo, emoji
    return _FAIXAS[-1][1], _FAIXAS[-1][2]


@dataclass
class SkillHit:
    """Uma skill exigida pela vaga, confrontada com o perfil."""

    required: str            # como aparece na vaga
    matched: str = ""        # skill do perfil que cobre (vazio = gap)
    level: str = ""          # tipo da evidência mais forte
    coverage: float = 0.0    # 0.0 (gap) .. 1.0 (cobertura total)
    transferable: bool = False
    critical: bool = True    # veio de requisitos obrigatórios?
    # Categoria da evidência: professional | practical | learned.
    # É o que separa "trabalhei com isso" de "implementei num projeto" e de
    # "estudei" — a distinção sobrevive até a mensagem do Telegram.
    category: str = ""
    evidence: str = ""       # resumo legível ("projeto próprio, em estudo")
    group: str = ""          # grupo da taxonomia que originou a transferência


@dataclass
class MatchResult:
    """Saída consolidada da análise. Preenchida por heurística, semântica e/ou LLM."""

    score: float = 0.0
    classification: str = ""
    emoji: str = "⚪"
    engine: str = "heuristic"          # heuristic | semantic | llm
    job_type: str = ""                 # Frontend, Full Stack, ...
    # Blocos de saída, separados por tipo de evidência (§11).
    strengths: list[str] = field(default_factory=list)            # 1. profissional
    practical_experience: list[str] = field(default_factory=list)  # 2. projetos / hands-on
    education: list[str] = field(default_factory=list)             # 3. formação / bootcamp
    related_knowledge: list[str] = field(default_factory=list)     # 4. estudo em andamento
    partial_matches: list[str] = field(default_factory=list)       # transferíveis
    gaps: list[str] = field(default_factory=list)                  # 5. sem evidência
    relevant_experiences: list[str] = field(default_factory=list)
    reason: str = ""
    # Diagnóstico interno, não vai para o Telegram.
    required_coverage: float = 0.0
    semantic_similarity: float = 0.0
    heuristic_score: float = 0.0
    emergent_bonus: float = 0.0
    # Exigências de tempo ("3+ anos de Java") encontradas na vaga. São tratadas
    # à parte da cobertura técnica: ter usado Java profissionalmente NÃO prova
    # três anos de Java, e o perfil não declara tempo de experiência nenhum.
    year_requirements: list[str] = field(default_factory=list)
    years_penalty: float = 0.0
    hits: list[SkillHit] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def finalize(self) -> "MatchResult":
        self.score = max(0.0, min(100.0, self.score))
        self.classification, self.emoji = classify(self.score)
        return self
