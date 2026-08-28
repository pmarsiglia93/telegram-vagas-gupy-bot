"""Matcher heurístico: aderência entre requisitos reais da vaga e o perfil.

A pergunta central não é "tem experiência profissional? sim/não", e sim
**qual é a evidência de competência?**. Cada skill do perfil carrega uma ou
mais evidências (profissional, projeto, hands-on, curso, estudo) e cada uma
vale um peso diferente — mas nenhuma vale zero.

Isso resolve o problema de uma vaga de AI Engineer pedindo RAG, LLMs e
embeddings ser classificada como baixa compatibilidade só porque essas
tecnologias não apareceram num emprego formal.

O que continua valendo:
  1. Lê a DESCRIÇÃO, não só o título.
  2. Separa requisito obrigatório de desejável.
  3. Estudo nunca vira experiência profissional — muda o peso E o rótulo.
  4. Crédito parcial por tecnologia vizinha da mesma família/grupo.
  5. Gaps são nomeados, nunca eliminatórios.
  6. Senioridade tem peso ZERO.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.evidence import (
    CATEGORY_EDUCATION,
    CATEGORY_LEARNED,
    CATEGORY_PRACTICAL,
    CATEGORY_PROFESSIONAL,
)
from ..domain.job import Job, WorkModel
from ..domain.match import MatchResult, SkillHit
from ..domain.profile import Profile
from ..domain.text import contains_phrase, normalize
from .experience_years import describe, detect_year_requirements, years_penalty
from .vocabulary import build_lookup, display_for

# Pesos dos componentes do score. Renormalizados quando um componente falta
# (ex.: sem embeddings, `semantic` some e os outros três se redistribuem).
COMPONENT_WEIGHTS: dict[str, float] = {
    "core": 0.50,      # requisitos obrigatórios cobertos
    "nice": 0.12,      # diferenciais cobertos
    "role": 0.18,      # aderência de cargo
    "semantic": 0.20,  # similaridade semântica perfil × vaga
}

# Peso, em "requisitos equivalentes", do palpite a priori sobre a vaga.
# A cobertura medida é encolhida na direção de um prior (a aderência de cargo)
# com força fixa: com 1 requisito legível o prior domina, com 10 ele quase
# some. Sem isso, "100% de aderência" sobre 2 requisitos vale tanto quanto
# sobre 10 — e uma descrição magra vira score inflado (§18).
PRIOR_PSEUDO_REQUIREMENTS = 2.0

# Cobertura mínima para considerar um grupo emergente realmente coberto.
EMERGENT_COVERAGE_MIN = 0.5
# Quantos grupos emergentes cobertos valem o bônus cheio.
EMERGENT_GROUPS_FULL = 2


@dataclass
class DetectedTech:
    display: str
    family: str
    critical: bool
    in_profile: bool
    groups: set[str] = field(default_factory=set)


def _detect(texto: str, profile: Profile, externo: dict, critical: bool) -> dict[str, DetectedTech]:
    """Encontra tecnologias citadas num trecho de texto.

    Três passes, nesta ordem de precedência:
      1. aliases do perfil       -> o candidato conhece de fato
      2. vocabulário externo     -> tecnologia conhecida que ele não tem
      3. membros da taxonomia    -> vizinhas (Pinecone, Zapier, FastAPI...)

    O terceiro passe é o que permite reconhecer uma tecnologia vizinha como
    compatibilidade parcial em vez de ela passar despercebida.
    """
    achados: dict[str, DetectedTech] = {}
    if not texto:
        return achados
    alvo = normalize(texto)

    for alias, skill in profile.alias_index.items():
        if len(alias) < 2:
            continue
        if contains_phrase(alvo, alias):
            achados.setdefault(skill.name, DetectedTech(
                skill.name, skill.family, critical, True, profile.groups_of(skill),
            ))

    for alias, (nome, familia) in externo.items():
        if contains_phrase(alvo, alias):
            grupos = profile.groups_of_term(alias) | profile.groups_of_term(nome)
            achados.setdefault(nome, DetectedTech(nome, familia, critical, False, grupos))

    for termo, grupos in profile.group_index.items():
        if len(termo) < 2 or termo in profile.alias_index or termo in externo:
            continue
        if contains_phrase(alvo, termo):
            nome = display_for(termo)
            achados.setdefault(nome, DetectedTech(nome, "", critical, False, set(grupos)))

    return achados


def _role_affinity(job: Job, profile: Profile) -> tuple[float, str]:
    """0..1 de aderência de cargo + rótulo do tipo de vaga.

    Título casando com cargo-alvo vale 1.0; só a descrição casando vale 0.6;
    nada casando ainda vale 0.35 — porque títulos genéricos ("Analista de
    Sistemas III") não podem eliminar a vaga (§5).
    """
    titulo = normalize(job.title)
    for role in profile.roles:
        if any(contains_phrase(titulo, kw) for kw in role.keywords):
            return 1.0, role.label
    corpo = normalize(job.description[:3000])
    for role in profile.roles:
        if any(contains_phrase(corpo, kw) for kw in role.keywords):
            return 0.6, role.label
    return 0.35, "Tecnologia"


def _work_model_bonus(job: Job, profile: Profile) -> float:
    """Bônus não-negativo. Presencial não é penalizado, só ganha menos (§3)."""
    prefs = profile.work_model_preference or {"remote": 3, "hybrid": 2, "onsite": 1, "unknown": 1}
    maximo = max(prefs.values()) or 1
    peso = prefs.get(job.work_model.value, prefs.get("unknown", 1))
    return profile.work_model_bonus * (peso / maximo)


def compose_score(componentes: dict[str, float]) -> float:
    """Combina componentes 0..1 em score 0..100, renormalizando o que existe."""
    presentes = {k: v for k, v in componentes.items() if v is not None}
    total_peso = sum(COMPONENT_WEIGHTS[k] for k in presentes if k in COMPONENT_WEIGHTS)
    if total_peso <= 0:
        return 0.0
    soma = sum(COMPONENT_WEIGHTS[k] * max(0.0, min(1.0, v)) for k, v in presentes.items() if k in COMPONENT_WEIGHTS)
    return 100.0 * soma / total_peso


class HeuristicMatcher:
    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self._externo = build_lookup(profile.alias_index)
        self._por_nome = {s.name: s for s in profile.skills}

    def match(self, job: Job, semantic_similarity: float | None = None) -> MatchResult:
        obrigatorios = _detect(job.requirements_text, self.profile, self._externo, True)
        desejaveis = _detect(job.nice_to_have_text, self.profile, self._externo, False)
        # Tecnologias do título e das tags contam como obrigatórias.
        titulo_tags = _detect(job.title + " " + " ".join(job.tags), self.profile, self._externo, True)
        obrigatorios.update({k: v for k, v in titulo_tags.items() if k not in obrigatorios})
        # Uma tech listada como obrigatória não é reclassificada como desejável.
        desejaveis = {k: v for k, v in desejaveis.items() if k not in obrigatorios}

        hits_core = [self._avaliar(t) for t in obrigatorios.values()]
        hits_nice = [self._avaliar(t) for t in desejaveis.values()]

        core = self._cobertura(hits_core)
        nice = self._cobertura(hits_nice)
        role, job_type = _role_affinity(job, self.profile)

        resultado = MatchResult(engine="heuristic", job_type=job_type)
        resultado.hits = hits_core + hits_nice
        resultado.required_coverage = core if core is not None else 0.0

        componentes: dict[str, float] = {"role": role}

        # A aderência medida é encolhida na direção do prior de cargo, com peso
        # proporcional a quantos requisitos realmente foram lidos. Com zero
        # requisitos sobra só o prior — o mesmo caminho da descrição ausente,
        # sem penalizar a vaga por um dado que a fonte não forneceu (§9).
        prior = role * 0.6
        n = float(len(hits_core))
        componentes["core"] = (
            ((core or 0.0) * n + prior * PRIOR_PSEUDO_REQUIREMENTS)
            / (n + PRIOR_PSEUDO_REQUIREMENTS)
        )
        if core is None:
            resultado.notes.append("descricao_indisponivel")
        elif n < 2 * PRIOR_PSEUDO_REQUIREMENTS:
            resultado.notes.append(f"poucos_requisitos:{int(n)}")
        if nice is not None:
            componentes["nice"] = nice
        if semantic_similarity is not None:
            componentes["semantic"] = semantic_similarity
            resultado.semantic_similarity = semantic_similarity
            resultado.engine = "semantic"

        resultado.emergent_bonus = self._emergent_bonus(resultado.hits)

        # Tempo de experiência é avaliado à parte da cobertura técnica: ter
        # usado a tecnologia profissionalmente não comprova N anos dela.
        anos = detect_year_requirements(job.requirements_text or job.description)
        resultado.year_requirements = describe(anos)
        resultado.years_penalty = years_penalty(anos)

        score = (
            compose_score(componentes)
            + _work_model_bonus(job, self.profile)
            + resultado.emergent_bonus
            - resultado.years_penalty
        )
        resultado.score = score
        resultado.heuristic_score = score

        self._preencher_listas(resultado)
        resultado.reason = self._explicar(job, resultado)
        return resultado.finalize()

    # --- internos ---

    def _avaliar(self, tech: DetectedTech) -> SkillHit:
        """Confronta uma tecnologia exigida com a evidência disponível no perfil."""
        if tech.in_profile:
            skill = self._por_nome.get(tech.display)
            if skill:
                return SkillHit(
                    required=tech.display,
                    matched=skill.name,
                    level=skill.level,
                    coverage=skill.weight,
                    critical=tech.critical,
                    category=skill.category,
                    evidence=skill.evidence_summary(),
                )

        # Não está no perfil: procura a vizinha mais forte na taxonomia.
        # Cada grupo tem seu próprio fator — ChromaDB→Pinecone vale muito mais
        # que Docker→Kubernetes, e é isso que impede inflar vaga de SRE (§18).
        melhor: SkillHit | None = None
        for grupo in tech.groups:
            vizinho = self.profile.melhor_do_grupo(grupo)
            if vizinho is None:
                continue
            cobertura = vizinho.weight * self.profile.transfer_factor(grupo)
            if melhor is None or cobertura > melhor.coverage:
                melhor = SkillHit(
                    required=tech.display,
                    matched=vizinho.name,
                    level=vizinho.level,
                    coverage=cobertura,
                    transferable=True,
                    critical=tech.critical,
                    category=vizinho.category,
                    evidence=vizinho.evidence_summary(),
                    group=grupo,
                )
        if melhor is not None:
            return melhor

        # Sem grupo: tenta a família (taxonomia antiga, ainda útil para
        # tecnologias externas que não entraram nos grupos).
        vizinho = self.profile.melhor_da_familia(tech.family) if tech.family else None
        if vizinho:
            return SkillHit(
                required=tech.display,
                matched=vizinho.name,
                level=vizinho.level,
                coverage=vizinho.weight * self.profile.transfer_factor(tech.family),
                transferable=True,
                critical=tech.critical,
                category=vizinho.category,
                evidence=vizinho.evidence_summary(),
                group=tech.family,
            )

        return SkillHit(required=tech.display, coverage=0.0, critical=tech.critical)

    def _emergent_bonus(self, hits: list[SkillHit]) -> float:
        """Bônus por competência emergente realmente coberta (§5).

        Pequeno e condicionado a cobertura real: é diferencial competitivo,
        não atalho para score alto.
        """
        emergentes = set(self.profile.emergent_groups)
        if not emergentes:
            return 0.0

        cobertos: set[str] = set()
        for hit in hits:
            if hit.coverage < EMERGENT_COVERAGE_MIN or not hit.matched:
                continue
            skill = self._por_nome.get(hit.matched)
            if skill is None:
                continue
            cobertos |= self.profile.groups_of(skill) & emergentes

        if not cobertos:
            return 0.0
        proporcao = min(1.0, len(cobertos) / EMERGENT_GROUPS_FULL)
        return self.profile.emergent_bonus * proporcao

    @staticmethod
    def _cobertura(hits: list[SkillHit]) -> float | None:
        if not hits:
            return None
        return sum(h.coverage for h in hits) / len(hits)

    @staticmethod
    def _preencher_listas(resultado: MatchResult) -> None:
        """Distribui os hits nos blocos da mensagem, por tipo de evidência (§11)."""
        for hit in sorted(resultado.hits, key=lambda h: (-h.coverage, h.required)):
            if hit.coverage == 0.0:
                resultado.gaps.append(hit.required)
            elif hit.transferable:
                resultado.partial_matches.append(f"{hit.required} ← {hit.matched}")
            elif hit.category == CATEGORY_PROFESSIONAL:
                resultado.strengths.append(hit.required)
            elif hit.category == CATEGORY_PRACTICAL:
                resultado.practical_experience.append(hit.required)
            elif hit.category == CATEGORY_EDUCATION:
                resultado.education.append(f"{hit.required} ({hit.evidence})")
            else:  # CATEGORY_LEARNED — estudo em andamento
                resultado.related_knowledge.append(f"{hit.required} ({hit.evidence})")

    def _explicar(self, job: Job, resultado: MatchResult) -> str:
        partes: list[str] = []
        criticos = [h for h in resultado.hits if h.critical]
        cobertos = [h for h in criticos if h.coverage > 0]
        if criticos:
            partes.append(f"{len(cobertos)}/{len(criticos)} requisitos técnicos com aderência")
        if resultado.practical_experience:
            partes.append(
                "evidência de projeto em " + ", ".join(resultado.practical_experience[:3])
            )
        if resultado.emergent_bonus > 0:
            partes.append("competências emergentes cobertas")
        if "descricao_indisponivel" in resultado.notes:
            partes.append("descrição não disponível — avaliação baseada no cargo")
        if job.work_model is not WorkModel.UNKNOWN:
            partes.append(job.work_model.label.lower())
        if not job.location_confirmed:
            partes.append("localização não confirmada")
        if resultado.year_requirements:
            partes.append(
                f"vaga pede {resultado.year_requirements[0]} — o perfil comprova uso "
                "profissional, mas não o tempo"
            )
        if resultado.gaps:
            partes.append("gaps: " + ", ".join(resultado.gaps[:3]))
        return "; ".join(partes) + "." if partes else "Análise heurística sem requisitos detectados."
