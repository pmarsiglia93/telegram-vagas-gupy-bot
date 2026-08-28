"""Modelo de evidência de competência.

A pergunta que o matcher faz deixou de ser "tem experiência profissional?
sim/não" e passou a ser "qual é a evidência de competência?".

Experiência profissional continua sendo a evidência mais forte — e o rótulo
dela nunca é aplicado a outra coisa. Mas projeto, hands-on, curso e estudo
passam a valer crédito real em vez de quase zero.
"""

from __future__ import annotations

from dataclasses import dataclass

# Peso por tipo de evidência (0..1). A ordem importa mais que os números
# absolutos: o que precisa se manter é professional > projeto > curso > estudo.
EVIDENCE_WEIGHT: dict[str, float] = {
    "professional": 1.00,   # usou em produção, remunerado, em vínculo formal
    "freelance": 0.95,      # usou em produção, remunerado, autônomo
    "production_project": 0.85,  # projeto próprio rodando de verdade, com uso real
    "project": 0.75,        # projeto próprio / portfólio publicado
    "academic_project": 0.62,
    "knowledge": 0.60,      # sabe aplicar; nível herdado da versão anterior
    "hands_on": 0.60,       # POC, laboratório, experimentação prática
    "bootcamp": 0.58,       # formação intensiva com projetos construídos
    "certification": 0.55,
    "course": 0.48,         # curso concluído
    "study": 0.40,          # estudando agora
    "interest": 0.20,       # acompanha o assunto, sem prática
}

# Rótulo humano — é o que aparece no chunk do RAG e no prompt do LLM.
# Nenhum destes textos pode sugerir emprego formal, exceto os dois primeiros.
EVIDENCE_LABEL: dict[str, str] = {
    "professional": "experiência profissional",
    "freelance": "experiência profissional (freelance)",
    "production_project": "projeto próprio em produção",
    "project": "projeto próprio",
    "academic_project": "projeto acadêmico",
    "knowledge": "conhecimento aplicado",
    "hands_on": "experimentação prática",
    "bootcamp": "formação/bootcamp",
    "certification": "certificação",
    "course": "curso concluído",
    "study": "em estudo",
    "interest": "interesse",
}

# Em qual bloco da análise a skill aparece. A separação existe para que
# "projeto" nunca seja apresentado como emprego — e para que "estudo" nunca
# desapareça da análise.
# Cinco níveis, do mais forte ao mais fraco. A separação entre FORMAÇÃO e
# ESTUDO existe porque bootcamp/curso concluído é evidência de coisa
# construída e avaliada — diferente de "estou estudando agora".
PROFESSIONAL_TYPES = frozenset({"professional", "freelance"})
PRACTICAL_TYPES = frozenset({"production_project", "project", "academic_project", "hands_on", "knowledge"})
EDUCATION_TYPES = frozenset({"bootcamp", "certification", "course"})
LEARNED_TYPES = frozenset({"study", "interest"})

CATEGORY_PROFESSIONAL = "professional"   # 1. experiência profissional
CATEGORY_PRACTICAL = "practical"         # 2. experiência prática / projetos
CATEGORY_EDUCATION = "education"         # 3. formação / bootcamp / certificação
CATEGORY_LEARNED = "learned"             # 4. conhecimento / estudo em andamento
#                                          5. sem evidência = ausência de Skill

# Ganho máximo por acumular evidências independentes. Existe para que
# "estudei + implementei + usei num projeto" valha mais que só "estudei",
# sem nunca alcançar o peso de experiência profissional.
MULTI_EVIDENCE_BONUS_CAP = 0.15
MULTI_EVIDENCE_FACTOR = 0.20

# Teto por origem da evidência.
#
# Sem isto, `production_project` (0.85) + curso + estudo somava exatamente
# 1.00 — numericamente idêntico a ter usado a tecnologia profissionalmente.
# O acúmulo continua valendo (0.92 é evidência forte), mas a ordem
# `professional > não-profissional` deixa de depender de sorte aritmética.
PROFESSIONAL_CAP = 1.00
FREELANCE_CAP = 0.95
NON_PROFESSIONAL_CAP = 0.92

# Níveis da versão anterior do profile.yaml → tipo de evidência equivalente.
LEGACY_LEVELS: dict[str, str] = {
    "professional": "professional",
    "project": "project",
    "knowledge": "knowledge",
    "study": "study",
}


@dataclass(frozen=True)
class Evidence:
    """Uma prova concreta de que a competência existe."""

    type: str
    note: str = ""       # o que sustenta a afirmação (projeto, curso, empresa)
    source: str = ""     # id do item do perfil que originou a evidência

    @property
    def weight(self) -> float:
        return EVIDENCE_WEIGHT.get(self.type, 0.4)

    @property
    def label(self) -> str:
        return EVIDENCE_LABEL.get(self.type, self.type)

    @property
    def category(self) -> str:
        if self.type in PROFESSIONAL_TYPES:
            return CATEGORY_PROFESSIONAL
        if self.type in PRACTICAL_TYPES:
            return CATEGORY_PRACTICAL
        if self.type in EDUCATION_TYPES:
            return CATEGORY_EDUCATION
        return CATEGORY_LEARNED


def ceiling_for(evidences: tuple[Evidence, ...] | list[Evidence]) -> float:
    """Teto de peso, definido pela evidência de MAIOR origem disponível.

    Só quem tem vínculo profissional pode chegar a 1.00. É o que garante
    `professional > não-profissional` por construção, e não por aritmética.
    """
    tipos = {e.type for e in evidences}
    if "professional" in tipos:
        return PROFESSIONAL_CAP
    if "freelance" in tipos:
        return FREELANCE_CAP
    return NON_PROFESSIONAL_CAP


def combine(evidences: tuple[Evidence, ...] | list[Evidence]) -> float:
    """Peso combinado de várias evidências para a mesma skill.

    A evidência mais forte manda. As demais somam um bônus pequeno e limitado,
    e o resultado respeita o teto da origem mais alta presente:

        só estudo                                  -> 0.40
        projeto + estudo + hands-on                -> 0.90
        projeto em produção + curso + estudo       -> 0.92  (teto não-profissional)
        profissional                               -> 1.00
    """
    if not evidences:
        return 0.0
    pesos = sorted((e.weight for e in evidences), reverse=True)
    base = pesos[0]
    extra = MULTI_EVIDENCE_FACTOR * sum(pesos[1:])
    return min(ceiling_for(evidences), base + min(extra, MULTI_EVIDENCE_BONUS_CAP))


def strongest(evidences: tuple[Evidence, ...] | list[Evidence]) -> Evidence | None:
    return max(evidences, key=lambda e: e.weight) if evidences else None


def category_of(evidences: tuple[Evidence, ...] | list[Evidence]) -> str:
    """Categoria do bloco de saída, definida pela evidência mais forte."""
    mais_forte = strongest(evidences)
    return mais_forte.category if mais_forte else CATEGORY_LEARNED
