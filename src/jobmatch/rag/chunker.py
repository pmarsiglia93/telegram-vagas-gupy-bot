"""Divisão do perfil em chunks SEMÂNTICOS.

Não há corte por N caracteres: o currículo já tem estrutura (experiências,
projetos, estudos, blocos de skill por família), e essa estrutura é a melhor
fronteira de chunk possível (§12).

Cada chunk carrega a EVIDÊNCIA de cada competência, não só um rótulo de nível.
É o que permite ao LLM dizer "implementou RAG em projeto próprio" em vez de
escolher entre "tem experiência com RAG" e "não conhece RAG".
"""

from __future__ import annotations

from ..domain.evidence import (
    CATEGORY_LEARNED,
    CATEGORY_PRACTICAL,
    CATEGORY_PROFESSIONAL,
    EVIDENCE_LABEL,
)
from ..domain.profile import Profile
from ..rag.vector_store import VectorRecord

# Ordem e rótulo dos blocos dentro de um chunk de skills.
CATEGORIA_ROTULO: list[tuple[str, str]] = [
    (CATEGORY_PROFESSIONAL, "Experiência profissional"),
    (CATEGORY_PRACTICAL, "Projetos e prática"),
    (CATEGORY_LEARNED, "Cursos e estudos"),
]


def build_chunks(profile: Profile) -> list[VectorRecord]:
    chunks: list[VectorRecord] = []

    # 1. Experiências, projetos e estudos — um chunk por item.
    # Uma experiência inteira vira UM chunk: separar por tecnologia destruiria
    # o contexto (empresa, produto, domínio) que o embedding precisa capturar.
    for item in profile.items:
        metadata = {
            "type": "professional_experience" if item.kind == "experience" else item.kind,
            "kind": item.kind,
            "title": item.title,
            "experience_level": item.level,
            "evidence_type": item.level,
            "experience_label": EVIDENCE_LABEL.get(item.level, item.level),
            "technologies": ", ".join(item.technologies),
            "domains": ", ".join(item.domains),
            "company": item.company,
            "role": item.role,
            **{k.lower(): v for k, v in item.extra.items() if v},
        }
        chunks.append(VectorRecord(
            id=item.id,
            text=item.to_text(),
            metadata={k: v for k, v in metadata.items() if v},
        ))

    # 2. Um chunk por família de skill, agrupado por categoria de evidência.
    por_familia: dict[str, list] = {}
    for skill in profile.skills:
        por_familia.setdefault(skill.family, []).append(skill)

    for familia, skills in por_familia.items():
        rotulo = profile.families.get(familia, familia.replace("_", " ").title())
        por_categoria: dict[str, list] = {}
        for s in skills:
            por_categoria.setdefault(s.category, []).append(s)

        linhas = [f"Bloco de competências: {rotulo}"]
        for categoria, titulo in CATEGORIA_ROTULO:
            grupo = sorted(por_categoria.get(categoria, []), key=lambda s: -s.weight)
            if not grupo:
                continue
            linhas.append(f"{titulo}:")
            for s in grupo:
                # A evidência acompanha a skill dentro do próprio chunk: é ela
                # que impede o LLM de promover "curso" a "experiência".
                detalhe = f"  - {s.name} ({s.evidence_summary()})"
                notas = [n for n in (ev.note for ev in s.evidence) if n]
                if notas:
                    detalhe += " — " + "; ".join(notas)
                linhas.append(detalhe)

        # Categoria representativa do bloco = a mais forte presente.
        categoria_bloco = next(
            (c for c, _ in CATEGORIA_ROTULO if por_categoria.get(c)), CATEGORY_LEARNED
        )
        nivel_bloco = max(skills, key=lambda s: s.weight).level

        chunks.append(VectorRecord(
            id=f"skill_{familia}",
            text="\n".join(linhas),
            metadata={
                "type": "skill_block",
                "title": rotulo,
                "family": familia,
                "category": categoria_bloco,
                "experience_level": nivel_bloco,
                "experience_label": EVIDENCE_LABEL.get(nivel_bloco, nivel_bloco),
                "technologies": ", ".join(sorted(s.name for s in skills)),
            },
        ))

    return chunks
