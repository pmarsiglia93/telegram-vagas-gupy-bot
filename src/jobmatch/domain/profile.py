"""Carregamento e modelagem do perfil profissional (profile.yaml)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .evidence import (
    CATEGORY_LEARNED,
    CATEGORY_PRACTICAL,
    CATEGORY_PROFESSIONAL,
    EVIDENCE_LABEL,
    EVIDENCE_WEIGHT,
    LEGACY_LEVELS,
    Evidence,
    category_of,
    combine,
    strongest,
)
from .text import normalize

# Aliases mantidos para compatibilidade com o resto do pipeline (chunker,
# analyzer, testes), agora derivados da tabela de evidências.
LEVEL_WEIGHT = EVIDENCE_WEIGHT
LEVEL_LABEL = EVIDENCE_LABEL

# Crédito quando a vaga pede uma tecnologia que o perfil não tem, mas o perfil
# domina outra do mesmo grupo (pedem Pinecone, tenho ChromaDB).
TRANSFERABLE_FACTOR = 0.55


@dataclass(frozen=True)
class Skill:
    name: str
    family: str
    aliases: tuple[str, ...]
    evidence: tuple[Evidence, ...] = ()
    # Quando o próprio NOME da skill é uma palavra comum ("Make", "Go"), casá-lo
    # contra a descrição gera falso positivo. Com isto, só os aliases contam.
    aliases_only: bool = False

    @property
    def weight(self) -> float:
        return combine(self.evidence)

    @property
    def level(self) -> str:
        """Tipo da evidência mais forte. Preservado para compatibilidade."""
        mais_forte = strongest(self.evidence)
        return mais_forte.type if mais_forte else "study"

    @property
    def level_label(self) -> str:
        return EVIDENCE_LABEL.get(self.level, self.level)

    @property
    def category(self) -> str:
        return category_of(self.evidence)

    @property
    def is_professional(self) -> bool:
        return self.category == CATEGORY_PROFESSIONAL

    def evidence_summary(self) -> str:
        """Ex.: 'projeto próprio, em estudo' — usado no prompt e nos chunks."""
        vistos: list[str] = []
        for ev in sorted(self.evidence, key=lambda e: -e.weight):
            if ev.label not in vistos:
                vistos.append(ev.label)
        return ", ".join(vistos)

    def evidence_notes(self) -> list[str]:
        return [f"{ev.label}: {ev.note}" for ev in self.evidence if ev.note]


@dataclass(frozen=True)
class ProfileItem:
    """Experiência, projeto ou estudo — unidade de chunking do RAG."""

    id: str
    kind: str  # experience | project | study
    title: str
    summary: str
    technologies: tuple[str, ...]
    level: str
    # Domínio de negócio/produto (e-commerce, edtech, ia...). Entra no texto do
    # chunk para o embedding capturar contexto, não só lista de tecnologias.
    domains: tuple[str, ...] = ()
    company: str = ""
    role: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    def to_text(self) -> str:
        linhas = [f"[{EVIDENCE_LABEL.get(self.level, self.level).upper()}] {self.title}"]
        for chave, valor in self.extra.items():
            if valor:
                linhas.append(f"{chave}: {valor}")
        if self.summary:
            linhas.append(self.summary.strip())
        if self.technologies:
            linhas.append("Tecnologias: " + ", ".join(self.technologies))
        if self.domains:
            linhas.append("Domínios: " + ", ".join(self.domains))
        return "\n".join(linhas)


@dataclass(frozen=True)
class Role:
    key: str
    label: str
    keywords: tuple[str, ...]
    search_terms: tuple[str, ...]


@dataclass
class Profile:
    name: str
    headline: str
    city: str
    state: str
    country: str
    metro_area: tuple[str, ...]
    country_aliases: tuple[str, ...]
    work_model_preference: dict[str, int]
    work_model_bonus: float
    emergent_bonus: float
    families: dict[str, str]
    skill_groups: dict[str, tuple[str, ...]]
    group_transfer: dict[str, float]
    emergent_groups: tuple[str, ...]
    skills: tuple[Skill, ...]
    items: tuple[ProfileItem, ...]
    roles: tuple[Role, ...]
    tech_signals: tuple[str, ...]

    # --- índices derivados ---

    def __post_init__(self) -> None:
        self._alias_index: dict[str, Skill] = {}
        for skill in self.skills:
            termos = skill.aliases if skill.aliases_only else (skill.name, *skill.aliases)
            for alias in termos:
                self._alias_index.setdefault(normalize(alias), skill)

        self._families_presentes: dict[str, list[Skill]] = {}
        for skill in self.skills:
            self._families_presentes.setdefault(skill.family, []).append(skill)

        # token normalizado -> grupos da taxonomia que o contêm
        self._group_index: dict[str, set[str]] = {}
        for grupo, membros in self.skill_groups.items():
            for membro in membros:
                self._group_index.setdefault(normalize(membro), set()).add(grupo)

        # grupo -> skills do perfil que pertencem a ele
        self._grupo_skills: dict[str, list[Skill]] = {}
        for skill in self.skills:
            for grupo in self.groups_of(skill):
                self._grupo_skills.setdefault(grupo, []).append(skill)

    @property
    def alias_index(self) -> dict[str, Skill]:
        return self._alias_index

    @property
    def group_index(self) -> dict[str, set[str]]:
        """termo normalizado -> grupos. Inclui tecnologias que o perfil não tem."""
        return self._group_index

    def groups_of(self, skill: Skill) -> set[str]:
        """Grupos da taxonomia aos quais uma skill do perfil pertence."""
        grupos: set[str] = set()
        for termo in (skill.name, *skill.aliases):
            grupos |= self._group_index.get(normalize(termo), set())
        return grupos

    def groups_of_term(self, termo: str) -> set[str]:
        """Grupos de um termo qualquer — inclusive de tecnologia que não tenho."""
        return set(self._group_index.get(normalize(termo), set()))

    def skills_do_grupo(self, grupo: str) -> list[Skill]:
        return self._grupo_skills.get(grupo, [])

    def melhor_do_grupo(self, grupo: str) -> Skill | None:
        candidatos = self.skills_do_grupo(grupo)
        return max(candidatos, key=lambda s: s.weight) if candidatos else None

    def transfer_factor(self, grupo: str) -> float:
        """Quanto de crédito uma tecnologia vizinha do grupo carrega.

        Configurável por grupo porque a proximidade real varia: ChromaDB→Pinecone
        é quase equivalente; Docker→Kubernetes é um salto muito maior. Sem isso,
        conhecer Docker inflaria o score de uma vaga de SRE (§18).
        """
        return self.group_transfer.get(grupo, TRANSFERABLE_FACTOR)

    def skills_da_familia(self, familia: str) -> list[Skill]:
        return self._families_presentes.get(familia, [])

    def melhor_da_familia(self, familia: str) -> Skill | None:
        candidatos = self.skills_da_familia(familia)
        return max(candidatos, key=lambda s: s.weight) if candidatos else None

    @property
    def has_professional_experience(self) -> bool:
        return any(i.kind == "experience" for i in self.items)

    @property
    def professional_skills(self) -> list[Skill]:
        return [s for s in self.skills if s.is_professional]

    def profile_summary_text(self) -> str:
        """Resumo do perfil por categoria de evidência (fallback sem RAG)."""
        por_categoria: dict[str, list[str]] = {}
        for s in self.skills:
            por_categoria.setdefault(s.category, []).append(s.name)
        rotulos = {
            CATEGORY_PROFESSIONAL: "Experiência profissional",
            CATEGORY_PRACTICAL: "Projetos e prática",
            CATEGORY_LEARNED: "Cursos e estudos",
        }
        linhas = [f"{self.name} — {self.headline}"]
        for categoria in (CATEGORY_PROFESSIONAL, CATEGORY_PRACTICAL, CATEGORY_LEARNED):
            nomes = por_categoria.get(categoria)
            if nomes:
                linhas.append(f"{rotulos[categoria]}: " + ", ".join(sorted(nomes)))
        return "\n".join(linhas)


def _tuple(valor) -> tuple[str, ...]:
    if not valor:
        return ()
    if isinstance(valor, str):
        return (valor,)
    return tuple(str(v) for v in valor)


def _parse_evidence(bruto: dict) -> tuple[Evidence, ...]:
    """Lê `evidence:` e cai para o `level:` da versão anterior se preciso."""
    itens = bruto.get("evidence")
    evidencias: list[Evidence] = []

    if itens:
        for item in itens:
            if isinstance(item, str):
                evidencias.append(Evidence(type=item))
            elif isinstance(item, dict) and item.get("type"):
                evidencias.append(Evidence(
                    type=str(item["type"]),
                    note=str(item.get("note", "")),
                    source=str(item.get("source", "")),
                ))

    if not evidencias:
        nivel = str(bruto.get("level", "study"))
        evidencias.append(Evidence(type=LEGACY_LEVELS.get(nivel, nivel)))

    # Descarta tipos desconhecidos em vez de dar peso arbitrário a eles.
    validas = tuple(e for e in evidencias if e.type in EVIDENCE_WEIGHT)
    return validas or (Evidence(type="study"),)


def load_profile(caminho: str) -> Profile:
    """Lê profile.yaml (ou .json). YAML é opcional: cai para JSON se PyYAML faltar."""
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Perfil não encontrado: {caminho}")

    with open(caminho, encoding="utf-8") as f:
        conteudo = f.read()

    if caminho.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - ambiente sem PyYAML
            raise RuntimeError(
                "profile.yaml requer PyYAML (`pip install pyyaml`) ou converta para profile.json"
            ) from exc
        dados = yaml.safe_load(conteudo) or {}
    else:
        dados = json.loads(conteudo)

    return _build_profile(dados)


def _build_profile(dados: dict) -> Profile:
    identity = dados.get("identity") or {}
    location = dados.get("location") or {}
    prefs = dados.get("preferences") or {}

    skills = tuple(
        Skill(
            name=str(s["name"]),
            family=str(s.get("family", "outros")),
            aliases=_tuple(s.get("aliases")),
            evidence=_parse_evidence(s),
            aliases_only=bool(s.get("aliases_only", False)),
        )
        for s in dados.get("skills") or []
        if s.get("name")
    )

    items: list[ProfileItem] = []
    for exp in dados.get("experiences") or []:
        items.append(ProfileItem(
            id=str(exp.get("id") or f"exp_{normalize(exp.get('company', ''))}"),
            kind="experience",
            title=f"{exp.get('role', 'Experiência')} — {exp.get('company', '')}".strip(" —"),
            summary=str(exp.get("summary", "")),
            technologies=_tuple(exp.get("technologies")),
            level=str(exp.get("level", "professional")),
            domains=_tuple(exp.get("domains")),
            company=str(exp.get("company", "")),
            role=str(exp.get("role", "")),
            extra={"Empresa": str(exp.get("company", "")), "Período": str(exp.get("period", ""))},
        ))
    for proj in dados.get("projects") or []:
        items.append(ProfileItem(
            id=str(proj.get("id") or f"proj_{normalize(proj.get('name', ''))}"),
            kind="project",
            title=str(proj.get("name", "Projeto")),
            summary=str(proj.get("summary", "")),
            technologies=_tuple(proj.get("technologies")),
            level=str(proj.get("level", "project")),
            domains=_tuple(proj.get("domains")),
            role=str(proj.get("role", "")),
            extra={"URL": str(proj.get("url", "")), "Tipo": str(proj.get("type", ""))},
        ))
    for est in dados.get("studies") or []:
        items.append(ProfileItem(
            id=str(est.get("id") or f"study_{normalize(est.get('topic', ''))}"),
            kind="study",
            title=str(est.get("topic", "Estudo")),
            summary=str(est.get("summary", "")),
            technologies=_tuple(est.get("technologies")),
            level=str(est.get("level", "study")),
            domains=_tuple(est.get("domains")),
        ))

    roles = tuple(
        Role(
            key=chave,
            label=str(valor.get("label", chave)),
            keywords=_tuple(valor.get("keywords")),
            search_terms=_tuple(valor.get("search_terms")),
        )
        for chave, valor in (dados.get("roles") or {}).items()
    )

    grupos: dict[str, tuple[str, ...]] = {}
    transferencia: dict[str, float] = {}
    for nome, valor in (dados.get("skill_groups") or {}).items():
        # Aceita a forma curta (lista de membros) e a longa (`members` +
        # `transfer_factor`), para não obrigar a anotar o fator em todo grupo.
        if isinstance(valor, dict):
            grupos[str(nome)] = _tuple(valor.get("members"))
            if valor.get("transfer_factor") is not None:
                transferencia[str(nome)] = float(valor["transfer_factor"])
        else:
            grupos[str(nome)] = _tuple(valor)

    return Profile(
        name=str(identity.get("name", "Candidato")),
        headline=str(identity.get("headline", "")),
        city=str(location.get("city", "")),
        state=str(location.get("state", "")),
        country=str(location.get("country", "Brasil")),
        metro_area=_tuple(location.get("metro_area")),
        country_aliases=_tuple(location.get("country_aliases")) or ("brasil", "brazil", "br"),
        work_model_preference={k: int(v) for k, v in (prefs.get("work_models") or {}).items()},
        work_model_bonus=float(prefs.get("work_model_bonus", 4)),
        emergent_bonus=float(prefs.get("emergent_bonus", 4)),
        families=dict(dados.get("families") or {}),
        skill_groups=grupos,
        group_transfer=transferencia,
        emergent_groups=_tuple(dados.get("emergent_groups")),
        skills=skills,
        items=tuple(items),
        roles=roles,
        tech_signals=_tuple(dados.get("tech_signals")),
    )
