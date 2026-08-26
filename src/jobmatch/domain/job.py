"""Representação normalizada de uma vaga, independente da fonte."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .text import contains_phrase, normalize, strip_html, truncate


class WorkModel(str, Enum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"

    @property
    def label(self) -> str:
        return {
            WorkModel.REMOTE: "Remoto",
            WorkModel.HYBRID: "Híbrido",
            WorkModel.ONSITE: "Presencial",
            WorkModel.UNKNOWN: "Modelo não informado",
        }[self]


# Termos de senioridade. IMPORTANTE: usados apenas como *informação contextual*.
# Nenhum ponto do pipeline pode eliminar ou penalizar uma vaga por causa disto.
_SENIORITY_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("estagio", ("estagio", "estagiario", "intern", "internship", "trainee")),
    ("junior", ("junior", "jr", "entry level", "entry-level", "i ")),
    ("pleno", ("pleno", "mid level", "mid-level", "midlevel", "intermediario")),
    ("senior", ("senior", "sr", "sênior", "especialista", "specialist", "iii")),
    ("staff", ("staff", "principal", "lead", "tech lead", "head", "arquiteto", "architect")),
]

_REMOTE_HINTS = ("remoto", "remota", "remote", "home office", "homeoffice", "100% remoto", "anywhere")
_HYBRID_HINTS = ("hibrido", "hybrid", "semi presencial", "semipresencial")
_ONSITE_HINTS = ("presencial", "on-site", "on site", "onsite", "no local", "alocado")


def detect_work_model(*textos: str) -> WorkModel:
    """Deriva o modelo de trabalho a partir de qualquer texto disponível.

    Ordem importa: 'híbrido' costuma aparecer junto de 'remoto' em descrições
    ("2x remoto, 3x presencial"), então híbrido tem prioridade sobre remoto.
    """
    alvo = normalize(" | ".join(t for t in textos if t))
    if not alvo:
        return WorkModel.UNKNOWN
    if any(contains_phrase(alvo, h) for h in _HYBRID_HINTS):
        return WorkModel.HYBRID
    if any(contains_phrase(alvo, h) for h in _REMOTE_HINTS):
        return WorkModel.REMOTE
    if any(contains_phrase(alvo, h) for h in _ONSITE_HINTS):
        return WorkModel.ONSITE
    return WorkModel.UNKNOWN


def detect_seniority(*textos: str) -> str:
    """Retorna a senioridade detectada — puramente informativa."""
    alvo = normalize(" | ".join(t for t in textos if t))
    encontrados = [
        nome for nome, termos in _SENIORITY_PATTERNS
        if any(contains_phrase(alvo, t.strip()) for t in termos if t.strip())
    ]
    if not encontrados:
        return "nao informado"
    # Se a vaga cita mais de um nível ("Pleno/Sênior"), reporta todos.
    return "/".join(encontrados)


# Cabeçalhos comuns em descrições PT/EN, mapeados para seções canônicas.
_SECTION_HEADINGS: list[tuple[str, tuple[str, ...]]] = [
    ("nice_to_have", (
        "diferenciais", "diferencial", "desejavel", "desejaveis", "sera um plus",
        "nice to have", "bonus", "plus", "o que seria legal", "vantagens tecnicas",
    )),
    ("requirements", (
        "requisitos", "requisitos e qualificacoes", "qualificacoes", "pre-requisitos",
        "o que esperamos", "o que voce precisa", "voce precisa ter", "must have",
        "requirements", "qualifications", "hard skills", "conhecimentos necessarios",
        "para se dar bem", "o que buscamos",
    )),
    ("responsibilities", (
        "responsabilidades", "responsabilidades e atribuicoes", "atribuicoes",
        "o que voce vai fazer", "suas entregas", "responsibilities", "sobre a posicao",
        "descricao da vaga", "principais atividades", "atividades",
    )),
    ("benefits", (
        "beneficios", "o que oferecemos", "nossos beneficios", "benefits", "perks",
        "etapas do processo", "processo seletivo",
    )),
]


def split_sections(descricao: str) -> dict[str, str]:
    """Fatia a descrição em seções canônicas por cabeçalho.

    Não é um parser perfeito de HTML — é uma heurística barata que funciona bem
    nas descrições de Gupy/LinkedIn, onde os cabeçalhos são linhas curtas. O
    texto não classificado cai em `other`, e nada é descartado.
    """
    if not descricao:
        return {}

    # Gupy concatena cabeçalhos ao texto sem separador ("Requisitos e qualificaçõesDomínio de...").
    # Insere quebra antes de um cabeçalho conhecido grudado na palavra anterior.
    texto = descricao
    for _, headings in _SECTION_HEADINGS:
        for h in headings:
            texto = re.sub(
                r"(?i)(?<=[a-zà-ú;.)])(" + re.escape(h).replace(r"\ ", r"\s+") + r")",
                r"\n\1",
                texto,
            )

    secoes: dict[str, list[str]] = {}
    atual = "other"
    for linha in texto.splitlines():
        limpa = linha.strip()
        if not limpa:
            continue
        cabecalho_n = normalize(limpa).rstrip(":").strip()
        achou = None
        for nome, headings in _SECTION_HEADINGS:
            # Só considera cabeçalho se a linha começa com ele e é curta.
            if any(cabecalho_n.startswith(h) for h in headings) and len(cabecalho_n) <= 90:
                achou = nome
                break
        if achou:
            atual = achou
            # Conteúdo pode vir na mesma linha, depois do cabeçalho.
            resto = limpa
            for h in sorted((x for _, hs in _SECTION_HEADINGS for x in hs), key=len, reverse=True):
                if normalize(resto).startswith(h):
                    resto = resto[len(h):].lstrip(":;-— ").strip()
                    break
            if resto:
                secoes.setdefault(atual, []).append(resto)
            continue
        secoes.setdefault(atual, []).append(limpa)

    return {k: "\n".join(v).strip() for k, v in secoes.items() if "".join(v).strip()}


@dataclass
class Job:
    """Vaga normalizada. Toda fonte converge para esta estrutura."""

    source: str
    title: str
    company: str
    url: str
    description: str = ""
    raw_location: str = ""
    city: str = ""
    state: str = ""
    country: str = "Brasil"
    work_model: WorkModel = WorkModel.UNKNOWN
    tags: list[str] = field(default_factory=list)
    job_type: str = ""
    salary: str = ""
    pcd: bool | None = None
    published_at: datetime | None = None
    external_id: str = ""
    search_label: str = ""
    # Preenchido pelos filtros: False quando a localização é ausente/ambígua.
    location_confirmed: bool = True
    sections: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.description = strip_html(self.description)
        if self.description and not self.sections:
            self.sections = split_sections(self.description)
        if self.work_model is WorkModel.UNKNOWN:
            self.work_model = detect_work_model(self.raw_location, self.title, self.description)

    # --- Views de texto usadas por filtros, matching e embeddings ---

    @property
    def seniority(self) -> str:
        return detect_seniority(self.title)

    @property
    def requirements_text(self) -> str:
        """Requisitos obrigatórios + responsabilidades — o núcleo técnico da vaga."""
        partes = [self.sections.get("requirements", ""), self.sections.get("responsibilities", "")]
        texto = "\n".join(p for p in partes if p).strip()
        # Sem seções identificadas, a descrição inteira é o melhor proxy.
        return texto or self.description

    @property
    def nice_to_have_text(self) -> str:
        return self.sections.get("nice_to_have", "")

    @property
    def full_text(self) -> str:
        return "\n".join(
            p for p in [self.title, self.company, " ".join(self.tags), self.description] if p
        )

    @property
    def location_label(self) -> str:
        if self.raw_location:
            return self.raw_location
        partes = [p for p in (self.city, self.state) if p]
        if partes:
            return " - ".join(partes)
        return self.country or "Não informado"

    @property
    def has_description(self) -> bool:
        return len(self.description) >= 180

    def fingerprint(self) -> str:
        """Identidade semântica da vaga, para dedup entre fontes diferentes.

        Empresa + título normalizado (sem senioridade/ruído) + cidade.
        URLs diferentes da mesma vaga colapsam neste hash.
        """
        titulo = normalize(self.title)
        titulo = re.sub(r"[^a-z0-9 ]+", " ", titulo)
        ruido = {
            "desenvolvedor", "desenvolvedora", "developer", "engenheiro", "engenheira",
            "engineer", "analista", "pessoa", "vaga", "profissional", "de", "do", "da",
            "e", "o", "a", "em", "para", "com", "i", "ii", "iii", "iv",
            "junior", "jr", "pleno", "senior", "sr", "especialista", "specialist",
            "staff", "principal", "lead", "efetivo", "clt", "pj", "remoto", "hibrido",
            "presencial", "afirmativa", "pcd", "banco", "talentos",
        }
        tokens = sorted({t for t in titulo.split() if t and t not in ruido})
        base = "|".join([
            normalize(self.company),
            " ".join(tokens),
            normalize(self.city) or normalize(self.state),
        ])
        return hashlib.sha1(base.encode("utf-8")).hexdigest()[:20]

    def short_description(self, limite: int = 400) -> str:
        return truncate(self.requirements_text.replace("\n", " "), limite)
