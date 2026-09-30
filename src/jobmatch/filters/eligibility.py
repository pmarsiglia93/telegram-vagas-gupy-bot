"""Portão de elegibilidade — o ÚNICO ponto do pipeline que descarta vagas.

Princípios (§2, §3, §4, §9):
  • Remoto vale para o Brasil inteiro.
  • Híbrido e presencial exigem São Paulo / Grande SP.
  • Localização ausente ou ambígua MANTÉM a vaga, marcada como não confirmada.

Foco (`focus` no profile.yaml) — decisões do candidato, não do algoritmo:
  • cargo fora dos cargos-alvo, senioridade fora da aceita, modelo de trabalho
    excluído, stack principal fora do perfil, empresa bloqueada e inglês acima
    do nível do candidato descartam.
  • Sem a seção `focus`, nada disso descarta: senioridade e stack voltam a ser
    só contexto e gap, como no modo amplo.
  • Nível ou modelo NÃO informado nunca descarta — ausência de dado não é
    evidência contra a vaga.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.job import Job, WorkModel
from ..domain.profile import Profile
from ..domain.text import contains_phrase, normalize
from .geo import ESTADOS_BR, ESTRANGEIROS, SP_INTERIOR
from .language import is_english_posting, required_english

# Nomes que identificam a Grande SP sem ambiguidade. A sigla "SP" fica de fora
# de propósito: sozinha ela só diz o estado, e "Campinas - SP" é estado de SP
# mas não é Grande SP.
SP_TERMOS_EXPLICITOS = (
    "sao paulo", "grande sao paulo", "regiao metropolitana de sao paulo", "capital paulista",
)


@dataclass
class Eligibility:
    eligible: bool
    reason: str = ""
    location_confirmed: bool = True
    detail: str = ""


def is_tech_job(job: Job, profile: Profile) -> bool:
    """Portão barato de relevância: isto é claramente uma vaga de tecnologia?

    Deliberadamente permissivo — na dúvida a vaga passa e o scoring decide.
    """
    titulo = normalize(job.title)
    if any(contains_phrase(titulo, sinal) for sinal in profile.tech_signals):
        return True
    if any(contains_phrase(titulo, kw) for role in profile.roles for kw in role.keywords):
        return True
    # Título genérico ("Analista II"): olha tags e descrição atrás de tecnologia.
    corpo = normalize(" ".join(job.tags) + " " + job.description[:4000])
    if corpo:
        aliases_encontrados = sum(
            1 for alias in profile.alias_index if len(alias) >= 3 and contains_phrase(corpo, alias)
        )
        if aliases_encontrados >= 2:
            return True
        if any(contains_phrase(corpo, sinal) for sinal in profile.tech_signals[:12]):
            return True
    return False


def _mentions_sp_metro_city(texto_n: str, profile: Profile) -> bool:
    """Município da Grande SP citado por nome (São Paulo, Osasco, Guarulhos...)."""
    if any(contains_phrase(texto_n, m) for m in profile.metro_area):
        return True
    return any(contains_phrase(texto_n, t) for t in SP_TERMOS_EXPLICITOS)


def _mentions_sp_state(texto_n: str) -> bool:
    """Apenas o estado ("SP"), sem cidade identificável."""
    return contains_phrase(texto_n, "sp")


def _mentions_other_br_location(texto_n: str) -> str:
    """Retorna o nome do local brasileiro fora da Grande SP, se houver."""
    for sigla, nomes in ESTADOS_BR.items():
        for nome in nomes:
            if contains_phrase(texto_n, nome):
                return nome
        if len(sigla) == 2 and contains_phrase(texto_n, sigla.lower()):
            return sigla
    for cidade in SP_INTERIOR:
        if contains_phrase(texto_n, cidade):
            return cidade
    return ""


def _is_foreign(texto_n: str, profile: Profile) -> str:
    if not texto_n:
        return ""
    if any(contains_phrase(texto_n, br) for br in profile.country_aliases):
        return ""
    for termo in ESTRANGEIROS:
        if contains_phrase(texto_n, termo):
            return termo
    return ""


def _off_stack_term(titulo_n: str, profile: Profile) -> str:
    """Termo de stack fora do perfil no título, se nenhuma stack do perfil aparecer junto.

    "Desenvolvedor Flutter" sai; "Mobile Developer (React Native / Flutter)"
    fica, porque React Native também está no título.
    """
    foco = profile.focus
    termo = next((t for t in foco.off_stack_title_terms if contains_phrase(titulo_n, t)), "")
    if not termo:
        return ""
    familias = set(foco.off_stack_unless_families)
    resgata = any(
        contains_phrase(titulo_n, alias)
        for alias, skill in profile.alias_index.items()
        if skill.family in familias and len(alias) >= 2
    )
    return "" if resgata else termo


def check_focus(job: Job, profile: Profile) -> Eligibility | None:
    """Regras de foco do candidato. None = a vaga passa."""
    foco = profile.focus
    titulo_n = normalize(job.title)

    empresa_n = normalize(job.company)
    for empresa in foco.excluded_companies:
        if contains_phrase(empresa_n, empresa):
            return Eligibility(False, "empresa_excluida", detail=job.company[:40])

    if job.work_model.value in foco.excluded_work_models:
        return Eligibility(False, "modelo_excluido", detail=job.work_model.label)

    excluido = next((t for t in foco.excluded_title_terms if contains_phrase(titulo_n, t)), "")
    if excluido:
        return Eligibility(False, "titulo_excluido", detail=excluido)

    if foco.require_role_in_title and not any(
        contains_phrase(titulo_n, kw) for role in profile.roles for kw in role.keywords
    ):
        return Eligibility(False, "cargo_fora_do_foco", detail=job.title[:60])

    if foco.seniority_accept:
        niveis = [n for n in job.seniority.split("/") if n != "nao informado"]
        if niveis and not any(n in foco.seniority_accept for n in niveis):
            return Eligibility(False, "senioridade_fora_do_foco", detail=job.seniority)

    termo = _off_stack_term(titulo_n, profile)
    if termo:
        return Eligibility(False, "stack_fora_do_foco", detail=termo)

    termo = required_english(job, foco.english_required_terms, foco.english_title_terms)
    if termo:
        return Eligibility(False, "ingles_exigido", detail=termo)
    if foco.exclude_english_postings and is_english_posting(job):
        return Eligibility(False, "vaga_em_ingles")

    return None


def check_eligibility(job: Job, profile: Profile) -> Eligibility:
    if not is_tech_job(job, profile):
        return Eligibility(False, "nao_e_tecnologia", detail=job.title[:60])

    fora_do_foco = check_focus(job, profile)
    if fora_do_foco is not None:
        return fora_do_foco

    # `country` é avaliado à parte: ele tem "Brasil" como valor padrão do
    # modelo, e misturá-lo ao texto do local mascararia uma vaga estrangeira
    # (o alias "brasil" cancelaria a detecção de "United States").
    texto_local = normalize(" ".join([job.raw_location, job.city, job.state]))
    texto_pais = normalize(job.country)
    texto_geo = f"{texto_local} {texto_pais}".strip()

    estrangeiro = _is_foreign(texto_local, profile) or _is_foreign(texto_pais, profile)
    if estrangeiro:
        return Eligibility(False, "estrangeiro", detail=estrangeiro)

    # Remoto: qualquer estado brasileiro serve.
    if job.work_model is WorkModel.REMOTE:
        return Eligibility(True, location_confirmed=bool(texto_geo))

    # Híbrido / presencial: precisa ser São Paulo ou Grande SP.
    if job.work_model in (WorkModel.HYBRID, WorkModel.ONSITE):
        # 1. Município da Grande SP citado por nome: aceita, mesmo que outra
        #    cidade apareça junto ("São Paulo / Campinas").
        if _mentions_sp_metro_city(texto_local, profile):
            return Eligibility(True, location_confirmed=True)
        # 2. Cidade identificável fora da Grande SP: não é elegível.
        outro = _mentions_other_br_location(texto_local)
        if outro:
            return Eligibility(
                False, "fora_da_grande_sp", detail=f"{job.work_model.label} em {outro}"
            )
        # 3. Só o estado ("SP"), sem cidade: mantém, mas sem confirmar (§4).
        if _mentions_sp_state(texto_local):
            return Eligibility(True, location_confirmed=False, detail="estado de SP, cidade não informada")
        # 4. Sem localização legível: mantém e sinaliza.
        return Eligibility(True, location_confirmed=False)

    # Modelo desconhecido: nunca descarta. Só confirma o local se der.
    if _mentions_sp_metro_city(texto_local, profile):
        return Eligibility(True, location_confirmed=True)
    outro = _mentions_other_br_location(texto_local)
    if outro:
        # Pode ser remoto não declarado — mantém, mas sem confirmação.
        return Eligibility(True, location_confirmed=False, detail=outro)
    return Eligibility(True, location_confirmed=bool(texto_geo))
