"""Portão de elegibilidade — o ÚNICO ponto do pipeline que descarta vagas.

Princípios (§2, §3, §4, §9):
  • Senioridade NUNCA elimina. Não há checagem de senioridade aqui, de propósito.
  • Tecnologia "fora do stack" NUNCA elimina — vira gap no scoring.
  • Remoto vale para o Brasil inteiro.
  • Híbrido e presencial exigem São Paulo / Grande SP.
  • Localização ausente ou ambígua MANTÉM a vaga, marcada como não confirmada.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.job import Job, WorkModel
from ..domain.profile import Profile
from ..domain.text import contains_phrase, normalize
from .geo import ESTADOS_BR, ESTRANGEIROS, SP_INTERIOR

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


def check_eligibility(job: Job, profile: Profile) -> Eligibility:
    if not is_tech_job(job, profile):
        return Eligibility(False, "nao_e_tecnologia", detail=job.title[:60])

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
