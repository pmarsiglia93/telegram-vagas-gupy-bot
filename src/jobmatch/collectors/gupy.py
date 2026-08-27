"""Coletor da Gupy — API pública do portal de empregabilidade.

A resposta do endpoint de listagem **já traz a descrição completa** da vaga
(campo `description`, ~3 mil caracteres), além de `workplaceType`, `city`,
`state`, `country` e `skills`. Isso significa que o matching por descrição sai
de graça aqui: nenhum request extra, nenhum scraping (§29.14).
"""

from __future__ import annotations

from datetime import datetime

from ..domain.job import Job, WorkModel
from .base import BRT, BaseCollector

API = "https://employability-portal.gupy.io/api/v1/jobs"

MODELO = {
    "remote": WorkModel.REMOTE,
    "hybrid": WorkModel.HYBRID,
    "on-site": WorkModel.ONSITE,
    "on_site": WorkModel.ONSITE,
}

TIPO_VAGA = {
    "vacancy_type_effective": "Efetivo",
    "vacancy_type_apprentice": "Jovem Aprendiz",
    "vacancy_type_internship": "Estágio",
    "vacancy_type_temporary": "Temporário",
    "vacancy_type_freelancer": "Freelancer",
}

PAGINAS_MAX = 3
POR_PAGINA = 20


class GupyCollector(BaseCollector):
    name = "gupy"

    def collect(self) -> list[Job]:
        vagas: list[Job] = []
        vistos: set[str] = set()
        headers = {"Accept": "application/json, text/plain, */*", "Origin": "https://portal.gupy.io"}

        # Duas varreduras por cargo: São Paulo (presencial/híbrido) e remoto
        # Brasil. As regras de localização são aplicadas depois, no filtro.
        buscas = [
            {"rotulo": "SP", "params": {"state": "São Paulo"}},
            {"rotulo": "REMOTO", "params": {"workplaceTypes": "remote"}},
        ]

        for termo in self.search_terms():
            for busca in buscas:
                rotulo = f"{termo.upper()} · {busca['rotulo']}"
                for pagina in range(PAGINAS_MAX):
                    params = {
                        **busca["params"],
                        "jobName": termo,
                        "limit": POR_PAGINA,
                        "offset": pagina * POR_PAGINA,
                    }
                    try:
                        resp = self._get(API, headers=headers, params=params)
                        if resp.status_code != 200:
                            self.errors += 1
                            self._log_http_error(rotulo, resp)
                            break
                        dados = resp.json().get("data", [])
                    except Exception as exc:
                        self.errors += 1
                        print(f"   ⚠️  Gupy [{rotulo}]: {type(exc).__name__}: {exc}")
                        break

                    if not dados:
                        break

                    antigas_seguidas = 0
                    for bruto in dados:
                        job = self._to_job(bruto, rotulo)
                        if job is None:
                            continue
                        if self._muito_antiga(job.published_at):
                            antigas_seguidas += 1
                            continue
                        if job.url in vistos:
                            continue
                        vistos.add(job.url)
                        vagas.append(job)

                    # A API devolve por data desc: página inteira velha = fim.
                    if antigas_seguidas >= len(dados):
                        break

        return vagas

    def _to_job(self, bruto: dict, rotulo: str) -> Job | None:
        link = bruto.get("jobUrl") or ""
        if not link:
            return None

        publicada = None
        iso = bruto.get("publishedDate") or ""
        if iso:
            try:
                publicada = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(BRT)
            except ValueError:
                publicada = None

        workplace = str(bruto.get("workplaceType") or "").lower()
        modelo = MODELO.get(workplace, WorkModel.UNKNOWN)
        if modelo is WorkModel.UNKNOWN and bruto.get("isRemoteWork"):
            modelo = WorkModel.REMOTE

        cidade = (bruto.get("city") or "").strip()
        estado = (bruto.get("state") or "").strip()
        pais = (bruto.get("country") or "Brasil").strip()
        local = " - ".join(p for p in (cidade, estado) if p) or pais

        skills = bruto.get("skills") or []
        tags = [str(s) for s in skills if s] if isinstance(skills, list) else []

        return Job(
            source="gupy",
            title=bruto.get("name") or "Título Indisponível",
            company=bruto.get("careerPageName") or "Empresa não informada",
            url=link,
            description=bruto.get("description") or "",
            raw_location=local,
            city=cidade,
            state=estado,
            country=pais,
            work_model=modelo,
            tags=tags,
            job_type=TIPO_VAGA.get(bruto.get("type") or "", ""),
            pcd=bool(bruto.get("disabilities")),
            published_at=publicada,
            external_id=str(bruto.get("id") or ""),
            search_label=rotulo,
        )
