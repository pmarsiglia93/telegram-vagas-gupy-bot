"""Coletor do LinkedIn — endpoints públicos "jobs-guest" (sem login).

São os mesmos endpoints que o bot original já usava; nada foi inventado.
A novidade é o segundo endpoint, `jobPosting/{id}`, que devolve a descrição
completa da vaga — chamado só depois da deduplicação, para não gastar request
com vaga já enviada (§21).
"""

from __future__ import annotations

import re
from datetime import datetime

from ..domain.job import Job, WorkModel, detect_work_model
from ..domain.text import strip_html
from .base import BRT, BaseCollector

BUSCA = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
DETALHE = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"

# f_TPR=r259200 → últimos 3 dias | f_WT=2 → remoto
JANELA_3_DIAS = "r259200"

_ID_RE = re.compile(r"(\d{6,})(?:\?|$)")

# Rodapé do widget "ver mais" que o LinkedIn injeta em toda descrição. Sem
# remover, o "less" final casava com o alias de LESS/CSS em 2 de cada 3 vagas.
_BOILERPLATE_RE = re.compile(
    r"(?i)\s*(show more\s*show less|mostrar mais\s*mostrar menos|ver mais\s*ver menos)\s*$"
)


def _limpar_boilerplate(texto: str) -> str:
    return _BOILERPLATE_RE.sub("", texto).strip()


class LinkedInCollector(BaseCollector):
    name = "linkedin"
    # O endpoint guest é o mais sensível das três fontes: sem esta pausa ele
    # passa a devolver 429 a partir do ~10º request da execução.
    request_delay = 2.5
    detail_delay = 0.8

    def __init__(self, profile, max_age_days: int = 3) -> None:
        super().__init__(profile, max_age_days)
        try:
            from bs4 import BeautifulSoup  # noqa: F401
            self.disponivel = True
        except ImportError:
            self.disponivel = False

    def collect(self) -> list[Job]:
        if not self.disponivel:
            print("   ⚠️  LinkedIn desativado: beautifulsoup4 não instalado")
            return []
        from bs4 import BeautifulSoup

        segundos = max(1, self.max_age_days) * 86400
        janela = f"r{segundos}"

        buscas = [
            {"rotulo": "SP", "params": {"location": "São Paulo, Brazil", "f_TPR": janela}},
            {"rotulo": "REMOTO", "params": {"location": "Brazil", "f_WT": "2", "f_TPR": janela}},
        ]

        vagas: list[Job] = []
        vistos: set[str] = set()

        for termo in self.search_terms():
            for busca in buscas:
                rotulo = f"{termo.upper()} · {busca['rotulo']}"
                params = {"keywords": termo, "start": 0, **busca["params"]}
                try:
                    resp = self._get(BUSCA, params=params, timeout=25)
                    if resp.status_code != 200:
                        self.errors += 1
                        self._log_http_error(rotulo, resp)
                        continue
                    cards = BeautifulSoup(resp.text, "html.parser").find_all("div", class_="base-card")
                except Exception as exc:
                    self.errors += 1
                    print(f"   ⚠️  LinkedIn [{rotulo}]: {type(exc).__name__}: {exc}")
                    continue

                for card in cards:
                    job = self._to_job(card, rotulo, remoto=busca["rotulo"] == "REMOTO")
                    if job is None or job.url in vistos:
                        continue
                    vistos.add(job.url)
                    vagas.append(job)

        return vagas

    def _to_job(self, card, rotulo: str, remoto: bool) -> Job | None:
        link_el = card.find("a", href=True)
        if not link_el:
            return None
        link = link_el["href"].split("?")[0]

        def texto(seletor: str) -> str:
            el = card.find(class_=lambda c: c and seletor in c)
            return el.get_text(strip=True) if el else ""

        titulo = texto("title") or "Título Indisponível"
        empresa = texto("subtitle") or "Empresa não informada"
        local = texto("location")

        publicada = None
        data_el = card.find("time")
        if data_el and data_el.get("datetime"):
            try:
                publicada = datetime.strptime(data_el["datetime"], "%Y-%m-%d").replace(tzinfo=BRT)
            except ValueError:
                publicada = None

        # A busca com f_WT=2 devolve apenas vagas remotas; fora dela o modelo
        # só é conhecido depois de ler a descrição.
        modelo = WorkModel.REMOTE if remoto else detect_work_model(local, titulo)

        m = _ID_RE.search(link)
        return Job(
            source="linkedin",
            title=titulo,
            company=empresa,
            url=link,
            raw_location=local,
            work_model=modelo,
            published_at=publicada,
            external_id=m.group(1) if m else "",
            search_label=rotulo,
        )

    def fetch_details(self, job: Job) -> bool:
        if not self.disponivel or not job.external_id or job.has_description:
            return False
        from bs4 import BeautifulSoup

        try:
            resp = self._get(
                DETALHE.format(job_id=job.external_id), timeout=25, delay=self.detail_delay
            )
            if resp.status_code != 200:
                return False
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception:
            self.errors += 1
            return False

        bloco = soup.select_one(".show-more-less-html__markup, .description__text")
        if not bloco:
            return False

        job.description = _limpar_boilerplate(strip_html(bloco.decode_contents()))
        job.sections = {}
        job.__post_init__()  # reprocessa seções e modelo com a descrição nova

        # Os critérios estruturados ("Seniority level", "Employment type") são
        # contexto, nunca critério de corte.
        for criterio in soup.select(".description__job-criteria-item"):
            rotulo = criterio.select_one(".description__job-criteria-subheader")
            valor = criterio.select_one(".description__job-criteria-text")
            if rotulo and valor and "employment" in rotulo.get_text(strip=True).lower():
                job.job_type = valor.get_text(strip=True)
        return True
