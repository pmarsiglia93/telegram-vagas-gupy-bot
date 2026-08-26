"""Coletor do ProgramaThor — scraping da listagem pública.

O site não expõe API. A listagem é estável e já traz empresa, local, nível,
salário e a stack em tags. A página de detalhe traz a descrição completa, mas
responde HTTP 500 em boa parte das vagas (falha do lado deles, verificada em
produção) — por isso `fetch_details` é best-effort e o matching usa as tags do
card quando a descrição não vem.
"""

from __future__ import annotations

from datetime import datetime

from ..domain.job import Job, detect_work_model
from ..domain.text import strip_html
from .base import BRT, BaseCollector

LISTAGEM = "https://programathor.com.br/jobs"
BASE = "https://programathor.com.br"
PAGINAS_MAX = 3


class ProgramaThorCollector(BaseCollector):
    name = "programathor"
    request_delay = 1.0
    detail_delay = 0.4

    def __init__(self, profile, max_age_days: int = 3) -> None:
        super().__init__(profile, max_age_days)
        try:
            from bs4 import BeautifulSoup  # noqa: F401
            self.disponivel = True
        except ImportError:
            self.disponivel = False

    def collect(self) -> list[Job]:
        if not self.disponivel:
            print("   ⚠️  ProgramaThor desativado: beautifulsoup4 não instalado")
            return []
        from bs4 import BeautifulSoup

        vagas: list[Job] = []
        vistos: set[str] = set()

        for termo in self.search_terms():
            for pagina in range(1, PAGINAS_MAX + 1):
                params = {"search": termo}
                if pagina > 1:
                    params["page"] = pagina
                try:
                    resp = self._get(LISTAGEM, params=params)
                    if resp.status_code != 200:
                        self.errors += 1
                        break
                    cards = BeautifulSoup(resp.text, "html.parser").find_all("div", class_="cell-list")
                except Exception as exc:
                    self.errors += 1
                    print(f"   ⚠️  ProgramaThor [{termo}]: {type(exc).__name__}: {exc}")
                    break

                if not cards:
                    break

                novos = 0
                for card in cards:
                    job = self._to_job(card, termo)
                    if job is None or job.url in vistos:
                        continue
                    vistos.add(job.url)
                    vagas.append(job)
                    novos += 1

                if novos == 0:
                    break

        return vagas

    def _to_job(self, card, termo: str) -> Job | None:
        link_el = card.find("a", href=lambda h: h and "/jobs/" in h)
        if not link_el:
            return None

        titulo_el = card.find("h3")
        titulo_raw = titulo_el.get_text(strip=True) if titulo_el else ""
        if "vencida" in titulo_raw.lower():
            return None
        titulo = titulo_raw.replace("NOVA", "").strip() or "Título Indisponível"

        spans = card.select(".cell-list-content-icon span")

        def span(i: int) -> str:
            return spans[i].get_text(strip=True) if len(spans) > i else ""

        empresa = span(0) or "Empresa não informada"
        local = span(1)
        salario = span(3)
        nivel = span(4)
        tipo = span(5)
        tags = [t.get_text(strip=True) for t in card.select("span.tag-list")]

        return Job(
            source="programathor",
            title=titulo,
            company=empresa,
            url=BASE + link_el["href"],
            raw_location=local,
            work_model=detect_work_model(local, titulo),
            tags=[t for t in tags if t],
            job_type=tipo,
            salary=salario,
            # `nivel` (Júnior/Pleno/Sênior) entra como contexto na descrição,
            # NUNCA como filtro — era exatamente esse campo que descartava
            # vagas sênior na versão anterior.
            description=f"Nível informado pela vaga: {nivel}\n" if nivel else "",
            published_at=datetime.now(BRT),
            search_label=termo.upper(),
        )

    def fetch_details(self, job: Job) -> bool:
        if not self.disponivel or job.has_description:
            return False
        from bs4 import BeautifulSoup

        try:
            resp = self._get(job.url, timeout=20, delay=self.detail_delay)
            if resp.status_code != 200:
                # ~500 é comum aqui e não conta como erro do bot.
                return False
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception:
            self.errors += 1
            return False

        bloco = soup.select_one(".wrapper-content-job-show")
        if not bloco:
            return False

        texto = strip_html(bloco.decode_contents())
        if len(texto) < 120:
            return False

        job.description = (job.description + "\n" + texto).strip()
        job.sections = {}
        job.__post_init__()
        return True
