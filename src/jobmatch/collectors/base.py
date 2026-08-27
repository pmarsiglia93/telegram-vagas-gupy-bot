"""Contrato comum dos coletores."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import requests

from ..domain.job import Job
from ..domain.profile import Profile

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

BRT = timezone(timedelta(hours=-3))


class BaseCollector:
    """Sessão HTTP compartilhada + contadores. Subclasses implementam `collect`."""

    name = "base"

    # Pausa entre requests de BUSCA. O LinkedIn devolve 429 depois de ~10
    # requests seguidos sem intervalo — medido em produção, com perda das
    # buscas de backend / software engineer / AI engineer.
    request_delay = 0.3
    # Pausa entre requests de DETALHE (descrição da vaga). É outro endpoint,
    # bem menos sensível, e são muito mais chamadas — usar o mesmo intervalo da
    # busca quadruplicaria o tempo de execução sem ganho nenhum.
    detail_delay = 0.3

    # Teto de avisos de erro HTTP por execução. Uma fonte bloqueada falha do
    # mesmo jeito em cada página/termo — repetir a mesma linha dezenas de
    # vezes só gera ruído no log do CI sem informação nova.
    MAX_AVISOS_HTTP = 3

    def __init__(self, profile: Profile, max_age_days: int = 3) -> None:
        self.profile = profile
        self.max_age_days = max_age_days
        self.errors = 0
        self.requests_made = 0
        self.rate_limited = 0
        self._ultimo_request = 0.0
        self._avisos_http = 0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "pt-BR,pt;q=0.9"})

    # --- helpers ---

    def _get(self, url: str, **kwargs):
        """GET com throttle e uma retentativa em caso de 429.

        `delay` sobrescreve o intervalo padrão (usado pelos `fetch_details`).
        """
        timeout = kwargs.pop("timeout", 20)
        intervalo = kwargs.pop("delay", None)
        if intervalo is None:
            intervalo = self.request_delay

        for tentativa in range(2):
            espera = intervalo - (time.monotonic() - self._ultimo_request)
            if espera > 0:
                time.sleep(espera)

            self.requests_made += 1
            self._ultimo_request = time.monotonic()
            resp = self.session.get(url, timeout=timeout, **kwargs)

            if resp.status_code != 429 or tentativa == 1:
                return resp

            # Respeita o Retry-After quando existe; senão, recua o suficiente
            # para a janela do LinkedIn abrir de novo.
            self.rate_limited += 1
            try:
                pausa = float(resp.headers.get("Retry-After", "") or 0)
            except ValueError:
                pausa = 0.0
            time.sleep(min(max(pausa, self.request_delay * 4), 30.0))

        return resp

    def _log_http_error(self, contexto: str, resp) -> None:
        """Loga o motivo real de uma falha HTTP, não só um contador.

        Sem isto, uma fonte bloqueada (ex.: anti-bot barrando o IP do runner
        do GitHub Actions) aparecia só como "0 vagas" e um número de erros,
        sem pista nenhuma do porquê — o mesmo código funcionando localmente e
        falhando só em CI virava mistério.
        """
        self._avisos_http += 1
        if self._avisos_http > self.MAX_AVISOS_HTTP:
            return
        corpo = (resp.text or "")[:300].lower()
        servidor = resp.headers.get("server", "").lower()
        if "cloudflare" in corpo or "checking your browser" in corpo or "cloudflare" in servidor:
            pista = " — parece bloqueio Cloudflare/anti-bot (comum em runners de CI)"
        elif resp.status_code == 403:
            pista = " — acesso negado, provável bloqueio por IP ou user-agent"
        elif resp.status_code == 429:
            pista = " — rate limit"
        else:
            pista = ""
        print(f"   ⚠️  {self.name} [{contexto}]: HTTP {resp.status_code}{pista}")

    def _muito_antiga(self, publicada: datetime | None) -> bool:
        if publicada is None:
            return False
        return datetime.now(BRT) - publicada > timedelta(days=self.max_age_days)

    def search_terms(self) -> list[str]:
        """Termos de busca vindos do perfil — cargos-alvo, não stack (§5)."""
        termos: list[str] = []
        for role in self.profile.roles:
            for termo in role.search_terms:
                if termo not in termos:
                    termos.append(termo)
        return termos

    # --- contrato ---

    def collect(self) -> list[Job]:  # pragma: no cover - implementado nas subclasses
        raise NotImplementedError

    def fetch_details(self, job: Job) -> bool:
        """Busca a descrição completa. Best-effort: False não é erro fatal.

        Chamado pelo pipeline SÓ para vagas que passaram na deduplicação, para
        não gastar request em vaga que já foi enviada (§21).
        """
        return False
