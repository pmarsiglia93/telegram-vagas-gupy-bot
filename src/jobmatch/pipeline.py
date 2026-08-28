"""Orquestração do pipeline JobMatch AI.

    Fontes → Coleta → Normalização → Dedup → Descrição completa
      → Filtros mínimos → Embeddings/RAG → Score → LLM → Ordenação → Telegram

Regras de custo (§21): a descrição só é buscada para vagas que passaram na
deduplicação; o LLM só é chamado para as N melhores vagas acima de um piso de
score. Regras de resiliência (§22): cada camada de IA degrada sozinha, sem
derrubar o envio.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .ai.analyzer import JobAnalyzer
from .ai.llm import create_llm_provider
from .collectors.base import BaseCollector
from .collectors.gupy import GupyCollector
from .collectors.linkedin import LinkedInCollector
from .collectors.programathor import ProgramaThorCollector
from .config.settings import Settings
from .domain.job import Job, WorkModel
from .domain.match import MatchResult
from .domain.profile import Profile, load_profile
from .filters.eligibility import check_eligibility
from .matching.heuristic import HeuristicMatcher
from .notifications.telegram import TelegramNotifier, format_message
from .persistence.sqlite import JobRepository
from .rag.embeddings import create_embedding_provider
from .rag.retriever import ProfileRetriever
from .rag.vector_store import create_vector_store

ORDEM_MODELO = {WorkModel.REMOTE: 3, WorkModel.HYBRID: 2, WorkModel.ONSITE: 1, WorkModel.UNKNOWN: 0}


@dataclass
class Metrics:
    coletadas: int = 0
    duplicadas: int = 0
    descartadas: int = 0
    analisadas: int = 0
    enviadas: int = 0
    chamadas_embedding: int = 0
    chamadas_llm: int = 0
    detalhes_buscados: int = 0
    erros: int = 0
    duracao_seg: float = 0.0
    motivos_descarte: dict[str, int] = field(default_factory=dict)
    avisos: list[str] = field(default_factory=list)

    def descartar(self, motivo: str) -> None:
        self.descartadas += 1
        self.motivos_descarte[motivo] = self.motivos_descarte.get(motivo, 0) + 1

    def to_dict(self) -> dict:
        return {
            "coletadas": self.coletadas,
            "descartadas": self.descartadas,
            "analisadas": self.analisadas,
            "enviadas": self.enviadas,
            "chamadas_embedding": self.chamadas_embedding,
            "chamadas_llm": self.chamadas_llm,
            "erros": self.erros,
            "duracao_seg": self.duracao_seg,
        }


def _sort_key(par: tuple[Job, MatchResult]):
    """§19: score → cobertura de requisitos → modelo de trabalho → data."""
    job, match = par
    # Vaga sem data conhecida vai para o fim do critério de desempate, sem
    # chamar timestamp() em datetime.min (que estoura em alguns sistemas).
    quando = job.published_at.timestamp() if job.published_at else float("-inf")
    return (
        -match.score,
        -match.required_coverage,
        -ORDEM_MODELO.get(job.work_model, 0),
        -quando,
    )


class Pipeline:
    def __init__(self, settings: Settings, profile: Profile | None = None) -> None:
        self.settings = settings
        self.profile = profile or load_profile(settings.profile_path)
        self.metrics = Metrics()
        self.matcher = HeuristicMatcher(self.profile)
        self.repo = JobRepository(settings.db_path)
        self.notifier = TelegramNotifier(settings.telegram_token, settings.chat_id, settings.dry_run)
        self.retriever = self._build_retriever()
        self.llm = self._build_llm()
        self.analyzer = JobAnalyzer(self.profile, self.retriever, self.llm)

    # --- construção das camadas opcionais ---

    def _build_retriever(self) -> ProfileRetriever | None:
        s = self.settings
        try:
            embedder = create_embedding_provider(
                s.embedding_provider, s.embedding_key, s.embedding_model
            )
            store, aviso = create_vector_store(s.vector_store, s.chroma_path)
            if aviso:
                self.metrics.avisos.append(aviso)
            retriever = ProfileRetriever(self.profile, embedder, store)
            # O índice do perfil é reconstruído a cada execução: são ~25 chunks,
            # o custo é desprezível e elimina a dependência de estado persistente
            # entre runs do GitHub Actions (§23).
            if not retriever.build():
                self.metrics.avisos.append(f"RAG indisponível: {retriever.error}")
                return None
            return retriever
        except Exception as exc:
            self.metrics.avisos.append(f"RAG indisponível: {type(exc).__name__}: {exc}")
            return None

    def _build_llm(self):
        provider, aviso = create_llm_provider(
            self.settings.llm_provider,
            self.settings.llm_key,
            self.settings.llm_model,
            self.settings.llm_timeout,
        )
        if aviso:
            self.metrics.avisos.append(aviso)
        return provider

    def _collectors(self) -> list[BaseCollector]:
        return [
            GupyCollector(self.profile, self.settings.max_age_days),
            LinkedInCollector(self.profile, self.settings.max_age_days),
            ProgramaThorCollector(self.profile, self.settings.max_age_days),
        ]

    # --- etapas ---

    def collect(self, collectors: list[BaseCollector]) -> list[tuple[Job, BaseCollector]]:
        coletadas: list[tuple[Job, BaseCollector]] = []
        for collector in collectors:
            print(f"\n🔎 {collector.name.upper()} — coletando...")
            try:
                vagas = collector.collect()
            except Exception as exc:
                self.metrics.erros += 1
                print(f"   ❌ {collector.name}: {type(exc).__name__}: {exc}")
                continue
            self.metrics.erros += collector.errors
            print(f"   → {len(vagas)} vagas em {collector.requests_made} requests")
            coletadas.extend((job, collector) for job in vagas)
        self.metrics.coletadas = len(coletadas)
        return coletadas

    def score(self, job: Job) -> MatchResult:
        similaridade = None
        if self.retriever is not None:
            resultado = self.retriever.retrieve(job.requirements_text or job.full_text, k=6)
            if resultado is not None:
                similaridade = resultado.similarity
        return self.matcher.match(job, similaridade)

    def prepare(self) -> list[tuple[Job, MatchResult]]:
        """Etapas 1-7: coleta → dedup → filtros → score → LLM.

        Separado de `run()` para que o diagnóstico execute exatamente o mesmo
        caminho sem enviar nada — evitando um pipeline paralelo que poderia
        divergir do real e mascarar o problema que se quer diagnosticar.
        """
        import datetime
        print(f"▶️  Execução iniciada em {datetime.datetime.now():%d/%m/%Y %H:%M:%S}")

        # 1. Coleta
        coletadas = self.collect(self._collectors())

        # 2. Deduplicação (URL + fingerprint, banco + sessão)
        novas: list[tuple[Job, BaseCollector]] = []
        for job, collector in coletadas:
            if self.repo.ja_enviada(job) or not self.repo.reservar_sessao(job):
                self.metrics.duplicadas += 1
                continue
            novas.append((job, collector))
        print(f"\n🧹 {len(novas)} novas ({self.metrics.duplicadas} duplicadas)")

        # 3. Descrição completa — só para as novas, e só quem ainda não tem
        for job, collector in novas:
            if job.has_description:
                continue
            try:
                if collector.fetch_details(job):
                    self.metrics.detalhes_buscados += 1
            except Exception:
                self.metrics.erros += 1

        # 4. Filtros mínimos + 5/6. score heurístico + semântico
        candidatas: list[tuple[Job, MatchResult]] = []
        for job, _ in novas:
            eleg = check_eligibility(job, self.profile)
            if not eleg.eligible:
                self.metrics.descartar(eleg.reason)
                continue
            job.location_confirmed = eleg.location_confirmed
            candidatas.append((job, self.score(job)))

        print(f"🎯 {len(candidatas)} elegíveis ({self.metrics.descartadas} descartadas "
              f"por filtros)")
        self.metrics.analisadas = len(candidatas)
        if self.retriever is not None:
            self.metrics.chamadas_embedding = getattr(self.retriever.embedder, "calls", 0)

        # 7. LLM só nas melhores acima do piso (§21)
        candidatas.sort(key=_sort_key)
        if self.llm is not None:
            elegiveis = [
                i for i, (_, m) in enumerate(candidatas)
                if m.score >= self.settings.llm_min_score
            ]
            for i in elegiveis[: self.settings.llm_max_jobs]:
                job, base = candidatas[i]
                candidatas[i] = (job, self.analyzer.analyze(job, base))
            self.metrics.chamadas_llm = getattr(self.llm, "calls", 0)
            self.metrics.erros += getattr(self.llm, "errors", 0)
            candidatas.sort(key=_sort_key)  # o LLM reordena as notas

        return candidatas

    def run(self) -> Metrics:
        inicio = time.monotonic()
        candidatas = self.prepare()

        # 8. Envio, do melhor para o pior
        selecionadas = candidatas[: self.settings.max_jobs_per_run]
        if not selecionadas:
            print("\n📭 Nenhuma vaga qualificada para envio nesta execução.")
        else:
            print(f"\n📨 Enviando {len(selecionadas)} de {len(candidatas)} candidatas...")

        for job, match in selecionadas:
            # A vaga só é marcada como enviada se o Telegram confirmar. Falha
            # de envio precisa deixar a vaga elegível para a próxima execução.
            if self.notifier.send(format_message(job, match)):
                self.repo.registrar(job, match.score)
                self.metrics.enviadas += 1
                print(f"   ✅ [{int(match.score):>3}%] {job.title[:55]}")
            else:
                print(f"   ❌ falha no envio, NÃO marcada como enviada: {job.title[:45]}")

        if selecionadas and self.metrics.enviadas == 0:
            print("   ⚠️  nenhuma mensagem chegou ao Telegram — verifique com "
                  "`python main.py --diagnose-telegram`")

        self.metrics.erros += self.notifier.erros
        self.metrics.duracao_seg = time.monotonic() - inicio
        self.repo.registrar_execucao(self.metrics.to_dict())
        return self.metrics

    def close(self) -> None:
        self.repo.close()

    # --- relatório ---

    def report(self) -> str:
        m = self.metrics
        linhas = [
            "",
            "📊 RESUMO DA EXECUÇÃO",
            f"   coletadas ............ {m.coletadas}",
            f"   duplicadas ........... {m.duplicadas}",
            f"   descartadas .......... {m.descartadas}",
            f"   analisadas ........... {m.analisadas}",
            f"   enviadas ............. {m.enviadas}",
            f"   descrições buscadas .. {m.detalhes_buscados}",
            f"   chamadas embedding ... {m.chamadas_embedding}",
            f"   chamadas LLM ......... {m.chamadas_llm}",
            f"   erros ................ {m.erros}",
            f"   duração .............. {m.duracao_seg:.1f}s",
        ]
        if m.motivos_descarte:
            linhas.append("   motivos de descarte:")
            for motivo, qtd in sorted(m.motivos_descarte.items(), key=lambda x: -x[1]):
                linhas.append(f"      - {motivo}: {qtd}")
        for aviso in m.avisos:
            linhas.append(f"   ⚠️  {aviso}")
        return "\n".join(linhas)
