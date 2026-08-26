"""Formatação e envio das mensagens do Telegram."""

from __future__ import annotations

import html
import time

import requests

from ..domain.job import Job
from ..domain.match import MatchResult
from ..domain.text import truncate

API = "https://api.telegram.org/bot{token}/sendMessage"
LIMITE_TELEGRAM = 4096

EMOJI_FONTE = {"GUPY": "🟣", "LINKEDIN": "🔷", "PROGRAMATHOR": "🟤"}


def esc(texto) -> str:
    """Escapa HTML. Herdado do bot original — evita quebrar o parse_mode."""
    return html.escape(str(texto)) if texto else ""


def _bloco(titulo: str, itens: list[str], limite: int = 4) -> list[str]:
    if not itens:
        return []
    linhas = [titulo]
    linhas += [f"• {esc(i)}" for i in itens[:limite]]
    restantes = len(itens) - limite
    if restantes > 0:
        linhas.append(f"• +{restantes}")
    return linhas


def format_message(job: Job, match: MatchResult) -> str:
    """Mensagem curta o suficiente para ser útil na timeline do Telegram (§18)."""
    score = int(round(match.score))
    linhas = [
        f"{match.emoji} <b>MATCH {score}%</b>",
        "",
        f"💼 <b>{esc(job.title)}</b>",
        f"🏢 {esc(job.company)}",
        "",
    ]

    local = job.location_label
    if not job.location_confirmed:
        local += " (não confirmado)"
    linhas.append(f"📍 {esc(job.work_model.label)} · {esc(local)}")
    linhas.append(f"🎯 {esc(match.classification)}")

    detalhes = [d for d in (job.job_type, match.job_type) if d]
    if detalhes:
        linhas.append(f"🧩 {esc(' · '.join(dict.fromkeys(detalhes)))}")
    if job.salary:
        linhas.append(f"💰 {esc(job.salary)}")
    if job.published_at:
        linhas.append(f"📅 {job.published_at.strftime('%d/%m/%Y às %H:%M')}")

    # Um bloco por tipo de evidência (§11/§16). A separação é o que impede
    # "implementei num projeto" de ser lido como "trabalhei com isso".
    for bloco in (
        _bloco("\n✅ <b>Experiência profissional</b>", match.strengths),
        _bloco("\n🧪 <b>Projetos / hands-on</b>", match.practical_experience, 4),
        _bloco("\n📚 <b>Conhecimento relacionado</b>", match.related_knowledge, 3),
        _bloco("\n🔄 <b>Transferível</b>", match.partial_matches, 3),
        _bloco("\n⚠️ <b>Gaps</b>", match.gaps, 4),
    ):
        linhas += bloco

    if match.reason:
        linhas += ["", "💡 <b>Análise</b>", esc(truncate(match.reason, 380))]

    fonte = EMOJI_FONTE.get(job.source.upper(), "🔗")
    linhas += ["", f"{fonte} <a href='{esc(job.url)}'>Ver vaga na {esc(job.source.title())}</a>"]

    mensagem = "\n".join(linhas)
    if len(mensagem) > LIMITE_TELEGRAM:
        mensagem = mensagem[: LIMITE_TELEGRAM - 40].rsplit("\n", 1)[0] + "\n…"
    return mensagem


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, dry_run: bool = False, pausa: float = 2.0) -> None:
        self.token = token
        self.chat_id = chat_id
        self.dry_run = dry_run
        self.pausa = pausa
        self.enviadas = 0
        self.erros = 0

    def send(self, mensagem: str) -> bool:
        if self.dry_run:
            print("\n--- DRY RUN ---\n" + mensagem + "\n---------------")
            self.enviadas += 1
            return True
        try:
            r = requests.post(
                API.format(token=self.token),
                json={
                    "chat_id": self.chat_id,
                    "text": mensagem,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=15,
            )
            if r.status_code != 200:
                print(f"   ⚠️  Telegram recusou: {r.text[:200]}")
                self.erros += 1
                return False
            self.enviadas += 1
            time.sleep(self.pausa)  # respeita o rate limit do Telegram
            return True
        except Exception as exc:
            print(f"   ❌ Erro Telegram: {exc}")
            self.erros += 1
            return False
