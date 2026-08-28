"""Diagnóstico do pipeline e da integração com o Telegram.

Existe porque "não recebi vagas" tem muitas causas possíveis — scheduler que
não disparou, todas as vagas filtradas, dedup engolindo tudo, token revogado,
bot removido do grupo — e sem isto era preciso adivinhar qual delas.

Roda exatamente as mesmas etapas do pipeline real (via `Pipeline.prepare`),
sem enviar nada. Nenhuma função aqui imprime o token completo.
"""

from __future__ import annotations

from .config.settings import Settings
from .notifications.telegram import TelegramNotifier, format_message
from .pipeline import Pipeline
from .scheduler_info import describe_schedule, read_cron_expressions

OK = "✅"
FALHA = "❌"
ALERTA = "⚠️ "

MENSAGEM_TESTE = "✅ Teste de integração: Telegram conectado com sucesso."


def _linha(rotulo: str, ok: bool, detalhe: str = "") -> str:
    marca = OK if ok else FALHA
    return f"  {marca} {rotulo:<28} {detalhe}"


def test_telegram(settings: Settings) -> int:
    """Envia uma única mensagem de teste. Não toca no banco nem coleta vagas."""
    print("🧪 TESTE DE ENVIO — Telegram\n" + "=" * 60)
    notifier = TelegramNotifier(settings.telegram_token, settings.chat_id, settings.dry_run)
    print(f"  token: {notifier.mask_token()}   chat_id: {settings.chat_id or '(ausente)'}")

    ok_token, detalhe_token = notifier.check_token()
    print(_linha("token", ok_token, detalhe_token))
    if not ok_token:
        return 1

    ok_chat, detalhe_chat = notifier.check_chat()
    print(_linha("chat", ok_chat, detalhe_chat))
    if not ok_chat:
        return 1

    if notifier.send(MENSAGEM_TESTE):
        print(_linha("envio", True, "mensagem entregue"))
        return 0
    print(_linha("envio", False, "a API aceitou token e chat, mas recusou a mensagem"))
    return 1


def diagnose(settings: Settings) -> int:
    """Percorre o pipeline inteiro sem enviar e aponta onde ele para."""
    print("🩺 DIAGNÓSTICO DO PIPELINE\n" + "=" * 60)

    # --- Agendador --------------------------------------------------------
    # Este projeto não roda um scheduler próprio: quem dispara é o `schedule:`
    # do GitHub Actions. Aqui só dá para verificar se ele está CONFIGURADO —
    # se o GitHub de fato criou a execução só se confirma no histórico do
    # Actions, não daqui.
    print("\n[1/5] Agendador (GitHub Actions)")
    crons = read_cron_expressions()
    print(_linha("agendamento configurado", bool(crons),
                 f"{len(crons)} expressão(ões) cron"))
    for linha in describe_schedule():
        print(f"     {linha}")
    if crons:
        print(f"  {ALERTA}execução efetiva depende do agendador do GitHub — "
              "confirme em Actions › histórico")

    # --- Configuração -----------------------------------------------------
    print("\n[2/5] Configuração")
    print(_linha("perfil", True, settings.profile_path.split("/")[-1]))
    print(_linha("banco", True, settings.db_path.split("/")[-1]))
    print(f"     embeddings={settings.embedding_provider}  vector_store={settings.vector_store}  "
          f"llm={settings.llm_provider}")
    print(f"     max_jobs_per_run={settings.max_jobs_per_run}  max_age_days={settings.max_age_days}")

    # --- Credenciais do Telegram (sem enviar nada) ------------------------
    print("\n[3/5] Credenciais do Telegram")
    notifier = TelegramNotifier(settings.telegram_token, settings.chat_id, dry_run=True)
    print(f"     token: {notifier.mask_token()}")
    ok_token, detalhe_token = notifier.check_token()
    print(_linha("TELEGRAM_TOKEN", ok_token, detalhe_token))
    ok_chat, detalhe_chat = notifier.check_chat()
    print(_linha("CHAT_ID_GRUPO", ok_chat, detalhe_chat))

    # --- Pipeline real, sem envio -----------------------------------------
    print("\n[4/5] Pipeline (execução real, sem enviar)")
    pipeline = Pipeline(settings)
    try:
        candidatas = pipeline.prepare()
        m = pipeline.metrics
        selecionadas = candidatas[: settings.max_jobs_per_run]

        print()
        print(_linha("busca de vagas", m.coletadas > 0, f"{m.coletadas} coletadas"))
        print(_linha("deduplicação", True,
                     f"{m.coletadas - m.duplicadas} restantes ({m.duplicadas} já enviadas)"))
        print(_linha("filtros", True,
                     f"{m.analisadas} elegíveis ({m.descartadas} descartadas)"))
        print(_linha("selecionadas p/ envio", len(selecionadas) > 0,
                     f"{len(selecionadas)} de {len(candidatas)} candidatas"))
        if candidatas:
            melhor = max(c[1].score for c in candidatas)
            print(f"     melhor score: {melhor:.1f}%")
        for aviso in m.avisos:
            print(f"  {ALERTA}{aviso}")
        if m.motivos_descarte:
            print("     motivos de descarte: " + ", ".join(
                f"{k}={v}" for k, v in sorted(m.motivos_descarte.items())))
    finally:
        pipeline.close()

    # --- Veredito ---------------------------------------------------------
    print("\n[5/5] Veredito")
    problemas: list[str] = []
    if not ok_token:
        problemas.append(f"token do Telegram: {detalhe_token}")
    if not ok_chat:
        problemas.append(f"chat do Telegram: {detalhe_chat}")
    if m.coletadas == 0:
        problemas.append("nenhuma vaga coletada — as fontes estão inacessíveis daqui")
    elif m.analisadas == 0 and m.duplicadas >= m.coletadas:
        problemas.append("todas as vagas coletadas já haviam sido enviadas (dedup)")
    elif m.analisadas == 0:
        problemas.append("todas as vagas foram descartadas pelos filtros de elegibilidade")
    elif not selecionadas:
        problemas.append("nenhuma vaga sobrou após ordenação/limite")

    if problemas:
        for p in problemas:
            print(f"  {FALHA} {p}")
        return 1

    print(f"  {OK} pipeline saudável: {len(selecionadas)} vaga(s) prontas para envio")
    if selecionadas:
        job, match = selecionadas[0]
        print("\n     prévia da melhor vaga:")
        for linha in format_message(job, match).splitlines()[:6]:
            print(f"       {linha}")
    print("\n  Se as vagas não chegam AUTOMATICAMENTE mas o pipeline está saudável,")
    print("  o problema está no agendador (cron), não na aplicação.")
    return 0
