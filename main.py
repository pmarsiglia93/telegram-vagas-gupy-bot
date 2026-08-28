#!/usr/bin/env python3
"""JobMatch AI — ponto de entrada.

Uso:
    python main.py                       # execução normal (coleta e envia)
    DRY_RUN=1 python main.py             # imprime as mensagens em vez de enviar
    python main.py --diagnose            # agendador + pipeline + Telegram, sem enviar
    python main.py --diagnose-telegram   # alias de --diagnose
    python main.py --test-telegram       # envia só uma mensagem de teste
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from jobmatch.config.settings import load_settings  # noqa: E402
from jobmatch.diagnostics import diagnose, test_telegram  # noqa: E402
from jobmatch.pipeline import Pipeline  # noqa: E402

AJUDA = __doc__


def _exige_perfil(settings) -> bool:
    if not os.path.exists(settings.profile_path):
        print(f"❌ ERRO: perfil não encontrado em {settings.profile_path}")
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    settings = load_settings()

    if "--help" in argv or "-h" in argv:
        print(AJUDA)
        return 0

    if "--test-telegram" in argv:
        return test_telegram(settings)

    if "--diagnose" in argv or "--diagnose-telegram" in argv:
        if not _exige_perfil(settings):
            return 1
        return diagnose(settings)

    desconhecidos = [a for a in argv if a.startswith("-")]
    if desconhecidos:
        print(f"❌ argumento desconhecido: {' '.join(desconhecidos)}\n")
        print(AJUDA)
        return 2

    # --- execução normal ---
    if not settings.telegram_ok and not settings.dry_run:
        print("❌ ERRO: TELEGRAM_TOKEN ou CHAT_ID_GRUPO não encontrados no .env")
        print("   (use DRY_RUN=1 para testar sem enviar, ou --diagnose-telegram)")
        return 1

    if not _exige_perfil(settings):
        return 1

    print("🤖 JobMatch AI")
    print(f"   perfil       : {os.path.basename(settings.profile_path)}")
    print(f"   embeddings   : {settings.embedding_provider}")
    print(f"   vector store : {settings.vector_store}")
    print(f"   LLM          : {settings.llm_provider}")
    if settings.dry_run:
        print("   modo         : DRY RUN (nada será enviado)")

    pipeline = Pipeline(settings)
    try:
        pipeline.run()
        print(pipeline.report())
    finally:
        pipeline.close()

    print("\n✅ Varredura completa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
