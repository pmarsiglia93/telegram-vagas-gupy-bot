#!/usr/bin/env python3
"""JobMatch AI — ponto de entrada.

Uso:
    python main.py             # execução normal
    DRY_RUN=1 python main.py   # imprime as mensagens em vez de enviar
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from jobmatch.config.settings import load_settings  # noqa: E402
from jobmatch.pipeline import Pipeline  # noqa: E402


def main() -> int:
    settings = load_settings()

    if not settings.telegram_ok and not settings.dry_run:
        print("❌ ERRO: TELEGRAM_TOKEN ou CHAT_ID_GRUPO não encontrados no .env")
        print("   (use DRY_RUN=1 para testar sem enviar)")
        return 1

    if not os.path.exists(settings.profile_path):
        print(f"❌ ERRO: perfil não encontrado em {settings.profile_path}")
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
