"""Configuração — tudo por variável de ambiente / .env. Nenhum segredo no código."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def load_env(caminho: str = "") -> None:
    """Lê o .env sem depender de python-dotenv (mantém o comportamento original).

    `setdefault`: variáveis já exportadas (GitHub Actions) têm precedência.
    """
    caminho = caminho or os.path.join(RAIZ, ".env")
    if not os.path.exists(caminho):
        return
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha and not linha.startswith("#") and "=" in linha:
                chave, valor = linha.split("=", 1)
                os.environ.setdefault(chave.strip(), valor.strip().strip('"').strip("'"))


def _bool(nome: str, padrao: bool = False) -> bool:
    valor = os.getenv(nome, "").strip().lower()
    if not valor:
        return padrao
    return valor in ("1", "true", "yes", "sim", "on")


def _int(nome: str, padrao: int) -> int:
    try:
        return int(os.getenv(nome, "").strip() or padrao)
    except ValueError:
        return padrao


@dataclass
class Settings:
    # --- Telegram (obrigatório) ---
    telegram_token: str = ""
    chat_id: str = ""

    # --- Caminhos ---
    db_path: str = ""
    profile_path: str = ""

    # --- Coleta ---
    max_age_days: int = 3
    max_jobs_per_run: int = 40

    # --- Embeddings / RAG ---
    embedding_provider: str = "hashing"   # hashing | openai | gemini
    embedding_model: str = ""
    vector_store: str = "memory"          # memory | chroma
    chroma_path: str = ""

    # --- LLM ---
    llm_provider: str = "none"            # none | anthropic | openai | gemini
    llm_model: str = ""
    llm_max_jobs: int = 15
    llm_min_score: float = 45.0
    llm_timeout: int = 60

    # --- Chaves ---
    api_keys: dict[str, str] = field(default_factory=dict)

    # --- Operação ---
    dry_run: bool = False
    verbose: bool = True

    @property
    def embedding_key(self) -> str:
        return self.api_keys.get(self.embedding_provider, "")

    @property
    def llm_key(self) -> str:
        return self.api_keys.get(self.llm_provider, "")

    @property
    def telegram_ok(self) -> bool:
        return bool(self.telegram_token and self.chat_id)


def load_settings() -> Settings:
    load_env()
    return Settings(
        telegram_token=os.getenv("TELEGRAM_TOKEN", ""),
        chat_id=os.getenv("CHAT_ID_GRUPO", ""),
        db_path=os.getenv("DB_PATH", os.path.join(RAIZ, "vagas_gupy.db")),
        profile_path=os.getenv("PROFILE_PATH", os.path.join(RAIZ, "profile.yaml")),
        max_age_days=_int("MAX_AGE_DAYS", 3),
        max_jobs_per_run=_int("MAX_JOBS_PER_RUN", 40),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "hashing").strip().lower(),
        embedding_model=os.getenv("EMBEDDING_MODEL", ""),
        vector_store=os.getenv("VECTOR_STORE", "memory").strip().lower(),
        chroma_path=os.getenv("CHROMA_PATH", ""),
        llm_provider=os.getenv("LLM_PROVIDER", "none").strip().lower(),
        llm_model=os.getenv("LLM_MODEL", ""),
        llm_max_jobs=_int("LLM_MAX_JOBS", 15),
        llm_min_score=float(os.getenv("LLM_MIN_SCORE", "45") or 45),
        llm_timeout=_int("LLM_TIMEOUT", 60),
        api_keys={
            "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
            "openai": os.getenv("OPENAI_API_KEY", ""),
            "gemini": os.getenv("GEMINI_API_KEY", ""),
        },
        dry_run=_bool("DRY_RUN"),
        verbose=not _bool("QUIET"),
    )
