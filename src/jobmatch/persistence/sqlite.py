"""Persistência operacional em SQLite.

O SQLite continua dono do histórico, das URLs já enviadas e da deduplicação
(§14). O ChromaDB só cuida de embeddings — as responsabilidades não se
misturam.

Compatibilidade: a tabela `vagas_enviadas` do bot original é preservada com o
mesmo esquema. As colunas novas são adicionadas por migração aditiva, então o
banco existente (48 vagas) continua válido.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from ..domain.job import Job

SCHEMA = """
CREATE TABLE IF NOT EXISTS vagas_enviadas (
    link TEXT PRIMARY KEY,
    data_publicacao TEXT,
    titulo TEXT
);
CREATE TABLE IF NOT EXISTS execucoes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    executado_em TEXT,
    coletadas INTEGER,
    descartadas INTEGER,
    analisadas INTEGER,
    enviadas INTEGER,
    chamadas_embedding INTEGER,
    chamadas_llm INTEGER,
    erros INTEGER,
    duracao_seg REAL
);
"""

# Colunas acrescentadas depois da v1 do bot. ALTER TABLE aditivo por coluna.
COLUNAS_NOVAS: list[tuple[str, str]] = [
    ("fingerprint", "TEXT"),
    ("empresa", "TEXT"),
    ("fonte", "TEXT"),
    ("score", "REAL"),
    ("modelo_trabalho", "TEXT"),
    ("enviado_em", "TEXT"),
]


class JobRepository:
    def __init__(self, caminho: str) -> None:
        self.conn = sqlite3.connect(caminho)
        self.conn.row_factory = sqlite3.Row
        self._migrar()
        self._fingerprints_sessao: set[str] = set()

    def _migrar(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(SCHEMA)
        existentes = {r["name"] for r in cur.execute("PRAGMA table_info(vagas_enviadas)")}
        for nome, tipo in COLUNAS_NOVAS:
            if nome not in existentes:
                cur.execute(f"ALTER TABLE vagas_enviadas ADD COLUMN {nome} {tipo}")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_vagas_fingerprint ON vagas_enviadas(fingerprint)"
        )
        self.conn.commit()

    # --- Deduplicação ---

    def ja_enviada(self, job: Job) -> bool:
        """True se a URL já foi enviada, OU se uma vaga equivalente já foi.

        A segunda checagem é a melhoria pedida em §20: a mesma vaga publicada na
        Gupy e no LinkedIn tem URLs diferentes mas o mesmo fingerprint
        (empresa + cargo normalizado + cidade).
        """
        fp = job.fingerprint()
        if fp in self._fingerprints_sessao:
            return True
        cur = self.conn.cursor()
        if cur.execute("SELECT 1 FROM vagas_enviadas WHERE link = ?", (job.url,)).fetchone():
            return True
        if cur.execute("SELECT 1 FROM vagas_enviadas WHERE fingerprint = ?", (fp,)).fetchone():
            return True
        return False

    def reservar_sessao(self, job: Job) -> bool:
        """Marca o fingerprint como visto nesta execução. False se repetido."""
        fp = job.fingerprint()
        if fp in self._fingerprints_sessao:
            return False
        self._fingerprints_sessao.add(fp)
        return True

    # --- Escrita ---

    def registrar(self, job: Job, score: float) -> None:
        data = job.published_at.strftime("%d/%m/%Y") if job.published_at else "Sem data"
        self.conn.execute(
            """INSERT OR IGNORE INTO vagas_enviadas
               (link, data_publicacao, titulo, fingerprint, empresa, fonte,
                score, modelo_trabalho, enviado_em)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job.url, data, job.title, job.fingerprint(), job.company, job.source,
                round(score, 1), job.work_model.value, datetime.now().isoformat(timespec="seconds"),
            ),
        )
        self.conn.commit()

    def registrar_execucao(self, metricas: dict) -> None:
        self.conn.execute(
            """INSERT INTO execucoes
               (executado_em, coletadas, descartadas, analisadas, enviadas,
                chamadas_embedding, chamadas_llm, erros, duracao_seg)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(timespec="seconds"),
                metricas.get("coletadas", 0), metricas.get("descartadas", 0),
                metricas.get("analisadas", 0), metricas.get("enviadas", 0),
                metricas.get("chamadas_embedding", 0), metricas.get("chamadas_llm", 0),
                metricas.get("erros", 0), round(metricas.get("duracao_seg", 0.0), 2),
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
