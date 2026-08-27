#!/usr/bin/env python3
"""Relatórios prontos sobre vagas_gupy.db — sem precisar lembrar SQL.

    python tools/db.py top [N]          # melhores vagas já analisadas (default 20)
    python tools/db.py execucoes [N]    # histórico de execuções (default 10)
    python tools/db.py fontes           # desempenho por fonte (Gupy/LinkedIn/ProgramaThor)
    python tools/db.py resumo           # visão geral do banco

Só lê o banco — nenhum comando aqui grava ou apaga nada.
"""

from __future__ import annotations

import os
import sqlite3
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PADRAO = os.path.join(RAIZ, "vagas_gupy.db")


def conectar(caminho: str = DB_PADRAO) -> sqlite3.Connection:
    if not os.path.exists(caminho):
        print(f"❌ banco não encontrado: {caminho}")
        sys.exit(1)
    conn = sqlite3.connect(caminho)
    conn.row_factory = sqlite3.Row
    return conn


def cmd_resumo(conn: sqlite3.Connection) -> None:
    total = conn.execute("SELECT COUNT(*) FROM vagas_enviadas").fetchone()[0]
    com_score = conn.execute(
        "SELECT COUNT(*) FROM vagas_enviadas WHERE score IS NOT NULL"
    ).fetchone()[0]
    execucoes = conn.execute("SELECT COUNT(*) FROM execucoes").fetchone()[0]

    print(f"📊 RESUMO DO BANCO\n{'='*60}")
    print(f"  total de vagas registradas .... {total}")
    print(f"  com score (arquitetura nova) .. {com_score}")
    print(f"  sem score (bot legado) ........ {total - com_score}")
    print(f"  execuções registradas ......... {execucoes}")

    if com_score:
        r = conn.execute(
            "SELECT MIN(score), MAX(score), AVG(score) FROM vagas_enviadas WHERE score IS NOT NULL"
        ).fetchone()
        print(f"\n  score: min {r[0]:.1f}  ·  média {r[2]:.1f}  ·  max {r[1]:.1f}")

    r = conn.execute(
        "SELECT MIN(enviado_em), MAX(enviado_em) FROM vagas_enviadas WHERE enviado_em IS NOT NULL"
    ).fetchone()
    if r[0]:
        print(f"  período registrado: {r[0]} → {r[1]}")


def cmd_top(conn: sqlite3.Connection, n: int = 20) -> None:
    linhas = conn.execute(
        """SELECT score, titulo, empresa, fonte, modelo_trabalho, enviado_em
           FROM vagas_enviadas
           WHERE score IS NOT NULL
           ORDER BY score DESC, enviado_em DESC
           LIMIT ?""",
        (n,),
    ).fetchall()
    if not linhas:
        print("Nenhuma vaga com score ainda — rode o bot com a arquitetura nova primeiro.")
        return
    print(f"🏆 TOP {len(linhas)} VAGAS\n{'='*90}")
    for r in linhas:
        modelo = (r["modelo_trabalho"] or "?")[:9]
        print(f"  {r['score']:5.1f}%  {r['titulo'][:44]:46} {r['empresa'][:20]:22} "
              f"{r['fonte'] or '?':12} {modelo}")


def cmd_execucoes(conn: sqlite3.Connection, n: int = 10) -> None:
    linhas = conn.execute(
        """SELECT executado_em, coletadas, duplicadas_ou_descartadas, analisadas,
                  enviadas, chamadas_embedding, chamadas_llm, erros, duracao_seg
           FROM (
             SELECT executado_em, coletadas, descartadas AS duplicadas_ou_descartadas,
                    analisadas, enviadas, chamadas_embedding, chamadas_llm, erros, duracao_seg
             FROM execucoes
           )
           ORDER BY executado_em DESC
           LIMIT ?""",
        (n,),
    ).fetchall()
    if not linhas:
        print("Nenhuma execução registrada ainda.")
        return
    print(f"🕒 ÚLTIMAS {len(linhas)} EXECUÇÕES\n{'='*100}")
    print(f"  {'quando':20} {'coletadas':>9} {'descart.':>9} {'analis.':>8} "
          f"{'enviadas':>9} {'embed':>6} {'llm':>4} {'erros':>6} {'seg':>7}")
    for r in linhas:
        print(f"  {r['executado_em'][:19]:20} {r['coletadas']:>9} {r['duplicadas_ou_descartadas']:>9} "
              f"{r['analisadas']:>8} {r['enviadas']:>9} {r['chamadas_embedding']:>6} "
              f"{r['chamadas_llm']:>4} {r['erros']:>6} {r['duracao_seg']:>7.1f}")


def cmd_fontes(conn: sqlite3.Connection) -> None:
    linhas = conn.execute(
        """SELECT fonte, COUNT(*) n, ROUND(AVG(score), 1) media,
                  ROUND(MIN(score), 1) minimo, ROUND(MAX(score), 1) maximo
           FROM vagas_enviadas
           WHERE score IS NOT NULL AND fonte IS NOT NULL
           GROUP BY fonte
           ORDER BY n DESC"""
    ).fetchall()
    if not linhas:
        print("Nenhuma vaga com fonte e score registrados ainda.")
        return
    print(f"📡 DESEMPENHO POR FONTE\n{'='*60}")
    for r in linhas:
        print(f"  {r['fonte']:14} {r['n']:>4} vagas   média {r['media']:>5.1f}%   "
              f"(min {r['minimo']:.1f} · max {r['maximo']:.1f})")


COMANDOS = {"resumo": cmd_resumo, "top": cmd_top, "execucoes": cmd_execucoes, "fontes": cmd_fontes}


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] not in COMANDOS:
        print(__doc__)
        return 0 if not args else 1

    conn = conectar()
    try:
        cmd = COMANDOS[args[0]]
        extra = [int(args[1])] if len(args) > 1 and args[1].isdigit() else []
        cmd(conn, *extra)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
