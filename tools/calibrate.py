#!/usr/bin/env python3
"""Calibração: compara o scoring ANTES e DEPOIS do modelo de evidência.

    python tools/calibrate.py            # coleta vagas reais (usa cache)
    python tools/calibrate.py --refresh  # força nova coleta

O "ANTES" é reconstruído fielmente: pesos por nível único (professional 1.00 /
project 0.80 / knowledge 0.65 / study 0.45), sem taxonomia de grupos, sem bônus
de competência emergente e sem encolhimento por confiança. É exatamente o
comportamento anterior a esta calibração.

Nada aqui roda em produção — é ferramenta de análise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

from jobmatch.collectors.gupy import GupyCollector  # noqa: E402
from jobmatch.collectors.linkedin import LinkedInCollector  # noqa: E402
from jobmatch.collectors.programathor import ProgramaThorCollector  # noqa: E402
from jobmatch.domain import evidence as EV  # noqa: E402
from jobmatch.domain.evidence import Evidence  # noqa: E402
from jobmatch.domain.job import Job, WorkModel  # noqa: E402
from jobmatch.domain.profile import Profile, load_profile  # noqa: E402
from jobmatch.filters.eligibility import check_eligibility  # noqa: E402
from jobmatch.matching import heuristic as H  # noqa: E402
from jobmatch.matching.heuristic import HeuristicMatcher  # noqa: E402

CACHE = "/tmp/jobmatch_calibracao.json"

# O modelo anterior tinha 4 níveis e um único por skill. Registramos esses
# pesos como tipos próprios ("legacy_*") em vez de mutar a tabela atual, para
# que a comparação seja fiel sem contaminar o scoring de produção.
NIVEL_ANTIGO = {
    "professional": "legacy_professional", "freelance": "legacy_professional",
    "production_project": "legacy_project", "project": "legacy_project",
    "academic_project": "legacy_project",
    "knowledge": "legacy_knowledge", "hands_on": "legacy_knowledge",
    "certification": "legacy_knowledge",
    "course": "legacy_study", "study": "legacy_study", "interest": "legacy_study",
}
EV.EVIDENCE_WEIGHT.update({
    "legacy_professional": 1.00,
    "legacy_project": 0.80,
    "legacy_knowledge": 0.65,
    "legacy_study": 0.45,
})


def perfil_legado(profile: Profile) -> Profile:
    """Reconstrói o perfil como ele era antes: nível único, sem grupos.

    O acúmulo de evidências (estudo + projeto + curso na mesma skill) não
    existia — só o nível mais forte contava.
    """
    skills = tuple(
        replace(s, evidence=(Evidence(type=NIVEL_ANTIGO.get(s.level, "legacy_study")),))
        for s in profile.skills
    )
    return replace(
        profile,
        skills=skills,
        skill_groups={},        # sem taxonomia -> sem transferência por grupo
        group_transfer={},
        emergent_groups=(),     # sem bônus de competência emergente
        emergent_bonus=0.0,
    )


class MatcherLegado(HeuristicMatcher):
    """Matcher anterior: cobertura crua, sem encolhimento por confiança."""

    def match(self, job, semantic_similarity=None):
        original = H.PRIOR_PSEUDO_REQUIREMENTS
        H.PRIOR_PSEUDO_REQUIREMENTS = 0.0  # sem prior => cobertura crua
        try:
            return super().match(job, semantic_similarity)
        finally:
            H.PRIOR_PSEUDO_REQUIREMENTS = original


def coletar(profile: Profile, refresh: bool) -> list[Job]:
    if os.path.exists(CACHE) and not refresh:
        with open(CACHE, encoding="utf-8") as f:
            brutos = json.load(f)
        print(f"📂 {len(brutos)} vagas do cache ({CACHE})")
        return [
            Job(
                source=b["source"], title=b["title"], company=b["company"], url=b["url"],
                description=b["description"], raw_location=b["raw_location"],
                city=b.get("city", ""), state=b.get("state", ""),
                country=b.get("country", "Brasil"),
                work_model=WorkModel(b["work_model"]), tags=b.get("tags", []),
            )
            for b in brutos
        ]

    vagas: list[Job] = []
    for cls in (GupyCollector, LinkedInCollector, ProgramaThorCollector):
        c = cls(profile, 3)
        print(f"🔎 coletando {c.name}...")
        novas = c.collect()
        for job in novas:
            if not job.has_description:
                try:
                    c.fetch_details(job)
                except Exception:
                    pass
        vagas.extend(novas)
        print(f"   {len(novas)} vagas")

    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump([{
            "source": j.source, "title": j.title, "company": j.company, "url": j.url,
            "description": j.description, "raw_location": j.raw_location,
            "city": j.city, "state": j.state, "country": j.country,
            "work_model": j.work_model.value, "tags": j.tags,
        } for j in vagas], f, ensure_ascii=False)
    return vagas


def categoria(job: Job, match) -> str:
    return match.job_type or "Tecnologia"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="força nova coleta")
    ap.add_argument("--min-jobs", type=int, default=20)
    ap.add_argument(
        "--baseline", metavar="PERFIL.yaml",
        help="compara MESMO algoritmo com outro perfil (mede o impacto dos DADOS)",
    )
    ap.add_argument(
        "--explain", metavar="TRECHO",
        help="audita uma vaga: requisitos, chunks recuperados e origem de cada evidência",
    )
    args = ap.parse_args()

    profile = load_profile(os.path.join(RAIZ, "profile.yaml"))
    vagas = coletar(profile, args.refresh)

    if args.explain:
        return explicar(profile, vagas, args.explain)

    elegiveis = []
    for job in vagas:
        el = check_eligibility(job, profile)
        if el.eligible:
            job.location_confirmed = el.location_confirmed
            elegiveis.append(job)

    if len(elegiveis) < args.min_jobs:
        print(f"⚠️  só {len(elegiveis)} vagas elegíveis (mínimo {args.min_jobs})")

    if args.baseline:
        # Mesmo algoritmo, perfis diferentes: isola o efeito dos dados.
        antes_matcher = HeuristicMatcher(load_profile(args.baseline))
        rotulo = f"perfil baseline ({os.path.basename(args.baseline)})"
    else:
        antes_matcher = MatcherLegado(perfil_legado(profile))
        rotulo = "algoritmo anterior"
    depois_matcher = HeuristicMatcher(profile)
    print(f"\n   ANTES = {rotulo}")

    linhas = []
    for job in elegiveis:
        a = antes_matcher.match(job)
        d = depois_matcher.match(job)
        linhas.append((job, a, d, d.score - a.score))

    print(f"\n{'='*78}")
    print(f"CALIBRAÇÃO — {len(linhas)} vagas reais")
    print("=" * 78)

    media_a = sum(l[1].score for l in linhas) / len(linhas)
    media_d = sum(l[2].score for l in linhas) / len(linhas)
    print(f"\nScore médio   ANTES {media_a:5.1f}   DEPOIS {media_d:5.1f}   "
          f"({media_d - media_a:+.1f})")

    # Distribuição por categoria de vaga
    print(f"\n{'CATEGORIA':<22} {'N':>3} {'ANTES':>7} {'DEPOIS':>7} {'Δ':>7}")
    print("-" * 78)
    por_cat: dict[str, list] = {}
    for job, a, d, delta in linhas:
        por_cat.setdefault(categoria(job, d), []).append((a.score, d.score))
    for cat, valores in sorted(por_cat.items(), key=lambda x: -len(x[1])):
        ma = sum(v[0] for v in valores) / len(valores)
        md = sum(v[1] for v in valores) / len(valores)
        print(f"{cat:<22} {len(valores):>3} {ma:>7.1f} {md:>7.1f} {md - ma:>+7.1f}")

    # Falsos negativos corrigidos
    print(f"\n{'='*78}\n🔼 MAIORES GANHOS — falsos negativos corrigidos\n{'='*78}")
    for job, a, d, delta in sorted(linhas, key=lambda x: -x[3])[:8]:
        print(f"\nANTES  {a.score:5.1f}%   DEPOIS {d.score:5.1f}%   ({delta:+.1f})")
        print(f"  {job.title[:66]}  [{job.company[:28]}]")
        print(f"  Motivo: {_motivo(a, d)}")

    # Possíveis falsos positivos
    print(f"\n{'='*78}\n🔽 MAIORES QUEDAS — proteção contra superestimação\n{'='*78}")
    quedas = [l for l in sorted(linhas, key=lambda x: x[3]) if l[3] < -1][:5]
    for job, a, d, delta in quedas:
        print(f"\nANTES  {a.score:5.1f}%   DEPOIS {d.score:5.1f}%   ({delta:+.1f})")
        print(f"  {job.title[:66]}  [{job.company[:28]}]")
        print(f"  Motivo: {_motivo(a, d)}")
    if not quedas:
        print("\n  (nenhuma queda relevante)")

    # O que mais importa num sistema de ranking não é o score absoluto, e sim
    # a posição: as vagas certas subiram para o topo da fila de envio?
    print(f"\n{'='*78}\n📊 DESLOCAMENTO NO RANKING (posição média, 1 = topo)\n{'='*78}")
    rank_antes = {id(j): i for i, (j, _, _, _) in enumerate(
        sorted(linhas, key=lambda x: -x[1].score), 1)}
    rank_depois = {id(j): i for i, (j, _, _, _) in enumerate(
        sorted(linhas, key=lambda x: -x[2].score), 1)}
    por_cat_rank: dict[str, list] = {}
    for job, a, d, _ in linhas:
        por_cat_rank.setdefault(categoria(job, d), []).append(
            (rank_antes[id(job)], rank_depois[id(job)])
        )
    print(f"\n{'CATEGORIA':<22} {'N':>3} {'ANTES':>7} {'DEPOIS':>7} {'Δ':>7}")
    print("-" * 78)
    for cat, valores in sorted(por_cat_rank.items(), key=lambda x: -len(x[1])):
        ra = sum(v[0] for v in valores) / len(valores)
        rd = sum(v[1] for v in valores) / len(valores)
        seta = "↑" if rd < ra - 0.5 else ("↓" if rd > ra + 0.5 else "=")
        print(f"{cat:<22} {len(valores):>3} {ra:>7.1f} {rd:>7.1f} {rd - ra:>+6.1f} {seta}")

    # Quantas vagas de IA/automação entram no top-15 (o que de fato é enviado)?
    top_n = 15
    topo_antes = [j for j, _, _, _ in sorted(linhas, key=lambda x: -x[1].score)[:top_n]]
    topo_depois = [j for j, _, _, _ in sorted(linhas, key=lambda x: -x[2].score)[:top_n]]
    cat_de = {id(j): categoria(j, d) for j, _, d, _ in linhas}
    ia_antes = sum(1 for j in topo_antes if cat_de[id(j)] == "AI Engineer")
    ia_depois = sum(1 for j in topo_depois if cat_de[id(j)] == "AI Engineer")
    print(f"\n  Vagas de AI Engineer no top-{top_n}: {ia_antes} -> {ia_depois}")

    # Auditoria de falso positivo: score alto com muitos gaps
    print(f"\n{'='*78}\n🚩 AUDITORIA — score alto com cobertura fraca\n{'='*78}")
    suspeitas = [
        (j, d) for j, a, d, _ in linhas
        if d.score >= 75 and d.required_coverage < 0.45
    ]
    for job, d in sorted(suspeitas, key=lambda x: -x[1].score)[:5]:
        print(f"\n  {d.score:5.1f}%  cobertura {d.required_coverage:.2f}  {job.title[:52]}")
        print(f"         gaps: {', '.join(d.gaps[:5]) or '—'}")
    if not suspeitas:
        print("\n  Nenhuma vaga com score alto e cobertura fraca. ✅")

    return 0


def explicar(profile, vagas, trecho: str) -> int:
    """Auditoria de uma vaga: por que o sistema deu esse score? (§15)"""
    from jobmatch.rag.embeddings import HashingEmbeddingProvider
    from jobmatch.rag.retriever import ProfileRetriever
    from jobmatch.rag.vector_store import InMemoryVectorStore

    alvo = trecho.lower()
    candidatas = [j for j in vagas if alvo in j.title.lower() or alvo in j.company.lower()]
    if not candidatas:
        print(f"❌ nenhuma vaga com '{trecho}' no título ou empresa")
        return 1
    job = candidatas[0]

    el = check_eligibility(job, profile)
    job.location_confirmed = el.location_confirmed

    retriever = ProfileRetriever(profile, HashingEmbeddingProvider(), InMemoryVectorStore())
    retriever.build()
    semantica = retriever.retrieve(job.requirements_text or job.full_text, k=5)
    match = HeuristicMatcher(profile).match(job, semantica.similarity if semantica else None)

    print(f"\n{'='*78}\nVAGA\n{'='*78}")
    print(f"  {job.title}")
    print(f"  {job.company} · {job.location_label} · {job.work_model.label} · {job.source}")
    print(f"  elegível: {el.eligible} ({el.reason or 'ok'})")

    print(f"\n{'='*78}\nSCORE  {match.score:.1f}%  —  {match.classification}\n{'='*78}")
    print(f"  cobertura de requisitos : {match.required_coverage:.2f}")
    print(f"  similaridade semântica  : {match.semantic_similarity:.2f}")
    print(f"  bônus emergente         : +{match.emergent_bonus:.1f}")
    print(f"  aderência de cargo      : {match.job_type}")
    if match.notes:
        print(f"  observações             : {', '.join(match.notes)}")

    print(f"\n{'='*78}\nCHUNKS RECUPERADOS (RAG)\n{'='*78}")
    if semantica:
        for i, hit in enumerate(semantica.hits, 1):
            md = hit.record.metadata
            print(f"  {i}. {md.get('title', hit.record.id)[:52]}")
            print(f"     similaridade {hit.score:.3f} | evidência: "
                  f"{md.get('experience_label', md.get('evidence_type', '?'))}"
                  + (f" | {md.get('company')}" if md.get("company") else ""))
    else:
        print("  (retrieval indisponível)")

    print(f"\n{'='*78}\nREQUISITOS DETECTADOS → EVIDÊNCIA\n{'='*78}")
    rotulo_cat = {
        "professional": "EXPERIÊNCIA PROFISSIONAL",
        "practical": "projeto / hands-on",
        "learned": "curso / estudo",
    }
    for hit in sorted(match.hits, key=lambda h: (-h.coverage, h.required)):
        marca = "obrigatório" if hit.critical else "desejável  "
        if hit.coverage == 0.0:
            print(f"  {hit.required:32} [{marca}]  →  GAP")
            continue
        origem = rotulo_cat.get(hit.category, hit.category)
        via = f" via {hit.matched}" if hit.transferable else ""
        grupo = f" (grupo {hit.group})" if hit.transferable and hit.group else ""
        print(f"  {hit.required:32} [{marca}]  →  {hit.coverage:.2f}  {origem}{via}{grupo}")
        if hit.evidence:
            print(f"  {'':32}      evidência: {hit.evidence}")

    print(f"\n{'='*78}\nBLOCOS DA MENSAGEM\n{'='*78}")
    for rotulo, itens in [
        ("✅ Experiência profissional", match.strengths),
        ("🧪 Projetos / hands-on", match.practical_experience),
        ("📚 Conhecimento relacionado", match.related_knowledge),
        ("🔄 Transferíveis", match.partial_matches),
        ("⚠️  Gaps", match.gaps),
    ]:
        print(f"  {rotulo}: {', '.join(itens) if itens else '—'}")
    print(f"\n  💡 {match.reason}\n")
    return 0


def _motivo(a, d) -> str:
    partes = []
    if d.practical_experience:
        partes.append("evidência de projeto: " + ", ".join(d.practical_experience[:4]))
    novos_parciais = set(d.partial_matches) - set(a.partial_matches)
    if novos_parciais:
        partes.append("transferíveis: " + ", ".join(sorted(novos_parciais)[:3]))
    if d.related_knowledge:
        partes.append("cursos/estudos: " + ", ".join(d.related_knowledge[:2]))
    if d.emergent_bonus > 0:
        partes.append(f"bônus emergente +{d.emergent_bonus:.1f}")
    gaps_resolvidos = set(a.gaps) - set(d.gaps)
    if gaps_resolvidos:
        partes.append("deixaram de ser gap: " + ", ".join(sorted(gaps_resolvidos)[:3]))
    return "; ".join(partes) or "ajuste de confiança sobre poucos requisitos"


if __name__ == "__main__":
    raise SystemExit(main())
