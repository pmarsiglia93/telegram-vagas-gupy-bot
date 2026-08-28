"""Análise de aderência por LLM, alimentada pelo contexto recuperado via RAG.

O LLM recebe:
  • os requisitos reais da vaga (não só o título);
  • os 4-6 chunks do perfil mais relevantes para AQUELA vaga;
  • a EVIDÊNCIA de cada competência (profissional, projeto, hands-on, curso,
    estudo), com instrução explícita de nunca promover um nível a outro.

O ponto central: conhecimento real conta no score, mas nunca é apresentado
como experiência profissional formal.

A saída é validada antes de ser usada. Saída inválida = fallback heurístico.
"""

from __future__ import annotations

from ..domain.job import Job
from ..domain.match import MatchResult
from ..domain.profile import LEVEL_LABEL, Profile
from ..domain.text import truncate
from ..rag.retriever import RetrievalResult

ANALYSIS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "description": "Aderência de 0 a 100"},
        "job_type": {"type": "string", "description": "Frontend, Backend, Full Stack, AI Engineer, ..."},
        "strengths": {
            "type": "array", "items": {"type": "string"},
            "description": "Requisitos cobertos por EXPERIÊNCIA PROFISSIONAL",
        },
        "practical_experience": {
            "type": "array", "items": {"type": "string"},
            "description": "Requisitos cobertos por projeto próprio, POC ou hands-on",
        },
        "education": {
            "type": "array", "items": {"type": "string"},
            "description": "Requisitos cobertos por formação/bootcamp/certificação concluída",
        },
        "related_knowledge": {
            "type": "array", "items": {"type": "string"},
            "description": "Requisitos cobertos por estudo em andamento",
        },
        "partial_matches": {
            "type": "array", "items": {"type": "string"},
            "description": "Tecnologias transferíveis, no formato 'Pedida ← Equivalente que possui'",
        },
        "gaps": {"type": "array", "items": {"type": "string"}},
        "relevant_experiences": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string", "description": "2 a 3 frases em português"},
    },
    "required": [
        "score", "job_type", "strengths", "practical_experience",
        "education", "related_knowledge", "partial_matches", "gaps",
        "relevant_experiences", "reason",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """Você avalia a aderência entre um candidato e uma vaga de tecnologia.

O QUE CONTA COMO COMPETÊNCIA

Experiência profissional é a evidência MAIS FORTE, mas NÃO é a única válida.
Projetos, POCs, cursos, certificações e experimentação prática demonstram
conhecimento real e devem contar no score.

Cada item do perfil vem rotulado com sua evidência. Ordem de peso:

  1. experiência profissional / freelance  — usou em produção, remunerado
  2. projeto próprio em produção           — construiu e roda de verdade
  3. projeto próprio / acadêmico           — construiu e publicou
  4. experimentação prática (hands-on)     — POC, laboratório
  5. formação / bootcamp / certificação    — concluída, com projetos avaliados
  6. em estudo
  7. tecnologia transferível               — equivalente da mesma família

REGRAS INEGOCIÁVEIS

1. NUNCA promova um nível de evidência a outro. "Estudou Java" jamais vira
   "tem experiência com Java". "Implementou RAG num projeto próprio" jamais
   vira "tem experiência profissional com RAG".

2. NUNCA invente anos de experiência, empresas, cargos ou datas. Se o perfil
   não traz esse dado, ele não existe — não estime, não arredonde, não sugira.
   Em especial: "usou X profissionalmente" NÃO é "tem N anos de X". Quando a
   vaga exigir tempo ("3+ anos de Java") e o perfil só comprovar uso
   profissional, trate como LACUNA PARCIAL de tempo — nunca como ausência da
   tecnologia, e nunca afirmando um número de anos que o perfil não declara.

3. Uma skill estudada ou de projeto vale MENOS que uma profissional, mas NÃO
   pode ser ignorada nem tratada como ausência. É errado concluir
   "não tem experiência profissional com RAG, portanto baixa compatibilidade".
   O correto é algo como "não há experiência profissional formal com RAG, mas
   há implementação prática em projeto diretamente relacionada aos requisitos".

4. Tecnologia equivalente ou relacionada é COMPATIBILIDADE PARCIAL, nunca
   ausência total. A vaga pede Pinecone e o perfil tem ChromaDB, embeddings e
   RAG? Isso é partial_match. A vaga pede Zapier e o perfil tem Make e n8n?
   partial_match. A vaga pede FastAPI e o perfil tem Python e Django?
   partial_match. Só é `gap` o que não tem nenhuma relação real.

5. Senioridade do título (júnior/pleno/sênior/staff) é NEUTRA. Não reduza o
   score porque a vaga diz "Sênior".

6. Modelo de trabalho não penaliza. Remoto, híbrido e presencial são válidos.

7. O score mede ADERÊNCIA aos requisitos, não probabilidade de contratação.
   Referência: 90+ excelente, 80-89 alta, 70-79 boa, 60-69 razoável,
   50-59 possível, abaixo de 50 baixa.

8. Não infle. Conhecimento transferível ajuda quando há proximidade real. Uma
   vaga que exige anos de Kubernetes, Terraform e SRE não fica com score alto
   porque o candidato conhece Docker.

CAMPOS DA RESPOSTA — a separação entre eles é o ponto central da análise

- strengths: requisitos cobertos por EXPERIÊNCIA PROFISSIONAL.
- practical_experience: cobertos por projeto próprio, POC ou hands-on.
- education: cobertos por formação, bootcamp ou certificação concluída.
- related_knowledge: cobertos por estudo em andamento.
- partial_matches: cobertos por tecnologia transferível. Use o formato
  "Pedida ← Equivalente que possui" (ex.: "Pinecone ← ChromaDB").
- gaps: exigidos pela vaga e sem nenhuma relação com o perfil.
- relevant_experiences: títulos dos itens do perfil que você realmente usou —
  copie do contexto, não invente.
- reason: 2 a 3 frases em português. Quando a cobertura vier de projeto ou
  estudo, diga isso explicitamente em vez de chamar de experiência.

Use apenas o contexto fornecido. Se a descrição da vaga for pobre, diga isso em
`reason` e seja conservador no score em vez de supor requisitos."""


def build_user_prompt(job: Job, profile: Profile, retrieval: RetrievalResult | None) -> str:
    linhas = [
        "## VAGA",
        f"Título: {job.title}",
        f"Empresa: {job.company}",
        f"Localização: {job.location_label}"
        + ("" if job.location_confirmed else " (não confirmada)"),
        f"Modelo de trabalho: {job.work_model.label}",
        f"Senioridade no título: {job.seniority} (informação contextual — não penalize)",
    ]
    if job.tags:
        linhas.append("Stack declarada: " + ", ".join(job.tags[:12]))

    requisitos = job.sections.get("requirements", "")
    responsabilidades = job.sections.get("responsibilities", "")
    desejaveis = job.nice_to_have_text

    if requisitos:
        linhas += ["", "### Requisitos obrigatórios", truncate(requisitos, 2500)]
    if responsabilidades:
        linhas += ["", "### Responsabilidades", truncate(responsabilidades, 1500)]
    if desejaveis:
        linhas += ["", "### Diferenciais", truncate(desejaveis, 900)]
    if not (requisitos or responsabilidades):
        corpo = job.description or "(descrição não disponível)"
        linhas += ["", "### Descrição", truncate(corpo, 3000)]

    linhas += ["", "## PERFIL DO CANDIDATO — TRECHOS RELEVANTES"]
    if retrieval and retrieval.hits:
        for hit in retrieval.hits:
            nivel = hit.record.metadata.get("experience_level", "knowledge")
            linhas.append(
                f"\n[{LEVEL_LABEL.get(nivel, nivel).upper()}] "
                f"{hit.record.metadata.get('title', hit.record.id)}"
            )
            linhas.append(truncate(hit.record.text, 700))
    else:
        # RAG indisponível: manda o resumo do perfil em vez de falhar (§22).
        linhas.append("(recuperação semântica indisponível — resumo do perfil)")
        linhas.append(profile.profile_summary_text())

    # O perfil pode marcar skills como `professional` sem trazer o histórico de
    # empregos. Nesse caso a competência é real, mas empresa, cargo e tempo de
    # casa simplesmente não existem no perfil — e não podem ser estimados.
    if not profile.has_professional_experience:
        if profile.professional_skills:
            linhas += [
                "",
                "OBSERVAÇÃO: o perfil marca competências como experiência profissional, "
                "mas NÃO detalha empregos. Não cite empresas, cargos, datas nem tempo "
                "de experiência — esses dados não existem no perfil.",
            ]
        else:
            linhas += [
                "",
                "OBSERVAÇÃO: o perfil não declara experiência profissional formal. "
                "Não afirme anos de experiência em lugar nenhum da resposta.",
            ]

    linhas += ["", "Responda no formato JSON especificado."]
    return "\n".join(linhas)


def _lista(valor, limite: int = 8) -> list[str]:
    if not isinstance(valor, list):
        return []
    saida = []
    for item in valor[:limite]:
        texto = str(item).strip()
        if texto:
            saida.append(truncate(texto, 80))
    return saida


def validate_analysis(bruto: dict) -> MatchResult:
    """Valida a saída do LLM. Levanta ValueError se inutilizável.

    Dataclasses + validação explícita em vez de Pydantic: são ~30 linhas, uma
    dependência a menos no CI, e o contrato é pequeno e estável.
    """
    if not isinstance(bruto, dict):
        raise ValueError("resposta não é um objeto JSON")

    try:
        score = float(bruto["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("campo 'score' ausente ou não numérico") from exc
    if not 0 <= score <= 100:
        raise ValueError(f"score fora da faixa 0-100: {score}")

    reason = str(bruto.get("reason", "")).strip()
    if not reason:
        raise ValueError("campo 'reason' vazio")

    resultado = MatchResult(
        score=score,
        engine="llm",
        job_type=truncate(str(bruto.get("job_type", "") or "Tecnologia"), 40),
        strengths=_lista(bruto.get("strengths")),
        practical_experience=_lista(bruto.get("practical_experience")),
        education=_lista(bruto.get("education")),
        related_knowledge=_lista(bruto.get("related_knowledge")),
        partial_matches=_lista(bruto.get("partial_matches")),
        gaps=_lista(bruto.get("gaps")),
        relevant_experiences=_lista(bruto.get("relevant_experiences"), 5),
        reason=truncate(reason, 420),
    )
    return resultado.finalize()


class JobAnalyzer:
    """Orquestra retriever + LLM, com fallback silencioso para o score base."""

    def __init__(self, profile: Profile, retriever, llm) -> None:
        self.profile = profile
        self.retriever = retriever
        self.llm = llm
        self.failures = 0

    def analyze(self, job: Job, base: MatchResult) -> MatchResult:
        """Devolve a análise do LLM, ou `base` inalterado se algo falhar."""
        if self.llm is None:
            return base

        retrieval = None
        if self.retriever is not None:
            retrieval = self.retriever.retrieve(job.requirements_text or job.full_text, k=6)

        try:
            bruto = self.llm.complete_json(
                SYSTEM_PROMPT,
                build_user_prompt(job, self.profile, retrieval),
                ANALYSIS_SCHEMA,
            )
            resultado = validate_analysis(bruto)
        except Exception as exc:
            self.failures += 1
            self.llm.errors = getattr(self.llm, "errors", 0) + 1
            base.notes.append(f"llm_falhou:{type(exc).__name__}")
            return base

        # Métricas do pipeline base seguem úteis para diagnóstico.
        resultado.heuristic_score = base.heuristic_score
        resultado.semantic_similarity = base.semantic_similarity
        resultado.required_coverage = base.required_coverage
        resultado.hits = base.hits
        resultado.notes = base.notes
        if retrieval and not resultado.relevant_experiences:
            resultado.relevant_experiences = retrieval.titles[:3]
        return resultado
