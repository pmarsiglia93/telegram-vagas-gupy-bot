"""Validação da saída do LLM, deduplicação, ordenação e mensagem do Telegram."""

import os
import tempfile

import pytest

from jobmatch.ai.analyzer import JobAnalyzer, build_user_prompt, validate_analysis
from jobmatch.domain.job import Job, WorkModel
from jobmatch.domain.match import MatchResult
from jobmatch.notifications.telegram import format_message
from jobmatch.persistence.sqlite import JobRepository
from jobmatch.pipeline import _sort_key


def vaga(titulo="Software Engineer", empresa="Empresa X", url="https://a.com/1",
         local="São Paulo - SP", modelo=WorkModel.REMOTE, descricao="React e TypeScript."):
    return Job(source="gupy", title=titulo, company=empresa, url=url,
               raw_location=local, work_model=modelo, description=descricao)


SAIDA_VALIDA = {
    "score": 87,
    "job_type": "Full Stack",
    "strengths": ["React", "TypeScript", "REST APIs"],
    "practical_experience": ["RAG", "Embeddings", "ChromaDB"],
    "related_knowledge": ["n8n (curso concluído)"],
    "partial_matches": ["Pinecone ← ChromaDB"],
    "gaps": ["AWS"],
    "relevant_experiences": ["Bloco de competências: Frameworks de interface SPA"],
    "reason": "Forte aderência aos requisitos centrais de frontend.",
}


# --------------------------------------------------------------------------
# §17 — validação da saída da IA
# --------------------------------------------------------------------------

def test_saida_valida_vira_matchresult():
    resultado = validate_analysis(SAIDA_VALIDA)
    assert resultado.score == 87
    assert resultado.classification == "Alta compatibilidade"
    assert resultado.emoji == "🟢"
    assert resultado.engine == "llm"
    assert resultado.strengths == ["React", "TypeScript", "REST APIs"]


@pytest.mark.parametrize("payload", [
    {},
    {"score": 87},                                  # sem reason
    {"score": "muito bom", "reason": "x"},          # score não numérico
    {"score": 150, "reason": "x"},                  # fora da faixa
    {"score": -5, "reason": "x"},
    "não é um objeto",
])
def test_saida_invalida_e_rejeitada(payload):
    with pytest.raises(ValueError):
        validate_analysis(payload)


def test_llm_quebrado_devolve_o_score_base(profile):
    class LLMQuebrado:
        calls = 0
        errors = 0

        def complete_json(self, system, user, schema):
            raise RuntimeError("429 rate limited")

    base = MatchResult(score=72.0, engine="semantic", reason="base").finalize()
    analyzer = JobAnalyzer(profile, None, LLMQuebrado())
    resultado = analyzer.analyze(vaga(), base)

    assert resultado is base
    assert resultado.score == 72.0
    assert analyzer.failures == 1
    assert any(n.startswith("llm_falhou") for n in resultado.notes)


def test_sem_llm_o_score_base_passa_intacto(profile):
    base = MatchResult(score=64.0, engine="heuristic").finalize()
    assert JobAnalyzer(profile, None, None).analyze(vaga(), base) is base


def test_prompt_nao_alerta_quando_ha_experiencia_declarada(profile):
    """Com `experiences` preenchido, o alerta de 'não invente' fica desnecessário."""
    assert profile.has_professional_experience
    prompt = build_user_prompt(vaga(), profile, None)
    assert "NÃO detalha empregos" not in prompt
    assert "não declara experiência profissional formal" not in prompt


def test_prompt_alerta_quando_nao_ha_experiencia_declarada(profile):
    """Perfil sem histórico de empregos continua recebendo o alerta."""
    from dataclasses import replace

    sem_experiencia = replace(
        profile, items=tuple(i for i in profile.items if i.kind != "experience")
    )
    prompt = build_user_prompt(vaga(), sem_experiencia, None)
    assert "NÃO detalha empregos" in prompt


def test_prompt_separa_evidencia_de_experiencia():
    """As regras que impedem promover projeto/estudo a emprego formal."""
    from jobmatch.ai.analyzer import SYSTEM_PROMPT

    assert "NUNCA promova um nível de evidência a outro" in SYSTEM_PROMPT
    assert "NUNCA invente anos de experiência" in SYSTEM_PROMPT
    assert "nem tratada como ausência" in SYSTEM_PROMPT
    assert "COMPATIBILIDADE PARCIAL, nunca" in SYSTEM_PROMPT


def test_saida_do_llm_separa_os_blocos_de_evidencia():
    resultado = validate_analysis(SAIDA_VALIDA)
    assert resultado.strengths == ["React", "TypeScript", "REST APIs"]
    assert resultado.practical_experience == ["RAG", "Embeddings", "ChromaDB"]
    assert resultado.related_knowledge == ["n8n (curso concluído)"]
    assert resultado.partial_matches == ["Pinecone ← ChromaDB"]
    # O que é projeto não pode vazar para o bloco de experiência profissional.
    assert "RAG" not in resultado.strengths


def test_prompt_marca_senioridade_como_contextual(profile):
    prompt = build_user_prompt(vaga(titulo="Senior Frontend Developer"), profile, None)
    assert "não penalize" in prompt


# --------------------------------------------------------------------------
# §20 — deduplicação
# --------------------------------------------------------------------------

@pytest.fixture
def repo():
    fd, caminho = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    r = JobRepository(caminho)
    yield r
    r.close()
    os.unlink(caminho)


def test_dedup_por_url(repo):
    job = vaga()
    assert not repo.ja_enviada(job)
    repo.registrar(job, 80.0)
    assert repo.ja_enviada(job)


def test_dedup_entre_fontes_diferentes(repo):
    """Mesma vaga na Gupy e no LinkedIn: URLs diferentes, fingerprint igual."""
    na_gupy = vaga(titulo="Desenvolvedor Full Stack Pleno", url="https://gupy.io/job/abc")
    no_linkedin = Job(
        source="linkedin",
        title="Desenvolvedor(a) Full Stack Sênior",  # senioridade não conta no fingerprint
        company="Empresa X",
        url="https://linkedin.com/jobs/view/999",
        raw_location="São Paulo - SP",
        work_model=WorkModel.REMOTE,
    )
    assert na_gupy.fingerprint() == no_linkedin.fingerprint()

    repo.registrar(na_gupy, 80.0)
    assert repo.ja_enviada(no_linkedin)


def test_dedup_nao_confunde_empresas_diferentes():
    a = vaga(empresa="Empresa A")
    b = vaga(empresa="Empresa B", url="https://a.com/2")
    assert a.fingerprint() != b.fingerprint()


def test_reserva_de_sessao(repo):
    job = vaga()
    assert repo.reservar_sessao(job) is True
    assert repo.reservar_sessao(job) is False


def test_migracao_preserva_banco_antigo():
    """O banco v1 (link, data_publicacao, titulo) continua abrindo."""
    import sqlite3

    fd, caminho = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    antigo = sqlite3.connect(caminho)
    antigo.execute(
        "CREATE TABLE vagas_enviadas (link TEXT PRIMARY KEY, data_publicacao TEXT, titulo TEXT)"
    )
    antigo.execute("INSERT INTO vagas_enviadas VALUES (?, ?, ?)", ("https://old/1", "01/01/2026", "Dev"))
    antigo.commit()
    antigo.close()

    repo = JobRepository(caminho)
    try:
        colunas = {r["name"] for r in repo.conn.execute("PRAGMA table_info(vagas_enviadas)")}
        assert {"fingerprint", "empresa", "score"} <= colunas
        assert repo.conn.execute("SELECT COUNT(*) FROM vagas_enviadas").fetchone()[0] == 1
        assert repo.ja_enviada(vaga(url="https://old/1"))
    finally:
        repo.close()
        os.unlink(caminho)


# --------------------------------------------------------------------------
# §19 — ordenação
# --------------------------------------------------------------------------

def test_ordena_por_score_depois_modelo_de_trabalho():
    alta = (vaga(url="https://a.com/1"), MatchResult(score=90.0).finalize())
    media_remota = (vaga(url="https://a.com/2", modelo=WorkModel.REMOTE),
                    MatchResult(score=70.0).finalize())
    media_presencial = (vaga(url="https://a.com/3", modelo=WorkModel.ONSITE),
                        MatchResult(score=70.0).finalize())

    ordenadas = sorted([media_presencial, media_remota, alta], key=_sort_key)
    assert [m.score for _, m in ordenadas] == [90.0, 70.0, 70.0]
    assert ordenadas[1][0].work_model is WorkModel.REMOTE  # remoto na frente


# --------------------------------------------------------------------------
# §18 — mensagem do Telegram
# --------------------------------------------------------------------------

def test_mensagem_tem_as_secoes_esperadas():
    mensagem = format_message(vaga(), validate_analysis(SAIDA_VALIDA))
    for trecho in ["MATCH 87%", "Software Engineer", "Empresa X", "Remoto",
                   "Experiência profissional", "Projetos / hands-on",
                   "Conhecimento relacionado", "Transferível",
                   "Gaps", "Análise", "Ver vaga"]:
        assert trecho in mensagem, f"faltou '{trecho}'"


def test_mensagem_cabe_no_telegram():
    gigante = dict(SAIDA_VALIDA)
    gigante["strengths"] = [f"Tecnologia {i}" for i in range(60)]
    gigante["gaps"] = [f"Gap {i}" for i in range(60)]
    gigante["reason"] = "análise muito longa. " * 200
    mensagem = format_message(vaga(titulo="X" * 300), validate_analysis(gigante))
    assert len(mensagem) <= 4096


def test_mensagem_escapa_html():
    job = vaga(titulo="Dev <script>alert(1)</script>", empresa="A & B")
    mensagem = format_message(job, validate_analysis(SAIDA_VALIDA))
    assert "<script>" not in mensagem
    assert "&lt;script&gt;" in mensagem
    assert "A &amp; B" in mensagem


def test_mensagem_sinaliza_localizacao_nao_confirmada():
    job = vaga(local="", modelo=WorkModel.HYBRID)
    job.location_confirmed = False
    assert "não confirmado" in format_message(job, validate_analysis(SAIDA_VALIDA))
