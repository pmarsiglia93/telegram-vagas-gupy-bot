"""Regras críticas: localização por modelo de trabalho e filtro de foco.

Dois modos, testados separadamente:
  • amplo (`profile_amplo`, sem `focus`): senioridade e stack nunca eliminam.
  • foco (`profile`, o profile.yaml real): as escolhas do candidato descartam.
"""

import pytest

from jobmatch.domain.job import Job, WorkModel, detect_seniority, detect_work_model
from jobmatch.domain.profile import Focus, _build_profile
from jobmatch.filters.eligibility import check_eligibility


def vaga(titulo, local="", modelo=WorkModel.UNKNOWN, descricao="", **kw):
    return Job(
        source="teste",
        title=titulo,
        company="Empresa X",
        url=f"https://exemplo.com/{abs(hash(titulo)) % 10**8}",
        raw_location=local,
        work_model=modelo,
        description=descricao or "Vaga de desenvolvimento de software com React e TypeScript.",
        **kw,
    )


# --------------------------------------------------------------------------
# §2 — sem foco, senioridade NUNCA elimina
# --------------------------------------------------------------------------

TITULOS_SENIORIDADE = [
    "Senior React Developer",
    "Desenvolvedor Front-end Sênior",
    "Junior React Developer",
    "Desenvolvedor Júnior",
    "Desenvolvedor Full Stack Jr",
    "Desenvolvedor Pleno",
    "Mid-Level Software Engineer",
    "Software Engineer Specialist",
    "Especialista em Desenvolvimento",
    "Staff Software Engineer",
    "Principal Engineer",
    "Tech Lead Frontend",
]


@pytest.mark.parametrize("titulo", TITULOS_SENIORIDADE)
def test_sem_foco_senioridade_nunca_elimina(profile_amplo, titulo):
    resultado = check_eligibility(vaga(titulo, "São Paulo - SP", WorkModel.REMOTE), profile_amplo)
    assert resultado.eligible, f"'{titulo}' foi eliminada por senioridade: {resultado.reason}"


# Sufixos de senioridade aplicados a um MESMO cargo. Comparar
# "Desenvolvedor" com "Principal Engineer" mediria cobertura de cargo, não
# senioridade — aqui só o nível varia.
SUFIXOS_SENIORIDADE = [
    "Júnior", "Jr", "Pleno", "Sênior", "Senior", "Sr", "Especialista",
    "Staff", "Principal", "Lead", "I", "II", "III",
]


@pytest.mark.parametrize("sufixo", SUFIXOS_SENIORIDADE)
def test_senioridade_nao_muda_o_score(profile, sufixo):
    """Mesmo cargo, só o nível muda → o score não pode cair, nem com foco.

    O foco descarta por nível; o que passa nunca é penalizado por ele.
    """
    from jobmatch.matching.heuristic import HeuristicMatcher

    matcher = HeuristicMatcher(profile)
    descricao = (
        "Requisitos e qualificações\n"
        "React\nTypeScript\nREST APIs\nGit\nPostgreSQL\n"
    )
    base = matcher.match(vaga("Desenvolvedor Front-end", "São Paulo - SP", WorkModel.REMOTE, descricao))
    atual = matcher.match(
        vaga(f"Desenvolvedor Front-end {sufixo}", "São Paulo - SP", WorkModel.REMOTE, descricao)
    )
    assert atual.score == pytest.approx(base.score), (
        f"'{sufixo}' alterou o score: {base.score:.2f} -> {atual.score:.2f}"
    )


def test_detecta_senioridade():
    assert "senior" in detect_seniority("Senior Frontend Developer")
    assert "junior" in detect_seniority("Desenvolvedor Júnior")
    assert detect_seniority("Desenvolvedor Front-end") == "nao informado"
    # Abreviações de pleno comuns nos títulos reais.
    assert detect_seniority("Engenheiro de Software Pl. (Full Stack)") == "pleno"
    assert detect_seniority("Software Engineer II, Frontend") == "pleno"
    assert "pleno" in detect_seniority("ReactJS Front-End Developer | Mid/Senior")
    # Terminologia LATAM: "Semi Senior" é pleno, não sênior.
    assert "pleno" in detect_seniority("Semi Senior React Native Developer")
    assert "pleno" in detect_seniority("Frontend Developer SSr")
    # "III" não pode casar com "II".
    assert detect_seniority("Frontend Engineer III") == "senior"


def test_sem_foco_gap_tecnologico_nao_elimina(profile_amplo):
    """Antes, '.NET' ou 'Kubernetes' no título descartavam a vaga inteira."""
    for titulo in ["Desenvolvedor .NET / C#", "Engenheiro Kafka e Kubernetes", "Dev Flutter"]:
        resultado = check_eligibility(vaga(titulo, "São Paulo - SP", WorkModel.REMOTE), profile_amplo)
        assert resultado.eligible


# --------------------------------------------------------------------------
# §4 — localização
# --------------------------------------------------------------------------

@pytest.mark.parametrize("local", [
    "Brasil", "Rio de Janeiro - RJ", "Minas Gerais", "Curitiba - PR",
    "Porto Alegre - RS", "São Paulo - SP", "Florianópolis - SC",
])
def test_remoto_vale_para_o_brasil_inteiro(profile, local):
    # Com o foco ativo: remoto é o modelo preferido e não pode ser afetado.
    resultado = check_eligibility(vaga("Frontend Developer", local, WorkModel.REMOTE), profile)
    assert resultado.eligible, f"Remoto em {local} deveria ser elegível: {resultado.reason}"


@pytest.mark.parametrize("local", [
    "São Paulo - SP", "Santo André - SP", "Osasco - SP", "Barueri - SP",
    "Guarulhos - SP", "Grande São Paulo", "São Bernardo do Campo",
])
def test_hibrido_e_presencial_na_grande_sp(profile_amplo, local):
    for modelo in (WorkModel.HYBRID, WorkModel.ONSITE):
        resultado = check_eligibility(vaga("Frontend Developer", local, modelo), profile_amplo)
        assert resultado.eligible, f"{modelo.label} em {local}: {resultado.reason}"
        assert resultado.location_confirmed


@pytest.mark.parametrize("local", [
    "Rio de Janeiro - RJ", "Curitiba - PR", "Belo Horizonte - MG",
    "Porto Alegre - RS", "Brasília - DF", "Recife - PE", "Campinas - SP",
])
def test_hibrido_e_presencial_fora_da_grande_sp(profile_amplo, local):
    for modelo in (WorkModel.HYBRID, WorkModel.ONSITE):
        resultado = check_eligibility(vaga("Frontend Developer", local, modelo), profile_amplo)
        assert not resultado.eligible, f"{modelo.label} em {local} não deveria passar"
        assert resultado.reason == "fora_da_grande_sp"


def test_localizacao_ausente_mantem_a_vaga(profile):
    """§4: na dúvida, manter e marcar como não confirmada — nunca descartar."""
    resultado = check_eligibility(vaga("Frontend Developer", "", WorkModel.HYBRID), profile)
    assert resultado.eligible
    assert resultado.location_confirmed is False


def test_estrangeiro_e_descartado(profile):
    for local in ["United States", "Remote - Canada", "Buenos Aires, Argentina", "Lisboa, Portugal"]:
        resultado = check_eligibility(vaga("Frontend Developer", local, WorkModel.REMOTE), profile)
        assert not resultado.eligible
        assert resultado.reason == "estrangeiro"


def test_sp_nao_casa_por_substring(profile_amplo):
    """Regressão: `"sp" in "jaspion"` era True na versão anterior."""
    resultado = check_eligibility(
        vaga("Frontend Developer", "Jaspion - RJ", WorkModel.ONSITE), profile_amplo
    )
    assert not resultado.eligible
    assert resultado.reason == "fora_da_grande_sp"


# --------------------------------------------------------------------------
# §3 — modelo de trabalho
# --------------------------------------------------------------------------

def test_detecta_modelo_de_trabalho():
    assert detect_work_model("100% Remoto") is WorkModel.REMOTE
    assert detect_work_model("Híbrido - São Paulo") is WorkModel.HYBRID
    assert detect_work_model("Presencial") is WorkModel.ONSITE
    assert detect_work_model("") is WorkModel.UNKNOWN
    # Híbrido vence remoto quando os dois aparecem.
    assert detect_work_model("Trabalho híbrido, 2x remoto por semana") is WorkModel.HYBRID


def test_presencial_em_sp_nao_e_penalizado(profile):
    """Presencial em SP continua válido e não leva penalidade forte (§10)."""
    from jobmatch.matching.heuristic import HeuristicMatcher

    matcher = HeuristicMatcher(profile)
    descricao = "Requisitos: React, TypeScript, Node.js, PostgreSQL."
    remoto = matcher.match(vaga("Software Engineer", "São Paulo - SP", WorkModel.REMOTE, descricao))
    presencial = matcher.match(vaga("Software Engineer", "São Paulo - SP", WorkModel.ONSITE, descricao))
    assert presencial.score > 0
    # A diferença é só o bônus de preferência, nunca uma penalidade.
    assert remoto.score - presencial.score <= profile.work_model_bonus + 0.01


# --------------------------------------------------------------------------
# Foco — as escolhas do candidato no profile.yaml
# --------------------------------------------------------------------------

def _foco(profile, titulo, modelo=WorkModel.REMOTE, local="São Paulo - SP", **kw):
    return check_eligibility(vaga(titulo, local, modelo, **kw), profile)


@pytest.mark.parametrize("titulo", [
    "Desenvolvedor(a) React - Pleno",
    "Desenvolvedor Front-end",                    # sem nível: passa
    "Engenheiro de Software Pl. (Full Stack)",
    "Software Engineer II, Frontend",
    "Desenvolvedor Full Stack Pleno/Sênior",      # aceita pleno
    "Frontend Developer | Mid/Senior",
    "Semi Senior React Native Developer",
])
def test_foco_aceita_pleno_e_nivel_nao_informado(profile, titulo):
    resultado = _foco(profile, titulo)
    assert resultado.eligible, f"'{titulo}': {resultado.reason} ({resultado.detail})"


@pytest.mark.parametrize("titulo", [
    "Senior React Developer",
    "Desenvolvedor Front-end Júnior",
    "Desenvolvedor Full Stack Jr",
    "Estágio em Desenvolvimento Front-end",
    "Especialista em Front-End (Angular)",
    "Tech Lead Frontend",
    "Lead Frontend Engineer",
])
def test_foco_descarta_senioridade_fora_de_pleno(profile, titulo):
    resultado = _foco(profile, titulo)
    assert not resultado.eligible
    assert resultado.reason == "senioridade_fora_do_foco"


def test_foco_usa_nivel_declarado_pela_fonte(profile):
    """ProgramaThor informa o nível num campo à parte, fora do título."""
    assert not _foco(profile, "Desenvolvedor(a) Front-end", declared_level="Sênior").eligible
    assert _foco(profile, "Desenvolvedor(a) Front-end", declared_level="Pleno").eligible
    # Título tem precedência sobre o campo estruturado.
    assert _foco(profile, "Desenvolvedor Front-end Pleno", declared_level="Sênior").eligible


@pytest.mark.parametrize("titulo", [
    "Desenvolvedor React Native",
    "Desenvolvedor Mobile",
    "Desenvolvedor Web",
    "Desenvolvedor VTEX",
    "Desenvolvedor Angular",
    "Desenvolvedor(a) Fullstack",
])
def test_foco_aceita_cargos_alvo(profile, titulo):
    resultado = _foco(profile, titulo)
    assert resultado.eligible, f"'{titulo}': {resultado.reason}"


@pytest.mark.parametrize("titulo", [
    "Desenvolvedor(a) Back-End Pleno",
    "Software Engineer",
    "AI Engineer",
    "Analista de Sistemas",
    "Desenvolvedor Java Pleno",
    "DESENVOLVEDOR DE T.I",
])
def test_foco_descarta_cargo_fora_do_alvo(profile, titulo):
    resultado = _foco(profile, titulo)
    assert not resultado.eligible
    assert resultado.reason == "cargo_fora_do_foco"


def test_foco_descarta_presencial_mesmo_em_sp(profile):
    resultado = _foco(profile, "Frontend Developer", WorkModel.ONSITE)
    assert not resultado.eligible
    assert resultado.reason == "modelo_excluido"


def test_foco_mantem_hibrido_em_sp_e_modelo_desconhecido(profile):
    assert _foco(profile, "Frontend Developer", WorkModel.HYBRID).eligible
    # Modelo não informado não é evidência de presencial.
    assert _foco(profile, "Frontend Developer", WorkModel.UNKNOWN).eligible


@pytest.mark.parametrize("titulo,esperado", [
    ("Desenvolvedor Mobile Flutter", False),
    ("Desenvolvedor Full Stack Ruby on Rails", False),
    ("Desenvolvedor Web Android", False),
    ("Mobile Developer (React Native / Flutter)", True),   # React Native resgata
    ("Full Stack Developer (Rails + React)", True),        # React resgata
    ("Desenvolvedor Full Stack .NET", True),               # .NET está no perfil
])
def test_foco_stack_principal_fora_do_perfil(profile, titulo, esperado):
    resultado = _foco(profile, titulo)
    assert resultado.eligible is esperado, f"'{titulo}': {resultado.reason}"
    if not esperado:
        assert resultado.reason == "stack_fora_do_foco"


@pytest.mark.parametrize("titulo", ["Analista de QA Front-end", "Instrutor de Front-end"])
def test_foco_descarta_titulo_que_nao_e_desenvolvimento(profile, titulo):
    resultado = _foco(profile, titulo)
    assert not resultado.eligible
    assert resultado.reason == "titulo_excluido"


def test_foco_descarta_empresa_bloqueada(profile):
    job = vaga("Desenvolvedor Front-End (Angular) - Trabalho Remoto", "Brasil", WorkModel.REMOTE)
    job.company = "BairesDev"
    resultado = check_eligibility(job, profile)
    assert not resultado.eligible
    assert resultado.reason == "empresa_excluida"


def test_sem_secao_focus_nada_e_descartado_por_foco():
    assert _build_profile({}).focus == Focus()


# --------------------------------------------------------------------------
# Foco — inglês acima do nível do candidato (intermediário)
# --------------------------------------------------------------------------

DESCRICAO_PT = (
    "Sobre a vaga\nVocê vai atuar no time de produto com React e TypeScript, "
    "desenvolvendo as telas da nossa plataforma de pagamentos para os clientes.\n"
)


@pytest.mark.parametrize("requisito", [
    "Inglês fluente", "Inglês avançado para reuniões com o time global",
    "Fluência em inglês", "Advanced English",
])
def test_foco_descarta_ingles_exigido(profile, requisito):
    descricao = DESCRICAO_PT + f"Requisitos\nReact\nTypeScript\n{requisito}\n"
    resultado = _foco(profile, "Desenvolvedor Front-end", descricao=descricao)
    assert not resultado.eligible
    assert resultado.reason == "ingles_exigido"


@pytest.mark.parametrize("trecho", [
    "Diferenciais\nInglês fluente\n",                      # seção de diferenciais
    "Requisitos\nReact\nInglês avançado será um diferencial\n",  # na mesma linha
    "Requisitos\nReact\nInglês intermediário\n",           # nível do candidato
])
def test_foco_mantem_ingles_como_diferencial_ou_intermediario(profile, trecho):
    resultado = _foco(profile, "Desenvolvedor Front-end", descricao=DESCRICAO_PT + trecho)
    assert resultado.eligible, resultado.reason


@pytest.mark.parametrize("titulo", [
    "Desenvolvedor(a) Front-end Angular | Pleno | Inglês",
    "Full-stack Engineer Node.js + React.js + AWS (USD-based pay)",
])
def test_foco_descarta_ingles_no_titulo(profile, titulo):
    resultado = _foco(profile, titulo)
    assert not resultado.eligible
    assert resultado.reason == "ingles_exigido"


def test_foco_descarta_vaga_escrita_em_ingles(profile):
    descricao = (
        "About the role\nWe are looking for a Full Stack Engineer to join our team. "
        "You will work with React and Node.js on the core of our product, and you "
        "will be responsible for the quality of the code that we ship to our customers. "
        "This is a remote role and the team is in the US, so you will talk with them "
        "every day. We value ownership and we are an async team with a strong culture "
        "of writing. You are expected to be comfortable with code review and to be "
        "able to mentor on the stack that we use in the company for the product.\n"
    )
    resultado = _foco(profile, "Full-stack Engineer Node.js + React.js", descricao=descricao)
    assert not resultado.eligible
    assert resultado.reason == "vaga_em_ingles"


def test_titulo_em_ingles_com_descricao_em_portugues_passa(profile):
    """'Fullstack Developer | Pleno' é título comum em vaga brasileira."""
    descricao = DESCRICAO_PT * 4 + "Requisitos\nReact\nNode.js\n"
    assert _foco(profile, "Fullstack Developer | Pleno", descricao=descricao).eligible
