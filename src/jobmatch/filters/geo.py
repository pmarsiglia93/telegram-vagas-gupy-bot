"""Dados geográficos para as regras de localização."""

from __future__ import annotations

# Estados brasileiros (nome + sigla). "sao paulo"/"sp" são tratados à parte.
ESTADOS_BR: dict[str, tuple[str, ...]] = {
    "AC": ("acre",), "AL": ("alagoas",), "AP": ("amapa",), "AM": ("amazonas",),
    "BA": ("bahia", "salvador"), "CE": ("ceara", "fortaleza"),
    "DF": ("distrito federal", "brasilia"), "ES": ("espirito santo", "vitoria", "vila velha"),
    "GO": ("goias", "goiania"), "MA": ("maranhao", "sao luis"),
    "MT": ("mato grosso", "cuiaba"), "MS": ("mato grosso do sul", "campo grande"),
    "MG": ("minas gerais", "belo horizonte", "uberlandia", "contagem", "juiz de fora"),
    # "PA": nome do estado omitido de propósito — "para" é preposição comum em PT
    # e geraria falso positivo. "belem" identifica o estado com segurança.
    "PA": ("belem",), "PB": ("paraiba", "joao pessoa"),
    "PR": ("parana", "curitiba", "londrina", "maringa"),
    "PE": ("pernambuco", "recife", "olinda", "jaboatao"),
    "PI": ("piaui", "teresina"),
    "RJ": ("rio de janeiro", "niteroi", "duque de caxias", "nova iguacu"),
    "RN": ("rio grande do norte", "natal"),
    "RS": ("rio grande do sul", "porto alegre", "caxias do sul", "canoas"),
    "RO": ("rondonia", "porto velho"), "RR": ("roraima", "boa vista"),
    "SC": ("santa catarina", "florianopolis", "joinville", "blumenau", "sao jose"),
    "SE": ("sergipe", "aracaju"), "TO": ("tocantins", "palmas"),
}

# Cidades do estado de SP que NÃO são Grande SP — relevantes para híbrido/presencial.
SP_INTERIOR: tuple[str, ...] = (
    "campinas", "sao jose dos campos", "ribeirao preto", "sorocaba", "santos",
    "bauru", "piracicaba", "jundiai", "sao jose do rio preto", "franca",
    "limeira", "americana", "indaiatuba", "araraquara", "marilia", "itu",
    "hortolandia", "sumare", "taubate", "praia grande", "guaruja",
)

# Sinais de vaga fora do Brasil. Precisam ser específicos: "us" e "es" sozinhos
# geram falso positivo, por isso só entram formas inequívocas.
ESTRANGEIROS: tuple[str, ...] = (
    "united states", "estados unidos", "united kingdom", "reino unido", "usa", "u.s.a",
    "u.k.", "canada", "argentina", "mexico", "espanha", "spain", "portugal", "lisboa",
    "porto, portugal", "colombia", "chile", "peru", "uruguay", "uruguai", "paraguay",
    "paraguai", "bolivia", "ecuador", "equador", "venezuela", "costa rica", "panama",
    "germany", "alemanha", "france", "franca", "italy", "italia", "netherlands",
    "holanda", "poland", "polonia", "romania", "india", "philippines", "filipinas",
    "worldwide", "anywhere in the world", "global remote", "latam", "europe", "emea",
    "north america", "san francisco", "new york", "london", "berlin", "madrid",
    "buenos aires", "bogota", "santiago", "lima, peru", "ciudad de mexico",
)
