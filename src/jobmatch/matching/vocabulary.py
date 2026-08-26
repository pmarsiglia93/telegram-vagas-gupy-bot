"""Vocabulário de tecnologias que o perfil NÃO possui.

Serve para dar nome aos gaps. Sem isto o sistema só saberia dizer o que casa,
nunca o que falta. As famílias permitem crédito parcial por transferência
(ex.: Kubernetes é da família `infra`, onde o perfil tem Docker).

Nada aqui elimina vaga — antes, `.NET` e `Kafka` no título descartavam a vaga
inteira; agora viram gaps informativos no score.
"""

from __future__ import annotations

# nome_exibido -> (familia, aliases)
EXTERNAL_TECH: dict[str, tuple[str, tuple[str, ...]]] = {
    "AWS": ("cloud", ("aws", "amazon web services", "ec2", "s3", "lambda", "cloudfront")),
    "Azure": ("cloud", ("azure", "microsoft azure")),
    "GCP": ("cloud", ("gcp", "google cloud", "google cloud platform")),
    "Kubernetes": ("orchestration", ("kubernetes", "k8s", "openshift", "eks", "helm")),
    "Terraform / IaC": ("iac", ("terraform", "iac", "infraestrutura como codigo",
                            "infrastructure as code", "ansible", "pulumi", "cloudformation")),
    # Família própria: antes caía em `infra`, cujo melhor skill é Git.
    "Observabilidade": ("observability", ("datadog", "grafana", "prometheus", "opentelemetry",
                                          "new relic", "splunk", "sentry", "observabilidade")),
    "Kafka": ("messaging", ("kafka", "rabbitmq", "sqs", "pub/sub", "mensageria", "event driven")),
    ".NET / C#": ("backend_dotnet", (".net", "dotnet", "c#", "csharp", "asp.net", "blazor")),
    "Go": ("backend_go", ("golang", "go lang")),
    "Ruby on Rails": ("backend_ruby", ("ruby", "rails", "ruby on rails")),
    "Rust": ("backend_rust", ("rust",)),
    "Elixir": ("backend_elixir", ("elixir", "phoenix framework")),
    "Scala": ("backend_scala", ("scala",)),
    "Flutter / Dart": ("mobile", ("flutter", "dart")),
    "React Native": ("mobile", ("react native", "expo")),
    "Swift / iOS": ("mobile", ("swift", "swiftui", "ios nativo", "objective-c")),
    "Kotlin / Android": ("mobile", ("kotlin", "android nativo", "jetpack compose")),
    "Salesforce": ("plataforma", ("salesforce", "apex", "sap", "abap", "servicenow", "dynamics")),
    "Power BI": ("dados", ("power bi", "powerbi", "tableau", "looker", "qlik")),
    "Spark / Big Data": ("dados", ("spark", "hadoop", "databricks", "airflow", "etl", "big data")),
    # ---- Engenharia de ML: família PRÓPRIA, separada de `ai` ---------------
    # Antes, tudo isto era uma única entrada na família `ai` — a mesma de
    # RAG/LLMs. Consequência medida: uma vaga com PyTorch, TensorFlow, MLOps,
    # SageMaker, Kubeflow, feature engineering, model training e deep learning
    # colapsava em UM requisito e herdava 55% do peso de RAG, pontuando 75.
    # Construir aplicação com LLM é competência diferente de treinar modelo;
    # `ml_engineering` não tem skill no perfil, então vira gap honesto.
    "Machine Learning": ("ml_engineering", ("machine learning", "aprendizado de maquina",
                                            "ml engineer", "engenheiro de machine learning")),
    "Deep Learning": ("ml_engineering", ("deep learning", "aprendizado profundo",
                                         "redes neurais", "neural network")),
    "PyTorch / TensorFlow": ("ml_engineering", ("pytorch", "tensorflow", "keras", "jax")),
    "MLOps": ("ml_engineering", ("mlops", "ml ops", "model serving", "model deployment",
                                 "mlflow", "kubeflow", "sagemaker", "vertex ai training",
                                 "feature store")),
    "Treinamento de modelos": ("ml_engineering", ("model training", "treinamento de modelos",
                                                  "fine tuning", "fine-tuning",
                                                  "feature engineering", "engenharia de features",
                                                  "hyperparameter", "model evaluation")),
    "Data Science": ("ml_engineering", ("data science", "ciencia de dados", "cientista de dados",
                                        "scikit-learn", "sklearn", "pandas", "numpy",
                                        "estatistica aplicada")),
    "NLP / Visão computacional": ("ml_engineering", ("nlp", "processamento de linguagem natural",
                                                     "visao computacional", "computer vision",
                                                     "ocr")),
    "Inglês fluente": ("idioma", ("ingles fluente", "ingles avancado", "fluent english",
                                  "english fluent", "advanced english", "ingles c1", "ingles c2")),
}


# Nome de exibição para termos que só existem na taxonomia de grupos. Sem isto
# "fastapi" apareceria como "Fastapi" na mensagem do Telegram.
GROUP_DISPLAY: dict[str, str] = {
    "fastapi": "FastAPI", "flask": "Flask", "celery": "Celery", "pydantic": "Pydantic",
    "pinecone": "Pinecone", "qdrant": "Qdrant", "weaviate": "Weaviate",
    "pgvector": "pgvector", "faiss": "FAISS", "milvus": "Milvus",
    "zapier": "Zapier", "power automate": "Power Automate", "airflow": "Airflow",
    "rpa": "RPA", "langchain": "LangChain", "llamaindex": "LlamaIndex",
    "langgraph": "LangGraph", "crewai": "CrewAI", "autogen": "AutoGen",
    "mcp": "MCP", "model context protocol": "MCP",
    "svelte": "Svelte", "sveltekit": "SvelteKit", "solidjs": "SolidJS",
    "remix": "Remix", "astro": "Astro", "ember": "Ember",
    "fastify": "Fastify", "koa": "Koa", "adonis": "AdonisJS",
    "symfony": "Symfony", "codeigniter": "CodeIgniter",
    "quarkus": "Quarkus", "micronaut": "Micronaut", "hibernate": "Hibernate",
    "jakarta ee": "Jakarta EE",
    "sql server": "SQL Server", "sqlserver": "SQL Server", "oracle": "Oracle",
    "cassandra": "Cassandra", "elasticsearch": "Elasticsearch",
    "opensearch": "OpenSearch", "firestore": "Firestore", "dynamodb": "DynamoDB",
    "helm": "Helm", "argocd": "ArgoCD", "podman": "Podman", "openshift": "OpenShift",
    "gitlab ci": "GitLab CI", "circleci": "CircleCI", "travis": "Travis CI",
    "hugging face": "Hugging Face", "ollama": "Ollama", "azure openai": "Azure OpenAI",
    "bedrock": "AWS Bedrock", "aws bedrock": "AWS Bedrock", "vertex ai": "Vertex AI",
    "grpc": "gRPC", "soap": "SOAP", "openapi": "OpenAPI", "swagger": "Swagger",
    "cypress": "Cypress", "playwright": "Playwright", "vitest": "Vitest",
    "testing library": "Testing Library", "e2e": "Testes E2E",
    "shadcn": "shadcn/ui", "design system": "Design System",
    "single page application": "SPA", "spa": "SPA",
}


def display_for(termo: str) -> str:
    """Nome apresentável para um membro de grupo."""
    return GROUP_DISPLAY.get(termo.lower(), termo.title())


def build_lookup(profile_aliases: dict[str, object]) -> dict[str, tuple[str, str]]:
    """alias normalizado -> (nome_exibido, familia) para tecnologias externas.

    Aliases que já existem no perfil têm precedência e não são sobrescritos.
    """
    lookup: dict[str, tuple[str, str]] = {}
    for nome, (familia, aliases) in EXTERNAL_TECH.items():
        for alias in aliases:
            if alias not in profile_aliases:
                lookup.setdefault(alias, (nome, familia))
    return lookup
