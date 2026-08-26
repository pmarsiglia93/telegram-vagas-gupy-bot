"""Abstração de LLM — provider definido por variável de ambiente.

Nenhum provider é obrigatório. Sem chave configurada, `create_llm_provider`
devolve None e o pipeline usa apenas o score heurístico + semântico (§22).

Anthropic usa o SDK oficial (`anthropic`); OpenAI e Gemini usam REST direto,
já que seus SDKs seriam dependências pesadas extras no GitHub Actions. Todos
os três são imports tardios — o bot roda sem nenhum deles instalado.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

import requests

# Defaults por provider. Sobrescreva com LLM_MODEL no .env.
# Custo importa aqui (§21): uma vaga = uma chamada. Veja .env.example para as
# alternativas mais baratas de cada provider.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
}


class LLMProvider(Protocol):
    name: str
    model: str
    calls: int
    errors: int

    def complete_json(self, system: str, user: str, schema: dict) -> dict: ...


def _extract_json(texto: str) -> dict:
    """Último recurso quando o provider não garante JSON puro."""
    texto = texto.strip()
    if texto.startswith("```"):
        texto = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto, flags=re.MULTILINE).strip()
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        inicio, fim = texto.find("{"), texto.rfind("}")
        if inicio >= 0 and fim > inicio:
            return json.loads(texto[inicio:fim + 1])
        raise


class AnthropicProvider:
    """Claude via SDK oficial, com structured outputs e prompt caching.

    O prompt de sistema é idêntico para todas as vagas de uma execução, então
    leva `cache_control`: a partir da segunda vaga ele é lido do cache a ~10%
    do preço de entrada.
    """

    name = "anthropic"

    def __init__(self, api_key: str, model: str = "", timeout: int = 60, effort: str = "low") -> None:
        self.model = model or DEFAULT_MODELS["anthropic"]
        self.effort = effort
        self.calls = 0
        self.errors = 0
        import anthropic  # import tardio: dependência opcional

        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=2)

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls += 1
        resposta = self._client.messages.create(
            model=self.model,
            max_tokens=8000,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            output_config={
                # `effort` baixo mantém o custo previsível numa tarefa de
                # classificação; a análise é estruturada, não exploratória.
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            messages=[{"role": "user", "content": user}],
        )
        if resposta.stop_reason == "refusal":
            raise RuntimeError("Claude recusou a análise desta vaga")
        texto = next((b.text for b in resposta.content if b.type == "text"), "")
        return _extract_json(texto)


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, model: str = "", timeout: int = 60) -> None:
        self.api_key = api_key
        self.model = model or DEFAULT_MODELS["openai"]
        self.timeout = timeout
        self.calls = 0
        self.errors = 0

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls += 1
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "job_match", "strict": True, "schema": schema},
                },
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return _extract_json(resp.json()["choices"][0]["message"]["content"])


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str = "", timeout: int = 60) -> None:
        self.api_key = api_key
        self.model = model or DEFAULT_MODELS["gemini"]
        self.timeout = timeout
        self.calls = 0
        self.errors = 0

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls += 1
        # O subset de JSON Schema do Gemini não aceita `additionalProperties`.
        limpo = {k: v for k, v in schema.items() if k != "additionalProperties"}
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            params={"key": self.api_key},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": limpo,
                },
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        partes = resp.json()["candidates"][0]["content"]["parts"]
        return _extract_json("".join(p.get("text", "") for p in partes))


def create_llm_provider(provider: str, api_key: str, model: str = "", timeout: int = 60):
    """Devolve (provider, aviso). Sem chave ou sem SDK → (None, motivo)."""
    escolha = (provider or "none").strip().lower()
    if escolha in ("", "none", "off"):
        return None, ""
    if not api_key:
        return None, f"LLM_PROVIDER={escolha} sem API key; análise por LLM desativada"

    try:
        if escolha in ("anthropic", "claude"):
            return AnthropicProvider(api_key, model, timeout), ""
        if escolha == "openai":
            return OpenAIProvider(api_key, model, timeout), ""
        if escolha in ("gemini", "google"):
            return GeminiProvider(api_key, model, timeout), ""
    except ImportError:
        return None, f"SDK do provider '{escolha}' não instalado (pip install anthropic)"
    except Exception as exc:
        return None, f"Falha ao criar provider '{escolha}': {type(exc).__name__}: {exc}"

    return None, f"LLM_PROVIDER desconhecido: {escolha}"
