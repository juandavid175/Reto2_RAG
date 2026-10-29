"""Un mismo contrato para Ollama local, Gemini y Anthropic."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

MODELOS = {
    "ollama": "llama3.2:3b",
    "gemini": "gemini-3.1-flash-lite",
    "anthropic": "claude-sonnet-5-5",
}


@dataclass(frozen=True)
class ConfigLLM:
    proveedor: str
    modelo: str
    timeout: float = 120
    url_ollama: str = "http://localhost:11434"

    @classmethod
    def desde_entorno(cls, proveedor: str | None = None, modelo: str | None = None):
        proveedor = (proveedor or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()
        if proveedor not in MODELOS:
            raise ValueError("El proveedor debe ser ollama, gemini o anthropic.")
        timeout = float(os.getenv("LLM_TIMEOUT", "120"))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("LLM_TIMEOUT debe ser un número positivo de segundos.")
        modelo = modelo or os.getenv(f"{proveedor.upper()}_MODEL") or MODELOS[proveedor]
        return cls(
            proveedor=proveedor,
            modelo=modelo,
            timeout=timeout,
            url_ollama=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        )


class ClienteLLM:
    def __init__(self, config: ConfigLLM):
        self.config = config
        if config.proveedor not in MODELOS:
            raise ValueError("Proveedor de LLM desconocido.")
        if config.proveedor != "ollama":
            variable = f"{config.proveedor.upper()}_API_KEY"
            if not os.getenv(variable, "").strip():
                raise ValueError(f"Falta {variable}. Configúrela en .env o use --proveedor ollama.")

    def __call__(self, sistema: str, mensajes: list[dict], max_tokens: int = 1024) -> str:
        if not mensajes:
            raise ValueError("La lista de mensajes no puede estar vacía.")
        if any(m.get("role") not in {"user", "assistant"}
               or not isinstance(m.get("content"), str) for m in mensajes):
            raise ValueError("Cada mensaje debe tener role user/assistant y content de texto.")
        try:
            if self.config.proveedor == "ollama":
                texto = self._ollama(sistema, mensajes, max_tokens)
            elif self.config.proveedor == "gemini":
                texto = self._gemini(sistema, mensajes, max_tokens)
            else:
                texto = self._anthropic(sistema, mensajes, max_tokens)
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"{self.config.proveedor} superó {self.config.timeout:g} segundos. "
                "Pruebe un modelo más pequeño, aumente LLM_TIMEOUT o cambie --proveedor."
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"No se pudo conectar con {self.config.proveedor}. "
                "Si usa Ollama, inícielo con 'ollama serve' y revise OLLAMA_BASE_URL."
            ) from exc
        if not texto or not texto.strip():
            raise RuntimeError(
                f"{self.config.proveedor} no devolvió texto. "
                "Revise el modelo, el límite de salida o los filtros del proveedor."
            )
        return texto.strip()

    def _ollama(self, sistema, mensajes, max_tokens):
        with httpx.Client(timeout=self.config.timeout, trust_env=False) as cliente:
            respuesta = cliente.post(
                f"{self.config.url_ollama}/api/chat",
                json={
                    "model": self.config.modelo,
                    "messages": [{"role": "system", "content": sistema}, *mensajes],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 8192},
                },
            )
        if respuesta.status_code == 404:
            raise RuntimeError(
                f"Ollama no encontró el modelo o la ruta. "
                f"Ejecute 'ollama pull {self.config.modelo}' y revise OLLAMA_BASE_URL."
            )
        if respuesta.is_error:
            raise RuntimeError(
                f"Ollama devolvió HTTP {respuesta.status_code}. "
                "Revise el servicio y la memoria disponible o use --proveedor gemini/anthropic."
            )
        try:
            return respuesta.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("Ollama devolvió una respuesta con formato inesperado.") from exc

    def _gemini(self, sistema, mensajes, max_tokens):
        from google import genai
        from google.genai import errors, types

        contenidos = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            ) for m in mensajes
        ]
        try:
            with genai.Client(
                api_key=os.environ["GEMINI_API_KEY"],
                http_options=types.HttpOptions(timeout=int(self.config.timeout * 1000)),
            ) as cliente:
                respuesta = cliente.models.generate_content(
                    model=self.config.modelo,
                    contents=contenidos,
                    config=types.GenerateContentConfig(
                        system_instruction=sistema,
                        max_output_tokens=max_tokens,
                    ),
                )
            return respuesta.text
        except errors.APIError as exc:
            raise RuntimeError(
                f"Gemini devolvió un error de API ({exc.code}). "
                "Revise GEMINI_API_KEY, GEMINI_MODEL y la cuota de su cuenta."
            ) from exc

    def _anthropic(self, sistema, mensajes, max_tokens):
        import anthropic

        try:
            with anthropic.Anthropic(timeout=self.config.timeout, max_retries=0) as cliente:
                respuesta = cliente.messages.create(
                    model=self.config.modelo,
                    max_tokens=max_tokens,
                    system=sistema,
                    messages=mensajes,
                )
            return "".join(b.text for b in respuesta.content if b.type == "text")
        except anthropic.APIError as exc:
            raise RuntimeError(
                "Falló la llamada a Anthropic. "
                "Revise ANTHROPIC_API_KEY, ANTHROPIC_MODEL, la conexión y los créditos."
            ) from exc


def llamar_llm(sistema: str, mensajes: list[dict], max_tokens: int = 1024) -> str:
    """También sirve a las memorias del tutorial original, sin respuestas simuladas."""
    return ClienteLLM(ConfigLLM.desde_entorno())(sistema, mensajes, max_tokens)
