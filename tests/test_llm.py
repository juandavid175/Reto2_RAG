import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import anthropic
import httpx
from google.genai import errors

from memoria_rag.llm import ClienteLLM, ConfigLLM


class TestLLM(unittest.TestCase):
    def setUp(self):
        self.entorno = patch.dict(os.environ, {}, clear=True)
        self.entorno.start()
        self.addCleanup(self.entorno.stop)
        self.mensajes = [{"role": "user", "content": "Pregunta sobre la página"}]

    def test_ollama_es_predeterminado_incluso_si_hay_clave_cloud(self):
        os.environ["ANTHROPIC_API_KEY"] = "clave-de-prueba"
        self.assertEqual(ConfigLLM.desde_entorno().proveedor, "ollama")

    def test_flags_tienen_prioridad_sobre_entorno(self):
        os.environ.update(LLM_PROVIDER="gemini", OLLAMA_MODEL="otro-modelo")
        config = ConfigLLM.desde_entorno("ollama", "llama3.2:3b")
        self.assertEqual((config.proveedor, config.modelo), ("ollama", "llama3.2:3b"))

    def test_proveedor_timeout_y_claves_invalidas(self):
        with self.assertRaises(ValueError):
            ConfigLLM.desde_entorno("desconocido")
        for timeout in ["0", "-1", "nan", "inf", "abc"]:
            os.environ["LLM_TIMEOUT"] = timeout
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                ConfigLLM.desde_entorno()
        for proveedor in ["gemini", "anthropic"]:
            with self.subTest(proveedor=proveedor), self.assertRaisesRegex(ValueError, "API_KEY"):
                ClienteLLM(ConfigLLM(proveedor, "modelo"))

    def test_ollama_recibe_sistema_historial_y_stream_false(self):
        cliente = Mock()
        cliente.post.return_value = httpx.Response(200, json={"message": {"content": "Respuesta local"}})
        with patch("memoria_rag.llm.httpx.Client") as fabrica:
            fabrica.return_value.__enter__.return_value = cliente
            respuesta = ClienteLLM(ConfigLLM("ollama", "llama3.2:3b"))("Evidencia", self.mensajes)
        self.assertEqual(respuesta, "Respuesta local")
        payload = cliente.post.call_args.kwargs["json"]
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "Evidencia"})
        self.assertEqual(payload["messages"][1:], self.mensajes)

    def test_ollama_falla_sin_simular_ni_cambiar_proveedor(self):
        for estado in [404, 500]:
            with self.subTest(estado=estado), patch("memoria_rag.llm.httpx.Client") as fabrica:
                fabrica.return_value.__enter__.return_value.post.return_value = httpx.Response(estado)
                with self.assertRaises(RuntimeError):
                    ClienteLLM(ConfigLLM("ollama", "modelo"))("Sistema", self.mensajes)

    def test_gemini_convierte_roles_y_preserva_prompt(self):
        os.environ["GEMINI_API_KEY"] = "clave-de-prueba"
        with patch("google.genai.Client") as fabrica:
            cliente = fabrica.return_value.__enter__.return_value
            cliente.models.generate_content.return_value = SimpleNamespace(text="Respuesta Gemini")
            mensajes = self.mensajes + [{"role": "assistant", "content": "Respuesta anterior"}, *self.mensajes]
            respuesta = ClienteLLM(ConfigLLM("gemini", "modelo-gemini"))("Evidencia", mensajes)
        self.assertEqual(respuesta, "Respuesta Gemini")
        argumentos = cliente.models.generate_content.call_args.kwargs
        self.assertEqual([c.role for c in argumentos["contents"]], ["user", "model", "user"])
        self.assertEqual(argumentos["config"].system_instruction, "Evidencia")
        self.assertEqual(argumentos["model"], "modelo-gemini")

    def test_anthropic_recibe_modelo_y_omite_bloques_no_textuales(self):
        os.environ["ANTHROPIC_API_KEY"] = "clave-de-prueba"
        with patch("anthropic.Anthropic") as fabrica:
            cliente = fabrica.return_value.__enter__.return_value
            cliente.messages.create.return_value = SimpleNamespace(content=[
                SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="Respuesta Claude")
            ])
            respuesta = ClienteLLM(ConfigLLM("anthropic", "modelo-claude"))("Evidencia", self.mensajes)
        self.assertEqual(respuesta, "Respuesta Claude")
        self.assertEqual(cliente.messages.create.call_args.kwargs["system"], "Evidencia")
        self.assertEqual(cliente.messages.create.call_args.kwargs["model"], "modelo-claude")

    def test_errores_de_apis_tienen_mensaje_util(self):
        os.environ.update(GEMINI_API_KEY="prueba", ANTHROPIC_API_KEY="prueba")
        with patch("google.genai.Client") as fabrica:
            cliente = fabrica.return_value.__enter__.return_value
            cliente.models.generate_content.side_effect = errors.ClientError(429, {"error": {"message": "cuota"}})
            with self.assertRaisesRegex(RuntimeError, "cuota"):
                ClienteLLM(ConfigLLM("gemini", "modelo"))("Sistema", self.mensajes)
        with patch("anthropic.Anthropic") as fabrica:
            cliente = fabrica.return_value.__enter__.return_value
            cliente.messages.create.side_effect = anthropic.APIConnectionError(request=httpx.Request("POST", "https://example.com"))
            with self.assertRaisesRegex(RuntimeError, "Anthropic"):
                ClienteLLM(ConfigLLM("anthropic", "modelo"))("Sistema", self.mensajes)

    def test_salida_vacia_y_timeout_no_producen_respuesta_ficticia(self):
        llm = ClienteLLM(ConfigLLM("ollama", "modelo"))
        with patch.object(llm, "_ollama", return_value=" "):
            with self.assertRaisesRegex(RuntimeError, "no devolvió texto"):
                llm("Sistema", self.mensajes)
        with patch.object(llm, "_ollama", side_effect=httpx.ReadTimeout("timeout")):
            with self.assertRaisesRegex(RuntimeError, "superó"):
                llm("Sistema", self.mensajes)


if __name__ == "__main__":
    unittest.main()
