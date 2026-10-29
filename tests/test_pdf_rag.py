"""Pruebas con PDFs reales y ChromaDB real, sin descargar modelos ni llamar APIs."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from chromadb.api.types import Documents, EmbeddingFunction
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from memoria_rag.almacen import crear_cliente
from memoria_rag.main import elegir_pagina, main
from memoria_rag.pdf import DocumentoPDF, PaginaPDF
from memoria_rag.rag_pdf import RAGPaginaPDF, fragmentar


def crear_pdf(ruta, paginas, clave=None):
    escritor = PdfWriter()
    for texto in paginas:
        pagina = escritor.add_blank_page(width=612, height=792)
        if texto:
            fuente = DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            })
            pagina[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): escritor._add_object(fuente)})
            })
            contenido = DecodedStreamObject()
            contenido.set_data(f"BT /F1 12 Tf 40 740 Td ({texto}) Tj ET".encode("ascii"))
            pagina[NameObject("/Contents")] = escritor._add_object(contenido)
    if clave:
        escritor.encrypt(clave)
    escritor.write(ruta)


class EmbeddingsPrueba(EmbeddingFunction[Documents]):
    """Vectores deterministas pequeños: se prueba la persistencia y el aislamiento."""

    def __init__(self):
        self.fallar = False

    def __call__(self, input):
        if self.fallar:
            raise RuntimeError("Encoder no disponible")
        return [[(b + 1) / 256 for b in hashlib.sha256(t.encode()).digest()[:8]] for t in input]

    @staticmethod
    def name():
        return "embeddings-prueba"

    def get_config(self):
        return {}


class TestPDF(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ruta = Path(self.tmp.name) / "documento.pdf"

    def test_solo_se_extrae_la_pagina_elegida(self):
        crear_pdf(self.ruta, ["SECRETO-UNO", "EVIDENCIA-DOS", "SECRETO-TRES"])
        pdf = DocumentoPDF(self.ruta)
        self.assertEqual(pdf.total_paginas, 3)
        original = type(pdf.lector.pages[0]).extract_text
        llamadas = []

        def registrar(pagina, *args, **kwargs):
            llamadas.append(pagina)
            return original(pagina, *args, **kwargs)

        with patch.object(type(pdf.lector.pages[0]), "extract_text", registrar):
            pagina = pdf.extraer_pagina(2)
        self.assertEqual(len(llamadas), 1)
        self.assertEqual(pagina.texto, "EVIDENCIA-DOS")
        self.assertEqual(pagina.numero, 2)

    def test_pdf_de_una_pagina_se_selecciona_solo(self):
        crear_pdf(self.ruta, ["UNICA"])
        self.assertEqual(DocumentoPDF(self.ruta).extraer_pagina().numero, 1)
        with patch("builtins.input", side_effect=AssertionError("No debe preguntar")):
            self.assertEqual(elegir_pagina(1, None), 1)

    def test_no_se_asume_pagina_en_pdf_multiple(self):
        crear_pdf(self.ruta, ["UNO", "DOS"])
        pdf = DocumentoPDF(self.ruta)
        for numero in [None, 0, -1, 3]:
            with self.subTest(numero=numero), self.assertRaises(ValueError):
                pdf.extraer_pagina(numero)

    def test_selector_rechaza_rangos_y_valores_fuera_de_limites(self):
        with patch("builtins.input", side_effect=["1-3", "1,2", "0", "4", "dos", "2"]):
            self.assertEqual(elegir_pagina(3, None), 2)

    def test_pdf_sin_texto_no_se_indexa(self):
        crear_pdf(self.ruta, [None])
        with self.assertRaisesRegex(ValueError, "OCR"):
            DocumentoPDF(self.ruta).extraer_pagina()

    def test_archivos_invalidos_vacios_y_con_clave(self):
        with self.assertRaisesRegex(ValueError, "No existe"):
            DocumentoPDF(self.ruta)
        self.ruta.write_text("esto no es un PDF")
        with self.assertRaisesRegex(ValueError, "PDF válido"):
            DocumentoPDF(self.ruta)
        crear_pdf(self.ruta, [])
        with self.assertRaisesRegex(ValueError, "no contiene páginas"):
            DocumentoPDF(self.ruta)
        crear_pdf(self.ruta, ["PRIVADO"], clave="secreto")
        with self.assertRaisesRegex(ValueError, "contraseña"):
            DocumentoPDF(self.ruta)

    def test_cli_pagina_invalida_no_abre_chroma(self):
        crear_pdf(self.ruta, ["UNO", "DOS"])
        with patch("memoria_rag.main.RAGPaginaPDF") as rag:
            with self.assertRaises(SystemExit) as error:
                main(["--pdf", str(self.ruta), "--pagina", "3", "--proveedor", "ollama"])
            self.assertEqual(error.exception.code, 1)
            rag.assert_not_called()


class TestRAG(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.emb = EmbeddingsPrueba()
        self.llm = Mock(return_value="El código es LUNA-742 [F1].")
        self.rag = RAGPaginaPDF(self.tmp.name, funcion_emb=self.emb, llm=self.llm)
        self.pagina = PaginaPDF("prueba.pdf", 2, 3, "El código del proyecto es LUNA-742.")

    def test_chroma_guarda_solo_la_pagina_y_no_duplica(self):
        self.rag.indexar(self.pagina)
        self.rag.indexar(self.pagina)
        datos = self.rag.coleccion.get()
        self.assertEqual(datos["documents"], [self.pagina.texto])
        self.assertEqual({m["pagina"] for m in datos["metadatas"]}, {2})
        self.assertEqual(self.rag.coleccion.count(), 1)

    def test_nueva_seleccion_reemplaza_anterior_y_limpia_historial(self):
        self.rag.indexar(self.pagina)
        self.rag.preguntar("¿Cuál es el código?")
        self.rag.indexar(PaginaPDF("otro.pdf", 1, 1, "NUEVA-PAGINA"))
        datos = self.rag.coleccion.get()
        self.assertEqual(datos["documents"], ["NUEVA-PAGINA"])
        self.assertEqual({m["archivo"] for m in datos["metadatas"]}, {"otro.pdf"})
        self.assertEqual(self.rag.historial.obtener(), [])
        self.assertEqual(self.rag.recuperar("código")[0].texto, "NUEVA-PAGINA")

    def test_una_pagina_larga_crea_varios_fragmentos_de_esa_pagina(self):
        larga = PaginaPDF("largo.pdf", 4, 8, "Evidencia de la pagina cuatro. " * 100)
        cantidad = self.rag.indexar(larga)
        self.assertGreater(cantidad, 1)
        datos = self.rag.coleccion.get()
        self.assertEqual({m["pagina"] for m in datos["metadatas"]}, {4})

    def test_recuperacion_llega_al_llm_con_metadatos(self):
        self.rag.indexar(self.pagina)
        respuesta = self.rag.preguntar("¿Cuál es el código?")
        sistema, mensajes = self.llm.call_args.args
        self.assertIn("LUNA-742", sistema)
        self.assertIn('"pagina": 2', sistema)
        self.assertEqual(mensajes[-1]["content"], "¿Cuál es el código?")
        self.assertEqual(respuesta.fuentes[0].pagina, 2)
        self.assertIn("[F1]", respuesta.texto)

    def test_historial_solo_en_ram_y_no_se_indexa(self):
        self.rag.indexar(self.pagina)
        self.rag.preguntar("¿Cuál es el código?")
        self.rag.preguntar("¿Y del mismo proyecto?")
        self.assertEqual(len(self.llm.call_args.args[1]), 3)
        self.assertEqual(self.rag.coleccion.get()["documents"], [self.pagina.texto])

    def test_no_toca_colecciones_de_memoria_del_tutorial(self):
        cliente = crear_cliente(self.tmp.name)
        memoria = cliente.create_collection("hechos", embedding_function=self.emb)
        memoria.add(ids=["recuerdo"], documents=["Hecho existente"])
        self.rag.indexar(self.pagina)
        self.assertEqual(memoria.get()["documents"], ["Hecho existente"])

    def test_persistencia_en_otro_objeto_chroma(self):
        self.rag.indexar(self.pagina)
        otro = RAGPaginaPDF(self.tmp.name, funcion_emb=self.emb, llm=self.llm)
        self.assertEqual(otro.coleccion.get()["documents"], [self.pagina.texto])

    def test_error_de_encoder_o_texto_vacio_conserva_pagina_anterior(self):
        self.rag.indexar(self.pagina)
        with self.assertRaises(ValueError):
            self.rag.indexar(PaginaPDF("vacio.pdf", 1, 1, " "))
        self.emb.fallar = True
        with self.assertRaises(RuntimeError):
            self.rag.indexar(PaginaPDF("otro.pdf", 1, 1, "Texto nuevo"))
        self.assertEqual(self.rag.coleccion.get()["documents"], [self.pagina.texto])

    def test_consulta_sin_pagina_o_pregunta_vacia_es_rechazada(self):
        with self.assertRaises(ValueError):
            self.rag.preguntar("Hola")
        self.rag.indexar(self.pagina)
        with self.assertRaises(ValueError):
            self.rag.preguntar(" ")
        self.llm.assert_not_called()

    def test_fragmentacion_no_pierde_texto_entre_limites(self):
        texto = "0123456789" * 300
        partes = fragmentar(texto)
        reconstruido = partes[0] + "".join(p[100:] for p in partes[1:])
        self.assertEqual(reconstruido, texto)


if __name__ == "__main__":
    unittest.main()
