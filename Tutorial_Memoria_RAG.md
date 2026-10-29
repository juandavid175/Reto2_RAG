# Adaptación del Tutorial Memoria RAG: una sola página PDF

Esta práctica extiende el tutorial de memoria a corto, medio y largo plazo con
una fuente documental: **una página elegida de un PDF**. El nuevo comando principal
se centra en consultar esa página. Los módulos originales de memoria siguen en
`src/memoria_rag/` para continuar las prácticas del tutorial.

## 1. El flujo que vamos a construir

1. El usuario indica un archivo PDF.
2. El programa lee la cantidad de páginas.
3. Si el PDF tiene una sola página, se elige automáticamente. Si tiene varias,
   el usuario introduce un único número, empezando desde 1.
4. Se extrae el texto **solo de esa página**.
5. Se divide ese texto en fragmentos con algo de solapamiento.
6. Un modelo de embeddings local convierte los fragmentos en vectores.
7. ChromaDB guarda los vectores, el texto y sus metadatos.
8. Cada pregunta se convierte en un vector y se buscan sus fragmentos más parecidos.
9. Se envían esos fragmentos y la pregunta al LLM, que responde y cita la evidencia.

Los pasos 4 a 7 son la **indexación**. Los pasos 8 y 9 son la **recuperación y
generación**, es decir, RAG. Guardar un texto en ChromaDB no entrena al LLM: la
información recuperada se incluye en cada petición.

## 2. Preparar el entorno

Desde la carpeta de este proyecto:

```bash
uv sync
ollama pull llama3.2:3b
```

Si el servicio local no está iniciado, ejecute `ollama serve` en otra terminal.
Para esta práctica no necesita una clave de API mientras use Ollama.

Las dependencias nuevas son `pypdf` para leer PDFs y `google-genai` para Gemini.
`httpx` conecta con Ollama y `python-dotenv` carga `.env`. ChromaDB,
sentence-transformers y Anthropic forman parte de la base del tutorial.

## 3. Leer exactamente una página

`src/memoria_rag/pdf.py` contiene `DocumentoPDF` y `PaginaPDF`:

```python
from memoria_rag.pdf import DocumentoPDF

documento = DocumentoPDF("mi_documento.pdf")
print(documento.total_paginas)
pagina = documento.extraer_pagina(2)
print(pagina.texto)
```

La selección 2 accede internamente a `lector.pages[1]`, porque Python empieza
a contar desde cero. La aplicación muestra números desde 1. El número corresponde
a la posición física de la página, aunque el documento tenga otra numeración impresa.

`extraer_pagina()` sin número solo está permitido cuando hay una página. Con varias
páginas, elegirla es obligatorio. El lector revisa la estructura del PDF para saber
su tamaño, pero solo llama a `extract_text()` en la página elegida.

Si no hay texto extraíble, la indexación se detiene. Un escaneo de una imagen
necesita reconocimiento óptico de caracteres (OCR) previo.

## 4. Convertir esa página en una colección vectorial

`src/memoria_rag/rag_pdf.py` contiene `RAGPaginaPDF`:

```python
from memoria_rag.rag_pdf import RAGPaginaPDF

rag = RAGPaginaPDF()
cantidad = rag.indexar(pagina)
print(f"Se guardaron {cantidad} fragmentos de la página {pagina.numero}.")
```

Una página extensa produce varios registros. Eso sigue siendo una sola página:
todos los fragmentos llevan los mismos metadatos `archivo`, `pagina` y `pagina_id`.
Cada fragmento tiene además su propio número.

Se usa la colección `pdf_pagina_actual` dentro de `datos/chroma`. Al indexar otra
selección, se insertan sus fragmentos y se retiran los de la página anterior.
Los identificadores se calculan a partir del contenido y la selección para evitar
duplicados. La extracción y los embeddings se completan antes de retirar la página
previa, de modo que esos errores no borran la selección que ya estaba almacenada.

Las colecciones `sesiones` y `hechos` del ejercicio original no intervienen en
este chat PDF. Aquí la evidencia documental procede únicamente de la página
seleccionada y el historial reciente se mantiene en RAM.

## 5. Elegir el LLM

El archivo `src/memoria_rag/llm.py` conserva el contrato
`llamar_llm(sistema, mensajes, max_tokens)` y ofrece `ClienteLLM` para fijar una
configuración durante el chat. Los tres proveedores reciben las mismas instrucciones
y el mismo historial, con la conversión de roles necesaria para cada API.

Ollama es la opción predeterminada. Este proyecto usa `llama3.2:3b` inicialmente;
se puede cambiar con `--modelo` o `OLLAMA_MODEL`. Los embeddings son independientes
del LLM, por lo que cambiar de proveedor no cambia cómo se indexa la página.

Para una alternativa en la nube, copie `.env.example` a `.env` y complete
`GEMINI_API_KEY` o `ANTHROPIC_API_KEY`. Luego elija el proveedor:

```bash
uv run memoria-rag --pdf "mi_documento.pdf" --pagina 2 --proveedor gemini
uv run memoria-rag --pdf "mi_documento.pdf" --pagina 2 --proveedor anthropic
```

La API ejecuta el LLM fuera del equipo; ChromaDB y los embeddings siguen siendo
locales. Solo se envían las preguntas, el historial reciente y los fragmentos
recuperados. Para dejar una alternativa por defecto, cambie `LLM_PROVIDER` en `.env`.
El programa no cambia automáticamente a una API ni genera respuestas simuladas
si falla Ollama.

## 6. Consultar la página con RAG

```bash
uv run memoria-rag --pdf "mi_documento.pdf" --pagina 2
```

Para una sola pregunta:

```bash
uv run memoria-rag --pdf "mi_documento.pdf" --pagina 2 --pregunta "¿Cuál es la idea principal?"
```

`recuperar()` busca hasta cuatro fragmentos de la selección actual. `preguntar()`
los incorpora al prompt con referencias `[F1]`, `[F2]`, etc. Las instrucciones piden
al modelo que responda con esa evidencia, que no obedezca instrucciones incrustadas
en el PDF y que reconozca cuándo la página no contiene una respuesta.

Las fuentes mostradas permiten revisar qué fragmentos se recuperaron. No son una
garantía automática de exactitud: conviene contrastar la respuesta con el texto
original, especialmente al usar modelos pequeños.

## 7. Comprobar qué quedó guardado

Después de consultar una página, puede inspeccionar la colección sin cargar un LLM:

```bash
uv run python - <<'PY'
import chromadb

cliente = chromadb.PersistentClient(path="datos/chroma")
coleccion = cliente.get_collection("pdf_pagina_actual", embedding_function=None)
datos = coleccion.get(include=["documents", "metadatas"])
for texto, metadatos in zip(datos["documents"], datos["metadatas"]):
    print(metadatos)
    print(texto[:200])
print("Páginas distintas:", {m["pagina"] for m in datos["metadatas"]})
print("Archivos distintos:", {m["archivo"] for m in datos["metadatas"]})
PY
```

En una colección cargada correctamente habrá un único archivo y una única página.
Puede haber varios fragmentos de ella. La próxima selección reemplazará esos
fragmentos, y ejecutar de nuevo la misma selección no los duplicará.

## 8. Prueba recomendada

Use un PDF con dos páginas que contengan datos diferentes. Elija la segunda y
pregunte primero por un dato de ella. Después pregunte por un dato que solo figure
en la primera: la respuesta debería indicar que falta evidencia en la página
seleccionada. Finalmente, seleccione la primera página y revise los metadatos para
comprobar que se reemplazó el contenido anterior.

La suite automatizada cubre estos límites sin descargar modelos ni consumir APIs:

```bash
uv run python -m unittest discover -s tests -v
```

## Referencias

- [Extracción de texto con pypdf](https://pypdf.readthedocs.io/en/stable/user/extract-text.html).
- [Cliente persistente de ChromaDB](https://docs.trychroma.com/reference/python).
- [API de chat de Ollama](https://docs.ollama.com/api/chat).
- [SDK oficial de Gemini](https://googleapis.github.io/python-genai/).
- [Modelo Gemini 3.1 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite).
- [SDK oficial de Anthropic](https://github.com/anthropics/anthropic-sdk-python).
- [Modelo Claude Sonnet 5.5](https://platform.claude.com/docs/en/models/sonnet-5-5/overview).
