# Tutorial Memoria RAG: consultar una página de un PDF

Seleccione un PDF, elija **una sola página** y consulte su contenido con RAG.
El programa extrae únicamente esa página, la divide en fragmentos, genera embeddings
locales y los guarda en ChromaDB. Ollama responde usando los fragmentos recuperados.
También puede elegir Gemini o Anthropic mediante API.

## Ejecutar con Ollama

Requiere Python 3.12+, [uv](https://docs.astral.sh/uv/) y [Ollama](https://ollama.com/).

```bash
uv sync
ollama pull llama3.2:3b
uv run memoria-rag
```

Si Ollama aún no está en ejecución, inícielo en otra terminal con `ollama serve`.
El programa pide la ruta del PDF y muestra cuántas páginas tiene. Si tiene una sola,
selecciona la página 1 automáticamente. Para varias páginas, solicita un número
entre 1 y el total. Escriba `salir` para terminar el chat.

También puede indicar la selección directamente:

```bash
uv run memoria-rag --pdf "/ruta/documento.pdf" --pagina 2
uv run memoria-rag --pdf "/ruta/documento.pdf" --pagina 2 --pregunta "¿Qué explica esta página?"
```

La numeración empieza en **1**, según la posición física en el PDF. `--pagina`
acepta un único número, no rangos ni listas. Para un PDF de una página puede omitirla.
También se puede ejecutar con `uv run python -m memoria_rag`.

## Usar Gemini o Anthropic

Copie `.env.example` a `.env` y complete la clave del proveedor que usará:

```bash
cp .env.example .env
```

En `.env`, configure `GEMINI_API_KEY` o `ANTHROPIC_API_KEY`. Luego ejecute:

```bash
uv run memoria-rag --pdf "/ruta/documento.pdf" --pagina 2 --proveedor gemini
uv run memoria-rag --pdf "/ruta/documento.pdf" --pagina 2 --proveedor anthropic
```

Puede dejar `LLM_PROVIDER=gemini` o `LLM_PROVIDER=anthropic` en `.env` para usarlo
por defecto. Los argumentos de consola tienen prioridad sobre `.env`.

| Proveedor | Modelo inicial | Variable para cambiarlo |
| --- | --- | --- |
| Ollama local | `llama3.2:3b` | `OLLAMA_MODEL` |
| Gemini | `gemini-3.1-flash-lite` | `GEMINI_MODEL` |
| Anthropic | `claude-sonnet-5-5` | `ANTHROPIC_MODEL` |

`--modelo NOMBRE` permite probar otro modelo del proveedor seleccionado.
`LLM_TIMEOUT=120` configura el tiempo de espera en segundos.
En el equipo donde se desarrolló esta adaptación, `llama3.2:3b` ya estaba instalado
y respondió correctamente en una prueba completa. Si resulta lento para su uso,
puede cambiar el proveedor con los comandos anteriores.

Con Gemini o Anthropic, las preguntas, el historial reciente y los fragmentos
recuperados se envían al proveedor elegido. El archivo PDF completo no se sube.
Las APIs requieren su propia clave y cuota o créditos. El cambio de proveedor es
explícito; los errores nunca se sustituyen por respuestas simuladas.

## Qué queda en ChromaDB

- Carpeta: `datos/chroma` (cámbiela con `--db OTRA_CARPETA`).
- Colección documental: `pdf_pagina_actual`.
- Contenido: fragmentos de **una única página**, con archivo, página y número de fragmento.
- Al seleccionar otra página o PDF, se retiran los fragmentos de la selección anterior.
- Repetir la misma selección no duplica registros.
- Las colecciones `sesiones` y `hechos` del tutorial original se conservan aparte.

Este chat PDF mantiene los últimos turnos en RAM para preguntas de seguimiento;
no guarda conversaciones ni resúmenes en ChromaDB. Cada respuesta muestra los
fragmentos recuperados como `[F1]`, `[F2]`, etc. El prompt pide responder solo con
evidencia de la página y señalar cuando falte información; la fidelidad de una
respuesta sigue dependiendo del modelo.

Los embeddings usan `paraphrase-multilingual-MiniLM-L12-v2` en CPU, igual que el
tutorial original. La primera ejecución descarga ese modelo si no está en caché;
después puede trabajar localmente con Ollama. Cambiar a una API libera al equipo
de ejecutar el LLM, pero los embeddings y ChromaDB continúan ejecutándose localmente.

Los PDFs escaneados sin texto requieren OCR previo. Un PDF vacío, dañado, protegido
con contraseña o una página fuera de rango produce un mensaje de error.

## Tutorial y pruebas

La explicación de la adaptación está en [Tutorial_Memoria_RAG.md](Tutorial_Memoria_RAG.md).

```bash
uv run python -m unittest discover -s tests -v
```

Las pruebas generan PDFs temporales y usan ChromaDB real con embeddings pequeños
de prueba. Verifican selección, extracción de una sola página, reemplazo,
persistencia y envío de contexto a los tres proveedores. No hacen llamadas de pago.
