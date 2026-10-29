"""Selección de un PDF y una sola página, indexación y chat RAG de consola."""

import argparse
import sys

from .llm import MODELOS, ClienteLLM, ConfigLLM
from .pdf import DocumentoPDF
from .rag_pdf import RAGPaginaPDF


def elegir_pagina(total: int, elegida: int | None) -> int:
    if elegida is not None:
        if not 1 <= elegida <= total:
            raise ValueError(f"La página debe estar entre 1 y {total}.")
        return elegida
    if total == 1:
        print("El PDF tiene una sola página: se selecciona la página 1.")
        return 1
    while True:
        valor = input(f"Elija UNA página (1-{total}): ").strip()
        try:
            numero = int(valor)
            if 1 <= numero <= total:
                return numero
        except ValueError:
            pass
        print(f"Ingrese un único número entero entre 1 y {total}.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", help="Ruta del PDF; se solicita si no se indica.")
    parser.add_argument("--pagina", type=int, help="Una página, numerada desde 1.")
    parser.add_argument("--proveedor", choices=MODELOS, help="Por defecto: LLM_PROVIDER u ollama.")
    parser.add_argument("--modelo", help="Modelo del proveedor elegido.")
    parser.add_argument("--db", default="datos/chroma", help="Carpeta local de ChromaDB.")
    parser.add_argument("--pregunta", help="Hacer una consulta y terminar, sin chat interactivo.")
    args = parser.parse_args(argv)
    try:
        config = ConfigLLM.desde_entorno(args.proveedor, args.modelo)
        llm = ClienteLLM(config)
        ruta = args.pdf or input("Ruta del archivo PDF: ").strip().strip("\"'")
        documento = DocumentoPDF(ruta)
        print(f"PDF: {documento.ruta.name} ({documento.total_paginas} páginas)")
        numero = elegir_pagina(documento.total_paginas, args.pagina)
        pagina = documento.extraer_pagina(numero)
        print(f"Página {numero}: {len(pagina.texto)} caracteres extraídos.")
        print("Preparando embeddings locales e indexando en ChromaDB...")
        rag = RAGPaginaPDF(args.db, llm=llm)
        cantidad = rag.indexar(pagina)
        print(f"Indexados {cantidad} fragmentos de UNA página en pdf_pagina_actual.")
        print(f"LLM: {config.proveedor} / {config.modelo}")
        if config.proveedor != "ollama":
            print("Las preguntas, el historial reciente y los fragmentos recuperados se enviarán al proveedor.")
        if args.pregunta is None:
            print("Pregunte sobre esta página. Escriba 'salir' para terminar.\n")
        while True:
            pregunta = args.pregunta if args.pregunta is not None else input("Usted: ").strip()
            if pregunta.lower() in {"salir", "exit"}:
                break
            if not pregunta.strip():
                if args.pregunta is not None:
                    raise ValueError("La pregunta no puede estar vacía.")
                continue
            print("Buscando evidencia y consultando al LLM...", flush=True)
            try:
                respuesta = rag.preguntar(pregunta)
            except RuntimeError as exc:
                if args.pregunta is not None:
                    raise
                print(f"Error: {exc}\n", file=sys.stderr)
                continue
            print(f"\nAsistente: {respuesta.texto}")
            for i, fuente in enumerate(respuesta.fuentes, 1):
                print(f"[F{i}] {fuente.archivo}, página {fuente.pagina}, fragmento {fuente.numero}")
            print()
            if args.pregunta is not None:
                break
    except (EOFError, KeyboardInterrupt):
        print("\nChat terminado. La página indexada permanece en ChromaDB.")
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
