"""Lectura de una única página, numerada desde 1 para el usuario."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass(frozen=True)
class PaginaPDF:
    archivo: str
    numero: int
    total_paginas: int
    texto: str


class DocumentoPDF:
    def __init__(self, ruta: str | Path):
        self.ruta = Path(ruta).expanduser()
        if not self.ruta.is_file():
            raise ValueError(f"No existe el archivo: {self.ruta}")
        if self.ruta.suffix.lower() != ".pdf":
            raise ValueError("Seleccione un archivo con extensión .pdf.")
        try:
            self.lector = PdfReader(self.ruta)
            if self.lector.is_encrypted and not self.lector.decrypt(""):
                raise ValueError("El PDF requiere contraseña. Use una copia desbloqueada.")
            self.total_paginas = len(self.lector.pages)
        except (PdfReadError, OSError) as exc:
            raise ValueError("No se pudo leer el PDF. Compruebe que sea un PDF válido.") from exc
        if self.total_paginas == 0:
            raise ValueError("El PDF no contiene páginas.")

    def extraer_pagina(self, numero: int | None = None) -> PaginaPDF:
        if numero is None:
            if self.total_paginas != 1:
                raise ValueError("Debe elegir una sola página de este PDF.")
            numero = 1
        if not 1 <= numero <= self.total_paginas:
            raise ValueError(f"La página debe estar entre 1 y {self.total_paginas}.")
        try:
            # Solo se extrae el texto de esta página; nunca se recorren las demás.
            texto = (self.lector.pages[numero - 1].extract_text() or "").strip()
        except (PdfReadError, OSError, ValueError) as exc:
            raise ValueError(f"No se pudo extraer el texto de la página {numero}.") from exc
        if not texto:
            raise ValueError(
                f"La página {numero} no tiene texto extraíble. "
                "Si es una imagen escaneada, aplique OCR antes de cargarla."
            )
        return PaginaPDF(self.ruta.name, numero, self.total_paginas, texto)
