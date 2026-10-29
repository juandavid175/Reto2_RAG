"""Memoria a medio plazo: resumen de la sesión actual y resúmenes de sesiones recientes."""
import time
import uuid

from .llm import llamar_llm

PROMPT_RESUMEN = (
    "Usted mantiene un resumen breve de una conversación. Integre los mensajes nuevos "
    "en el resumen existente. Conserve decisiones, tareas pendientes, datos concretos "
    "y preferencias. Elimine saludos y repeticiones. Máximo 150 palabras."
)

def formatear(mensajes: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in mensajes)

class MemoriaMedioPlazo:
    def __init__(self, coleccion, dias_vigencia: int = 14):
        self.coleccion = coleccion
        self.dias_vigencia = dias_vigencia
        self.resumen_sesion = ""
        self.id_sesion = str(uuid.uuid4())

    def incorporar(self, mensajes: list[dict]) -> None:
        """Resumen progresivo: integra en el resumen los mensajes que salen del corto plazo."""
        if not mensajes:
            return
        contenido = (f"Resumen actual:\n{self.resumen_sesion or '(vacío)'}\n\n"
                    f"Mensajes nuevos:\n{formatear(mensajes)}")
        self.resumen_sesion = llamar_llm(PROMPT_RESUMEN, [{"role": "user", "content": contenido}])

    def cerrar_sesion(self) -> None:
        """Guarda el resumen de la sesión en la base vectorial, con fecha."""
        if not self.resumen_sesion:
            return
        self.coleccion.add(
            ids=[self.id_sesion],
            documents=[self.resumen_sesion],
            metadatas=[{"timestamp": time.time(), "tipo": "resumen_sesion"}],
        )
        self.resumen_sesion = ""
        self.id_sesion = str(uuid.uuid4())

    def recuperar(self, consulta: str, k: int = 3, distancia_max: float = 0.6) -> list[str]:
        """RAG con filtro temporal: solo resúmenes de los últimos N días."""
        if self.coleccion.count() == 0:
            return []
        limite = time.time() - self.dias_vigencia * 86400
        r = self.coleccion.query(
            query_texts=[consulta],
            n_results=min(k, self.coleccion.count()),
            where={"timestamp": {"$gte": limite}},
        )
        return [doc for doc, dist in zip(r["documents"][0], r["distances"][0])
                if dist <= distancia_max]

    def purgar_caducados(self) -> None:
        """Borra los resúmenes que superan el periodo de vigencia."""
        limite = time.time() - self.dias_vigencia * 86400
        self.coleccion.delete(where={"timestamp": {"$lt": limite}})
