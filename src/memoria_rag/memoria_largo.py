"""Memoria a largo plazo: hechos duraderos recuperados con RAG."""
import json
import math
import time
import uuid

from .llm import llamar_llm

PROMPT_EXTRACCION = (
    "Extraiga de la conversación los hechos que conviene recordar de forma permanente: "
    "datos del usuario, preferencias estables, decisiones y conocimientos importantes. "
    "Ignore lo trivial o temporal. Responda SOLO con un JSON válido con esta forma: "
    '[{"hecho": "texto breve en tercera persona", "importancia": 1-10}]. '
    "Si no hay nada que recordar, responda []."
)


class MemoriaLargoPlazo:
    def __init__(self, coleccion, umbral_duplicado: float = 0.15):
        self.coleccion = coleccion
        self.umbral_duplicado = umbral_duplicado

    def guardar_hecho(self, hecho: str, importancia: int = 5) -> str:
        """Guarda un hecho. Si ya existe uno casi idéntico, lo actualiza."""
        ahora = time.time()
        id_hecho = str(uuid.uuid4())
        if self.coleccion.count() > 0:
            r = self.coleccion.query(query_texts=[hecho], n_results=1)
            if r["distances"][0] and r["distances"][0][0] < self.umbral_duplicado:
                id_hecho = r["ids"][0][0]  # se reutiliza el id: actualización
        self.coleccion.upsert(
            ids=[id_hecho],
            documents=[hecho],
            metadatas=[{"importancia": importancia, "creado": ahora, "ultimo_acceso": ahora}],
        )
        return id_hecho

    def extraer_y_guardar(self, mensajes: list[dict]) -> list[str]:
        """Pide al LLM que extraiga hechos de la conversación y los guarda."""
        texto = "\n".join(f"{m['role']}: {m['content']}" for m in mensajes)
        salida = llamar_llm(PROMPT_EXTRACCION, [{"role": "user", "content": texto}])
        try:
            limpio = salida.replace("```json", "").replace("```", "").strip()
            hechos = json.loads(limpio)
        except json.JSONDecodeError:
            return []
        guardados = []
        for h in hechos:
            if isinstance(h, dict) and h.get("hecho"):
                self.guardar_hecho(h["hecho"], int(h.get("importancia", 5)))
                guardados.append(h["hecho"])
        return guardados

    def recuperar(self, consulta: str, k: int = 5, candidatos: int = 15) -> list[str]:
        """RAG con puntuación combinada: similitud + importancia + recencia."""
        total = self.coleccion.count()
        if total == 0:
            return []
        r = self.coleccion.query(query_texts=[consulta], n_results=min(candidatos, total))
        ahora = time.time()
        puntuados = []
        for id_, doc, meta, dist in zip(r["ids"][0], r["documents"][0],
                                        r["metadatas"][0], r["distances"][0]):
            similitud = 1 - dist                            # 1 = idéntico
            importancia = meta.get("importancia", 5) / 10   # 0..1
            dias = (ahora - meta.get("ultimo_acceso", ahora)) / 86400
            recencia = math.exp(-dias / 30)                 # decae con los días
            puntuacion = 0.6 * similitud + 0.25 * importancia + 0.15 * recencia
            puntuados.append((puntuacion, id_, doc, meta))
        puntuados.sort(reverse=True)
        elegidos = [p for p in puntuados[:k] if p[0] > 0.35]
        # Refuerzo: lo que se recuerda se "refresca"
        if elegidos:
            self.coleccion.update(
                ids=[p[1] for p in elegidos],
                metadatas=[{**p[3], "ultimo_acceso": ahora} for p in elegidos],
            )
        return [p[2] for p in elegidos]

    def olvidar(self, consulta: str) -> int:
        """Borra los hechos muy parecidos a la consulta (derecho al olvido)."""
        if self.coleccion.count() == 0:
            return 0
        r = self.coleccion.query(query_texts=[consulta], n_results=3)
        ids = [i for i, d in zip(r["ids"][0], r["distances"][0]) if d < 0.3]
        if ids:
            self.coleccion.delete(ids=ids)
        return len(ids)


