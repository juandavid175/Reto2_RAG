"""Memoria a corto plazo: los últimos mensajes de la conversación."""


class MemoriaCortoPlazo:
    def __init__(self, max_mensajes: int = 10, max_caracteres: int = 8000):
        self.max_mensajes = max_mensajes
        self.max_caracteres = max_caracteres
        self.mensajes: list[dict] = []

    def _tamano(self) -> int:
        return sum(len(m["content"]) for m in self.mensajes)

    def agregar(self, rol: str, contenido: str) -> list[dict]:
        """Añade un mensaje y devuelve los mensajes que salen de la ventana."""
        self.mensajes.append({"role": rol, "content": contenido})
        expulsados = []
        while (len(self.mensajes) > self.max_mensajes
            or self._tamano() > self.max_caracteres) and len(self.mensajes) > 2:
            # Se expulsan en pares (usuario + asistente) para no romper la alternancia
            expulsados.extend(self.mensajes[:2])
            self.mensajes = self.mensajes[2:]
        return expulsados

    def obtener(self) -> list[dict]:
        return list(self.mensajes)

    def vaciar(self) -> list[dict]:
        restantes, self.mensajes = self.mensajes, []
        return restantes
