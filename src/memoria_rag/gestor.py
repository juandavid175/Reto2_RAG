"""Gestor que coordina los tres niveles de memoria."""
from .almacen import crear_cliente, obtener_coleccion
from .llm import llamar_llm
from .memoria_corto import MemoriaCortoPlazo
from .memoria_largo import MemoriaLargoPlazo
from .memoria_medio import MemoriaMedioPlazo

SISTEMA_BASE = "Usted es un asistente útil. Responda en español y de forma concisa."
class GestorMemoria:
    def __init__(self, ruta: str = "datos/chroma", funcion_emb=None, extraer_cada: int = 6):
        cliente = crear_cliente(ruta)
        self.corto = MemoriaCortoPlazo(max_mensajes=10)
        self.medio = MemoriaMedioPlazo(obtener_coleccion(cliente, "sesiones", funcion_emb))
        self.largo = MemoriaLargoPlazo(obtener_coleccion(cliente, "hechos", funcion_emb))
        self.extraer_cada = extraer_cada
        self.pendientes_extraccion: list[dict] = []

    def construir_sistema(self, consulta: str) -> str:
        partes = [SISTEMA_BASE]
        hechos = self.largo.recuperar(consulta)
        if hechos:
            partes.append("## Lo que sabe del usuario (memoria a largo plazo)\n"
                        + "\n".join(f"- {h}" for h in hechos))
        sesiones = self.medio.recuperar(consulta)
        if sesiones:
            partes.append("## Conversaciones recientes relacionadas (medio plazo)\n"
                        + "\n".join(f"- {s}" for s in sesiones))
        if self.medio.resumen_sesion:
            partes.append("## Resumen de lo hablado antes en esta sesión\n"
                        + self.medio.resumen_sesion)
        partes.append("Use esta información solo si es relevante para la pregunta.")
        return "\n\n".join(partes)

    def conversar(self, mensaje_usuario: str) -> str:
        sistema = self.construir_sistema(mensaje_usuario)
        mensajes = self.corto.obtener() + [{"role": "user", "content": mensaje_usuario}]
        respuesta = llamar_llm(sistema, mensajes)

        for rol, texto in (("user", mensaje_usuario), ("assistant", respuesta)):
            expulsados = self.corto.agregar(rol, texto)
            self.medio.incorporar(expulsados)          # corto -> medio
            self.pendientes_extraccion.append({"role": rol, "content": texto})

        if len(self.pendientes_extraccion) >= self.extraer_cada:
            self.largo.extraer_y_guardar(self.pendientes_extraccion)  # -> largo
            self.pendientes_extraccion = []
        return respuesta

    def cerrar_sesion(self) -> None:
        """Consolida todo al terminar la sesión."""
        restantes = self.corto.vaciar()
        self.medio.incorporar(restantes)
        if self.pendientes_extraccion:
            self.largo.extraer_y_guardar(self.pendientes_extraccion)
            self.pendientes_extraccion = []
        self.medio.cerrar_sesion()



