"""Almacén vectorial compartido por las memorias de medio y largo plazo."""
import chromadb
from chromadb.utils import embedding_functions
 
 
def crear_cliente(ruta: str = "datos/chroma") -> chromadb.ClientAPI:
    """Crea (o abre) una base vectorial persistente en disco."""
    return chromadb.PersistentClient(path=ruta)

 
def funcion_embeddings():
    """Modelo multilingüe: funciona bien con textos en español."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
        normalize_embeddings=True,
    )
 
 
def obtener_coleccion(cliente, nombre: str, funcion_emb=None):
    """Devuelve una colección que mide la similitud con la distancia coseno."""
    return cliente.get_or_create_collection(
        name=nombre,
        embedding_function=funcion_emb or funcion_embeddings(),
        metadata={"hnsw:space": "cosine"},
    )
