import os
import time
from dotenv import load_dotenv
import argparse
from typing import List, Tuple, Dict, Any

# Importar los componentes que creaste
from rag.retrieve import QdrantRetriever
from providers.deepseek import DeepSeekProvider
from providers.openrouter import OpenRouterProvider

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# -------------------------------
# Inicialización global (1 sola vez)
# -------------------------------
retriever = QdrantRetriever()
deepseek_provider = DeepSeekProvider()
openrouter_provider = OpenRouterProvider()

# Definimos el tipo para los metadatos de citación
CitationMetadata = Dict[str, Any]


def rag_pipeline(query: str, provider: str = "openrouter", k: int = 4) -> Tuple[str, List[str], List[CitationMetadata], int]:
    """
    Ejecuta el pipeline RAG completo para una consulta de usuario.
    """
    # Seleccionar el proveedor ya inicializado
    if provider == "deepseek":
        llm = deepseek_provider
    else:
        llm = openrouter_provider

    # Paso de Recuperación (Retrieval)
    chunks = retriever.retrieve(query, k=k)

    if not chunks:
        return (
            "No pude encontrar información relevante en la base de datos para responder a tu pregunta.",
            [],
            [],
            0,
        )

    # 1. Extraer el contenido de texto para el prompt
    retrieved_texts = [chunk["text"] for chunk in chunks]

    # 2. LÓGICA DE CITACIÓN: Extraer y filtrar metadatos de citación
    unique_citations_set = set()
    citation_metadata: List[CitationMetadata] = []

    for chunk in chunks:
        source_key = (chunk.get("title"), chunk.get("page"), chunk.get("url"))
        if source_key not in unique_citations_set:
            unique_citations_set.add(source_key)
            citation_metadata.append({
                "title": chunk.get("title", "Documento Desconocido"),
                "page": chunk.get("page", "N/D"),
                "url": chunk.get("url", "#")
            })

    # Paso de Aumento de Contexto (Augmentation)
    context = "\n\n".join(retrieved_texts)

    # 🎯 PROMPT MEJORADO: Más útil y menos restrictivo
    augmented_prompt = (
            f"Eres un asistente especializado en la normativa de la UFRO. "
            f"Responde la pregunta del usuario usando la información proporcionada en el contexto.\n\n"
            f"INSTRUCCIONES ESPECÍFICAS:\n"
            f"1. Responde de manera ÚTIL y COMPLETA con la información disponible\n"
            f"2. Si la pregunta es general, proporciona un RESUMEN de la información relevante que encuentres\n"
            f"3. Si el contexto no cubre algún aspecto específico, indica claramente QUÉ INFORMACIÓN SÍ ESTÁ DISPONIBLE\n"
            f"4. ORGANIZA tu respuesta de manera clara y estructurada usando:\n"
            f"   - Párrafos separados para diferentes temas\n"
            f"   - Listas con puntos (•) o números cuando sea apropiado\n"
            f"   - Negritas para conceptos importantes\n"
            f"   - Títulos claros para secciones diferentes\n"
            f"5. Incluye detalles concretos, artículos, fechas o procedimientos cuando estén disponibles\n"
            f"6. NO inventes información, pero SÉ PROACTIVO en ayudar con lo que sí sabes\n\n"
            f"FORMATO DE RESPUESTA PREFERIDO:\n"
            f"- Comienza con un resumen general si la pregunta es amplia\n"
            f"- Separa en párrafos diferentes temas\n"
            f"- Usa listas para enumerar puntos importantes\n"
            f"- Destaca conceptos clave\n"
            f"- Termina con las referencias si es necesario\n\n"
            f"CONTEXTO DISPONIBLE:\n{context}\n\n"
            f"PREGUNTA DEL USUARIO: {query}\n\n"
            f"Por favor, proporciona una respuesta bien estructurada y formateada."
        )

    # Paso de Generación (Generation)
    response = llm.chat(messages=[{"role": "user", "content": augmented_prompt}])

    # Estimación de tokens
    tokens_used = len(augmented_prompt.split()) + len(response.split())

    return response, retrieved_texts, citation_metadata, tokens_used


def call_rag_chatgpt(query: str, k: int = 4) -> Tuple[str, List[str], List[CitationMetadata], int]:
    """Función de envoltura para llamar al pipeline RAG con el proveedor OpenRouter (ChatGPT)."""
    return rag_pipeline(query=query, provider="openrouter", k=k)


def call_rag_deepseek(query: str, k: int = 4) -> Tuple[str, List[str], List[CitationMetadata], int]:
    """Función de envoltura para llamar al pipeline RAG con el proveedor DeepSeek."""
    return rag_pipeline(query=query, provider="deepseek", k=k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline RAG de la UFRO.")
    parser.add_argument("query", type=str, help="La pregunta del usuario a procesar.")
    parser.add_argument(
        "--provider",
        type=str,
        default="openrouter",
        choices=["openrouter", "deepseek"],
        help="El proveedor LLM a usar ('openrouter' o 'deepseek').",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Número de fragmentos (chunks) a recuperar de la base de datos (top-k).",
    )

    args = parser.parse_args()

    print("--- 1. Inicializando componentes RAG ---")
    start_time = time.time()

    final_response, retrieved_texts, citations, tokens = rag_pipeline(
        query=args.query, provider=args.provider, k=args.k
    )

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    print("\n--- ¡Proceso Completado! ---")
    print("\nRespuesta Final:")
    print(final_response)

    # LÓGICA DE CITAS
    if citations:
        print("\n### Referencias:")
        for citation in citations:
            print(f"  - [{citation['title']}, p.{citation['page']}] (URL: {citation['url']})")
    else:
        print("\n### Referencias: No se encontraron fuentes.")

    # MÉTRICAS FINALES
    print(f"\nModelo usado: {args.provider.upper()}")
    print(f"Fragmentos recuperados (k): {len(retrieved_texts)}")
    print(f"Tokens usados (estimado): {tokens}")
    print(f"Latencia: {latency_ms:.2f} ms")