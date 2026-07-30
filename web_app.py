from flask import Flask, render_template, request, jsonify
import os
import time
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Importar tu pipeline RAG existente
from rag.retrieve import QdrantRetriever
from providers.deepseek import DeepSeekProvider
from providers.openrouter import OpenRouterProvider

app = Flask(__name__)

# Inicializar componentes
retriever = QdrantRetriever()

# Inicialización perezosa de providers
providers = {}

def get_provider(provider_name):
    """Inicializa el provider solo cuando se necesita"""
    if provider_name not in providers:
        try:
            if provider_name == "deepseek":
                providers[provider_name] = DeepSeekProvider()
            elif provider_name == "openrouter":
                providers[provider_name] = OpenRouterProvider()
            else:
                return None
            print(f"✅ {provider_name} configurado")
        except Exception as e:
            print(f"❌ Error inicializando {provider_name}: {e}")
            return None
    return providers[provider_name]

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint para el chat RAG"""
    try:
        # Obtener datos del request
        data = request.json
        query = data.get('query', '')
        provider = data.get('provider', 'openrouter')
        k = data.get('k', 3)
        
        if not query:
            return jsonify({'error': 'La consulta no puede estar vacía'}), 400
        
        # Inicializar proveedor
        llm = get_provider(provider)
        if not llm:
            return jsonify({'error': f'Proveedor {provider} no disponible'}), 400
        
        # Paso de Recuperación
        start_time = time.time()
        chunks = retriever.retrieve(query, k=k)
        
        if not chunks:
            return jsonify({
                'response': 'No encontré información relevante para responder tu pregunta.',
                'sources': [],
                'latency': 0,
                'tokens': 0
            })
        
        # Procesar chunks y citas
        retrieved_texts = [chunk["text"] for chunk in chunks]
        
        unique_citations = set()
        citation_metadata = []
        for chunk in chunks:
            source_key = (chunk.get("title"), chunk.get("page"), chunk.get("url"))
            if source_key not in unique_citations:
                unique_citations.add(source_key)
                citation_metadata.append({
                    "title": chunk.get("title", "Documento Desconocido"),
                    "page": chunk.get("page", "N/D"),
                    "url": chunk.get("url", "#")
                })
        
        # Construir contexto
        context = "\n\n".join(retrieved_texts)
        
        # 🎯 PROMPT MEJORADO: Más útil y menos restrictivo
        # 🎯 PROMPT MEJORADO CON INSTRUCCIONES DE FORMATEO
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

        # Generar respuesta
        response = llm.chat(messages=[{"role": "user", "content": augmented_prompt}])
        tokens_used = len(augmented_prompt.split()) + len(response.split())
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        return jsonify({
            'response': response,
            'sources': citation_metadata,
            'latency': round(latency_ms, 2),
            'tokens': tokens_used,
            'provider': provider
        })
        
    except Exception as e:
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/health')
def health():
    """Endpoint de salud"""
    return jsonify({'status': 'healthy', 'service': 'RAG UFRO'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)