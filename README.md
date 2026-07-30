# 🤖 Sistema RAG - Normativa UFRO

Sistema de Retrieval-Augmented Generation (RAG) para consultas sobre normativa y reglamentos de la Universidad de La Frontera (UFRO). Implementa un chatbot inteligente que responde preguntas basándose en documentos oficiales de la universidad.

---

## 🚀 Características

- **🔍 Búsqueda Semántica**: Recuperación de documentos relevantes usando Qdrant.
- **🤖 Múltiples Proveedores LLM**: Soporte para ChatGPT, DeepSeek y OpenRouter.
- **📊 Sistema de Evaluación**: Métricas de exactitud, latencia y costo.
- **🌐 Interfaz Web**: API Flask y una interfaz web moderna.
- **📚 Citas Automáticas**: Referencias a documentos fuente con formato estándar.

---

## 🏗️ Arquitectura

PDFs UFRO → Chunking → Embeddings → Qdrant → RAG Pipeline → LLM → Respuesta + Citas

---

## 📦 Instalación

### Prerrequisitos

- Python 3.11+
- Docker
- Git

### 1. Clonar el repositorio

```bash
git clone [https://github.com/NicoLas21lagos/esp3Rag.git](https://github.com/NicoLas21lagos/esp3Rag.git)
cd esp3Rag
2. Configurar entorno virtual
Bash

python -m venv venv
# En Linux/Mac
source venv/bin/activate
# En Windows
venv\Scripts\activate
3. Instalar dependencias
Bash

pip install -r requirements.txt
4. Configurar variables de entorno
Crea un archivo llamado .env en la raíz del proyecto con el siguiente contenido:

Fragmento de código

# API Keys (obtener en las plataformas respectivas)
OPENAI_API_KEY=tu_key_openai
DEEPSEEK_API_KEY=tu_key_deepseek
OPENROUTER_API_KEY=tu_key_openrouter

# Configuración Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
5. Levantar Qdrant con Docker
Bash

docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
6. Preparar datos
Coloca los documentos PDF en el directorio data/raw/ y actualiza el archivo data/sources.csv con los metadatos correspondientes.

7. Ejecutar ingesta de datos
Este comando procesará los PDFs y los cargará en la base de datos vectorial Qdrant.

Bash

python -m rag.ingest_qdrant
🎯 Uso
CLI Interactiva
Puedes hacer consultas directamente desde la línea de comandos.

Bash

# Consulta básica usando OpenRouter
python app.py "¿Cuáles son los requisitos de admisión?" --provider openrouter --k 3

# Consulta con otro proveedor y más documentos
python app.py "Normativa de convivencia universitaria" --provider deepseek --k 5

# Ver todas las opciones disponibles
python app.py --help
Interfaz Web
Para una experiencia más amigable, puedes usar la interfaz web.

Bash

python web_app.py
Luego, abre tu navegador y ve a: http://localhost:5000

Evaluación del Sistema
Para correr las métricas de evaluación sobre el sistema:

Bash

python eval/evaluate.py
🔧 Estructura del Proyecto
esp3Rag/
├── app.py                # CLI principal
├── web_app.py            # Interfaz web Flask
├── requirements.txt      # Dependencias del proyecto
├── .env.example          # Plantilla para variables de entorno
├── data/
│   ├── raw/              # PDFs originales
│   ├── sources.csv       # Metadatos de los documentos
│   └── gold_set.csv      # Preguntas y respuestas para evaluación
├── providers/            # Módulos para los proveedores de LLM
│   ├── base.py           # Clase base
│   ├── chatgpt.py        # OpenAI/ChatGPT
│   ├── deepseek.py       # DeepSeek API
│   └── openrouter.py     # OpenRouter
├── rag/                  # Lógica principal del sistema RAG
│   ├── ingest.py         # Procesamiento de PDFs (chunking)
│   ├── ingest_qdrant.py  # Ingesta de datos a Qdrant
│   ├── retrieve.py       # Lógica de recuperación semántica
│   └── vector_store.py   # Cliente para interactuar con Qdrant
└── eval/
    └── evaluate.py       # Script para calcular métricas de evaluación
```
