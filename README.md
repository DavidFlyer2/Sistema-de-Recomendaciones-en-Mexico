#  Sistema Híbrido de Recomendación de Destinos Turísticos en México

Sistema inteligente de recomendación que combina **Filtrado Colaborativo** y **Filtrado Basado en Contenido** para sugerir destinos turísticos personalizados en México. Utiliza procesamiento de lenguaje natural avanzado y búsqueda vectorial para entender las intenciones del usuario.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

##  Características Principales

- ** Modelo Híbrido Inteligente**: Fusión adaptativa de Filtrado Colaborativo (SVD) y Basado en Contenido
- ** Búsqueda Semántica**: Embeddings con Sentence-Transformers + FAISS para búsquedas vectoriales ultrarrápidas
- ** Expansión NLP**: Integración con Ollama/Llama 2 para expandir consultas simples en palabras clave ricas
- ** API REST de Alto Rendimiento**: FastAPI + Uvicorn para peticiones asíncronas
- ** Ponderación Asimétrica**: El sistema ajusta automáticamente el peso entre historial e intención del usuario
- ** Mitigación de Cold Start**: Recomendaciones efectivas incluso para usuarios nuevos

---

##  Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI + Uvicorn                       │
│                        (API REST)                            │
└──────────────┬────────────────────────────┬─────────────────┘
               │                            │
       ┌───────▼────────┐          ┌────────▼────────┐
       │ Filtrado       │          │   Filtrado      │
       │ Colaborativo   │          │ Basado en       │
       │    (SVD)       │          │  Contenido      │
       └───────┬────────┘          └────────┬────────┘
               │                            │
       ┌───────▼────────┐          ┌────────▼────────┐
       │     MySQL      │          │  FAISS Index    │
       │  (Historial)   │          │  (Embeddings)   │
       └────────────────┘          └────────┬────────┘
                                            │
                                   ┌────────▼────────┐
                                   │ Sentence-       │
                                   │ Transformers    │
                                   └────────┬────────┘
                                            │
                                   ┌────────▼────────┐
                                   │ Ollama/Llama 2  │
                                   │ (Expansión NLP) │
                                   └─────────────────┘
```

###  Componentes Clave

| Componente | Tecnología | Propósito |
|-----------|-----------|-----------|
| **Base de Datos Relacional** | MySQL | Almacena usuarios, destinos y valoraciones |
| **Base de Datos Vectorial** | FAISS | Búsqueda kNN ultrarrápida de embeddings |
| **Modelo de Lenguaje** | Sentence-Transformers | Genera embeddings semánticos (384 dimensiones) |
| **LLM Local** | Ollama/Llama 2 | Expande consultas NLP con palabras clave ricas |
| **Filtrado Colaborativo** | Surprise (SVD) | Predice gustos basados en similitud de usuarios |
| **API Service** | FastAPI + Uvicorn | Expone endpoints REST con alta concurrencia |

---

## 📐 Modelo Híbrido: Fusión de Scores

El sistema utiliza una **ponderación asimétrica** que se ajusta según el contexto:

```
Score_Final = α · Score_CF + (1 - α) · Score_CB
```

| Modo | α (Peso CF) | Comportamiento |
|------|-------------|----------------|
| **Sin Consulta** (Navegación) | 0.5 | Equilibra historial y preferencias estáticas |
| **Con Consulta NLP** | 0.2 | Prioriza la intención actual (80% CB) |

### ¿Por qué Híbrido?

1. **Mitiga Cold Start**: CB garantiza recomendaciones para usuarios/destinos nuevos
2. **Aumenta Diversidad**: CF introduce el factor "sorpresa" basado en la comunidad
3. **Gestiona Intención**: Responde dinámicamente a las búsquedas del usuario

---



##  Estructura del Proyecto

```
Recommender-project/
├── main.py                    # Punto de entrada (FastAPI + Uvicorn)
├── index.html                 # Interface web
├── data/
│   └── pueblosmagicos.csv     # Dataset de destinos
├── src/
│   ├── cf_model.py            # Filtrado Colaborativo (SVD)
│   ├── cb_model.py            # Filtrado Basado en Contenido (FAISS)
│   ├── hybrid_model.py        # Lógica de fusión de scores
│   ├── llm_processor.py       # Expansión semántica (Ollama)
│   ├── database.py            # Conexión MySQL
│   └── etl.py                 # Carga de datos
├── models/
│   ├── cf_svd_model.pkl       # Modelo SVD serializado
│   ├── faiss_index.idx        # Índice FAISS
│   └── dest_ids_map.pkl       # Mapeo de IDs
├── .gitignore
├── requirements.txt
└── README.md
```

---

##  Ejemplo de Flujo

### Flujo con Consulta NLP

1. **Usuario escribe**: `"Quiero playas con vida nocturna"`
2. **LLM expande**: `"fiesta, bares, discotecas, ambiente joven, costa"`
3. **Vectorización**: Se genera embedding de 384 dimensiones
4. **FAISS busca**: Encuentra destinos semánticamente similares
5. **Hibridación**: Combina score CB (80%) + score CF (20%)
6. **Resultado**: Lista ordenada de playas con vida nocturna

---

##  Tecnologías Utilizadas

- **Backend**: Python 3.8+, FastAPI, Uvicorn
- **Machine Learning**: 
  - Surprise (SVD)
  - Sentence-Transformers (paraphrase-multilingual-MiniLM-L12-v2)
  - FAISS (Facebook AI Similarity Search)
- **NLP**: Ollama, Llama 2
- **Base de Datos**: MySQL, Pandas
- **Científicas**: NumPy, Scikit-learn

---

##  Mejoras Futuras

- [ ] Implementar caché Redis para búsquedas frecuentes
- [ ] Agregar filtros por categoría y precio
- [ ] Sistema de retroalimentación implícita (clics, tiempo)
- [ ] Dashboard de métricas en tiempo real
- [ ] Soporte multi-idioma
- [ ] Integración con APIs de turismo

##  Autor

**David Flyer**

- GitHub: [@DavidFlyer2](https://github.com/DavidFlyer2)
- Proyecto: [Sistema de Recomendaciones en México](https://github.com/DavidFlyer2/Sistema-de-Recomendaciones-en-Mexico)
