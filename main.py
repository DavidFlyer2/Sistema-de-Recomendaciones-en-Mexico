import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.hybrid_model import get_hybrid_recommendations
from src.database import create_tables
from mysql.connector import Error

# 1. Asegurarse de que las tablas existan al inicio
try:
    create_tables() 
except Error as e:
    print(f"ATENCIÓN: No se pudo conectar a la BD o crear tablas. Verifique MySQL. Error: {e}")
except Exception as e:
    print(f"Error inesperado al inicializar la BD: {e}")

app = FastAPI(
    title="Sistema de Recomendación de Destinos Turísticos en México",
    description="API híbrida con LLM/FAISS para recomendaciones personalizadas"
)

# Habilitar CORS para el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status", tags=["Admin"])
def get_status():
    """Verifica que la API esté funcionando"""
    return {
        "status": "ok", 
        "message": "API de Recomendación está en línea.",
        "endpoints": {
            "docs": "/docs",
            "user_recommendations": "/recommend/user/{user_id}",
            "query_recommendations": "/recommend/query"
        }
    }

@app.get("/recommend/user/{user_id}", tags=["Recomendación"])
async def get_user_recommendations(user_id: int, n: int = 10):
    """
    Genera recomendaciones basadas en el historial del usuario (prioriza CF/preferencias estáticas).
    
    Args:
        user_id: ID del usuario en la base de datos
        n: Número de recomendaciones a devolver (default: 10)
    
    Returns:
        Lista de destinos recomendados con scores
    """
    try:
        recommendations = get_hybrid_recommendations(user_id=user_id, top_n=n)
        
        if not recommendations:
            raise HTTPException(
                status_code=404, 
                detail=f"No se encontraron recomendaciones para el usuario {user_id}. Verifique que exista en la BD."
            )
        
        return {
            "user_id": user_id,
            "total_recommendations": len(recommendations),
            "recommendations": recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error interno en get_user_recommendations: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error al generar recomendación: {str(e)}"
        )

@app.post("/recommend/query", tags=["Recomendación"])
async def get_query_recommendations(query_text: str, user_id: int, n: int = 10):
    """
    Genera recomendaciones basadas en una consulta de lenguaje natural.
    Usa Ollama para expansión de query y prioriza CB sobre CF.
    
    Args:
        query_text: Consulta en lenguaje natural (ej: "playas tranquilas")
        user_id: ID del usuario (para personalización CF)
        n: Número de recomendaciones (default: 10)
    
    Returns:
        Lista de destinos recomendados con scores
    """
    if not query_text or not query_text.strip():
        raise HTTPException(
            status_code=400, 
            detail="El campo 'query_text' es obligatorio y no puede estar vacío."
        )
    
    try:
        recommendations = get_hybrid_recommendations(
            user_id=user_id, 
            top_n=n, 
            query_text=query_text
        )
        
        if not recommendations:
            raise HTTPException(
                status_code=404, 
                detail="No se encontraron destinos relevantes para esta consulta."
            )
        
        return {
            "query": query_text,
            "user_id": user_id,
            "total_recommendations": len(recommendations),
            "recommendations": recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error interno en get_query_recommendations: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error al procesar consulta NLP: {str(e)}"
        )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SISTEMA DE RECOMENDACIÓN HÍBRIDO - SERVIDOR INICIANDO")
    print("="*60)
    print("\n📍 Documentación interactiva: http://127.0.0.1:8000/docs")
    print("📍 Estado del servidor: http://127.0.0.1:8000/status")
    print("\n⚠️  REQUISITOS:")
    print("   1. MySQL corriendo en localhost")
    print("   2. Ollama corriendo en localhost:11434")
    print("   3. Modelos CF y FAISS generados (ejecutar etl.py)")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )