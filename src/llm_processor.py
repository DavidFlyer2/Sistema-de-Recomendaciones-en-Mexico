import requests
import json
import os

# --- CONFIGURACIÓN DE OLLAMA ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# Usa el modelo Águila especificado [cite: 35]
MODEL_NAME = "llama2:7b"

def get_expanded_query(user_query: str) -> str:
    """
    Usa el LLM Águila (via Ollama) para interpretar y expandir la consulta 
    en lenguaje natural del usuario. [cite: 35]
    """
    
    # Prompt que define la personalidad y la tarea del LLM
    prompt = f"""
    Eres un experto en turismo en México. Tu tarea es analizar la siguiente consulta de usuario y expandirla 
    en una lista concisa de palabras clave y temas turísticos relevantes para usarse en una búsqueda semántica 
    (ej: "cultura", "playa", "aventura", "historia", "gastronomía"). 
    Solo devuelve la lista de palabras clave separadas por coma, sin texto adicional.
    
    Consulta del Usuario: "{user_query}"
    
    Palabras clave expandidas:
    """
    
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3 # Baja temperatura para respuestas más determinísticas
        }
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=data)
        response.raise_for_status()
        
        result = response.json()
        expanded_text = result['response'].strip()
        
        # Limpieza simple de la salida
        return expanded_text.split("Palabras clave expandidas:")[-1].strip().replace(':', '')
    
    except requests.exceptions.RequestException as e:
        print(f"Error al comunicarse con Ollama. Asegúrate de que el modelo {MODEL_NAME} esté cargado y Ollama esté corriendo en http://localhost:11434.")
        # Fallback de emergencia
        return user_query