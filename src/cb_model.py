import pandas as pd
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from src.database import get_db_connection
from mysql.connector import Error

# --- CONFIGURACIÓN DE EMBEDDINGS ---
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2' 
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# --- ARCHIVOS DE PERSISTENCIA FAISS (BD Vectorial) ---
FAISS_INDEX_FILENAME = 'faiss_index.idx'
DEST_IDS_FILENAME = 'dest_ids_map.pkl'
MODEL_DIR = 'models'

def generate_and_store_embeddings():
    """
    1. Genera embeddings de las descripciones de destino.
    2. Almacena los embeddings como BLOBs en MySQL.
    3. Construye y guarda el índice FAISS (BD Vectorial).
    """
    conn = None
    try:
        conn = get_db_connection()
        destinos_df = pd.read_sql_query("SELECT id_destino, full_description FROM destinos", conn)
        
        if destinos_df.empty:
            print("Error: No se encontraron destinos en la base de datos para generar embeddings.")
            return

        print(f"Generando embeddings para {len(destinos_df)} destinos usando {EMBEDDING_MODEL_NAME}...")
        
        descriptions = destinos_df['full_description'].tolist()
        embeddings = model.encode(descriptions, convert_to_numpy=True)
        embeddings = embeddings.astype('float32')

        cursor = conn.cursor()
        
        for index, row in destinos_df.iterrows():
            embedding_blob = embeddings[index].tobytes()
            cursor.execute(
                "UPDATE destinos SET embedding = %s WHERE id_destino = %s",
                (embedding_blob, row['id_destino'])
            )
        conn.commit()
        
        dimension = embeddings.shape[1]
        faiss.normalize_L2(embeddings) 
        index = faiss.IndexFlatIP(dimension) 
        index.add(embeddings)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        faiss.write_index(index, os.path.join(MODEL_DIR, FAISS_INDEX_FILENAME))
        
        with open(os.path.join(MODEL_DIR, DEST_IDS_FILENAME), 'wb') as f:
            pickle.dump(destinos_df['id_destino'].tolist(), f)
            
        print("Embeddings y Índice FAISS (BD Vectorial) construidos y guardados.")

    except Error as e:
        print(f"Error de MySQL en generate_and_store_embeddings: {e}")
        raise e 
    except Exception as e:
        print(f"Error general en generate_and_store_embeddings: {e}")
        raise e
    finally:
        if conn and conn.is_connected():
            conn.close()


def load_faiss_index():
    """Carga el índice FAISS y el mapeo de IDs."""
    try:
        index = faiss.read_index(os.path.join(MODEL_DIR, FAISS_INDEX_FILENAME))
        with open(os.path.join(MODEL_DIR, DEST_IDS_FILENAME), 'rb') as f:
            dest_ids_map = pickle.load(f)
        return index, dest_ids_map
    except FileNotFoundError:
        raise FileNotFoundError(f"Índice FAISS no encontrado en {MODEL_DIR}. Por favor, ejecute la generación.") 


def get_cb_scores(query_expanded_text: str, top_k: int = 50) -> pd.DataFrame:
    """
    Calcula los scores de similitud (CB) usando el índice FAISS (BD Vectorial).
    """
    index, dest_ids_map = load_faiss_index()
    
    query_embedding = model.encode(query_expanded_text, convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_embedding.reshape(1, -1))

    D, I = index.search(query_embedding.reshape(1, -1), top_k) 

    similarities = D.flatten() 
    recommended_ids = [dest_ids_map[i] for i in I.flatten()]
    
    min_score = similarities.min()
    max_score = similarities.max()
    
    normalized_scores = 1 + 4 * (similarities - min_score) / (max_score - min_score + 1e-6)

    cb_scores_df = pd.DataFrame({
        'id_destino': recommended_ids,
        'score_contenido': normalized_scores
    })
    
    # CORRECCIÓN FINAL: Limpiar el DataFrame de salida de Inf/NaN
    cb_scores_df.replace([np.inf, -np.inf, np.nan], 3.0, inplace=True)
    
    return cb_scores_df.set_index('id_destino')

if __name__ == '__main__':
    test_query_expanded = "cultura, historia, pirámides, arquitectura prehispánica"
    
    print("--- INICIANDO GENERACIÓN DE EMBEDDINGS Y FAISS ---")
    try:
        generate_and_store_embeddings() 
        print("--- GENERACIÓN DE EMBEDDINGS COMPLETADA ---")
    except Exception as e:
        print(f"\nERROR CRÍTICO DURANTE LA GENERACIÓN DE EMBEDDINGS. Causa: {e}")
        exit() 

    try:
        scores = get_cb_scores(test_query_expanded)
        print("\nScores Basados en Contenido (FAISS):")
        print(scores.sort_values(by='score_contenido', ascending=False).head(5))
    except Exception as e:
        print(f"\nERROR: Falló la ejecución de la prueba CB/FAISS. Causa: {e}")