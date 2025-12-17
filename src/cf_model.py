import pandas as pd
import numpy as np # Importación necesaria para manejar np.nan y np.inf
import os
import pickle
from surprise import Dataset, Reader
from surprise import SVD
from src.database import get_db_connection
from mysql.connector import Error

# --- CONFIGURACIÓN ---
MODEL_FILENAME = 'cf_svd_model.pkl'
MODEL_PATH = os.path.join('models', MODEL_FILENAME)
RATING_SCALE = (1, 5) 

def load_ratings_data():
    """
    Carga los datos de valoraciones (Usuario, Destino, Puntuación) desde MySQL.
    """
    conn = None
    try:
        conn = get_db_connection()
        ratings_df = pd.read_sql_query("SELECT id_usuario, id_destino, puntuacion FROM valoraciones", conn)
        return ratings_df
    except Error as e:
        print(f"Error al cargar datos de valoraciones: {e}")
        return pd.DataFrame() 
    finally:
        if conn and conn.is_connected():
            conn.close()


def train_cf_model(data_df: pd.DataFrame, save_model: bool = True):
    """
    Entrena el modelo SVD (Factorización de Matrices).
    """
    if data_df.empty:
        print("No hay datos para entrenar el modelo CF.")
        return None
        
    reader = Reader(rating_scale=RATING_SCALE)
    data = Dataset.load_from_df(data_df[['id_usuario', 'id_destino', 'puntuacion']], reader)
    trainset = data.build_full_trainset()
    
    algo = SVD(n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    algo.fit(trainset)
    
    if save_model:
        os.makedirs('models', exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(algo, f)
        print(f"Modelo CF (SVD) entrenado y guardado en models/{MODEL_FILENAME}")
        
    return algo


def load_cf_model():
    """Carga el modelo SVD entrenado; si no existe, lo entrena."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            print("Modelo CF cargado desde disco.")
            return pickle.load(f)
    except FileNotFoundError:
        print("Modelo CF no encontrado. Entrenando uno nuevo...")
        ratings_df = load_ratings_data()
        return train_cf_model(ratings_df, save_model=True)


def get_cf_scores(user_id: int) -> pd.DataFrame:
    """
    Genera predicciones (scores_cf) para todos los destinos no calificados por el usuario.
    """
    algo = load_cf_model()
    conn = None
    
    try:
        conn = get_db_connection()
        all_destinos_df = pd.read_sql_query("SELECT id_destino FROM destinos", conn)
        all_destinos = all_destinos_df['id_destino'].tolist()
        
        rated_destinos = pd.read_sql_query(f"SELECT id_destino FROM valoraciones WHERE id_usuario = {user_id}", conn)
        rated_destinos_ids = rated_destinos['id_destino'].tolist()
        
    except Error as e:
        print(f"Error al obtener datos en get_cf_scores: {e}")
        return pd.DataFrame()
    finally:
        if conn and conn.is_connected():
            conn.close()

    unrated_destinos = [d for d in all_destinos if d not in rated_destinos_ids]
    predictions = []
    
    for destino_id in unrated_destinos:
        pred = algo.predict(uid=user_id, iid=destino_id)
        predictions.append({'id_destino': destino_id, 'score_cf': pred.est})
        
    cf_scores_df = pd.DataFrame(predictions)
    
    # --- Manejo del problema Cold Start (Usuario Nuevo) ---
    full_ratings_data = load_ratings_data()
    
    if cf_scores_df.empty or user_id not in full_ratings_data['id_usuario'].unique():
        print(f"Advertencia: Usuario {user_id} es un usuario nuevo (Cold Start). CF devolverá scores promedio.")
        mean_rating = full_ratings_data['puntuacion'].mean() if not full_ratings_data.empty else 3.0
        all_destinos_df['score_cf'] = mean_rating
        
        # Corrección: Asegurar que el DataFrame devuelto no tenga Inf/NaN
        all_destinos_df.replace([np.inf, -np.inf, np.nan], 3.0, inplace=True)
        return all_destinos_df.set_index('id_destino')
    
    # CORRECCIÓN FINAL: Limpiar el DataFrame de salida de Inf/NaN
    cf_scores_df.replace([np.inf, -np.inf, np.nan], 3.0, inplace=True)
    return cf_scores_df.set_index('id_destino')


if __name__ == '__main__':
    train_cf_model(load_ratings_data())
    test_user_id = 1 
    scores = get_cf_scores(test_user_id)
    print(f"\nScores de CF predichos para el Usuario {test_user_id}:")
    print(scores.sort_values(by='score_cf', ascending=False).head(5))