import pandas as pd
import numpy as np
import os
from src.database import get_db_connection, create_tables
import mysql.connector
from mysql.connector import Error

# --- CONFIGURACIÓN DE ARCHIVOS ---
INPUT_CSV = os.path.join('data', 'pueblosmagicos.csv')

def load_and_clean_data():
    """
    1. Extracción (E): Carga el CSV.
    2. Transformación (T): Limpia, normaliza y enriquece las columnas.
    """
    try:
        # [cite_start]1. Carga del CSV [cite: 1]
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: No se encuentra el archivo {INPUT_CSV}. Asegúrate de que esté en la carpeta 'data'.")
        raise
    
    # --- Tareas de Transformación ---
    
    # A. Limpieza de Lat/Lng: Convertir 'N/A' a NaN y luego a numérico
    # Esto prepara los datos para una futura geolocalización o imputación.
    df['lat'] = df['lat'].replace('N/A', np.nan)
    df['lng'] = df['lng'].replace('N/A', np.nan)
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
    
    # B. Enriquecimiento de Contenido (Texto Inicial)
    # Esta descripción será la base para generar los embeddings en el CB.
    df['full_description'] = (
        "El Pueblo Mágico de " + df['city'] + ", ubicado en el estado de " + df['state'] + 
        ". Destaca por su belleza y atractivos turísticos."
    )
    
    # C. Renombrar y seleccionar columnas finales
    df.rename(columns={'index': 'id_destino', 'city': 'city', 'state': 'state'}, inplace=True)
    
    # Columnas necesarias para el esquema de la BD (database.py)
    return df[['id_destino', 'city', 'state', 'lat', 'lng', 'full_description']]

def simulate_user_data(num_users=100, min_rating=1, max_rating=5):
    """
    Genera datos sintéticos de usuarios y valoraciones para el Filtrado Colaborativo (CF).
    """
    
    destinos_df = load_and_clean_data()
    num_destinos = len(destinos_df)
    
    # 1. Usuarios sintéticos
    users_data = pd.DataFrame({
        'id_usuario': range(1, num_users + 1),
        'nombre': [f'User_{i}' for i in range(1, num_users + 1)],
        # Simulación de preferencias de texto (necesario para CB cuando no hay query)
        'preferencias_texto': ['cultura, historia, comida tradicional' if i % 2 == 0 else 'playa, aventura, naturaleza, relax' for i in range(1, num_users + 1)]
    })
    
    # 2. Valoraciones sintéticas (Matriz de Interacción para CF)
    np.random.seed(42) # Fija la semilla para reproducibilidad
    valoraciones = []
    
    for user_id in range(1, num_users + 1):
        # Simula entre 3 y 8 valoraciones por usuario
        num_ratings = np.random.randint(3, 8) 
        rated_destinos = np.random.choice(destinos_df['id_destino'], num_ratings, replace=False)
        
        for destino_id in rated_destinos:
            score = np.random.uniform(min_rating, max_rating)
            
            # Introducir un sesgo: Usuarios pares (cultura) puntúan alto a destinos culturales (ej. Teotihuacán, Cholula, Malinalco)
            if user_id % 2 == 0 and destino_id in [3, 5, 27]: 
                 score = np.random.uniform(4.0, 5.0)
            
            valoraciones.append((user_id, destino_id, round(score, 2))) # Redondeo para la BD
    
    valoraciones_df = pd.DataFrame(valoraciones, columns=['id_usuario', 'id_destino', 'puntuacion'])
    
    return destinos_df, users_data, valoraciones_df

def clear_data_for_reload(conn, cursor):
    """
    Borra los datos de las tablas en el orden inverso de la dependencia
    para evitar errores de Foreign Key. Usamos TRUNCATE para reiniciar los IDs.
    """
    print("\n--- Limpieza de Datos (TRUNCATE) ---")
    
    # 1. Eliminar la tabla "hijo" que depende de los demás (Valoraciones)
    try:
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
        cursor.execute("TRUNCATE TABLE valoraciones")
        print("TRUNCATE TABLE valoraciones: OK")
    except Error as e:
        print(f"Error en TRUNCATE valoraciones: {e}")
        
    # 2. Eliminar las tablas "padre"
    try:
        cursor.execute("TRUNCATE TABLE usuarios")
        print("TRUNCATE TABLE usuarios: OK")
        cursor.execute("TRUNCATE TABLE destinos")
        print("TRUNCATE TABLE destinos: OK")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
    except Error as e:
        print(f"Error en TRUNCATE destinos/usuarios: {e}")
        
    conn.commit()


def load_data_to_db(destinos_df, users_df, valoraciones_df):
    """
    3. Carga (L): Limpia las tablas y carga los DataFrames en MySQL.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # **********************************************
    # PASO A: Limpieza Forzada
    # **********************************************
    clear_data_for_reload(conn, cursor) 
    
    def insert_data(df, table_name):
        """Función genérica para insertar/reemplazar datos en MySQL."""
        columns = df.columns.tolist()
        
        # --- CORRECCIÓN CRÍTICA DE MANEJO DE NaN (Mantenemos esta lógica) ---
        data = []
        for index, row in df.iterrows():
            clean_row = []
            for value in row.values:
                if pd.isna(value):
                    clean_row.append(None)
                elif isinstance(value, np.generic):
                    clean_row.append(value.item())
                else:
                    clean_row.append(value)
            data.append(tuple(clean_row))
        # ----------------------------------------------
        
        sql = f"REPLACE INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))})"
        try:
            cursor.executemany(sql, data)
            print(f"Carga en '{table_name}': {len(df)} filas.")
        except Error as e:
            print(f"Error al ejecutar execuremany en {table_name}: {e}")
            raise e

    print("\n--- Carga de Datos (Orden de Dependencia) ---")

    # **********************************************
    # PASO B: Carga en orden (Padres primero)
    # **********************************************
    print("Cargando tabla 'destinos' (Padre 1)...")
    insert_data(destinos_df, 'destinos')
    
    print("Cargando tabla 'usuarios' (Padre 2)...")
    insert_data(users_df, 'usuarios')
    
    # 3. Cargar la tabla "hijo" (Valoraciones) al final
    print("Cargando tabla 'valoraciones' (Hijo)...")
    insert_data(valoraciones_df, 'valoraciones')
    
    conn.commit()
    conn.close()
    print("\nTodos los datos de ETL cargados exitosamente a MySQL.")

if __name__ == '__main__':
    print("--- 1. Creación de Tablas ---")
    create_tables()
    
    print("\n--- 2. Extracción, Limpieza y Simulación ---")
    destinos, users, valoraciones = simulate_user_data()
    
    print("\n--- 3. Carga de Datos a MySQL ---")
    try:
        load_data_to_db(destinos, users, valoraciones)
    except Error:
        print("\n**¡ERROR CRÍTICO!** Falló la carga de datos. Revisa la configuración en src/database.py.")