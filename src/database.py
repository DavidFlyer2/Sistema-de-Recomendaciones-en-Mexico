import mysql.connector
from mysql.connector import Error


DB_CONFIG = {
    'host': 'localhost',
    'database': 'tourist_recommender_db', 
    'user': 'root',          
    'password': ''    
}

def get_db_connection():
    """Establece y devuelve la conexión a MySQL."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            return conn
    except Error as e:
        print(f"Error al conectar a MySQL: {e}")
        raise e

def create_tables():
    """Crea las tablas en MySQL."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Usar ENGINE=InnoDB para soportar claves foráneas
        
        # 1. Tabla de Destinos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS destinos (
                id_destino INT PRIMARY KEY,
                city VARCHAR(255) NOT NULL,
                state VARCHAR(255) NOT NULL,
                lat DECIMAL(10, 8),
                lng DECIMAL(11, 8),
                full_description TEXT, 
                embedding BLOB 
            ) ENGINE=InnoDB;
        """)
        
        # 2. Tabla de Usuarios
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usuarios (
                id_usuario INT PRIMARY KEY,
                nombre VARCHAR(255) NOT NULL,
                preferencias_texto TEXT
            ) ENGINE=InnoDB;
        """)
        
        # 3. Tabla de Valoraciones (CF)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS valoraciones (
                id_usuario INT,
                id_destino INT,
                puntuacion DECIMAL(3, 2),
                PRIMARY KEY (id_usuario, id_destino),
                FOREIGN KEY (id_usuario) REFERENCES usuarios(id_usuario),
                FOREIGN KEY (id_destino) REFERENCES destinos(id_destino)
            ) ENGINE=InnoDB;
        """)

        conn.commit()
        print("Tablas de la BD creadas o verificadas en MySQL.")

    except Error as e:
        print(f"Error al crear tablas: {e}")
    finally:
        if conn and conn.is_connected():
            conn.close()

if __name__ == '__main__':
    create_tables()