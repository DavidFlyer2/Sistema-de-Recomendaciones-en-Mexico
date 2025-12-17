import pandas as pd
import numpy as np
from src.cf_model import get_cf_scores 
from src.cb_model import get_cb_scores
from src.llm_processor import get_expanded_query
from src.database import get_db_connection
from mysql.connector import Error

ALPHA_DEFAULT = 0.5 

def clean_dataframe_for_json(df):
    """
    Limpia un DataFrame de valores problemáticos para JSON.
    """
    # Reemplazar infinitos con NaN primero
    df = df.replace([np.inf, -np.inf], np.nan)
    # Reemplazar NaN con un valor seguro (3.0)
    df = df.fillna(3.0)
    return df

def get_hybrid_recommendations(user_id: int, top_n: int = 10, query_text: str = None) -> list:
    """
    Implementa el modelo híbrido de recomendación.
    Score Final = alpha * Score_CF + (1 - alpha) * Score_contenido
    """
    
    # 1. Ajuste Dinámico de Alpha y Expansión de Consulta
    if query_text:
        alpha_dynamic = 0.2
        try:
            expanded_query = get_expanded_query(query_text) 
            if not expanded_query.strip():
                expanded_query = query_text
        except Exception as e:
            print(f"ATENCIÓN: Fallo al llamar al LLM. Usando query original. Error: {e}")
            expanded_query = query_text
    else:
        alpha_dynamic = ALPHA_DEFAULT 
        
        conn = None
        try:
            conn = get_db_connection()
            user_pref_df = pd.read_sql_query(
                "SELECT preferencias_texto FROM usuarios WHERE id_usuario = %s", 
                conn, 
                params=(user_id,)
            )
            expanded_query = user_pref_df['preferencias_texto'].iloc[0] if not user_pref_df.empty else "cultura, naturaleza, turismo"
        except Exception as e:
            print(f"ATENCIÓN: Fallo al obtener preferencias del usuario. Usando fallback. Error: {e}")
            expanded_query = "cultura, naturaleza, turismo" 
        finally:
            if conn:
                conn.close()
    
    # 2. Obtener Scores
    cf_scores = get_cf_scores(user_id) 
    cb_scores = get_cb_scores(expanded_query)
    
    # DEBUG: Verificar Inf INMEDIATAMENTE después de obtener scores
    print(f"DEBUG - CF tiene Inf: {np.isinf(cf_scores.values).any()}")
    print(f"DEBUG - CB tiene Inf: {np.isinf(cb_scores.values).any()}")
    
    # Limpiar INMEDIATAMENTE después de obtener
    cf_scores = cf_scores.replace([np.inf, -np.inf], 3.0).fillna(3.0)
    cb_scores = cb_scores.replace([np.inf, -np.inf], 3.0).fillna(3.0)
    
    print(f"DEBUG - CF después de limpiar tiene Inf: {np.isinf(cf_scores.values).any()}")
    print(f"DEBUG - CB después de limpiar tiene Inf: {np.isinf(cb_scores.values).any()}")
    
    # 3. Fusión y Normalización de Scores
    merged_scores = pd.merge(
        cf_scores.reset_index(), 
        cb_scores.reset_index(), 
        on='id_destino', 
        how='outer'
    )
    
    # Rellenar faltantes con valor neutro
    merged_scores['score_cf'] = merged_scores['score_cf'].fillna(3.0)
    merged_scores['score_contenido'] = merged_scores['score_contenido'].fillna(3.0)
    
    # Aplicar la fórmula híbrida
    merged_scores['score_final'] = (
        alpha_dynamic * merged_scores['score_cf'] + 
        (1 - alpha_dynamic) * merged_scores['score_contenido']
    )
    
    print(f"DEBUG - merged_scores tiene Inf: {np.isinf(merged_scores.values).any()}")
    
    # CRÍTICO: Limpiar Inf/NaN ANTES de ordenar
    merged_scores = clean_dataframe_for_json(merged_scores)
    
    print(f"DEBUG - merged_scores después de clean tiene Inf: {np.isinf(merged_scores.values).any()}")
    
    # 4. Obtener los destinos top
    final_recommendations = merged_scores.sort_values(
        by='score_final', 
        ascending=False
    ).head(top_n)
    
    # 5. Preparar para merge
    final_scores_for_merge = final_recommendations[['id_destino', 'score_final']].copy()
    
    print(f"DEBUG - final_scores_for_merge tiene Inf: {np.isinf(final_scores_for_merge.values).any()}")
    
    # 6. Obtener datos geográficos
    conn = None
    try:
        conn = get_db_connection()
        ids_to_fetch = final_scores_for_merge['id_destino'].tolist()
        
        if not ids_to_fetch:
            return []
        
        format_strings = ','.join(['%s'] * len(ids_to_fetch))
        recommendation_list = pd.read_sql_query(
            f"SELECT id_destino, city, state, lat, lng FROM destinos WHERE id_destino IN ({format_strings})", 
            conn, 
            params=ids_to_fetch
        )
    except Exception as e:
        print(f"Error al obtener datos geográficos: {e}")
        return []
    finally:
        if conn:
            conn.close()
    
    print(f"DEBUG - recommendation_list tiene Inf: {np.isinf(recommendation_list.select_dtypes(include=[np.number]).values).any()}")
    
    # 7. Merge final
    final_results = pd.merge(recommendation_list, final_scores_for_merge, on='id_destino')
    
    print(f"DEBUG - final_results tiene Inf: {np.isinf(final_results.select_dtypes(include=[np.number]).values).any()}")
    
    # 8. CRÍTICO: Limpieza final completa - reemplazar TODOS los valores problemáticos
    final_results = final_results.replace([np.inf, -np.inf], 3.0)
    final_results = final_results.fillna(3.0)
    
    print(f"DEBUG - final_results después de limpieza final tiene Inf: {np.isinf(final_results.select_dtypes(include=[np.number]).values).any()}")
    
    # 9. Convertir a lista de diccionarios con manejo explícito
    result_list = []
    for _, row in final_results.iterrows():
        item = {}
        for col in final_results.columns:
            val = row[col]
            # Convertir tipos numpy a tipos nativos Python
            if isinstance(val, (np.integer, np.floating)):
                if np.isnan(val) or np.isinf(val):
                    val = 3.0
                else:
                    val = float(val)
            elif pd.isna(val):
                val = None
            item[col] = val
        result_list.append(item)
    
    print(f"DEBUG - Longitud de result_list: {len(result_list)}")
    
    return result_list