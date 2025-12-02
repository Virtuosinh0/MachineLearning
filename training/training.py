import os
from datetime import date, datetime
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import numpy as np

from utils.constants import MODEL_CACHE_PATH, ITEM_FEATURES_PATH, POPULARITY_PATH, TYPE_WEIGHTS, HALF_LIFE_DAYS, PRICE_WEIGHT
from db import get_db_connection

_item_df = None
_similarity_matrix = None
_popularity = None


def days_since(d):
    if isinstance(d, (pd.Timestamp, datetime)): d = d.date()
    if isinstance(d, date): return (date.today() - d).days
    try: return (date.today() - datetime.strptime(str(d), "%Y-%m-%d").date()).days
    except Exception: return 0

def time_decay(days, half_life_days=HALF_LIFE_DAYS):
    if days < 0: days = 0
    lam = math.log(2) / half_life_days
    return math.exp(-lam * days)

def interaction_score(row):
    interaction_type = row.get("type", "").lower()
    base = TYPE_WEIGHTS.get(interaction_type, 0.0)
    days = days_since(row.get("created_at", date.today()))
    return base * time_decay(days)

def load_items_from_db():
    """
    Carrega itens, aplica transformação logarítmica no preço, faz One-Hot Encoding
    e padroniza as features.
    """
    conn = get_db_connection()
    if conn is None:
        raise RuntimeError("Não foi possível conectar ao banco para carregar itens.")
    try:
        query = """
            SELECT id, metal_id, gemstone_id, supplier_id, jewelry_type,
            COALESCE(supplier_price, 0) + COALESCE(process_price, 0) + COALESCE(profit, 0) as total_price
            FROM jewelries
            WHERE quantity > 0
            -- REMOVIDO PARA TESTE: AND to_sell = true; 
        """
        df = pd.read_sql_query(query, conn)
        if df.empty:
            print("[recommendation_logic] A query SQL retornou zero itens. Verifique sua tabela 'jewelries'.") 
            return pd.DataFrame()
        
        df['id'] = df['id'].astype(str)

        categorical_features = ['metal_id', 'gemstone_id', 'supplier_id', 'jewelry_type']
        numerical_features = ['total_price']

        df[categorical_features] = df[categorical_features].fillna(-1).astype(int)
        df[numerical_features] = df[numerical_features].fillna(0)
        
        df['total_price'] = np.log1p(df['total_price'])

        df_encoded = pd.get_dummies(df, columns=categorical_features, prefix_sep='_')
        
        scaler = StandardScaler()
        if not df_encoded.empty:
            df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

        return df_encoded
    finally:
        conn.close()

def compute_item_similarity(df):
    """
    Recebe dataframe com features, aplica um peso customizado ao preço,
    e retorna a matriz de similaridade.
    """
    if df.empty:
        return None
    
    df_weighted = df.copy()

    numerical_features = ['total_price']
    
    for feature in numerical_features:
        if feature in df_weighted.columns:
            df_weighted[feature] = df_weighted[feature] * PRICE_WEIGHT

    feature_cols = [col for col in df_weighted.columns if col != 'id']
    
    X = df_weighted[feature_cols].values.astype(float)
    
    sim = cosine_similarity(X, X)
    return sim

def compute_popularity(limit_days=None):
    conn = get_db_connection()
    if conn is None: return []
    try:
        q = "SELECT jewelry_id, created_at, type FROM user_interaction"
        if limit_days is not None:
            q += " WHERE created_at >= current_date - %s"
            params = (limit_days,)
        else:
            params = None
        df = pd.read_sql_query(q, conn, params=params)
        if df.empty: return []
        df['score'] = df.apply(interaction_score, axis=1)
        agg = df.groupby('jewelry_id')['score'].sum().reset_index().sort_values('score', ascending=False)
        agg['jewelry_id'] = agg['jewelry_id'].astype(str)
        return agg['jewelry_id'].tolist()
    finally:
        conn.close()

def train_model(limit_days=365):
    global _item_df, _similarity_matrix, _popularity
    print("[recommendation_logic] Iniciando treinamento/ETL do modelo.")
    df = load_items_from_db()
    if df.empty:
        print("[recommendation_logic] Nenhum item encontrado na tabela 'jewelries'.")
        return
    sim = compute_item_similarity(df)
    
    if sim is not None:
        os.makedirs(MODEL_CACHE_PATH, exist_ok=True) # MODIFICAÇÃO: Garante que o diretório existe ANTES de salvar o primeiro arquivo.
        print(f"[recommendation_logic] Criando diretório cache: {MODEL_CACHE_PATH}") # MODIFICAÇÃO: Log do diretório.
        sim_path = os.path.join(MODEL_CACHE_PATH, "similarity_matrix.pkl")
        joblib.dump(sim, sim_path)
        print(f"[recommendation_logic] Matriz de Similaridade salva em: {sim_path}") # MODIFICAÇÃO: Log do salvamento.
    
    pop = compute_popularity(limit_days=limit_days)
    os.makedirs(MODEL_CACHE_PATH, exist_ok=True) # MODIFICAÇÃO: Mantenha aqui também, por segurança.
    joblib.dump(df, ITEM_FEATURES_PATH)
    joblib.dump(pop, POPULARITY_PATH)
    print(f"[recommendation_logic] Features salvas em: {ITEM_FEATURES_PATH}") # MODIFICAÇÃO: Log do salvamento.
    print(f"[recommendation_logic] Popularidade salva em: {POPULARITY_PATH}") # MODIFICAÇÃO: Log do salvamento.
    _item_df = df
    _similarity_matrix = sim
    _popularity = pop
    print(f"[recommendation_logic] Treinamento concluído. Items: {len(df)}, Popularidade len: {len(pop)}")

def _load_cache_if_needed():
    global _item_df, _similarity_matrix, _popularity
    print("[recommendation_logic] Tentando carregar cache...")
    if _item_df is None and os.path.exists(ITEM_FEATURES_PATH):
        _item_df = joblib.load(ITEM_FEATURES_PATH)
        print(f"[recommendation_logic] Item Features carregadas de: {ITEM_FEATURES_PATH}")
    elif _item_df is None:
        print(f"[recommendation_logic] Item Features NÃO encontradas em: {ITEM_FEATURES_PATH}")
        
    if _similarity_matrix is None and _item_df is not None:
        sim_path = os.path.join(MODEL_CACHE_PATH, "similarity_matrix.pkl")
        if os.path.exists(sim_path):
            _similarity_matrix = joblib.load(sim_path)
            print(f"[recommendation_logic] Matriz de Similaridade carregada de: {sim_path}")
        else:
            print(f"[recommendation_logic] Matriz de Similaridade NÃO encontrada em: {sim_path}. Recalculando...")
            _similarity_matrix = compute_item_similarity(_item_df)
            if _similarity_matrix is not None:
                joblib.dump(_similarity_matrix, sim_path)
    if _popularity is None:
        if os.path.exists(POPULARITY_PATH):
            _popularity = joblib.load(POPULARITY_PATH)
            print(f"[recommendation_logic] Popularidade carregada de: {POPULARITY_PATH}")
        else:
            print(f"[recommendation_logic] Popularidade NÃO encontrada. Recalculando...")
            _popularity = compute_popularity()


def explain_similarity(seed_id: str, candidate_id: str):
    """
    Função de diagnóstico para mostrar quais features contribuem para a similaridade
    entre dois itens.
    """
    global _item_df
    if _item_df is None:
        return "DataFrame de features não carregado."

    try:
        feature_cols = [col for col in _item_df.columns if col != 'id']
        
        seed_row = _item_df.loc[_item_df['id'] == seed_id]
        candidate_row = _item_df.loc[_item_df['id'] == candidate_id]

        if seed_row.empty or candidate_row.empty:
            return f"Não foi possível encontrar o ID {seed_id} ou {candidate_id} no DataFrame de features."

        seed_vector = seed_row.iloc[0]
        candidate_vector = candidate_row.iloc[0]

        categorical_cols = [col for col in feature_cols if not col.startswith('total_price')]
        numerical_col = 'total_price'

        shared_categories = []
        for col in categorical_cols:
            if col in seed_vector.index and seed_vector[col] == 1 and candidate_vector[col] == 1:
                shared_categories.append(col)

        price_contribution = 0
        if numerical_col in seed_vector.index and numerical_col in candidate_vector.index:
            seed_price_w = seed_vector[numerical_col] * PRICE_WEIGHT
            candidate_price_w = candidate_vector[numerical_col] * PRICE_WEIGHT
            price_contribution = seed_price_w * candidate_price_w 

        dot_product = len(shared_categories) + price_contribution

        return {
            "seed": seed_id,
            "candidate": candidate_id,
            "shared_categories": shared_categories,
            "price_contribution": f"{price_contribution:.4f}",
            "dot_product_score": f"{dot_product:.4f}"
        }

    except IndexError:
        return f"Não foi possível encontrar o ID {seed_id} ou {candidate_id}."
    except Exception as e:
        return f"Erro na explicação: {e}"