import os
import psycopg2
from psycopg2.extras import register_uuid

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            database=os.getenv("PGDATABASE", "postgres"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"),
            port=os.getenv("PGPORT", "5432")
        )
        register_uuid()
        return conn
    except Exception as e:
        print(f"[recommendation_logic] Erro ao conectar ao DB: {e}")
        return None