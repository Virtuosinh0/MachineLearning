import os
import psycopg

def get_db_connection():
    try:
        conn = psycopg.connect(
            host=os.getenv("PGHOST", "localhost"),
            dbname=os.getenv("PGDATABASE", "postgres"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "banana"),
            port=os.getenv("PGPORT", "5433"),
            connect_timeout=10,
            options="-c statement_timeout=30000"
        )
        return conn
    except Exception as e:
        print(f"[db] Erro ao conectar ao DB: {e}")
        return None
