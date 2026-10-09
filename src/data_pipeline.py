import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sqlalchemy.engine import URL  # add this to your imports at the top


# 1. RESOLVE ABSOLUTE PATH TO THE .ENV FILE TO PREVENT STREAMLIT LOAD FAILURES
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)

def get_db_engine():
    """Builds a secure, isolated connection pool to the PostgreSQL database."""
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port_raw = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    
    # SAFE FALLBACK: If environment variables failed to load, prevent app crash
    if not all([db_user, db_password, db_host, db_port_raw, db_name]):
        raise RuntimeError(
            "Critical Error: Environment credentials failed to load. "
            "Ensure your hidden '.env' file or Streamlit Cloud Secrets are properly configured."
        )
        
    db_port = int(db_port_raw)
    
    # FORCE EXPLICIT PSYCOPG2 DRIVER MAPPING TO FIX THE MISSING CLOUD DRIVER ERROR
    # connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    # return create_engine(connection_string)

    url = URL.create(
        drivername="postgresql+psycopg2",
        username=db_user,
        password=db_password,
        host=db_host,
        port=db_port,
        database=db_name,
    )
    return create_engine(
        url,
        connect_args={"sslmode": "require"},
        pool_pre_ping=True,
    )

def extract_data_from_db():
    """Queries live log tables from pgAdmin views into a reusable DataFrame."""
    try:
        engine = get_db_engine()
        query_string = "SELECT * FROM public.fleet_telematics_logs;"
        
        # PASSING THE ENGINE DIRECTLY COMPLIES PERFECTLY WITH MODERN PANDAS SPECIFICATIONS
        df = pd.read_sql_query(sql=query_string, con=engine)
        return df
    except Exception as e:
        raise RuntimeError(f"Database extraction pipeline failed. Details: {e}")

def load_csv_to_db(csv_path="fleet_telematics_database.csv"):
    """ETL step to safely stream local CSV generation snapshots straight to SQL."""
    engine = get_db_engine()
    df = pd.read_csv(csv_path)
    df.columns = [col.lower() for col in df.columns]
    
    # Secure engine management context block for inserting tables
    df.to_sql(
        name='fleet_telematics_logs', 
        con=engine, 
        if_exists='replace', 
        index=False,
        schema='public'
    )
    print(f"ETL Complete: {len(df)} logs written to database.")

# =====================================================================
# PRODUCTION EXECUTION TRIGGER
# =====================================================================
if __name__ == "__main__":
    print("🚀 Triggering Data Pipeline ETL Stream to PostgreSQL...")
    try:
        target_csv = os.path.join(BASE_DIR, "fleet_telematics_database.csv")
        
        if not os.path.exists(target_csv):
            # Fallback to local string check if path building resolves differently
            target_csv = "fleet_telematics_database.csv"
            
        load_csv_to_db(target_csv)
    except Exception as e:
        print(f"❌ ETL Pipeline failed. Details: {e}")
