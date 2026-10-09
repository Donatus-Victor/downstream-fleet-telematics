import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load background credentials from hidden environment file
load_dotenv()

def get_db_engine():
    """Builds a secure, isolated connection pool to the PostgreSQL database."""
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(connection_string)

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
        # Resolves file paths correctly even if running from subdirectories
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target_csv = os.path.join(base_dir, "fleet_telematics_database.csv")
        
        if not os.path.exists(target_csv):
            # Fallback to local string check if path building resolves differently
            target_csv = "fleet_telematics_database.csv"
            
        load_csv_to_db(target_csv)
    except Exception as e:
        print(f"❌ ETL Pipeline failed. Details: {e}")
