from sqlalchemy import create_engine, text
import pandas as pd

def make_engine(cfg: dict):
    conn_str = (
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(conn_str, pool_recycle=3600)


def test_connection(engine) -> None:
    with engine.connect() as conn:
        ver = conn.execute(text("SELECT VERSION()")).fetchone()[0]
    print(f"Connected — MySQL {ver}")
    
    
def load_main_dataset(query, engine) -> pd.DataFrame:
    df = pd.read_sql(query, engine)
    return df


def load_main_dataset_chunked(query,engine,
                               chunksize: int = 500) -> pd.DataFrame:
    """
    Load dataset in chunks.
    Use when data is too large to fit in RAM at once.
    """
    chunks = []
    for i, chunk in enumerate(
        pd.read_sql(query, engine,
                    chunksize=chunksize)
    ):
        chunks.append(chunk)
        print(f"  chunk {i+1:02d}: {len(chunk)} rows")

    df = pd.concat(chunks, ignore_index=True)
    print(f"  total : {len(df):,} rows")
    return df