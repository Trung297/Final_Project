from sqlalchemy import URL, create_engine, text
import pandas as pd


def make_engine(cfg: dict):
    url = URL.create(
        "mysql+pymysql",
        username=cfg["user"],
        password=cfg["password"],
        host=cfg["host"],
        port=int(cfg["port"]),
        database=cfg["database"],
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)


def test_connection(engine) -> None:
    with engine.connect() as conn:
        ver = conn.execute(text("SELECT VERSION()")).fetchone()[0]
    print(f"Connected — MySQL {ver}")


def load_main_dataset(query, engine) -> pd.DataFrame:
    return pd.read_sql(query, engine)


def load_main_dataset_chunked(query, engine, chunksize: int = 500) -> pd.DataFrame:
    """Load dataset in chunks."""
    chunks = []
    for i, chunk in enumerate(pd.read_sql(query, engine, chunksize=chunksize)):
        chunks.append(chunk)
        print(f"  chunk {i+1:02d}: {len(chunk)} rows")
    df = pd.concat(chunks, ignore_index=True)
    print(f"  total : {len(df):,}")
    return df
