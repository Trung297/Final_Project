import numpy as np
import pandas as pd


def optimize_dtypes(df: pd.DataFrame,
                    cat_threshold: float = 0.5) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        t = out[col].dtype
        if t in ["int64", "int32"]:
            vmin, vmax = out[col].min(), out[col].max()
            for itype in [np.int8, np.int16, np.int32]:
                if vmin >= np.iinfo(itype).min and vmax <= np.iinfo(itype).max:
                    out[col] = out[col].astype(itype)
                    break
        elif t == "float64":
            out[col] = out[col].astype(np.float32)
        elif t == "object" or t == "str":
            if out[col].nunique() / len(out) < cat_threshold:
                out[col] = out[col].astype("category")
    return out

def optimize_dtypes_chunked(df: pd.DataFrame,
                             chunksize: int = 1000,
                             cat_threshold: float = 0.5) -> pd.DataFrame:
    """
    Same as optimize_dtypes but processes in chunks.
    Use when DataFrame is very large.
    """
    chunks = []
    total  = len(df)
    for start in range(0, total, chunksize):
        end   = min(start + chunksize, total)
        chunk = optimize_dtypes(df.iloc[start:end].copy(), cat_threshold)
        chunks.append(chunk)
        print(f"  optimized rows {start:>6} – {end:>6}")
    out = pd.concat(chunks, ignore_index=True)
    print(f"  total : {len(out):,} rows")
    return out


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s      = df[col]
        n_miss = s.isnull().sum()
        rows.append({
            "column"  : col,
            "dtype"   : str(s.dtype),
            "missing" : n_miss,
            "missing%": round(n_miss / len(s) * 100, 1),
            "unique"  : s.nunique(),
            "sample"  : s.dropna().iloc[0] if n_miss < len(s) else None,
        })
    return (pd.DataFrame(rows)
              .set_index("column")
              .sort_values("missing%", ascending=False))


def check_duplicates(df: pd.DataFrame,
                     key_cols: list = None) -> None:
    n_full = df.duplicated().sum()
    status = "OK" if n_full == 0 else f"{n_full} rows"
    print(f"Full duplicates    : {status}")

    if key_cols:
        n_key = df.duplicated(subset=key_cols).sum()
        status = "OK" if n_key == 0 else f"{n_key} rows"
        print(f"Key duplicates {key_cols}: {status}")

        if n_key > 0:
            dup_rows = df[df.duplicated(subset=key_cols, keep=False)]
            print(dup_rows.sort_values(key_cols).head(10).to_string())


def check_data_issues(df: pd.DataFrame, checks: dict) -> None:
    for name, condition in checks.items():
        n      = int(condition.sum())
        status = "OK" if n == 0 else f"{n} rows"
        print(f"{name:<35}: {status}")


def add_datetime_features(df: pd.DataFrame,
                           date_col: str = "orderDate") -> pd.DataFrame:
    out = df.copy()
    out["year"]    = out[date_col].dt.year
    out["month"]   = out[date_col].dt.month
    out["quarter"] = out[date_col].dt.quarter
    return out

# def add_shipping_features(df: pd.DataFrame) -> pd.DataFrame:
#     out = df.copy()
#     out["isShipped"]    = out["shippedDate"].notna().astype(int)
#     out["deliveryDays"] = ((out["shippedDate"] - out["orderDate"])
#                            .dt.days
#                            .fillna(-1)
#                            .astype(int))
#     return out


def add_features_chunked(df: pd.DataFrame,
                          chunksize: int = 1000) -> pd.DataFrame:
    """
    Add datetime + shipping features in chunks.
    Use when DataFrame is very large.
    """
    chunks = []
    total  = len(df)
    for start in range(0, total, chunksize):
        end   = min(start + chunksize, total)
        chunk = df.iloc[start:end].copy()
        chunk = add_shipping_features(chunk)
        chunk = add_datetime_features(chunk)
        chunks.append(chunk)
        print(f"  features added rows {start:>6} – {end:>6}")
    out = pd.concat(chunks, ignore_index=True)
    print(f"  total : {len(out):,} rows")
    return out