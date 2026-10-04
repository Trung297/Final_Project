import pandas as pd


def describe_all(df: pd.DataFrame) -> None:
    num_df = df.select_dtypes(include=["number"])
    if not num_df.empty:
        print("=== Numeric ===")
        print(num_df.describe().round(2).to_string())

    cat_df = df.select_dtypes(include=["str", "category"])
    if not cat_df.empty:
        print("\n=== Categorical ===")
        print(cat_df.describe().to_string())

    dt_df = df.select_dtypes(include=["datetime64[ns]"])
    if not dt_df.empty:
        print("\n=== Datetime ===")
        print(dt_df.describe().to_string())