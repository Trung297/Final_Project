import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def detect_outliers(series: pd.Series,
                    multiplier: float = 1.5) -> dict:
    Q1    = series.quantile(0.25)
    Q3    = series.quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    mask  = (series < lower) | (series > upper)
    return {
        "lower"  : lower,
        "upper"  : upper,
        "mask"   : mask,
        "n_out"  : int(mask.sum()),
        "pct_out": round(mask.mean() * 100, 1),
    }


def outlier_summary(df: pd.DataFrame, cols: list) -> None:
    print(f"{'column':<22} {'lower':>10} {'upper':>10} {'n_out':>8} {'pct':>6}")
    print("-" * 60)
    for col in cols:
        if col not in df.columns:
            continue
        r = detect_outliers(df[col].dropna())
        print(f"{col:<22} {r['lower']:>10,.1f} {r['upper']:>10,.1f} "
              f"{r['n_out']:>8,} {r['pct_out']:>5.1f}%")


def plot_outlier(df: pd.DataFrame, col: str,
                 title: str = None) -> None:
    data = df[col].dropna().reset_index(drop=True)
    r    = detect_outliers(data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = ["red" if v else "steelblue" for v in r["mask"].values]
    axes[0].scatter(range(len(data)), data.values,
                    c=colors, s=10, alpha=0.5)
    axes[0].axhline(r["upper"], color="red",    lw=2, ls="--",
                    label=f"upper = {r['upper']:,.0f}")
    axes[0].axhline(r["lower"], color="orange", lw=2, ls="--",
                    label=f"lower = {r['lower']:,.0f}")
    axes[0].set_title(f"Scatter — {r['n_out']} outliers ({r['pct_out']}%)")
    axes[0].set_ylabel(col)
    axes[0].legend(fontsize=8)

    axes[1].hist(data[~r["mask"]], bins=30,
                 color="steelblue", alpha=0.7, label="normal")
    axes[1].hist(data[r["mask"]],  bins=10,
                 color="red",      alpha=0.7, label="outlier")
    axes[1].set_title("Distribution")
    axes[1].set_xlabel(col)
    axes[1].legend()

    plt.suptitle(title or col, fontweight="bold")
    plt.tight_layout()
    plt.show()


def cap_outliers(df: pd.DataFrame,
                 cols: list,
                 multiplier: float = 1.5) -> pd.DataFrame:
    out = df.copy()
    print(f"{'column':<22} {'before_max':>12} {'after_max':>12} {'n_changed':>10}")
    print("-" * 60)
    for col in cols:
        if col not in out.columns:
            continue
        r         = detect_outliers(out[col].dropna(), multiplier)
        lb        = max(0, r["lower"])
        n         = int(((out[col] < lb) | (out[col] > r["upper"])).sum())
        before_max = out[col].max()
        out[col]  = out[col].clip(lower=lb, upper=r["upper"])
        after_max = out[col].max()
        print(f"{col:<22} {before_max:>12,.1f} {after_max:>12,.1f} {n:>10,}")
    print(f"\nRow count unchanged: {len(out):,}")
    return out


def cap_outliers_chunked(df: pd.DataFrame,
                          cols: list,
                          multiplier: float = 1.5,
                          chunksize: int = 1000) -> pd.DataFrame:
    """
    Same as cap_outliers but processes in chunks.

    Important: bounds are computed on the FULL dataset first
    so they are consistent across all chunks.
    """
    # Step 1: compute bounds on full data
    bounds = {}
    for col in cols:
        if col not in df.columns:
            continue
        r           = detect_outliers(df[col].dropna(), multiplier)
        bounds[col] = {"lower": max(0, r["lower"]), "upper": r["upper"]}
        print(f"  bounds [{col}]: "
              f"lower={bounds[col]['lower']:,.1f}  "
              f"upper={bounds[col]['upper']:,.1f}")

    print()

    # Step 2: clip chunk by chunk
    chunks = []
    total  = len(df)
    for start in range(0, total, chunksize):
        end   = min(start + chunksize, total)
        chunk = df.iloc[start:end].copy()
        for col, b in bounds.items():
            chunk[col] = chunk[col].clip(lower=b["lower"],
                                          upper=b["upper"])
        chunks.append(chunk)
        print(f"  capped rows {start:>6} – {end:>6}")

    out = pd.concat(chunks, ignore_index=True)
    print(f"\n  total : {len(out):,} rows")
    return out