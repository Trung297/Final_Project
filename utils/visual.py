import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "figure.dpi"       : 110,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.titlesize"   : 12,
    "axes.titleweight" : "bold",
    "axes.labelsize"   : 10,
    "figure.figsize"   : (10, 4),
})


def plot_histogram(df: pd.DataFrame, col: str,
                   bins: int = 30,
                   title: str = None,
                   color: str = "steelblue") -> None:
    data   = df[col].dropna()
    mean_v = data.mean()
    med_v  = data.median()

    fig, ax = plt.subplots()
    ax.hist(data, bins=bins, color=color,
            edgecolor="white", alpha=0.8)
    ax.axvline(mean_v, color="red",   lw=2, ls="--",
               label=f"mean = {mean_v:,.1f}")
    ax.axvline(med_v,  color="green", lw=2, ls="-",
               label=f"median = {med_v:,.1f}")
    ax.set_title(title or col)
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_bar(df: pd.DataFrame, col: str, value: str,
             agg: str = "sum",
             top_n: int = 10,
             title: str = None,
             color: str = "steelblue") -> None:
    data = (df.groupby(col)[value]
              .agg(agg)
              .sort_values()
              .tail(top_n))

    fig, ax = plt.subplots()
    ax.barh(data.index.astype(str), data.values,
            color=color, edgecolor="white")
    ax.set_title(title or f"{agg}({value}) by {col}")
    ax.set_xlabel(value)
    plt.tight_layout()
    plt.show()


def plot_countbar(df: pd.DataFrame, col: str,
                  title: str = None,
                  color: str = "steelblue") -> None:
    counts = df[col].value_counts()

    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.values,
           color=color, edgecolor="white")
    ax.set_title(title or f"{col} distribution")
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.show()


def plot_boxplot(df: pd.DataFrame, num_col: str, cat_col: str,
                 title: str = None) -> None:
    order = (df.groupby(cat_col)[num_col]
               .median()
               .sort_values(ascending=False)
               .index.tolist())

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=cat_col, y=num_col,
                order=order, palette="tab10", ax=ax,
                flierprops={"marker"          : ".",
                            "markersize"      : 4,
                            "markerfacecolor" : "gray",
                            "alpha"           : 0.5})
    ax.set_title(title or f"{num_col} by {cat_col}")
    ax.set_xlabel(cat_col)
    ax.set_ylabel(num_col)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.show()


def plot_line(df: pd.DataFrame, date_col: str, value_col: str,
              freq: str = "ME", agg: str = "sum",
              title: str = None,
              color: str = "steelblue") -> None:
    ts = (df.set_index(date_col)[value_col]
            .resample(freq)
            .agg(agg))

    fig, ax = plt.subplots()
    ax.plot(ts.index, ts.values, marker="o",
            ms=4, lw=2, color=color)
    ax.fill_between(ts.index, ts.values,
                    alpha=0.1, color=color)
    ax.set_title(title or f"{value_col} over time")
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.show()


def plot_heatmap_corr(df: pd.DataFrame, cols: list,
                       title: str = "Correlation matrix") -> None:
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, mask=mask,
                annot=True, fmt=".2f",
                cmap="RdYlGn", vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()