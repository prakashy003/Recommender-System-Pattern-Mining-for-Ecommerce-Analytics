from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.metrics.pairwise import cosine_similarity


NOTEBOOK6_RESULTS = [
    {
        "Model": "Hybrid (User-CF + Content)",
        "UsersEvaluated": 50,
        "PrecisionK": 0.044,
        "RecallK": 0.22,
        "MAPK": 0.145667,
        "Coverage": 0.71,
        "Diversity": 0.818396,
    },
    {
        "Model": "User-Based CF",
        "UsersEvaluated": 50,
        "PrecisionK": 0.036,
        "RecallK": 0.18,
        "MAPK": 0.109000,
        "Coverage": 0.76,
        "Diversity": 0.804878,
    },
    {
        "Model": "Popularity Baseline",
        "UsersEvaluated": 50,
        "PrecisionK": 0.016,
        "RecallK": 0.08,
        "MAPK": 0.048000,
        "Coverage": 0.10,
        "Diversity": 0.815457,
    },
    {
        "Model": "Item-Based CF",
        "UsersEvaluated": 50,
        "PrecisionK": 0.012,
        "RecallK": 0.06,
        "MAPK": 0.016667,
        "Coverage": 0.80,
        "Diversity": 0.838859,
    },
]


def resolve_processed_dir() -> Path | None:
    candidates = [
        settings.SMARTCART_ROOT / "data" / "processed",
        Path.cwd() / "data" / "processed",
        Path.cwd().parent / "data" / "processed",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def required_processed_files() -> list[str]:
    return [
        "user_data_clean.csv",
        "product_data_clean.csv",
        "user_item_matrix_filled.csv",
        "user_category_agg.csv",
    ]


def build_user_item_matrix(user_data: pd.DataFrame) -> pd.DataFrame:
    if user_data.empty:
        return pd.DataFrame()
    matrix = user_data.pivot_table(
        index="UserID",
        columns="ProductID",
        values="Rating",
        aggfunc="mean",
    )
    return matrix.fillna(0).astype(np.float64)


def build_user_category_agg(user_data: pd.DataFrame) -> pd.DataFrame:
    if user_data.empty:
        return pd.DataFrame(columns=["UserID", "Category", "TotalInteractions", "AverageRating", "LastInteraction"])

    return (
        user_data.groupby(["UserID", "Category"], as_index=False)
        .agg(
            TotalInteractions=("Rating", "count"),
            AverageRating=("Rating", "mean"),
            LastInteraction=("Timestamp", "max"),
        )
        .sort_values(["UserID", "TotalInteractions"], ascending=[True, False])
        .reset_index(drop=True)
    )


def load_data() -> dict[str, pd.DataFrame] | None:
    processed_dir = resolve_processed_dir()
    if processed_dir is None:
        return None

    missing = [f for f in required_processed_files() if not (processed_dir / f).exists()]
    if missing:
        return None

    base_user = pd.read_csv(processed_dir / "user_data_clean.csv", parse_dates=["Timestamp"])
    app_user_path = processed_dir / "user_data_app.csv"
    user_data = pd.read_csv(app_user_path, parse_dates=["Timestamp"]) if app_user_path.exists() else base_user

    product_data = pd.read_csv(processed_dir / "product_data_clean.csv")
    user_item_matrix_filled = build_user_item_matrix(user_data)
    user_category_agg = build_user_category_agg(user_data)

    return {
        "user_data": user_data,
        "product_data": product_data,
        "user_item_matrix_filled": user_item_matrix_filled,
        "user_category_agg": user_category_agg,
        "processed_dir_df": pd.DataFrame({"path": [str(processed_dir)]}),
    }


def save_app_interaction(user_data: pd.DataFrame, processed_dir: Path) -> None:
    user_data.to_csv(processed_dir / "user_data_app.csv", index=False)


def safe_cosine_similarity(matrix_like):
    matrix = np.asarray(matrix_like, dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return cosine_similarity(matrix)


def recommend_user_cf(target_user: str, ratings_matrix: pd.DataFrame, n: int = 5, neighbor_top_m: int = 10) -> pd.DataFrame:
    if ratings_matrix.empty or target_user not in ratings_matrix.index:
        return pd.DataFrame(columns=["ProductID", "PredictedScore"])

    user_sim = pd.DataFrame(
        safe_cosine_similarity(ratings_matrix),
        index=ratings_matrix.index,
        columns=ratings_matrix.index,
    )

    neighbors = (
        user_sim.loc[target_user]
        .drop(index=target_user, errors="ignore")
        .sort_values(ascending=False)
        .head(neighbor_top_m)
    )
    neighbors = neighbors[neighbors > 0]
    if neighbors.empty:
        return pd.DataFrame(columns=["ProductID", "PredictedScore"])

    target_ratings = ratings_matrix.loc[target_user]
    unrated = target_ratings[target_ratings <= 0].index
    if len(unrated) == 0:
        return pd.DataFrame(columns=["ProductID", "PredictedScore"])

    neighbor_ratings = ratings_matrix.loc[neighbors.index, unrated]
    weighted_sum = neighbor_ratings.T.dot(neighbors.values)
    denom = np.abs(neighbors.values).sum()
    if denom == 0:
        return pd.DataFrame(columns=["ProductID", "PredictedScore"])

    pred = (weighted_sum / denom).sort_values(ascending=False)
    return pred.head(n).rename("PredictedScore").reset_index().rename(columns={"index": "ProductID"})


def build_transactions(user_df: pd.DataFrame) -> list[list[str]]:
    return (
        user_df.groupby("UserID")["ProductID"]
        .apply(lambda s: sorted(set(s.astype(str))))
        .tolist()
    )


def try_import_mlxtend():
    try:
        from mlxtend.preprocessing import TransactionEncoder  # type: ignore
        from mlxtend.frequent_patterns import apriori, association_rules  # type: ignore

        return TransactionEncoder, apriori, association_rules
    except Exception:
        return None, None, None


def mine_pair_rules_fallback(transactions: list[list[str]], min_support: float = 0.08, min_conf: float = 0.35) -> pd.DataFrame:
    tx_sets = [set(t) for t in transactions if len(t) > 0]
    n = len(tx_sets)
    if n == 0:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    item_count: dict[str, int] = {}
    pair_count: dict[tuple[str, str], int] = {}
    for tx in tx_sets:
        for i in tx:
            item_count[i] = item_count.get(i, 0) + 1
        for a, b in combinations(sorted(tx), 2):
            pair_count[(a, b)] = pair_count.get((a, b), 0) + 1

    rows = []
    for (a, b), c_ab in pair_count.items():
        sup_ab = c_ab / n
        if sup_ab < min_support:
            continue

        sup_a = item_count[a] / n
        sup_b = item_count[b] / n

        conf_a_b = sup_ab / sup_a if sup_a else 0.0
        lift_a_b = conf_a_b / sup_b if sup_b else 0.0
        if conf_a_b >= min_conf:
            rows.append({
                "antecedents": frozenset([a]),
                "consequents": frozenset([b]),
                "support": sup_ab,
                "confidence": conf_a_b,
                "lift": lift_a_b,
            })

        conf_b_a = sup_ab / sup_b if sup_b else 0.0
        lift_b_a = conf_b_a / sup_a if sup_a else 0.0
        if conf_b_a >= min_conf:
            rows.append({
                "antecedents": frozenset([b]),
                "consequents": frozenset([a]),
                "support": sup_ab,
                "confidence": conf_b_a,
                "lift": lift_b_a,
            })

    if not rows:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    return pd.DataFrame(rows).sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)


def mine_rules(user_df: pd.DataFrame, min_support: float = 0.08, min_conf: float = 0.35) -> pd.DataFrame:
    transactions = build_transactions(user_df)

    TransactionEncoder, apriori, association_rules = try_import_mlxtend()
    if TransactionEncoder is not None and apriori is not None and association_rules is not None:
        te = TransactionEncoder()
        tx_matrix = te.fit(transactions).transform(transactions)
        tx_df = pd.DataFrame(tx_matrix, columns=te.columns_)
        fi = apriori(tx_df, min_support=min_support, use_colnames=True, max_len=3)
        if fi.empty:
            return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

        rules = association_rules(fi, metric="confidence", min_threshold=min_conf)
        if rules.empty:
            return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

        return (
            rules[["antecedents", "consequents", "support", "confidence", "lift"]]
            .sort_values(["lift", "confidence", "support"], ascending=False)
            .reset_index(drop=True)
        )

    return mine_pair_rules_fallback(transactions, min_support=min_support, min_conf=min_conf)
