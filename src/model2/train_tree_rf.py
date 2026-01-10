# 12_train_model2_tree_rf.py
"""
Treina Modelo 2 (descontinuidade) com:
- Decision Tree (explicável)
- RandomForest (robusto)

Entrada: data/processed/model/model2_dataset_YYYYMMDD.parquet
Saídas:
- models/model2_<tree|rf>_YYYYMMDD.joblib
- reports/model2_metrics_YYYYMMDD.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS = PROJECT_ROOT / "reports"
RUN_DATE = datetime.now().strftime("%Y%m%d")


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.1) -> float:
    n = len(y_true)
    cut = max(1, int(n * k))
    idx = np.argsort(-y_score)[:cut]
    base_rate = y_true.mean()
    top_rate = y_true[idx].mean() if cut > 0 else 0.0
    return (top_rate / base_rate) if base_rate > 0 else np.nan


def precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.1) -> tuple[float, float]:
    n = len(y_true)
    cut = max(1, int(n * k))
    idx = np.argsort(-y_score)[:cut]
    precision = (y_true[idx].sum() / cut) if cut else 0.0
    recall = (y_true[idx].sum() / y_true.sum()) if y_true.sum() else 0.0
    return precision, recall


def main(
    in_path: Path,
    model_type: str = "rf",
    sample_n: int = 0,
    test_size: float = 0.2,
    n_estimators: int = 300,
    max_depth: int = 12,
    min_leaf_tree: int = 200,
    min_leaf_rf: int = 50,
    random_state: int = 42,
) -> None:
    if not in_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {in_path}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    df = pq.read_table(in_path).to_pandas()

    if sample_n and sample_n < len(df):
        df = df.sample(sample_n, random_state=random_state)

    y = df["y_descontinuidade"].astype(int).values

    drop_cols = {
    "cnpj_raiz", "y_descontinuidade",
    "dt_inscricao_min", "dt_inscricao_max",
    "dt_inicio_min", "dt_inicio_max", "dt_situacao_max",

    # LEAKAGE (alvo foi definido por qtd_ativos==0)
    "qtd_ativos",
    "pct_ativos",

    # recomendado (quase colinear e pode “entregar” o alvo)
    "qtd_inativos",
}

    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Para árvore/forest: imputação basta (escala não é necessária)
    prep = SimpleImputer(strategy="median")

    if model_type == "tree":
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_leaf_tree,
            random_state=random_state,
        )
    elif model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_leaf_rf,
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        raise ValueError("model_type deve ser: tree ou rf")

    pipe = Pipeline(steps=[("prep", prep), ("clf", clf)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    lift10 = lift_at_k(y_test, proba, k=0.10)
    p10, r10 = precision_recall_at_k(y_test, proba, k=0.10)

    model_path = MODEL_DIR / f"model2_{model_type}_{RUN_DATE}.joblib"
    dump({"model": pipe, "features": feat_cols}, model_path)

    metrics_path = REPORTS / f"model2_metrics_{model_type}_{RUN_DATE}.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Modelo 2 — Descontinuidade (proxy: qtd_ativos==0)\n")
        f.write(f"Dataset: {in_path}\n")
        f.write(f"Amostra usada: {len(df):,} linhas (0=full)\n")
        f.write(f"Model: {model_type}\n")
        if model_type == "rf":
            f.write(f"n_estimators={n_estimators} | max_depth={max_depth} | min_leaf={min_leaf_rf}\n\n")
        else:
            f.write(f"max_depth={max_depth} | min_leaf={min_leaf_tree}\n\n")
        f.write(f"ROC AUC: {auc:.4f}\n")
        f.write(f"PR-AUC:  {ap:.4f}\n")
        f.write(f"Lift@10%: {lift10:.3f}\n")
        f.write(f"Precision@10%: {p10:.3f}\n")
        f.write(f"Recall@10%: {r10:.3f}\n\n")
        f.write("Obs.: métricas @10% simulam priorização comercial (top decil).\n")

    print(f"[DONE] Modelo salvo: {model_path}")
    print(f"[DONE] Métricas: {metrics_path}")
    print(f"[INFO] ROC AUC={auc:.4f} | PR-AUC={ap:.4f} | Lift@10%={lift10:.3f} | P@10%={p10:.3f} | R@10%={r10:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=str, required=True)
    ap.add_argument("--model", dest="model_type", choices=["tree", "rf"], default="rf")
    ap.add_argument("--sample_n", type=int, default=0)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--max_depth", type=int, default=12)
    ap.add_argument("--min_leaf_tree", type=int, default=200)
    ap.add_argument("--min_leaf_rf", type=int, default=50)
    args = ap.parse_args()

    main(
        Path(args.in_path),
        model_type=args.model_type,
        sample_n=args.sample_n,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_leaf_tree=args.min_leaf_tree,
        min_leaf_rf=args.min_leaf_rf,
    )
