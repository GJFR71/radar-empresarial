# 09_train_model1_logistica.py
"""
Treina Modelo 1 (Saúde fiscal) com regressão logística regularizada.
Entrada: data/processed/model/model1_dataset_YYYYMMDD.parquet
Saídas:
- models/model1_logit_YYYYMMDD.joblib
- reports/model1_metrics_YYYYMMDD.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    sample_n: int = 0,
    test_size: float = 0.2,
    penalty: str = "elasticnet",
    C: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 5000,
    tol: float = 1e-4,
) -> None:
    if not in_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {in_path}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Lê parquet agregado (1 linha por cnpj_raiz)
    df = pq.read_table(in_path).to_pandas()

    # Amostra opcional (para rodar rápido)
    if sample_n and sample_n < len(df):
        df = df.sample(sample_n, random_state=42)

    y = df["y_risco_fiscal"].astype(int).values

    # Features numéricas (mantém simples e explicável)
    drop_cols = {"cnpj_raiz", "y_risco_fiscal", "dt_inscricao_min", "dt_inscricao_max"}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].copy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    num_cols = feat_cols

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ]
                ),
                num_cols,
            )
        ],
        remainder="drop",
    )

    # Logística regularizada (solver saga suporta elasticnet/l1/l2)
    if penalty == "elasticnet":
        clf = LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            l1_ratio=l1_ratio,
            C=C,
            max_iter=max_iter,
            tol=tol,
            n_jobs=-1,
            random_state=42,
        )
    elif penalty in {"l1", "l2"}:
        clf = LogisticRegression(
            solver="saga",
            penalty=penalty,
            C=C,
            max_iter=max_iter,
            tol=tol,
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError("penalty deve ser: elasticnet, l1, l2")

    pipe = Pipeline(steps=[("prep", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    # Scores
    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    lift10 = lift_at_k(y_test, proba, k=0.10)
    prec10, rec10 = precision_recall_at_k(y_test, proba, k=0.10)

    # Salvar modelo
    model_path = MODEL_DIR / f"model1_logit_{RUN_DATE}.joblib"
    dump({"model": pipe, "features": feat_cols}, model_path)

    # Salvar métricas
    metrics_path = REPORTS / f"model1_metrics_{RUN_DATE}.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Modelo 1 — Regressão Logística (saúde fiscal)\n")
        f.write(f"Dataset: {in_path}\n")
        f.write(f"Amostra usada: {len(df):,} linhas (0=full)\n")
        f.write(f"penalty={penalty} | C={C} | l1_ratio={l1_ratio}\n")
        f.write(f"max_iter={max_iter} | tol={tol}\n\n")
        f.write(f"ROC AUC: {auc:.4f}\n")
        f.write(f"Average Precision (PR-AUC): {ap:.4f}\n")
        f.write(f"Lift@10%: {lift10:.3f}\n")
        f.write(f"Precision@10%: {prec10:.3f}\n")
        f.write(f"Recall@10%: {rec10:.3f}\n\n")
        f.write("Observação: métricas @10% simulam priorização comercial (top decil).\n")

    print(f"[DONE] Modelo salvo: {model_path}")
    print(f"[DONE] Métricas: {metrics_path}")
    print(
        f"[INFO] ROC AUC={auc:.4f} | PR-AUC={ap:.4f} | Lift@10%={lift10:.3f} | "
        f"P@10%={prec10:.3f} | R@10%={rec10:.3f}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=str, required=True, help="Parquet do model1_dataset_YYYYMMDD.parquet")
    ap.add_argument("--sample_n", type=int, default=0, help="0=full; senão amostra aleatória")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--penalty", type=str, default="elasticnet", choices=["elasticnet", "l1", "l2"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--l1_ratio", type=float, default=0.5)
    ap.add_argument("--max_iter", type=int, default=5000, help="Iterações máximas do solver (saga)")
    ap.add_argument("--tol", type=float, default=1e-4, help="Tolerância de convergência")
    args = ap.parse_args()

    main(
        Path(args.in_path),
        sample_n=args.sample_n,
        test_size=args.test_size,
        penalty=args.penalty,
        C=args.C,
        l1_ratio=args.l1_ratio,
        max_iter=args.max_iter,
        tol=args.tol,
    )
