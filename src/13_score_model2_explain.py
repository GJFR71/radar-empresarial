# 13_score_model2_explain.py
"""
Gera score do Modelo 2 por empresa (cnpj_raiz) + top fatores (explicação leve).

Entradas:
- dataset: data/processed/model/model2_dataset_YYYYMMDD.parquet
- modelo:  models/model2_<tree|rf>_YYYYMMDD.joblib

Saídas (em reports/ ou pasta informada):
- model2_scores_YYYYMMDD.csv
- model2_top_factors_YYYYMMDD.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import load


RUN_DATE = datetime.now().strftime("%Y%m%d")


def main(data_path: Path, model_path: Path, out_dir: Path) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    pack = load(model_path)
    pipe = pack["model"]
    feat_cols = pack["features"]

    df = pq.read_table(data_path).to_pandas().reset_index(drop=True)

    X = df[feat_cols].copy()
    proba = pipe.predict_proba(X)[:, 1]

    scores = pd.DataFrame({
        "cnpj_raiz": df["cnpj_raiz"].astype(str).values,
        "score_descontinuidade": proba,
        "y_descontinuidade": df["y_descontinuidade"].astype(int).values,
    }).sort_values("score_descontinuidade", ascending=False)

    score_path = out_dir / f"model2_scores_{RUN_DATE}.csv"
    scores.to_csv(score_path, index=False, encoding="utf-8")

    # -------- explicação leve: zscore * importance --------
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        raise ValueError("Modelo não possui feature_importances_. Use tree ou rf.")

    importances = np.array(clf.feature_importances_, dtype=float)
    importances = np.where(np.isfinite(importances), importances, 0.0)

    X_num = X.apply(pd.to_numeric, errors="coerce")
    med = X_num.median(numeric_only=True)
    mad = (X_num - med).abs().median(numeric_only=True).replace(0, 1.0)

    z = ((X_num - med) / mad).fillna(0.0)

    contrib = z.values * importances.reshape(1, -1)
    topk = 5

    feat = np.array(feat_cols, dtype=object)
    cnpj_vals = df["cnpj_raiz"].astype(str).values

    rows = []
    for i in range(len(df)):
        idx = np.argsort(-np.abs(contrib[i]))[:topk]
        rows.append({
            "cnpj_raiz": cnpj_vals[i],
            "score_descontinuidade": float(proba[i]),
            "top_features": "; ".join([str(feat[j]) for j in idx]),
        })

    top_df = pd.DataFrame(rows).sort_values("score_descontinuidade", ascending=False)
    top_path = out_dir / f"model2_top_factors_{RUN_DATE}.csv"
    top_df.to_csv(top_path, index=False, encoding="utf-8")

    print(f"[DONE] Scores: {score_path}")
    print(f"[DONE] Top fatores: {top_path}")
    print("[INFO] Interpretação: top_features = maior impacto aproximado (z-score × importance).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Parquet do model2_dataset_YYYYMMDD.parquet")
    ap.add_argument("--model", required=True, help="Joblib do model2_<tree|rf>_YYYYMMDD.joblib")
    ap.add_argument("--out", default="reports", help="Pasta de saída")
    args = ap.parse_args()

    main(Path(args.data), Path(args.model), Path(args.out))

