# 10_score_model1_explain.py
"""
Gera score do Modelo 1 (saúde fiscal) + explicação por coeficientes (logística).
Entradas:
- --data : parquet do model1_dataset_YYYYMMDD.parquet
- --model: joblib do model1_logit_YYYYMMDD.joblib
Saídas:
- reports/model1_scores_YYYYMMDD.csv (cnpj_raiz + score)
- reports/model1_top_factors_YYYYMMDD.csv (top coeficientes globais)
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
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pq.read_table(data_path).to_pandas()
    cnpj = df["cnpj_raiz"].astype(str)

    y_col = "y_risco_fiscal"
    drop_cols = {"cnpj_raiz", y_col, "dt_inscricao_min", "dt_inscricao_max"}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].copy()

    payload = load(model_path)
    pipe = payload["model"]
    features = payload.get("features", feat_cols)

    # Score (probabilidade de y=1)
    proba = pipe.predict_proba(X[features])[:, 1]

    scores = pd.DataFrame({"cnpj_raiz": cnpj, "score_risco_fiscal": proba})
    scores = scores.sort_values("score_risco_fiscal", ascending=False)

    scores_path = out_dir / f"model1_scores_{RUN_DATE}.csv"
    scores.to_csv(scores_path, index=False, encoding="utf-8")

    # Coeficientes globais (explicação simples)
    clf = pipe.named_steps["clf"]
    coefs = clf.coef_.ravel()

    coef_df = pd.DataFrame({"feature": features, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    top_path = out_dir / f"model1_top_factors_{RUN_DATE}.csv"
    coef_df[["feature", "coef"]].head(30).to_csv(top_path, index=False, encoding="utf-8")

    print(f"[DONE] Scores: {scores_path}")
    print(f"[DONE] Top fatores: {top_path}")
    print("[INFO] Interpretação: coef>0 aumenta o score; coef<0 reduz o score.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Parquet model1_dataset_YYYYMMDD.parquet")
    ap.add_argument("--model", required=True, help="Joblib model1_logit_YYYYMMDD.joblib")
    ap.add_argument("--out", default="reports", help="Diretório de saída (default=reports)")
    args = ap.parse_args()
    main(Path(args.data), Path(args.model), Path(args.out))
