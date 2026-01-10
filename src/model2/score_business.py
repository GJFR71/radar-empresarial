"""
Score de Descontinuidade — Versão Negócio (Modelo 2)

Objetivo:
- Permitir digitar um CNPJ raiz
- Retornar score de descontinuidade
- Classificar o risco (baixo / moderado / alto)
- Explicar, em linguagem simples, os principais fatores

Entradas:
- Dataset: data/processed/model/model2_dataset_YYYYMMDD.parquet
- Modelo:  models/model2_<tree|rf>_YYYYMMDD.joblib

Uso:
python 13b_score_model2_business.py \
  --data data/processed/model/model2_dataset_YYYYMMDD.parquet \
  --model models/model2_rf_YYYYMMDD.joblib \
  --cnpj 96199989
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import load


# -----------------------------
# Tradução técnica → negócio
# -----------------------------
FEATURE_TRANSLATION = {
    "qtd_estabelecimentos": "Quantidade total de estabelecimentos",
    "qtd_ativos": "Quantidade de estabelecimentos ativos",
    "qtd_inativos": "Quantidade de estabelecimentos inativos",
    "idade_empresa_dias": "Idade da empresa",
    "pgfn_linhas": "Quantidade de registros na dívida ativa",
    "pgfn_qtd_inscricoes": "Quantidade de inscrições em dívida ativa",
    "pgfn_valor_sum": "Valor total da dívida ativa",
    "pgfn_valor_mean": "Valor médio das dívidas",
    "pgfn_valor_max": "Maior valor individual em dívida",
    "pgfn_qtd_ajuizadas": "Quantidade de dívidas ajuizadas",
    "pgfn_pct_ajuizadas": "Percentual de dívidas ajuizadas",
    "pgfn_receitas_distintas": "Diversidade de tipos de tributo",
    "pgfn_situacoes_distintas": "Diversidade de situações da dívida",
    "pct_ativos": "Proporção de estabelecimentos ativos",
    "log_qtd_estabelecimentos": "Escala de porte da empresa",
    "idade_empresa_anos": "Idade da empresa (anos)",
    "log_pgfn_valor_sum": "Escala do valor total da dívida",
}


def risk_label(score: float) -> str:
    if score < 0.30:
        return "Baixo risco de descontinuidade"
    elif score < 0.60:
        return "Risco moderado de descontinuidade"
    else:
        return "Alto risco de descontinuidade"


def main(data_path: Path, model_path: Path, cnpj_raiz: str) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    # Carrega modelo
    pack = load(model_path)
    pipe = pack["model"]
    feat_cols = pack["features"]

    # Carrega dados
    df = pq.read_table(data_path).to_pandas()
    df["cnpj_raiz"] = df["cnpj_raiz"].astype(str)

    row = df[df["cnpj_raiz"] == cnpj_raiz]
    if row.empty:
        raise ValueError(f"CNPJ raiz {cnpj_raiz} não encontrado no dataset.")

    X = row[feat_cols]
    score = float(pipe.predict_proba(X)[:, 1][0])
    label = risk_label(score)
    y_obs = int(row["y_descontinuidade"].iloc[0])

    # -----------------------------
    # Explicação (heurística)
    # -----------------------------
    clf = pipe.named_steps["clf"]
    importances = clf.feature_importances_
    X_num = X.apply(pd.to_numeric, errors="coerce")

    med = df[feat_cols].median(numeric_only=True)
    mad = (df[feat_cols] - med).abs().median(numeric_only=True).replace(0, 1.0)

    z = ((X_num - med) / mad).fillna(0.0).values.flatten()
    contrib = z * importances

    top_idx = np.argsort(-np.abs(contrib))[:5]

    fatores = []
    for i in top_idx:
        var = feat_cols[i]
        direction = "aumenta" if contrib[i] > 0 else "reduz"
        fatores.append(f"- {FEATURE_TRANSLATION.get(var, var)} ({direction} o risco)")

    # -----------------------------
    # Saída amigável
    # -----------------------------
    print("\n" + "=" * 60)
    print(f"CNPJ raiz avaliado: {cnpj_raiz}")
    print(f"Score de descontinuidade (0–1): {score:.3f}")
    print(f"Classificação: {label}")
    print(f"Situação observada no snapshot (proxy): {y_obs}")
    print("\nPrincipais fatores considerados pelo modelo:")
    for f in fatores:
        print(f)
    print("\nObservação:")
    print(
        "Este score é um indicador estatístico baseado em dados públicos.\n"
        "Não representa certeza de encerramento, mas uma priorização de risco\n"
        "para fins analíticos e de gestão."
    )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Parquet do model2_dataset_YYYYMMDD.parquet")
    ap.add_argument("--model", required=True, help="Modelo treinado model2_<tree|rf>_YYYYMMDD.joblib")
    ap.add_argument("--cnpj", required=True, help="CNPJ raiz (8 dígitos)")
    args = ap.parse_args()

    main(Path(args.data), Path(args.model), args.cnpj)

