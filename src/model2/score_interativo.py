# 13d_score_model2_interativo.py
"""
Produto demo (Modelo 2):
- Usuário só digita o CNPJ raiz (8 dígitos).
- Dataset e modelo ficam fixos.
- Saída em linguagem de negócio.
- "Principais fatores" mostram efeito LOCAL no score:
  (aumenta o risco) / (reduz o risco) com base no sinal de (z-score × importance).

Uso:
  python -u .\src\13d_score_model2_interativo.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import load

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Fixos (produto final)
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "model" / "model2_dataset_20260110.parquet"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "model2_rf_20260110.joblib"


def clean_cnpj_raiz(x: str) -> str:
    x = "".join([c for c in str(x) if c.isdigit()])
    return x.zfill(8)[:8]


def risk_band(score: float) -> str:
    if score >= 0.80:
        return "Alto risco de descontinuidade"
    if score >= 0.50:
        return "Risco moderado"
    return "Baixo risco"


def human_feature_name(col: str) -> str:
    mapping = {
        "qtd_estabelecimentos": "Qtd estabelecimentos",
        "qtd_ativos": "Qtd estabelecimentos ativos",
        "qtd_inativos": "Qtd estabelecimentos inativos",
        "idade_empresa_dias": "Idade da empresa (dias)",
        "idade_empresa_anos": "Idade da empresa (anos)",
        "pgfn_linhas": "Qtd de registros na PGFN",
        "pgfn_qtd_inscricoes": "Qtd de inscrições na PGFN",
        "pgfn_valor_sum": "Valor total da dívida (PGFN)",
        "pgfn_valor_mean": "Valor médio por dívida (PGFN)",
        "pgfn_valor_max": "Maior dívida individual (PGFN)",
        "pgfn_qtd_ajuizadas": "Qtd de dívidas ajuizadas",
        "pgfn_pct_ajuizadas": "Percentual de dívidas ajuizadas",
        "pct_ativos": "Percentual de estabelecimentos ativos",
        "log_pgfn_valor_sum": "Log do valor total da dívida",
        "log_qtd_estabelecimentos": "Log da qtd de estabelecimentos",
    }
    return mapping.get(col, col.replace("_", " ").title())


def format_value(feature: str, value: float) -> str:
    # formata números para leitura rápida
    if feature in {"pgfn_valor_sum", "pgfn_valor_mean", "pgfn_valor_max"}:
        return f"{value:,.0f}".replace(",", ".")
    if feature in {"pgfn_pct_ajuizadas", "pct_ativos", "pct_irregular", "pct_negociacao", "pct_garantia",
                   "pct_beneficio_fiscal", "pct_suspenso_judicial"}:
        return f"{value:.3f}"
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.2f}"


def main() -> None:
    data_path = DEFAULT_DATA
    model_path = DEFAULT_MODEL

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    cnpj_in = input("Digite o CNPJ raiz (8 dígitos, sem separadores): ").strip()
    cnpj = clean_cnpj_raiz(cnpj_in)

    pack = load(model_path)
    pipe = pack["model"]
    feat_cols = pack["features"]

    df = pq.read_table(data_path).to_pandas()

    mask = df["cnpj_raiz"].astype(str).str.zfill(8) == cnpj
    if not mask.any():
        raise SystemExit(f"CNPJ raiz {cnpj} não encontrado no dataset do Modelo 2.")

    row = df.loc[mask].iloc[0]

    # score
    x_one = pd.DataFrame([row[feat_cols].to_dict()])
    score = float(pipe.predict_proba(x_one)[0, 1])

    # proxy observado
    y_obs = int(row["y_descontinuidade"])
    status_txt = "Empresa sem estabelecimentos ativos" if y_obs == 1 else "Empresa com pelo menos 1 estabelecimento ativo"

    # importâncias (tree/rf)
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        raise SystemExit("Modelo não possui feature_importances_. Use árvore ou RandomForest.")
    importances = np.array(clf.feature_importances_, dtype=float)
    importances = np.where(np.isfinite(importances), importances, 0.0)

    # -------- contrib LOCAL: z-score robusto (mediana/MAD) × importance --------
    X = df[feat_cols].copy()
    X_num = X.apply(pd.to_numeric, errors="coerce")

    med = X_num.median(numeric_only=True)
    mad = (X_num - med).abs().median(numeric_only=True).replace(0, 1.0)

    z = ((X_num - med) / mad).fillna(0.0).values
    contrib = z * importances.reshape(1, -1)

    # índice (linha) do CNPJ dentro do df
    idx_row = int(df.index[mask][0])

    # top features por |contrib|
    topk = 5
    order = np.argsort(-np.abs(contrib[idx_row]))  # maior impacto absoluto primeiro
    selected = []
    for j in order:
        f = feat_cols[j]

        # evita redundância (idade em dias e anos) no mesmo top
        if f == "idade_empresa_anos" and "idade_empresa_dias" in selected:
            continue
        if f == "idade_empresa_dias" and "idade_empresa_anos" in selected:
            continue

        selected.append(f)
        if len(selected) == topk:
            break

    # monta linhas já com (aumenta/reduz) pelo sinal LOCAL
    fator_lines = []
    for f in selected:
        v = float(pd.to_numeric(row[f], errors="coerce")) if pd.notna(row[f]) else 0.0
        sign = float(contrib[idx_row][feat_cols.index(f)])  # contrib local

        efeito = "aumenta o risco" if sign > 0 else "reduz o risco"
        # Se contrib for ~0, deixa neutro
        if abs(sign) < 1e-12:
            efeito = "impacto pequeno"

        fator_lines.append(f"- {human_feature_name(f)}: {format_value(f, v)} ({efeito})")

    # -------- saída “bonita” --------
    print("=" * 60)
    print(f"CNPJ raiz avaliado: {cnpj}")
    print(f"Score de descontinuidade (0–1): {score:.3f}")
    print(f"Classificação: {risk_band(score)}\n")

    print("Última atualização oficial disponível:")
    print(status_txt)
    print("\nPrincipais fatores considerados pelo modelo:")
    for ln in fator_lines:
        print(ln)

    print("\nObservação:")
    print("Este score é um indicador estatístico baseado em dados públicos.")
    print("Não representa certeza de encerramento, mas uma priorização de risco")
    print("para fins analíticos e de gestão.")
    print("=" * 60)


if __name__ == "__main__":
    # mantém compatível com uso simples (sem args)
    main()
