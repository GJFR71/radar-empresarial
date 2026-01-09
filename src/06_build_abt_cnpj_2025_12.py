"""
06_build_abt_cnpj_2025_12.py

Cria ABT enxuta do CNPJ (snapshot 2025-12) agregada por cnpj_raiz (8 dígitos),
a partir de data/processed/cnpj/cnpj_core_2025_12.parquet.

Saídas:
- data/processed/cnpj/cnpj_abt_2025_12.parquet
- reports/cnpj_abt_head_YYYYMMDD.csv
- reports/cnpj_abt_schema_YYYYMMDD.csv
- reports/cnpj_etapa6_summary_YYYYMMDD.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_PATH = PROJECT_ROOT / "data" / "processed" / "cnpj" / "cnpj_core_2025_12.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "cnpj" / "cnpj_abt_2025_12.parquet"
REPORTS = PROJECT_ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

RUN_DATE = datetime.now().strftime("%Y%m%d")

# Snapshot fixo: Dez/2025 (para idade em meses)
SNAPSHOT_DATE = pd.Timestamp("2025-12-31")


def _mode(series: pd.Series) -> str | None:
    """Moda robusta para strings/categorias; devolve None se vazio."""
    s = series.dropna()
    if s.empty:
        return None
    vc = s.value_counts(dropna=True)
    return str(vc.index[0]) if not vc.empty else None


def _to_int_safe(x) -> int | None:
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    # normaliza chave raiz
    if "cnpj_raiz" in df.columns:
        df["cnpj_raiz"] = (
            df["cnpj_raiz"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(8)
        )
    # UF
    if "uf" in df.columns:
        df["uf"] = df["uf"].astype(str).str.strip().str.upper().replace({"NAN": np.nan, "": np.nan})
    # CNAE
    if "cnae_fiscal_principal" in df.columns:
        df["cnae_fiscal_principal"] = (
            df["cnae_fiscal_principal"].astype(str).str.replace(r"\D", "", regex=True)
        ).replace({"": np.nan, "NAN": np.nan})
    return df


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computa features por cnpj_raiz dentro do pedaço (chunk).
    Retorna um DF já agregado por raiz.
    """
    df = df.copy()
    df = _normalize_keys(df)

    # flags
    if "situacao_cadastral" in df.columns:
        sc = df["situacao_cadastral"].astype(str).str.strip()
        df["flag_ativa"] = sc.eq("02")  # em geral "02" = ATIVA no layout do CNPJ
    else:
        df["flag_ativa"] = np.nan

    # idade (meses) - usa data_inicio_atividade se existir
    if "data_inicio_atividade" in df.columns:
        dt = pd.to_datetime(df["data_inicio_atividade"], errors="coerce")
        df["idade_meses"] = ((SNAPSHOT_DATE - dt).dt.days / 30.44).round(1)
    else:
        df["idade_meses"] = np.nan

    # agregações por raiz
    g = df.groupby("cnpj_raiz", dropna=False)

    out = pd.DataFrame({
        "cnpj_raiz": g.size().index,
        "qtd_estabelecimentos": g.size().values,
        "qtd_ufs_distintas": g["uf"].nunique(dropna=True).values if "uf" in df.columns else np.nan,
        "qtd_cnaes_distintos": g["cnae_fiscal_principal"].nunique(dropna=True).values if "cnae_fiscal_principal" in df.columns else np.nan,
        "prop_ativos": g["flag_ativa"].mean().values if "flag_ativa" in df.columns else np.nan,
        "idade_meses_mediana": g["idade_meses"].median().values if "idade_meses" in df.columns else np.nan,
        "idade_meses_min": g["idade_meses"].min().values if "idade_meses" in df.columns else np.nan,
        "idade_meses_max": g["idade_meses"].max().values if "idade_meses" in df.columns else np.nan,
    })

    # moda UF e CNAE (feito à parte para não ficar caro dentro do dict acima)
    if "uf" in df.columns:
        out["uf_modal"] = g["uf"].apply(_mode).values
    else:
        out["uf_modal"] = None

    if "cnae_fiscal_principal" in df.columns:
        out["cnae_modal"] = g["cnae_fiscal_principal"].apply(_mode).values
    else:
        out["cnae_modal"] = None

    # porte modal (vem de empresas -> pode ser útil como “proxy de porte”)
    if "porte_empresa" in df.columns:
        out["porte_modal"] = g["porte_empresa"].apply(_mode).values
    else:
        out["porte_modal"] = None

    return out


def build_abt(max_groups: int = 80) -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Parquet não encontrado: {IN_PATH}")

    pf = pq.ParquetFile(IN_PATH)
    n_groups = pf.num_row_groups
    read_groups = min(max_groups, n_groups)

    print(f"[INFO] Input: {IN_PATH}")
    print(f"[INFO] Row groups: {n_groups} | Processando: {read_groups}")

    # colunas necessárias (evita carregar razão social)
    cols_wanted = [
        "cnpj_raiz",
        "porte_empresa",
        "situacao_cadastral",
        "cnae_fiscal_principal",
        "uf",
        "data_inicio_atividade",
    ]
    # filtra só as que existem
    schema_cols = set(pf.schema.names)
    cols = [c for c in cols_wanted if c in schema_cols]

    parts = []
    for rg in range(read_groups):
        table = pf.read_row_group(rg, columns=cols)
        df = table.to_pandas()
        agg = _compute_features(df)
        parts.append(agg)

        if (rg + 1) % 10 == 0:
            print(f"  [INFO] grupos processados: {rg+1}/{read_groups}", flush=True)

    # junta agregados e re-agrega (porque cada group tem raízes repetidas)
    all_agg = pd.concat(parts, ignore_index=True)

    # re-aggregate final por raiz (soma e médias ponderadas)
    # prop_ativos: média ponderada por qtd_estabelecimentos
    def wavg(x, w):
        x = x.astype(float)
        w = w.astype(float)
        m = ~(x.isna() | w.isna())
        if not m.any():
            return np.nan
        return (x[m] * w[m]).sum() / w[m].sum()

    g = all_agg.groupby("cnpj_raiz", dropna=False)

    abt = pd.DataFrame({
        "cnpj_raiz": g.size().index,
        "qtd_estabelecimentos": g["qtd_estabelecimentos"].sum().values,
        "qtd_ufs_distintas": g["qtd_ufs_distintas"].max().values,
        "qtd_cnaes_distintos": g["qtd_cnaes_distintos"].max().values,
        "idade_meses_mediana": g["idade_meses_mediana"].median().values,
        "idade_meses_min": g["idade_meses_min"].min().values,
        "idade_meses_max": g["idade_meses_max"].max().values,
        "prop_ativos": g.apply(lambda x: wavg(x["prop_ativos"], x["qtd_estabelecimentos"])).values,
        "uf_modal": g["uf_modal"].apply(_mode).values,
        "cnae_modal": g["cnae_modal"].apply(_mode).values,
        "porte_modal": g["porte_modal"].apply(_mode).values,
    })

    # salva parquet
    table_out = pa.Table.from_pandas(abt, preserve_index=False)
    pq.write_table(table_out, OUT_PATH, compression="snappy")
    print(f"[DONE] ABT CNPJ salva: {OUT_PATH} | linhas: {len(abt):,}")

    # reports
    head_path = REPORTS / f"cnpj_abt_head_{RUN_DATE}.csv"
    abt.head(30).to_csv(head_path, index=False, encoding="utf-8")

    schema_path = REPORTS / f"cnpj_abt_schema_{RUN_DATE}.csv"
    sch = pq.ParquetFile(OUT_PATH).schema_arrow
    pd.DataFrame({"field": sch.names, "type": [str(sch.field(i).type) for i in range(len(sch.names))]}).to_csv(
        schema_path, index=False, encoding="utf-8"
    )

    summary_path = REPORTS / f"cnpj_etapa6_summary_{RUN_DATE}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Etapa 6 — ABT CNPJ (snapshot 2025-12)\n")
        f.write(f"Input:  {IN_PATH}\n")
        f.write(f"Output: {OUT_PATH}\n")
        f.write(f"Row groups total: {n_groups} | processados: {read_groups}\n")
        f.write(f"Linhas ABT (cnpj_raiz): {len(abt):,}\n\n")
        f.write("Features:\n")
        for c in abt.columns:
            f.write(f"- {c}\n")

    print(f"[DONE] Reports: {head_path.name}, {schema_path.name}, {summary_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_groups", type=int, default=80, help="Quantos row groups processar (0 = todos)")
    args = ap.parse_args()

    max_groups = args.max_groups
    if max_groups == 0:
        # 0 = todos
        pf = pq.ParquetFile(IN_PATH)
        max_groups = pf.num_row_groups

    build_abt(max_groups=max_groups)


if __name__ == "__main__":
    main()
