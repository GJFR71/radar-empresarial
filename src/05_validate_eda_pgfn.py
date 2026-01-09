"""
ETAPA 5 – Validação estrutural + EDA inicial (sem carregar o Parquet inteiro)

Objetivo:
- Confirmar volume e estrutura via metadata
- Gerar head para visualização
- Montar amostra controlada (ex.: 200k linhas)
- Diagnosticar nulos, tipos e colunas candidatas a remoção (tabela de decisão)
- Checar representatividade de uf_devedor ao longo do Parquet (row groups espaçados)

Saídas (reports/):
- pgfn_schema_YYYYMMDD.csv
- pgfn_head_eda_YYYYMMDD.csv
- pgfn_info_YYYYMMDD.txt
- pgfn_nulos_amostra_YYYYMMDD.csv
- pgfn_tabela_decisao_YYYYMMDD.csv
- pgfn_uf_devedor_check_YYYYMMDD.csv
- pgfn_etapa5_summary_YYYYMMDD.txt
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import io

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# -------- Config --------
PARQUET_PATH = Path("data/processed/pgfn/pgfn_sida_2024_2025.parquet")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

RUN_DATE = datetime.now().strftime("%Y%m%d")
RUN_TS = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

HEAD_ROWS = 100
MAX_SAMPLE_ROWS = 200_000

# colunas que podem ser constantes em amostra e AINDA assim serem úteis como dimensões/linhagem
PROTECTED_CONSTANT_COLS = {"ano", "trimestre", "uf", "arquivo_origem"}


def read_row_group_df(pq_file: pq.ParquetFile, rg_index: int, columns: list[str] | None = None) -> pd.DataFrame:
    """Lê um row group (ou subconjunto de colunas) e converte para pandas."""
    table = pq_file.read_row_group(rg_index, columns=columns)
    return table.to_pandas()


def build_sample(pq_file: pq.ParquetFile, max_rows: int) -> pd.DataFrame:
    """Monta uma amostra concatenando row groups até atingir max_rows."""
    num_row_groups = pq_file.num_row_groups
    if num_row_groups == 0:
        return pd.DataFrame()

    rg0 = read_row_group_df(pq_file, 0)
    parts = [rg0]
    count = len(rg0)

    rg = 1
    while count < max_rows and rg < num_row_groups:
        part = read_row_group_df(pq_file, rg)
        parts.append(part)
        count += len(part)
        rg += 1

    sample_df = pd.concat(parts, ignore_index=True)
    if len(sample_df) > max_rows:
        sample_df = sample_df.head(max_rows)

    return sample_df


def decision_table(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Gera tabela objetiva de decisão (manter/remover + tipo sugerido) com base na amostra."""
    N = len(sample_df)
    rows = []

    for col in sample_df.columns:
        s = sample_df[col]

        n_null = int(s.isna().sum())
        pct_null = round((n_null / N * 100), 2) if N else 0.0

        nunique = int(s.nunique(dropna=True))
        is_constant = (nunique == 1)
        dtype_atual = str(s.dtype)

        # decisão objetiva
        if pct_null == 100.0:
            acao = "remover"
            tipo_destino = "N/A"
            justificativa = "100% nulo na amostra"
        elif is_constant and col not in PROTECTED_CONSTANT_COLS:
            acao = "remover"
            tipo_destino = "N/A"
            justificativa = "coluna constante (1 valor) na amostra"
        else:
            acao = "manter"

            # sugestão de tipo (sem “decidir ML”; só estrutura)
            if col == "valor_consolidado":
                tipo_destino = "float64"
                justificativa = "métrica financeira"
            elif col in {"ano", "trimestre"}:
                tipo_destino = "int"
                justificativa = "dimensão temporal"
            elif "data" in col.lower():
                tipo_destino = "datetime"
                justificativa = "variável temporal"
            elif dtype_atual in {"object", "string"} and nunique <= 50:
                tipo_destino = "category"
                justificativa = "baixa cardinalidade"
            else:
                tipo_destino = "string"
                justificativa = "texto/identificador"

        rows.append(
            {
                "coluna": col,
                "tipo_atual": dtype_atual,
                "tipo_sugerido": tipo_destino,
                "acao_sugerida": acao,
                "nulos_%": pct_null,
                "n_valores_distintos": nunique,
                "constante": is_constant,
                "justificativa": justificativa,
            }
        )

    decisao_df = (
        pd.DataFrame(rows)
        .sort_values(["acao_sugerida", "nulos_%"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return decisao_df


def uf_devedor_check(pq_file: pq.ParquetFile, num_checks: int = 10) -> pd.DataFrame:
    """
    Checa a taxa de nulos de uf_devedor em row groups espaçados.
    Útil para demonstrar que a amostra não “enganou”.
    """
    if pq_file.num_row_groups == 0:
        return pd.DataFrame(columns=["row_group", "linhas", "nulos", "pct_nulos"])

    # se não existir a coluna, devolve vazio (não quebra a execução)
    schema_names = set(pq_file.schema_arrow.names)
    if "uf_devedor" not in schema_names:
        return pd.DataFrame(columns=["row_group", "linhas", "nulos", "pct_nulos"])

    rg_indices = np.linspace(0, pq_file.num_row_groups - 1, num=num_checks, dtype=int)

    out = []
    for rg in rg_indices:
        part = read_row_group_df(pq_file, rg, columns=["uf_devedor"])
        total = len(part)
        n_null = int(part["uf_devedor"].isna().sum())
        pct = round((n_null / total * 100), 2) if total else 0.0

        out.append({"row_group": int(rg), "linhas": int(total), "nulos": n_null, "pct_nulos": pct})

    return pd.DataFrame(out)


def main() -> None:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet não encontrado: {PARQUET_PATH}")

    pq_file = pq.ParquetFile(PARQUET_PATH)
    md = pq_file.metadata
    schema = pq_file.schema_arrow

    total_rows = md.num_rows
    num_row_groups = pq_file.num_row_groups
    colnames = schema.names

    print(f"[INFO] Execução: {RUN_TS}")
    print(f"[INFO] Parquet: {PARQUET_PATH}")
    print(f"[INFO] Linhas (metadata): {total_rows:,}")
    print(f"[INFO] Colunas: {len(colnames)}")
    print(f"[INFO] Row groups: {num_row_groups}")

    # -------- schema report --------
    schema_rows = [{"coluna": name, "tipo_parquet": str(field.type)} for name, field in zip(schema.names, schema)]
    schema_df = pd.DataFrame(schema_rows)
    schema_path = REPORTS_DIR / f"pgfn_schema_{RUN_DATE}.csv"
    schema_df.to_csv(schema_path, index=False, encoding="utf-8")
    print(f"[OK] Schema salvo: {schema_path}")

    # -------- head (via row_group 0) --------
    if num_row_groups == 0:
        head_df = pd.DataFrame(columns=colnames)
    else:
        rg0 = read_row_group_df(pq_file, 0)
        head_df = rg0.head(HEAD_ROWS)

    head_path = REPORTS_DIR / f"pgfn_head_eda_{RUN_DATE}.csv"
    head_df.to_csv(head_path, index=False, encoding="utf-8")
    print(f"[OK] Head salvo: {head_path}")

    # -------- sample --------
    sample_df = build_sample(pq_file, MAX_SAMPLE_ROWS)
    print(f"[INFO] Amostra: {sample_df.shape[0]:,} linhas x {sample_df.shape[1]} colunas")

    # -------- info (amostra) --------
    buf = io.StringIO()
    sample_df.info(buf=buf)
    info_txt = buf.getvalue()

    info_path = REPORTS_DIR / f"pgfn_info_{RUN_DATE}.txt"
    info_path.write_text(info_txt, encoding="utf-8")
    print(f"[OK] Info (amostra) salvo: {info_path}")

    # -------- nulos (amostra) --------
    na_abs = sample_df.isna().sum()
    na_pct = (na_abs / len(sample_df) * 100).round(2) if len(sample_df) else na_abs.astype(float)

    na_tbl = (
        pd.DataFrame({"nulos": na_abs, "pct": na_pct})
        .sort_values("nulos", ascending=False)
    )

    na_path = REPORTS_DIR / f"pgfn_nulos_amostra_{RUN_DATE}.csv"
    na_tbl.to_csv(na_path, encoding="utf-8")
    print(f"[OK] Nulos (amostra) salvo: {na_path}")

    # -------- tabela de decisão --------
    decisao_df = decision_table(sample_df)
    decisao_path = REPORTS_DIR / f"pgfn_tabela_decisao_{RUN_DATE}.csv"
    decisao_df.to_csv(decisao_path, index=False, encoding="utf-8")
    print(f"[OK] Tabela de decisão salva: {decisao_path}")

    # -------- uf_devedor representatividade (row groups espaçados) --------
    uf_check_df = uf_devedor_check(pq_file, num_checks=10)
    uf_check_path = REPORTS_DIR / f"pgfn_uf_devedor_check_{RUN_DATE}.csv"
    uf_check_df.to_csv(uf_check_path, index=False, encoding="utf-8")
    print(f"[OK] Check uf_devedor salvo: {uf_check_path}")

    # -------- summary --------
    summary = []
    summary.append("PGFN – ETAPA 5 – Validação estrutural e diagnóstico inicial")
    summary.append(f"Execução: {RUN_TS}")
    summary.append("")
    summary.append("Resumo do Parquet (metadata)")
    summary.append(f"- arquivo: {PARQUET_PATH}")
    summary.append(f"- linhas: {total_rows:,}")
    summary.append(f"- colunas: {len(colnames)}")
    summary.append(f"- row_groups: {num_row_groups}")
    summary.append("")
    summary.append("Amostra usada para diagnóstico")
    summary.append(f"- linhas: {len(sample_df):,} (limite: {MAX_SAMPLE_ROWS:,})")
    summary.append("")
    summary.append("Artefatos gerados em reports/")
    summary.append(f"- {schema_path.name}")
    summary.append(f"- {head_path.name}")
    summary.append(f"- {info_path.name}")
    summary.append(f"- {na_path.name}")
    summary.append(f"- {decisao_path.name}")
    summary.append(f"- {uf_check_path.name}")
    summary.append("")

    summary_path = REPORTS_DIR / f"pgfn_etapa5_summary_{RUN_DATE}.txt"
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    print(f"[OK] Summary salvo: {summary_path}")

    print("[DONE] ETAPA 5 concluída (sem carregar o dataset inteiro).")


if __name__ == "__main__":
    main()
