"""
ETAPA 5 (CNPJ) – Validação estrutural + EDA inicial (sem carregar o Parquet inteiro)

Objetivo:
- Confirmar volume e estrutura via metadata
- Gerar head para visualização
- Montar amostra controlada (ex.: 200k linhas)
- Diagnosticar nulos, tipos e colunas candidatas a ajuste (tabela de decisão)
- Checar integridade da chave (cnpj_raiz e, se existir, cnpj)

Saídas (reports/):
- cnpj_schema_YYYYMMDD.csv
- cnpj_head_eda_YYYYMMDD.csv
- cnpj_info_YYYYMMDD.txt
- cnpj_nulos_amostra_YYYYMMDD.csv
- cnpj_tabela_decisao_YYYYMMDD.csv
- cnpj_domains_YYYYMMDD.csv
- cnpj_keycheck_YYYYMMDD.txt
- cnpj_etapa5_summary_YYYYMMDD.txt
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import io
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# -------- Config --------
PARQUET_PATH = Path("data/processed/cnpj/cnpj_core_2025_12.parquet")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

RUN_DATE = datetime.now().strftime("%Y%m%d")
RUN_TS = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

HEAD_ROWS = 100
MAX_SAMPLE_ROWS = 200_000

PROTECTED_CONSTANT_COLS = {"cnpj_raiz"}  # chave nunca “remover”


def read_row_group_df(pq_file: pq.ParquetFile, rg_index: int, columns: list[str] | None = None) -> pd.DataFrame:
    table = pq_file.read_row_group(rg_index, columns=columns)
    return table.to_pandas()


def build_sample(pq_file: pq.ParquetFile, max_rows: int) -> pd.DataFrame:
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
    N = len(sample_df)
    rows = []

    for col in sample_df.columns:
        s = sample_df[col]
        n_null = int(s.isna().sum())
        pct_null = round((n_null / N * 100), 2) if N else 0.0
        nunique = int(s.nunique(dropna=True))
        is_constant = (nunique == 1)
        dtype_atual = str(s.dtype)

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

            # sugestão de tipo (estrutura, não modelagem)
            low_card = nunique <= 80
            if "data" in col.lower():
                tipo_destino = "datetime"
                justificativa = "variável temporal"
            elif dtype_atual in {"object", "string"} and low_card:
                tipo_destino = "category"
                justificativa = "baixa/média cardinalidade"
            elif col in {"idade_anos"}:
                tipo_destino = "float"
                justificativa = "métrica contínua"
            else:
                tipo_destino = "string"
                justificativa = "texto/identificador"

        rows.append(
            dict(
                coluna=col,
                tipo_atual=dtype_atual,
                tipo_sugerido=tipo_destino,
                acao_sugerida=acao,
                nulos_pct=pct_null,
                n_valores_distintos=nunique,
                constante=is_constant,
                justificativa=justificativa,
            )
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["acao_sugerida", "nulos_pct"], ascending=[True, False])
        .reset_index(drop=True)
    )


def key_checks(sample_df: pd.DataFrame) -> str:
    out = []
    out.append("CNPJ – Key checks (amostra)")
    out.append("")

    if "cnpj_raiz" not in sample_df.columns:
        out.append("[ERRO] coluna cnpj_raiz não existe no parquet.")
        return "\n".join(out)

    raiz = sample_df["cnpj_raiz"].astype("string")
    raiz_len_ok = raiz.str.len().fillna(0).eq(8).mean() * 100
    raiz_num_ok = raiz.str.fullmatch(r"\d{8}").fillna(False).mean() * 100
    out.append(f"- cnpj_raiz len==8 (%): {raiz_len_ok:.2f}")
    out.append(f"- cnpj_raiz só dígitos (%): {raiz_num_ok:.2f}")
    out.append(f"- nulos cnpj_raiz: {int(raiz.isna().sum())}")
    out.append("")

    if "cnpj" in sample_df.columns:
        cnpj = sample_df["cnpj"].astype("string")
        cnpj_len_ok = cnpj.str.len().fillna(0).eq(14).mean() * 100
        cnpj_num_ok = cnpj.str.fullmatch(r"\d{14}").fillna(False).mean() * 100
        out.append(f"- cnpj len==14 (%): {cnpj_len_ok:.2f}")
        out.append(f"- cnpj só dígitos (%): {cnpj_num_ok:.2f}")
        out.append(f"- nulos cnpj: {int(cnpj.isna().sum())}")
        out.append("")

    return "\n".join(out)


def domains_report(sample_df: pd.DataFrame) -> pd.DataFrame:
    # Ajuste aqui conforme as colunas que seu cnpj_core trouxe
    cols = [c for c in [
        "porte_empresa",
        "situacao_cadastral",
        "uf",
        "cnae_fiscal_principal",
    ] if c in sample_df.columns]

    rows = []
    for c in cols:
        vc = sample_df[c].astype("string").value_counts(dropna=False).head(30)
        for k, v in vc.items():
            rows.append({"coluna": c, "valor": str(k), "count": int(v)})

    return pd.DataFrame(rows)


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

    # schema
    schema_df = pd.DataFrame([{"coluna": name, "tipo_parquet": str(field.type)} for name, field in zip(schema.names, schema)])
    schema_path = REPORTS_DIR / f"cnpj_schema_{RUN_DATE}.csv"
    schema_df.to_csv(schema_path, index=False, encoding="utf-8")
    print(f"[OK] Schema salvo: {schema_path}")

    # head
    head_df = read_row_group_df(pq_file, 0).head(HEAD_ROWS) if num_row_groups else pd.DataFrame(columns=colnames)
    head_path = REPORTS_DIR / f"cnpj_head_eda_{RUN_DATE}.csv"
    head_df.to_csv(head_path, index=False, encoding="utf-8")
    print(f"[OK] Head salvo: {head_path}")

    # sample
    sample_df = build_sample(pq_file, MAX_SAMPLE_ROWS)
    print(f"[INFO] Amostra: {sample_df.shape[0]:,} linhas x {sample_df.shape[1]} colunas")

    # info
    buf = io.StringIO()
    sample_df.info(buf=buf)
    info_path = REPORTS_DIR / f"cnpj_info_{RUN_DATE}.txt"
    info_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"[OK] Info salvo: {info_path}")

    # nulos
    na_abs = sample_df.isna().sum()
    na_pct = (na_abs / len(sample_df) * 100).round(2) if len(sample_df) else na_abs.astype(float)
    na_tbl = pd.DataFrame({"nulos": na_abs, "pct": na_pct}).sort_values("nulos", ascending=False)
    na_path = REPORTS_DIR / f"cnpj_nulos_amostra_{RUN_DATE}.csv"
    na_tbl.to_csv(na_path, encoding="utf-8")
    print(f"[OK] Nulos salvo: {na_path}")

    # decisão
    decisao_df = decision_table(sample_df)
    decisao_path = REPORTS_DIR / f"cnpj_tabela_decisao_{RUN_DATE}.csv"
    decisao_df.to_csv(decisao_path, index=False, encoding="utf-8")
    print(f"[OK] Tabela de decisão salva: {decisao_path}")

    # domínios
    dom_df = domains_report(sample_df)
    dom_path = REPORTS_DIR / f"cnpj_domains_{RUN_DATE}.csv"
    dom_df.to_csv(dom_path, index=False, encoding="utf-8")
    print(f"[OK] Domains salvo: {dom_path}")

    # key checks
    key_txt = key_checks(sample_df)
    key_path = REPORTS_DIR / f"cnpj_keycheck_{RUN_DATE}.txt"
    key_path.write_text(key_txt, encoding="utf-8")
    print(f"[OK] Keycheck salvo: {key_path}")

    # summary
    summary = []
    summary.append("CNPJ – ETAPA 5 – Validação estrutural e diagnóstico inicial")
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
    summary.append(f"- {dom_path.name}")
    summary.append(f"- {key_path.name}")
    summary.append("")

    summary_path = REPORTS_DIR / f"cnpj_etapa5_summary_{RUN_DATE}.txt"
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    print(f"[OK] Summary salvo: {summary_path}")

    print("[DONE] CNPJ – validação concluída (sem carregar o dataset inteiro).")


if __name__ == "__main__":
    main()
