from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMP_LOOKUP = PROJECT_ROOT / "data" / "processed" / "cnpj" / "empresas_lookup_2025_12.parquet"
EST_CORE   = PROJECT_ROOT / "data" / "processed" / "cnpj" / "estabelecimentos_core_2025_12.parquet"

OUT_DIR    = PROJECT_ROOT / "data" / "processed" / "cnpj"
OUT_PATH   = OUT_DIR / "cnpj_core_2025_12.parquet"
TMP_PATH   = OUT_DIR / "tmp_cnpj_core_2025_12.parquet"

REPORTS    = PROJECT_ROOT / "reports"
RUN_DATE   = datetime.now().strftime("%Y%m%d")


def main() -> None:
    print(f"[INFO] Project: {PROJECT_ROOT}", flush=True)
    print(f"[INFO] EMP_LOOKUP: {EMP_LOOKUP}", flush=True)
    print(f"[INFO] EST_CORE:   {EST_CORE}", flush=True)
    print(f"[INFO] OUT:        {OUT_PATH}", flush=True)

    if not EMP_LOOKUP.exists():
        raise FileNotFoundError(f"Não encontrado: {EMP_LOOKUP}")
    if not EST_CORE.exists():
        raise FileNotFoundError(f"Não encontrado: {EST_CORE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # remove saídas quebradas
    if OUT_PATH.exists():
        OUT_PATH.unlink()
    if TMP_PATH.exists():
        TMP_PATH.unlink()

    con = duckdb.connect(database=":memory:")

    # Ajustes úteis (sem prometer milagre, mas ajuda)
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA enable_progress_bar=false;")  # vamos logar manualmente

    # TEMP em disco do projeto (melhor que temp do Windows em alguns casos)
    tmpdir = str((PROJECT_ROOT / "data" / "tmp_duckdb").resolve())
    os.makedirs(tmpdir, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{tmpdir}';")

    print("[STEP] Registrando fontes parquet...", flush=True)
    con.execute(f"CREATE VIEW est AS SELECT * FROM read_parquet('{EST_CORE.as_posix()}');")
    con.execute(f"CREATE VIEW emp_raw AS SELECT * FROM read_parquet('{EMP_LOOKUP.as_posix()}');")

    print("[STEP] Deduplicando empresas (cnpj_basico)...", flush=True)
    # Mantém 1 linha por cnpj_basico (qualquer uma). Para nosso join é suficiente.
    con.execute("""
        CREATE TABLE emp AS
        SELECT
            cnpj_basico,
            any_value(razao_social) AS razao_social,
            any_value(porte_empresa) AS porte_empresa
        FROM emp_raw
        WHERE cnpj_basico IS NOT NULL AND length(cnpj_basico) = 8
        GROUP BY cnpj_basico
    """)

    emp_cnt = con.execute("SELECT COUNT(*) FROM emp;").fetchone()[0]
    est_cnt = con.execute("SELECT COUNT(*) FROM est;").fetchone()[0]
    print(f"[INFO] emp (dedup): {emp_cnt:,} linhas", flush=True)
    print(f"[INFO] est:         {est_cnt:,} linhas", flush=True)

    print("[STEP] Join est + emp -> TMP parquet...", flush=True)
    # Seleção final enxuta (ajuste se quiser mais campos)
    con.execute(f"""
        COPY (
            SELECT
                est.cnpj,
                est.cnpj_raiz,
                emp.razao_social,
                emp.porte_empresa,
                est.situacao_cadastral,
                est.data_situacao_cadastral,
                est.cnae_fiscal_principal,
                est.uf,
                est.municipio,
                est.data_inicio_atividade
            FROM est
            LEFT JOIN emp
                ON est.cnpj_raiz = emp.cnpj_basico
        )
        TO '{TMP_PATH.as_posix()}'
        (FORMAT PARQUET, COMPRESSION 'SNAPPY');
    """)

    # move TMP -> OUT
    if not TMP_PATH.exists() or TMP_PATH.stat().st_size == 0:
        raise RuntimeError("Falha: parquet TMP não foi gerado (0 bytes).")

    TMP_PATH.replace(OUT_PATH)
    print(f"[DONE] cnpj_core gerado: {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)", flush=True)

    # reports leves
    print("[STEP] Reports...", flush=True)
    head_path = REPORTS / f"cnpj_core_head_{RUN_DATE}.csv"
    schema_path = REPORTS / f"cnpj_core_schema_{RUN_DATE}.csv"

    con.execute(f"""
        COPY (SELECT * FROM read_parquet('{OUT_PATH.as_posix()}') LIMIT 20)
        TO '{head_path.as_posix()}'
        (HEADER, DELIMITER ',', QUOTE '"');
    """)

    # schema simples via DESCRIBE
    schema_df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{OUT_PATH.as_posix()}');").fetchdf()
    schema_df.to_csv(schema_path, index=False, encoding="utf-8")

    print(f"[DONE] reports: {head_path.name}, {schema_path.name}", flush=True)


if __name__ == "__main__":
    main()
