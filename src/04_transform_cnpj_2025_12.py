"""
03_transform_cnpj_2025_12.py
Transforma Dados Abertos do CNPJ (snapshot 2025-12) em um parquet enxuto (cnpj_core).

Entradas (extraídas):
- data/staging/cnpj/2025-12/empresas/*.*
- data/staging/cnpj/2025-12/estabelecimentos/*.*

Saídas:
- data/processed/cnpj/cnpj_core_2025_12.parquet
- reports/cnpj_core_head_YYYYMMDD.csv
- reports/cnpj_core_schema_YYYYMMDD.csv

Notas:
- Layout CNPJ RFB: arquivos sem header, separados por ';', encoding latin1.
- Mantemos apenas colunas-chave para join e análise.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMP_DIR = PROJECT_ROOT / "data" / "staging" / "cnpj" / "2025-12" / "empresas"
EST_DIR = PROJECT_ROOT / "data" / "staging" / "cnpj" / "2025-12" / "estabelecimentos"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "cnpj" / "cnpj_core_2025_12.parquet"
REPORTS = PROJECT_ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

RUN_DATE = datetime.now().strftime("%Y%m%d")


# Layout (RFB) — nomes mínimos usados aqui
EMP_COLS = [
    "cnpj_basico", "razao_social", "natureza_juridica", "qualif_resp",
    "capital_social", "porte_empresa", "ente_federativo_resp"
]

EST_COLS = [
    "cnpj_basico", "cnpj_ordem", "cnpj_dv", "matriz_filial", "nome_fantasia",
    "situacao_cadastral", "data_situacao_cadastral", "motivo_situacao",
    "nome_cidade_exterior", "pais", "data_inicio_atividade",
    "cnae_fiscal_principal", "cnae_fiscal_secundaria",
    "tipo_logradouro", "logradouro", "numero", "complemento", "bairro",
    "cep", "uf", "municipio",
    "ddd1", "telefone1", "ddd2", "telefone2", "ddd_fax", "fax",
    "email", "situacao_especial", "data_situacao_especial"
]

def list_data_files(folder: Path) -> list[Path]:
    # nos zips do CNPJ, os extraídos podem vir como .CSV/.csv/.TXT etc.
    files = [p for p in folder.iterdir() if p.is_file()]
    # ignore arquivos pequenos “de controle” (raros), priorize grandes
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return files


def read_empresas_map(path: Path, chunksize: int = 500_000) -> pd.DataFrame:
    """
    Lê Empresas*.EMPRECSV em streaming e retorna mapa enxuto por cnpj_basico
    (razao_social, porte_empresa). Evita estouro de memória.
    """
    usecols = [0, 1, 5]  # cnpj_basico, razao_social, porte_empresa
    names = [
        "cnpj_basico",
        "razao_social",
        "natureza_juridica",
        "qualif_resp",
        "capital_social",
        "porte_empresa",
        "ente_federativo",
    ]

    parts = []

    reader = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=names,
        usecols=usecols,
        dtype=str,
        encoding="latin1",
        chunksize=chunksize,
        engine="python",        # engine tolerante
        on_bad_lines="skip",    # ignora linhas ruins
    )

    for chunk in reader:
        chunk["cnpj_basico"] = (
            chunk["cnpj_basico"]
            .str.replace(r"\D", "", regex=True)
            .str.zfill(8)
        )
        parts.append(chunk[["cnpj_basico", "razao_social", "porte_empresa"]])

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates("cnpj_basico")

    return df


def transform(chunksize: int = 500_000) -> None:
    emp_files = list_data_files(EMP_DIR)
    est_files = list_data_files(EST_DIR)

    if not emp_files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em: {EMP_DIR}")
    if not est_files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em: {EST_DIR}")

    print(f"[INFO] Empresas files: {len(emp_files)}")
    print(f"[INFO] Estabelecimentos files: {len(est_files)}")
    print(f"[INFO] Output: {OUT_PATH}")

    emp = []
    for idx, f in enumerate(emp_files, start=1):
        print(f"[READ] Empresas ({idx}/{len(emp_files)}): {f.name}", flush=True)
        try:
            df = read_empresas_map(f, chunksize=chunksize)
        except Exception as e:
            print(f"[ERR] Falha lendo Empresas {f.name}: {type(e).__name__}: {e}", flush=True)
            raise

        if df is None:
            raise ValueError(f"[ERR] read_empresas_map retornou None em {f.name}")

        if df.empty:
            raise ValueError(f"[ERR] DataFrame vazio em {f.name}")

        emp.append(df)

    print("[INFO] Concat empresas...", flush=True)
    emp_df = pd.concat(emp, ignore_index=True).drop_duplicates("cnpj_basico")

    print(f"[INFO] Empresas distinct cnpj_basico: {len(emp_df):,}", flush=True)
    print("[INFO] Iniciando streaming de Estabelecimentos...", flush=True)

      
    # Writer parquet em streaming
    writer: pq.ParquetWriter | None = None
    written_chunks = 0
    written_rows = 0
    
    try:
        print("[INFO] Iniciando streaming de Estabelecimentos...")    
        for f in est_files:
            print(f"[STREAM] Estab: {f.name}")

            reader = pd.read_csv(
                f,
                sep=";",
                header=None,
                names=EST_COLS,
                dtype=str,
                encoding="latin1",
                low_memory=False,
                chunksize=chunksize,
            )

            for i, chunk in enumerate(reader, start=1):
                # DEBUG: inspeciona o 1º chunk de cada arquivo
                if i == 1:
                    print(f"  [DEBUG] chunk shape: {chunk.shape}")
                    print(f"  [DEBUG] cols (first 12): {list(chunk.columns)[:12]} | total={len(chunk.columns)}")
                    # se o layout estiver errado, aqui já aparece

                # chaves
                chunk["cnpj_basico"] = (
                    chunk["cnpj_basico"].astype("string")
                    .str.replace(r"\D", "", regex=True)
                    .str.zfill(8)
                )
                chunk["cnpj_ordem"] = (
                    chunk["cnpj_ordem"].astype("string")
                    .str.replace(r"\D", "", regex=True)
                    .str.zfill(4)
                )
                chunk["cnpj_dv"] = (
                    chunk["cnpj_dv"].astype("string")
                    .str.replace(r"\D", "", regex=True)
                    .str.zfill(2)
                )

                chunk["cnpj"] = (chunk["cnpj_basico"] + chunk["cnpj_ordem"] + chunk["cnpj_dv"]).astype("string")
                chunk["cnpj_raiz"] = chunk["cnpj_basico"]

                # join leve com Empresas (traz razão social/porte)
                out = chunk.merge(emp_df, on="cnpj_basico", how="left")

                # DEBUG: taxa de match no primeiro chunk
                if i == 1 and "razao_social" in out.columns:
                    match_pct = out["razao_social"].notna().mean() * 100
                    print(f"  [DEBUG] match Empresas (razao_social notna) no 1º chunk: {match_pct:.2f}%")

                # parse datas (se falhar vira NaT)
                out["data_inicio_atividade"] = pd.to_datetime(
                    out["data_inicio_atividade"], errors="coerce", format="%Y%m%d"
                )
                out["data_situacao_cadastral"] = pd.to_datetime(
                    out["data_situacao_cadastral"], errors="coerce", format="%Y%m%d"
                )

                # seleciona colunas finais (enxuto)
                out = out[
                    [
                        "cnpj",
                        "cnpj_raiz",
                        "razao_social",
                        "porte_empresa",
                        "situacao_cadastral",
                        "data_situacao_cadastral",
                        "cnae_fiscal_principal",
                        "uf",
                        "municipio",
                        "data_inicio_atividade",
                    ]
                ]

                # se por algum motivo vier vazio, pula
                if out.shape[0] == 0:
                    if i == 1:
                        print("  [WARN] 1º chunk resultou vazio após processamento (verificar parse/layout).")
                    continue

                table = pa.Table.from_pandas(out, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(str(OUT_PATH), table.schema, compression="snappy")
                    print(f"[INFO] writer criado. cols={len(table.schema.names)}")

                writer.write_table(table)
                written_chunks += 1
                written_rows += table.num_rows

                if i % 10 == 0:
                    print(f"  [INFO] chunks lidos: {i} | chunks gravados total: {written_chunks} | linhas gravadas: {written_rows:,}")

    finally:
        if writer is not None:
            writer.close()
            print("[INFO] writer fechado com sucesso")

    # Se nada foi escrito, falha com mensagem útil
    if written_chunks == 0:
        raise RuntimeError(
            "Nenhum chunk foi gravado no parquet. "
            "Causas prováveis: parse/layout incorreto (sep/encoding/colunas), "
            "ou problema em EST_COLS (qtd/ordem). Veja logs [DEBUG] do 1º chunk."
        )

    # Confirma arquivo no disco antes de reports
    if not OUT_PATH.exists():
        raise FileNotFoundError(f"Writer finalizou, mas o arquivo não existe: {OUT_PATH}")

    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"[INFO] Parquet criado: {OUT_PATH} ({size_mb:.2f} MB)")
    print(f"[INFO] Total gravado: {written_rows:,} linhas em {written_chunks} chunks")

    # reports leves
    head_df = pq.read_table(OUT_PATH).to_pandas().head(20)
    head_path = REPORTS / f"cnpj_core_head_{RUN_DATE}.csv"
    head_df.to_csv(head_path, index=False, encoding="utf-8")
    print(f"[OK] Report head: {head_path}")

    schema_path = REPORTS / f"cnpj_core_schema_{RUN_DATE}.csv"
    schema = pq.ParquetFile(OUT_PATH).schema_arrow
    pd.DataFrame(
        {"field": schema.names, "type": [str(schema.field(i).type) for i in range(len(schema.names))]}
    ).to_csv(schema_path, index=False, encoding="utf-8")
    print(f"[OK] Report schema: {schema_path}")

    print(f"[DONE] cnpj_core gerado: {OUT_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunksize", type=int, default=500_000)
    args = ap.parse_args()
    transform(chunksize=args.chunksize)


if __name__ == "__main__":
    main()
