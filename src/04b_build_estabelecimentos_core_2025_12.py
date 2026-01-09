from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EST_DIR = PROJECT_ROOT / "data" / "staging" / "cnpj" / "2025-12" / "estabelecimentos"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "cnpj" / "estabelecimentos_core_2025_12.parquet"

# Layout RFB (sem header; ';'; latin1)
EST_COLS = [
    "cnpj_basico","cnpj_ordem","cnpj_dv","identificador_matriz_filial","nome_fantasia",
    "situacao_cadastral","data_situacao_cadastral","motivo_situacao_cadastral","nome_cidade_exterior",
    "pais","data_inicio_atividade","cnae_fiscal_principal","cnae_fiscal_secundaria",
    "tipo_logradouro","logradouro","numero","complemento","bairro","cep","uf","municipio",
    "ddd1","telefone1","ddd2","telefone2","ddd_fax","fax","correio_eletronico",
    "situacao_especial","data_situacao_especial"
]

def list_files(d: Path) -> list[Path]:
    return sorted([p for p in d.glob("*") if p.is_file()])

def build_core(chunksize: int = 200_000) -> None:
    files = list_files(EST_DIR)
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo em {EST_DIR}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    written_rows = 0

    try:
        for k, f in enumerate(files, start=1):
            print(f"[STREAM] ({k}/{len(files)}) {f.name}", flush=True)

            reader = pd.read_csv(
                f,
                sep=";",
                header=None,
                names=EST_COLS,
                dtype=str,
                encoding="latin1",
                chunksize=chunksize,
                engine="c",
                low_memory=False,
                on_bad_lines="skip",
            )

            for i, chunk in enumerate(reader, start=1):
                # chaves
                chunk["cnpj_basico"] = chunk["cnpj_basico"].str.replace(r"\D","",regex=True).str.zfill(8)
                chunk["cnpj_ordem"]  = chunk["cnpj_ordem"].str.replace(r"\D","",regex=True).str.zfill(4)
                chunk["cnpj_dv"]     = chunk["cnpj_dv"].str.replace(r"\D","",regex=True).str.zfill(2)

                out = pd.DataFrame({
                    "cnpj": (chunk["cnpj_basico"] + chunk["cnpj_ordem"] + chunk["cnpj_dv"]).astype(str),
                    "cnpj_raiz": chunk["cnpj_basico"],
                    "situacao_cadastral": chunk["situacao_cadastral"],
                    "data_situacao_cadastral": pd.to_datetime(chunk["data_situacao_cadastral"], errors="coerce", format="%Y%m%d"),
                    "cnae_fiscal_principal": chunk["cnae_fiscal_principal"],
                    "uf": chunk["uf"],
                    "municipio": chunk["municipio"],
                    "data_inicio_atividade": pd.to_datetime(chunk["data_inicio_atividade"], errors="coerce", format="%Y%m%d"),
                })

                table = pa.Table.from_pandas(out, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(str(OUT_PATH), table.schema, compression="snappy")
                    print("[INFO] writer criado", flush=True)

                writer.write_table(table)
                written_rows += table.num_rows

                if i % 20 == 0:
                    print(f"  [INFO] chunks: {i} | linhas gravadas: {written_rows:,}", flush=True)

        print(f"[DONE] estabelecimentos_core: {written_rows:,} linhas -> {OUT_PATH}", flush=True)

    finally:
        if writer is not None:
            writer.close()
            print("[INFO] writer fechado", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()
    build_core(chunksize=args.chunksize)

if __name__ == "__main__":
    main()
