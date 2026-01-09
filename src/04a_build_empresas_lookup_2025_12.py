from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMP_DIR = PROJECT_ROOT / "data" / "staging" / "cnpj" / "2025-12" / "empresas"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "cnpj" / "empresas_lookup_2025_12.parquet"

def list_files(d: Path) -> list[Path]:
    return sorted([p for p in d.glob("*") if p.is_file()])

def build_lookup(chunksize: int = 200_000) -> None:
    files = list_files(EMP_DIR)
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo em {EMP_DIR}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    written = 0

    names = [
        "cnpj_basico",
        "razao_social",
        "natureza_juridica",
        "qualif_resp",
        "capital_social",
        "porte_empresa",
        "ente_federativo",
    ]
    usecols = [0, 1, 5]

    try:
        for k, f in enumerate(files, start=1):
            print(f"[READ] ({k}/{len(files)}) {f.name}", flush=True)
            reader = pd.read_csv(
                f,
                sep=";",
                header=None,
                names=names,
                usecols=usecols,
                dtype=str,
                encoding="latin1",
                chunksize=chunksize,
                engine="c",
                low_memory=False,
                on_bad_lines="skip",
            )

            for chunk in reader:
                chunk["cnpj_basico"] = (
                    chunk["cnpj_basico"].astype(str)
                    .str.replace(r"\D", "", regex=True)
                    .str.zfill(8)
                )

                out = chunk[["cnpj_basico", "razao_social", "porte_empresa"]]
                table = pa.Table.from_pandas(out, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(str(OUT_PATH), table.schema, compression="snappy")
                    print("[INFO] writer criado", flush=True)

                writer.write_table(table)
                written += table.num_rows

        print(f"[DONE] lookup bruto escrito: {written:,} linhas -> {OUT_PATH}", flush=True)

    finally:
        if writer is not None:
            writer.close()
            print("[INFO] writer fechado", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()
    build_lookup(chunksize=args.chunksize)

if __name__ == "__main__":
    main()
