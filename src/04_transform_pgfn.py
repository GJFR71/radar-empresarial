"""
04_transform_pgfn.py
Transformação dos dados públicos da PGFN (SIDA Não Previdenciário) da área de staging
para um único arquivo parquet em data/processed.

Entrada:
- data/staging/pgfn/ano=YYYY/trimestre=T/*.csv

Saídas:
- data/processed/pgfn/pgfn_sida.parquet (único)
- reports/pgfn_head_YYYYMMDD.csv (preview das primeiras linhas)
- reports/pgfn_profile_YYYYMMDD.txt (perfil descritivo básico)

Observação:
- Processa em streaming (chunks) para não estourar memória.
- A análise descritiva é feita em amostra (por padrão 100k linhas),
  e o total de linhas vem do metadata do parquet.
- Encoding escolhido por "taxa de sucesso" em uma amostra (ex.: 20 arquivos),
  para evitar que erros pontuais derrubem todo o processo.
- Schema do Parquet é estabilizado com união de colunas (headers) + dtype=str
  e conversão explícita de valor_consolidado para float.
- Metadados de partição: ano e trimestre são persistidos como int32 (facilita KPIs).
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.errors import ParserError


# =========================
# Paths do projeto
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGING_DIR = PROJECT_ROOT / "data" / "staging" / "pgfn"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "pgfn"
REPORTS_DIR = PROJECT_ROOT / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_DATE = datetime.now().strftime("%Y%m%d")
RUN_TS = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =========================
# Utilitários
# =========================

def _slugify_col(name: str) -> str:
    """Converte nome de coluna para snake_case simples."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = s.strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_{2,}", "_", s).strip("_")
    return s


def list_csv_files(staging_root: Path) -> list[Path]:
    # Esperado: .../ano=YYYY/trimestre=T/*.csv
    return sorted(staging_root.rglob("*.csv"))


def parse_partitions_from_path(csv_path: Path) -> tuple[int, int]:
    """Extrai ano e trimestre do path estilo: .../ano=2020/trimestre=1/arquivo.csv"""
    ano: int | None = None
    tri: int | None = None

    for p in csv_path.parts:
        if p.startswith("ano="):
            ano = int(p.split("=", 1)[1])
        elif p.startswith("trimestre="):
            tri = int(p.split("=", 1)[1])

    if ano is None or tri is None:
        raise ValueError(f"Não consegui extrair ano/trimestre do path: {csv_path}")

    return ano, tri


_UF_RE = re.compile(r"_SIDA_([A-Z]{2})_", re.IGNORECASE)


def parse_uf_from_filename(csv_path: Path) -> str:
    """A partir do padrão observado: arquivo_lai_SIDA_SP_202002.csv -> UF=SP"""
    m = _UF_RE.search(csv_path.name)
    if not m:
        return "NA"
    return m.group(1).upper()


def detect_separator(path: Path) -> str:
    """Heurística simples para identificar separador (; vs ,)."""
    raw = path.read_bytes()[:4096]
    text = raw.decode("latin-1", errors="ignore")
    return ";" if text.count(";") > text.count(",") else ","


def choose_working_encoding(
    csv_files: list[Path],
    encodings: list[str],
    sample_files: int = 20,
    min_success_rate: float = 0.80,
) -> str:
    """
    Escolhe encoding por taxa de sucesso em uma amostra.
    - testa até `sample_files` arquivos (nrows=1)
    - conta sucesso quando NÃO ocorre UnicodeDecodeError
    - escolhe encoding com maior sucesso
    - exige taxa mínima; se ninguém atingir, usa latin-1
    """
    test_files = csv_files[: min(sample_files, len(csv_files))]
    if not test_files:
        return "latin-1"

    n = len(test_files)
    results: dict[str, int] = {}

    for enc in encodings:
        ok = 0
        for p in test_files:
            sep = detect_separator(p)
            try:
                pd.read_csv(
                    p,
                    sep=sep,
                    encoding=enc,
                    nrows=1,
                    dtype=str,
                    low_memory=False,
                )
                ok += 1
            except UnicodeDecodeError:
                pass
            except Exception:
                # ParserError/linhas ruins não são evidência contra o encoding
                ok += 1

        results[enc] = ok
        print(f"[INFO] Probe encoding={enc}: ok={ok}/{n} ({ok/n:.0%})")

    best_enc, best_ok = max(results.items(), key=lambda kv: kv[1])

    if best_ok / n >= min_success_rate:
        print(f"[INFO] Encoding padrão escolhido: {best_enc} (sucesso {best_ok}/{n})")
        return best_enc

    print(
        f"[WARN] Nenhum encoding atingiu taxa mínima ({min_success_rate:.0%}). "
        f"Melhor foi {best_enc} com {best_ok}/{n}. Usando latin-1 (fallback)."
    )
    return "latin-1"


def iter_csv_chunks(
    path: Path,
    chunksize: int,
    primary_encoding: str,
    fallback_encodings: list[str],
    sep: str,
) -> Iterator[pd.DataFrame]:
    """
    Lê CSV em chunks.
    - Tenta primeiro `primary_encoding`
    - Se falhar (UnicodeDecodeError), tenta fallback_encodings
    - Último recurso: engine=python + latin-1 + on_bad_lines=skip
    Força dtype=str para estabilizar schema do parquet.
    """
    encodings_to_try = [primary_encoding] + [e for e in fallback_encodings if e != primary_encoding]
    last_exc: Exception | None = None

    for enc in encodings_to_try:
        try:
            reader = pd.read_csv(
                path,
                sep=sep,
                encoding=enc,
                low_memory=False,
                chunksize=chunksize,
                dtype=str,
            )
            for chunk in reader:
                yield chunk
            return
        except UnicodeDecodeError as e:
            last_exc = e
            continue
        except Exception as e:
            last_exc = e
            continue

    # Último recurso: parser python
    try:
        reader = pd.read_csv(
            path,
            sep=sep,
            encoding="latin-1",
            engine="python",
            low_memory=False,
            chunksize=chunksize,
            dtype=str,
            on_bad_lines="skip",
        )
        for chunk in reader:
            yield chunk
        return
    except (UnicodeDecodeError, ParserError) as e:
        raise RuntimeError(f"Falha total ao ler CSV {path.name} (encoding/parser).") from e
    except Exception as e:
        raise RuntimeError(f"Falha total ao ler CSV {path.name}.") from (last_exc or e)


def normalize_chunk_types(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizações mínimas.
    - valor_consolidado -> float (normaliza milhar/decimal BR)
    """
    if "valor_consolidado" in chunk.columns:
        s = chunk["valor_consolidado"].astype("string").str.strip()
        s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})

        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        chunk["valor_consolidado"] = pd.to_numeric(s, errors="coerce")

    return chunk


def union_schema_columns(
    csv_files: list[Path],
    primary_encoding: str,
    fallback_encodings: list[str],
) -> list[str]:
    """
    Lê só o cabeçalho (nrows=0) para obter a união das colunas (já slugificadas).
    Isso fixa um schema estável para o Parquet.
    """
    cols: list[str] = []
    seen = set()

    for p in csv_files:
        sep = detect_separator(p)
        encs = [primary_encoding] + [e for e in fallback_encodings if e != primary_encoding]
        df0 = None

        for enc in encs:
            try:
                df0 = pd.read_csv(p, sep=sep, encoding=enc, nrows=0, dtype=str, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
            except Exception:
                # tenta parser mais tolerante só pro header
                try:
                    df0 = pd.read_csv(
                        p,
                        sep=sep,
                        encoding=enc,
                        nrows=0,
                        dtype=str,
                        engine="python",
                        on_bad_lines="skip",
                    )
                    break
                except Exception:
                    continue

        if df0 is None:
            continue

        for c in df0.columns:
            c2 = _slugify_col(c)
            if c2 and c2 not in seen:
                cols.append(c2)
                seen.add(c2)

    return cols


def ensure_columns(df: pd.DataFrame, all_cols: list[str]) -> pd.DataFrame:
    """Garante DF com exatamente as colunas do schema (ordem fixa)."""
    for c in all_cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[all_cols]


# =========================
# Escrita parquet (streaming)
# =========================

@dataclass(frozen=True)
class TransformConfig:
    chunksize: int = 200_000
    sample_rows: int = 100_000
    force: bool = False
    encoding_probe_files: int = 20
    encoding_min_success_rate: float = 0.80


def transform_to_parquet(cfg: TransformConfig) -> Path:
    out_path = PROCESSED_DIR / "pgfn_sida.parquet"
    tmp_path = PROCESSED_DIR / "pgfn_sida.parquet.tmp"

    if out_path.exists() and not cfg.force:
        print(f"[INFO] Parquet já existe: {out_path} (use --force para recriar)")
        return out_path

    csv_files = list_csv_files(STAGING_DIR)
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em: {STAGING_DIR}")

    print(f"[INFO] Projeto: {PROJECT_ROOT}")
    print(f"[INFO] Staging: {STAGING_DIR}")
    print(f"[INFO] Output: {out_path}")
    print(f"[INFO] CSVs encontrados: {len(csv_files)}")
    print("-" * 70)

    # se recriar, apaga o final; tmp sempre recomeça
    if out_path.exists() and cfg.force:
        out_path.unlink(missing_ok=True)
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)

    primary_encoding = choose_working_encoding(
        csv_files=csv_files,
        encodings=["utf-8", "cp1252", "latin-1"],
        sample_files=cfg.encoding_probe_files,
        min_success_rate=cfg.encoding_min_success_rate,
    )
    fallback_encodings = ["cp1252", "latin-1", "utf-8"]

    # schema fixo: união de colunas + metadados
    base_cols = union_schema_columns(csv_files, primary_encoding, fallback_encodings)
    meta_cols = ["ano", "trimestre", "uf", "arquivo_origem"]

    all_cols = base_cols.copy()
    for c in meta_cols:
        if c not in all_cols:
            all_cols.append(c)

    # schema arrow: string para tudo, exceto:
    # - valor_consolidado: float64
    # - ano/trimestre: int32
    fields = []
    for c in all_cols:
        if c == "valor_consolidado":
            fields.append(pa.field(c, pa.float64()))
        elif c in ("ano", "trimestre"):
            fields.append(pa.field(c, pa.int32()))
        else:
            fields.append(pa.field(c, pa.string()))
    arrow_schema = pa.schema(fields)

    print(f"[INFO] Schema estabilizado com {len(all_cols)} colunas (união de headers + metadados).")
    print(f"[INFO] Escrevendo em arquivo temporário: {tmp_path}")

    writer: pq.ParquetWriter | None = None
    total_rows_written = 0

    sample_accum: list[pd.DataFrame] = []
    sample_count = 0

    try:
        for i, csv_path in enumerate(csv_files, start=1):
            ano, tri = parse_partitions_from_path(csv_path)
            uf = parse_uf_from_filename(csv_path)
            sep = detect_separator(csv_path)

            print(
                f"[INFO] ({i}/{len(csv_files)}) Processando: "
                f"ano={ano} tri={tri} uf={uf} sep='{sep}' file={csv_path.name}"
            )

            for chunk in iter_csv_chunks(
                path=csv_path,
                chunksize=cfg.chunksize,
                primary_encoding=primary_encoding,
                fallback_encodings=fallback_encodings,
                sep=sep,
            ):
                # slug das colunas vindas do CSV
                chunk.columns = [_slugify_col(c) for c in chunk.columns]

                # metadados (tipos já compatíveis com schema)
                chunk["ano"] = int(ano)
                chunk["trimestre"] = int(tri)
                chunk["uf"] = str(uf)
                chunk["arquivo_origem"] = str(csv_path.name)

                # normalizações
                chunk = normalize_chunk_types(chunk)

                # schema estável
                chunk = ensure_columns(chunk, all_cols)

                # garante compatibilidade (pandas às vezes mantém object)
                # ano/trimestre: int32 (nullable não é necessário aqui, sempre preenchido)
                chunk["ano"] = pd.to_numeric(chunk["ano"], errors="coerce").astype("Int32").astype("int32")
                chunk["trimestre"] = pd.to_numeric(chunk["trimestre"], errors="coerce").astype("Int32").astype("int32")

                table = pa.Table.from_pandas(chunk, schema=arrow_schema, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(tmp_path, arrow_schema, compression="snappy")

                writer.write_table(table)
                total_rows_written += len(chunk)

                if sample_count < cfg.sample_rows:
                    missing = cfg.sample_rows - sample_count
                    take_n = min(len(chunk), missing)
                    sample_accum.append(chunk.head(take_n))
                    sample_count += take_n

        print("-" * 70)

    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        # em erro, remove apenas o tmp (não destrói um parquet final válido antigo)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    finally:
        if writer is not None:
            writer.close()

    # promove para o nome final
    tmp_path.replace(out_path)

    print(f"[OK] Parquet gerado: {out_path}")
    print(f"[OK] Linhas escritas (contagem do processo): {total_rows_written:,}")

    sample_df = pd.concat(sample_accum, ignore_index=True) if sample_accum else pd.DataFrame()
    build_reports(out_path, sample_df, cfg)

    return out_path


# =========================
# Relatórios (head + perfil)
# =========================

def build_reports(parquet_path: Path, sample_df: pd.DataFrame, cfg: TransformConfig) -> None:
    head_path = REPORTS_DIR / f"pgfn_head_{RUN_DATE}.csv"
    profile_path = REPORTS_DIR / f"pgfn_profile_{RUN_DATE}.txt"

    pq_file = pq.ParquetFile(parquet_path)
    total_rows = pq_file.metadata.num_rows
    num_row_groups = pq_file.num_row_groups
    schema = pq_file.schema_arrow

    # HEAD
    if num_row_groups > 0:
        rg0 = pq_file.read_row_group(0)
        head_df = rg0.to_pandas().head(20)
    else:
        head_df = pd.DataFrame()

    head_df.to_csv(head_path, index=False, encoding="utf-8")

    # Perfil (amostra)
    lines: list[str] = []
    lines.append("PGFN - Perfil Descritivo Básico")
    lines.append(f"Execução: {RUN_TS}")
    lines.append("")
    lines.append("Resumo do Parquet")
    lines.append(f"- arquivo: {parquet_path}")
    lines.append(f"- linhas (parquet metadata): {total_rows:,}")
    lines.append(f"- colunas: {len(schema.names)}")
    lines.append(f"- row_groups: {num_row_groups}")
    lines.append("")
    lines.append("Colunas (schema)")
    lines.append(", ".join(schema.names))
    lines.append("")

    if sample_df.empty:
        lines.append("Amostra: vazia (não foi possível gerar estatísticas descritivas).")
    else:
        lines.append("Amostra usada para estatísticas")
        lines.append(f"- linhas na amostra: {len(sample_df):,} (limite configurado: {cfg.sample_rows:,})")
        lines.append("")

        lines.append("Tipos (amostra)")
        lines.append(sample_df.dtypes.astype(str).to_string())
        lines.append("")

        null_counts = sample_df.isna().sum().sort_values(ascending=False)
        null_pct = (null_counts / len(sample_df) * 100).round(2)
        null_top = pd.DataFrame({"nulos": null_counts, "pct": null_pct}).head(30)
        lines.append("Nulos (top 30 colunas na amostra)")
        lines.append(null_top.to_string())
        lines.append("")

        num_cols = sample_df.select_dtypes(include=["number"]).columns.tolist()
        if num_cols:
            desc_num = sample_df[num_cols].describe().transpose()
            lines.append("Describe numérico (amostra)")
            lines.append(desc_num.to_string())
            lines.append("")
        else:
            lines.append("Describe numérico: não há colunas numéricas detectadas na amostra.")
            lines.append("")

        for col in ["ano", "trimestre", "uf"]:
            if col in sample_df.columns:
                vc = sample_df[col].value_counts(dropna=False).head(30)
                lines.append(f"Distribuição (amostra) - {col}")
                lines.append(vc.to_string())
                lines.append("")

    profile_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Head salvo: {head_path}")
    print(f"[OK] Perfil salvo: {profile_path}")


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--chunksize", type=int, default=200_000, help="Linhas por chunk ao ler cada CSV")
    p.add_argument("--sample-rows", type=int, default=100_000, help="Linhas máximas para a amostra do profiling")
    p.add_argument("--force", action="store_true", help="Recria o parquet mesmo se já existir")
    p.add_argument("--encoding-probe-files", type=int, default=20, help="Arquivos para probe de encoding")
    p.add_argument(
        "--encoding-min-success-rate",
        type=float,
        default=0.80,
        help="Taxa mínima (0-1) para aceitar o melhor encoding na amostra",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TransformConfig(
        chunksize=args.chunksize,
        sample_rows=args.sample_rows,
        force=args.force,
        encoding_probe_files=args.encoding_probe_files,
        encoding_min_success_rate=args.encoding_min_success_rate,
    )

    print(f"[INFO] Projeto: {PROJECT_ROOT}")
    print(f"[INFO] Staging: {STAGING_DIR}")
    print(f"[INFO] Processed: {PROCESSED_DIR}")
    print(f"[INFO] Reports: {REPORTS_DIR}")
    print("-" * 70)

    transform_to_parquet(cfg)
    print("[DONE] Transformação PGFN concluída.")


if __name__ == "__main__":
    main()

