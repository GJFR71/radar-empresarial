from __future__ import annotations

"""
06_build_abt_pgfn_2024_2025.py

ETAPA 6 — Build da ABT (PGFN 2024–2025)

Objetivo:
- Transformar a base consolidada (Parquet) em uma Base Analítica Tratada (ABT),
  pronta para análises agregadas por empresa (CNPJ) e para modelagem/insights.

Princípios:
- Processamento em streaming por batches (RecordBatches) usando PyArrow Dataset.
- Evita conversões para pandas durante o pipeline de transformação/escrita.
- Normalizações mínimas e rastreáveis (chaves, datas e domínios).
- Gera artefatos em reports/:
  - tabela de decisões (dicionário → ABT)
  - schema final
  - head (amostra)
  - summary

Decisões do projeto (fechadas):
- ABT será restrita a PESSOAS JURÍDICAS (PJ) para aderência ao objetivo por empresa (CNPJ).
- tipo_devedor: categórica com 3 níveis (PRINCIPAL/CORRESPONSAVEL/SOLIDARIO) via dictionary encoding.
- ano e trimestre: dimensões discretas (mantidas e dictionary encoding).
- uf: NÃO entra na ABT.
- uf_devedor: NÃO entra na ABT.
- situacao_inscricao, receita_principal: categóricas (dictionary encoding).
- tipo_situacao_inscricao: normalizada em 5 níveis + dictionary encoding.
- Chave de merge com CNPJ: cria colunas cnpj (14 dígitos) e cnpj_raiz (8 dígitos).
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


# -----------------------------
# Paths / Config
# -----------------------------
@dataclass(frozen=True)
class Paths:
    project_dir: Path
    parquet_in: Path
    parquet_out: Path
    reports_dir: Path


def now_run_date() -> str:
    return datetime.now().strftime("%Y%m%d")


def build_paths(project_dir: Path) -> Paths:
    project_dir = project_dir.resolve()
    parquet_in = project_dir / "data" / "processed" / "pgfn" / "pgfn_sida_2024_2025.parquet"
    parquet_out = project_dir / "data" / "processed" / "pgfn" / "pgfn_abt_2024_2025.parquet"
    reports_dir = project_dir / "reports"
    return Paths(project_dir, parquet_in, parquet_out, reports_dir)


def ensure_dirs(paths: Paths) -> None:
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    paths.parquet_out.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helpers (Arrow)
# -----------------------------
def _to_string(arr: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
        return arr
    return pc.cast(arr, pa.string())


def _trim_and_nullify_empty(arr: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    s = _to_string(arr)
    s = pc.utf8_trim_whitespace(s)
    is_empty = pc.equal(s, "")
    return pc.if_else(is_empty, pa.scalar(None, pa.string()), s)


def _dict_encode(arr: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return pc.dictionary_encode(arr)


def _map_sim_nao_to_int(arr: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    s = _trim_and_nullify_empty(arr)
    s_up = pc.utf8_upper(s)
    is_sim = pc.equal(s_up, "SIM")
    is_nao = pc.equal(s_up, "NAO")
    return pc.if_else(
        is_sim,
        pa.scalar(1, pa.int8()),
        pc.if_else(is_nao, pa.scalar(0, pa.int8()), pa.scalar(None, pa.int8())),
    )


def _parse_ddmmyyyy(arr: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    # Se já vier timestamp no parquet, mantém
    if pa.types.is_timestamp(arr.type):
        return arr
    s = _trim_and_nullify_empty(arr)
    return pc.strptime(s, format="%d/%m/%Y", unit="ms", error_is_null=True)


def _normalize_tipo_situacao_inscricao(arr: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """
    Normaliza tipo_situacao_inscricao para 5 níveis:
    irregular, beneficio_fiscal, negociacao, suspenso_judicial, garantia
    """
    s = _trim_and_nullify_empty(arr)
    s_low = pc.utf8_lower(s)

    def has(sub: str):
        # compatível com versões antigas: padrão como string "pura"
        return pc.match_substring(s_low, sub)

    out = pa.scalar("outros", pa.string())
    out = pc.if_else(has("cobran"), "irregular", out)
    out = pc.if_else(has("benef"), "beneficio_fiscal", out)
    out = pc.if_else(has("negocia"), "negociacao", out)
    out = pc.if_else(has("suspens"), "suspenso_judicial", out)
    out = pc.if_else(has("garant"), "garantia", out)

    return pc.cast(out, pa.string())


def _digits_only(arr: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """
    Remove tudo que não for dígito.
    """
    s = _trim_and_nullify_empty(arr)
    # regex_replace remove não-dígitos
    return pc.replace_substring_regex(s, pattern=r"[^0-9]", replacement="")


def _cnpj_14(arr: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """
    Produz CNPJ limpo (14 dígitos). Se não tiver 14, vira null.
    """
    d = _digits_only(arr)
    ln = pc.utf8_length(d)
    ok = pc.equal(ln, 14)
    return pc.if_else(ok, d, pa.scalar(None, pa.string()))


def _cnpj_raiz_8(cnpj14: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """
    Raiz do CNPJ (8 dígitos). Se CNPJ for null, raiz será null.
    """
    return pc.utf8_slice_codeunits(cnpj14, start=0, stop=8)


# -----------------------------
# ABT spec (colunas e tipos)
# -----------------------------
# ABT final é por empresa (PJ): tipo_pessoa vira apenas filtro e NÃO entra
ABT_KEEP_COLS = [
    "cnpj",
    "cnpj_raiz",
    "tipo_devedor",
    "numero_inscricao",
    "situacao_inscricao",
    "tipo_situacao_inscricao",
    "receita_principal",
    "data_inscricao",
    "indicador_ajuizado",
    "valor_consolidado",
    "unidade_responsavel",
    "ano",
    "trimestre",
    # opcionais consultáveis (pode tirar se quiser enxugar mais)
    "nome_devedor",
]

ABT_DROP_COLS = {
    "uf",
    "uf_devedor",
    "arquivo_origem",
    "uf_unidade_responsavel",
    # tipo_pessoa será usado para filtro, mas não entra na ABT final
}

CAT_DICT_COLS = {
    "tipo_devedor",
    "tipo_situacao_inscricao",
    "situacao_inscricao",
    "receita_principal",
    "ano",
    "trimestre",
}

STRING_COLS = {
    "numero_inscricao",
    "unidade_responsavel",
    "nome_devedor",
}


# -----------------------------
# Reports (decisão ABT)
# -----------------------------
def write_decision_table(paths: Paths, run_date: str) -> Path:
    rows = [
        {
            "elemento": "filtro_tipo_pessoa",
            "decisao": "Manter apenas Pessoa jurídica",
            "justificativa": "Escopo do desafio é análise por empresa (CNPJ). Reduz custo e evita ruído de PF.",
        },
        {"coluna": "cnpj", "entra_na_abt": "SIM", "tipo_final": "string", "transformacao": "digits_only + valida 14", "justificativa": "chave p/ merge com base CNPJ"},
        {"coluna": "cnpj_raiz", "entra_na_abt": "SIM", "tipo_final": "string", "transformacao": "slice 0:8 do cnpj", "justificativa": "agregação por grupo econômico/raiz"},
        {"coluna": "tipo_devedor", "entra_na_abt": "SIM", "tipo_final": "category(dict)", "transformacao": "trim + dict_encode", "justificativa": "preserva semântica jurídica (3 níveis)"},
        {"coluna": "numero_inscricao", "entra_na_abt": "SIM", "tipo_final": "string", "transformacao": "trim", "justificativa": "identificador da inscrição"},
        {"coluna": "situacao_inscricao", "entra_na_abt": "SIM", "tipo_final": "category(dict)", "transformacao": "trim + dict_encode", "justificativa": "variável central de status"},
        {"coluna": "tipo_situacao_inscricao", "entra_na_abt": "SIM", "tipo_final": "category(dict)", "transformacao": "normaliza 5 níveis + dict_encode", "justificativa": "domínio enxuto p/ análise"},
        {"coluna": "receita_principal", "entra_na_abt": "SIM", "tipo_final": "category(dict)", "transformacao": "trim + dict_encode", "justificativa": "segmentação econômica"},
        {"coluna": "data_inscricao", "entra_na_abt": "SIM", "tipo_final": "timestamp(ms)", "transformacao": "strptime dd/mm/yyyy", "justificativa": "dimensão temporal do crédito"},
        {"coluna": "indicador_ajuizado", "entra_na_abt": "SIM", "tipo_final": "int8 (0/1)", "transformacao": "SIM/NAO->1/0", "justificativa": "indicador de judicialização"},
        {"coluna": "valor_consolidado", "entra_na_abt": "SIM", "tipo_final": "float64", "transformacao": "cast float64", "justificativa": "métrica principal"},
        {"coluna": "unidade_responsavel", "entra_na_abt": "SIM", "tipo_final": "string", "transformacao": "trim", "justificativa": "segmentação operacional"},
        {"coluna": "ano", "entra_na_abt": "SIM", "tipo_final": "category(dict)", "transformacao": "cast int32 + dict_encode", "justificativa": "dimensão discreta"},
        {"coluna": "trimestre", "entra_na_abt": "SIM", "tipo_final": "category(dict)", "transformacao": "cast int32 + dict_encode", "justificativa": "dimensão discreta"},
        {"coluna": "uf", "entra_na_abt": "NAO", "tipo_final": "-", "transformacao": "drop", "justificativa": "baixa utilidade no recorte"},
        {"coluna": "uf_devedor", "entra_na_abt": "NAO", "tipo_final": "-", "transformacao": "drop", "justificativa": "não essencial para objetivo por CNPJ"},
        {"coluna": "arquivo_origem", "entra_na_abt": "NAO", "tipo_final": "-", "transformacao": "drop", "justificativa": "rastreio técnico, não ABT"},
        {"coluna": "tipo_pessoa", "entra_na_abt": "NAO", "tipo_final": "-", "transformacao": "usada apenas como filtro", "justificativa": "ABT é PJ-only; variável seria constante"},
    ]

    df = pd.DataFrame(rows)
    out = paths.reports_dir / f"pgfn_abt_dict_decisions_{run_date}.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"[OK] Tabela de decisões ABT salva: {out}")
    return out


def write_reports(paths: Paths, run_date: str) -> None:
    head_path = paths.reports_dir / f"pgfn_abt_head_{run_date}.csv"
    schema_path = paths.reports_dir / f"pgfn_abt_schema_{run_date}.csv"
    summary_path = paths.reports_dir / f"pgfn_etapa6_summary_{run_date}.txt"

    pf = pq.ParquetFile(paths.parquet_out)
    total_rows = pf.metadata.num_rows
    num_row_groups = pf.num_row_groups
    schema = pf.schema_arrow

    if num_row_groups > 0:
        rg0 = pf.read_row_group(0)
        head_df = rg0.to_pandas().head(50)
    else:
        head_df = pd.DataFrame()

    head_df.to_csv(head_path, index=False, encoding="utf-8")

    schema_rows = [{"coluna": f.name, "tipo": str(f.type)} for f in schema]
    pd.DataFrame(schema_rows).to_csv(schema_path, index=False, encoding="utf-8")

    lines = [
        "[ETAPA 6] Build ABT PGFN (2024–2025) — PJ-only",
        f"- Execução: {run_date}",
        f"- Input:  {paths.parquet_in.as_posix()}",
        f"- Output: {paths.parquet_out.as_posix()}",
        f"- Linhas (metadata): {total_rows:,}",
        f"- Colunas: {len(schema.names)}",
        f"- Row groups: {num_row_groups}",
        f"- Head: {head_path.as_posix()}",
        f"- Schema: {schema_path.as_posix()}",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Head salvo: {head_path}")
    print(f"[OK] Schema salvo: {schema_path}")
    print(f"[OK] Summary salvo: {summary_path}")


# -----------------------------
# Core build
# -----------------------------
def process_batch(rb: pa.RecordBatch) -> Optional[pa.Table]:
    """
    Aplica regras da ABT em um RecordBatch e devolve uma Table pronta para escrita.
    Retorna None se o batch ficar vazio após o filtro PJ.
    """
    tbl = pa.Table.from_batches([rb])

    existing = set(tbl.schema.names)

    # 0) Filtro PJ (antes de todo o resto)
    if "tipo_pessoa" in existing:
        tp = _trim_and_nullify_empty(tbl["tipo_pessoa"])
        mask_pj = pc.equal(tp, "Pessoa jurídica")
        tbl = tbl.filter(mask_pj)

        if tbl.num_rows == 0:
            return None
    else:
        # Se não existir tipo_pessoa, não dá para garantir PJ-only
        # Melhor falhar do que gerar ABT incoerente.
        raise RuntimeError("Coluna 'tipo_pessoa' não encontrada no parquet de entrada. Não é possível filtrar PJ-only.")

    # 1) Cria chave CNPJ limpa (a partir de cpf_cnpj)
    if "cpf_cnpj" not in existing:
        raise RuntimeError("Coluna 'cpf_cnpj' não encontrada. Não é possível derivar 'cnpj'.")

    cnpj = _cnpj_14(tbl["cpf_cnpj"])
    cnpj_raiz = _cnpj_raiz_8(cnpj)

    # 2) Seleciona colunas necessárias (depois do filtro)
    wanted = set(ABT_KEEP_COLS) | {"cpf_cnpj"}  # cpf_cnpj só para derivar cnpj (não sai na ABT)
    keep = [c for c in tbl.schema.names if c in wanted and c not in ABT_DROP_COLS]
    tbl = tbl.select(keep)

    # 3) Normalizações / typing
    cols = {}

    # colunas derivadas primeiro
    cols["cnpj"] = cnpj
    cols["cnpj_raiz"] = cnpj_raiz

    for c in tbl.schema.names:
        if c == "cpf_cnpj":
            continue  # não entra na ABT final

        arr = tbl[c]

        if c in STRING_COLS:
            cols[c] = _trim_and_nullify_empty(arr)

        elif c == "data_inscricao":
            cols[c] = _parse_ddmmyyyy(arr)

        elif c == "indicador_ajuizado":
            cols[c] = _map_sim_nao_to_int(arr)

        elif c == "valor_consolidado":
            cols[c] = pc.cast(arr, pa.float64())

        elif c == "tipo_situacao_inscricao":
            norm = _normalize_tipo_situacao_inscricao(arr)
            cols[c] = _dict_encode(norm)

        elif c in CAT_DICT_COLS:
            if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
                arr2 = _trim_and_nullify_empty(arr)
            else:
                if c in {"ano", "trimestre"}:
                    arr2 = pc.cast(arr, pa.int32())
                else:
                    arr2 = arr
            cols[c] = _dict_encode(arr2)

        else:
            cols[c] = _trim_and_nullify_empty(arr)

    out_tbl = pa.table(cols)

    # 4) Ordem final garantida
    final_cols = [c for c in ABT_KEEP_COLS if c in out_tbl.schema.names]
    out_tbl = out_tbl.select(final_cols)

    return out_tbl


def build_abt(paths: Paths, run_date: str, batch_size: int, force: bool) -> None:
    if not paths.parquet_in.exists():
        raise SystemExit(f"[ERRO] Input não encontrado: {paths.parquet_in}")

    if paths.parquet_out.exists():
        if not force:
            raise SystemExit(f"[ERRO] Output já existe: {paths.parquet_out} (use --force)")
        paths.parquet_out.unlink()
        print(f"[INFO] Removido output anterior: {paths.parquet_out}")

    print(f"[INFO] Input:  {paths.parquet_in}")
    print(f"[INFO] Output: {paths.parquet_out}")
    print(f"[INFO] Execução: {run_date}")
    print(f"[INFO] Batch size: {batch_size:,}")

    dataset = ds.dataset(paths.parquet_in, format="parquet")

    # lê apenas o necessário + tipo_pessoa + cpf_cnpj (para filtro e chave)
    required_in = set(ABT_KEEP_COLS) | {"tipo_pessoa", "cpf_cnpj"} | {
        "tipo_situacao_inscricao",  # pode estar em keep já, mas garantimos
    }

    cols_in = [c for c in dataset.schema.names if c in required_in and c not in ABT_DROP_COLS]

    # garante que tipo_pessoa e cpf_cnpj vieram
    if "tipo_pessoa" not in cols_in:
        cols_in.append("tipo_pessoa")
    if "cpf_cnpj" not in cols_in:
        cols_in.append("cpf_cnpj")

    scanner = dataset.scanner(columns=cols_in, batch_size=batch_size)

    writer: Optional[pq.ParquetWriter] = None
    rows_written = 0
    batches = 0
    batches_skipped = 0

    for rb in scanner.to_batches():
        batches += 1
        out_tbl = process_batch(rb)

        if out_tbl is None:
            batches_skipped += 1
            continue

        if writer is None:
            writer = pq.ParquetWriter(
                where=str(paths.parquet_out),
                schema=out_tbl.schema,
                compression="snappy",
                use_dictionary=True,
            )

        writer.write_table(out_tbl)
        rows_written += out_tbl.num_rows

        if batches % 50 == 0:
            print(f"[INFO] Batches: {batches} | Linhas escritas: {rows_written:,} | Batches vazios: {batches_skipped}")

    if writer is not None:
        writer.close()

    print(f"[OK] ABT Parquet gerado: {paths.parquet_out}")
    print(f"[OK] Linhas escritas: {rows_written:,}")
    print(f"[OK] Batches vazios (após filtro PJ): {batches_skipped}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETAPA 6 - Build ABT PGFN (2024–2025) — PJ-only")
    p.add_argument("--project-dir", default=".", help="Diretório raiz do projeto")
    p.add_argument("--batch-size", type=int, default=25_000, help="Tamanho do batch (RecordBatch) no scanner")
    p.add_argument("--force", action="store_true", help="Sobrescreve o parquet de saída se existir")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_paths(Path(args.project_dir))
    ensure_dirs(paths)

    run_date = now_run_date()

    # 1) Tabela de decisões (artefato rastreável)
    write_decision_table(paths, run_date)

    # 2) Build ABT
    build_abt(paths, run_date, batch_size=args.batch_size, force=args.force)

    # 3) Reports pós-escrita
    write_reports(paths, run_date)

    print("[DONE] ETAPA 6 concluída.")


if __name__ == "__main__":
    main()
