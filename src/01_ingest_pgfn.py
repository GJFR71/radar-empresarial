"""
01_ingest_pgfn.py
Ingestão em lote dos dados públicos da PGFN (Dívida Ativa Geral / Não Previdenciário)

Fonte de verdade dos períodos:
- config/pgfn_periodos.csv  (ano,trimestre,url_zip)

Saídas:
- data/raw/pgfn/  (ZIPs baixados; diretório ignorado no Git)
- reports/manifest_pgfn_YYYYMMDD.csv  (registro do que foi baixado/erros; versionável)

Observação:
- Este script só faz ingestão. Não extrai nem transforma.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests


# =========================
# Paths do projeto
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
REPORTS_DIR = PROJECT_ROOT / "reports"
RAW_PGFN_DIR = PROJECT_ROOT / "data" / "raw" / "pgfn"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_PGFN_DIR.mkdir(parents=True, exist_ok=True)

RUN_DATE = datetime.now().strftime("%Y%m%d")
RUN_TS = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =========================
# Fontes auxiliares
# =========================

DICIONARIO_CAMPOS_URL = (
    "https://www.gov.br/pgfn/pt-br/assuntos/divida-ativa-da-uniao/"
    "transparencia-fiscal-1/arquivos-dados-abertos/dicionario_de_campos.xlsx"
)


@dataclass(frozen=True)
class Periodo:
    ano: int
    trimestre: int
    url_zip: str


# =========================
# Utilitários
# =========================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest_path: Path, timeout: int = 300) -> tuple[int, str]:
    """
    Baixa arquivo via streaming. Retorna (status_code, final_url).
    Lança exceção apenas para erros de conexão/timeout; HTTP != 200 é tratado pelo status_code.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Data Science Challenge)"}

    with requests.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True) as r:
        status = r.status_code
        final_url = str(r.url)

        if status != 200:
            return status, final_url

        with dest_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return 200, final_url


def read_periodos_csv(path: Path) -> list[Periodo]:
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}. "
            "Crie e preencha config/pgfn_periodos.csv"
        )

    periodos: list[Periodo] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"ano", "trimestre", "url_zip"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV inválido. Esperado cabeçalho com: {sorted(required)}. "
                f"Encontrado: {reader.fieldnames}"
            )

        for row in reader:
            ano = int(row["ano"])
            tri = int(row["trimestre"])
            url = (row["url_zip"] or "").strip()
            if not url:
                continue
            periodos.append(Periodo(ano=ano, trimestre=tri, url_zip=url))

    # ordena para execução previsível
    periodos.sort(key=lambda p: (p.ano, p.trimestre))
    return periodos


# =========================
# Manifest (evidência)
# =========================

MANIFEST_HEADERS = [
    "run_date",
    "run_ts",
    "dataset",
    "ano",
    "trimestre",
    "url",
    "final_url",
    "output_path",
    "status",
    "http_status",
    "size_mb",
    "sha256_16",
    "message",
]


def append_manifest_row(manifest_path: Path, row: dict) -> None:
    new_file = not manifest_path.exists()
    with manifest_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADERS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


# =========================
# Pipeline
# =========================

def download_periodo_zip(periodo: Periodo, manifest_path: Path) -> None:
    dataset = "pgfn_nao_previdenciario"

    filename = f"{dataset}_ano={periodo.ano}_tri={periodo.trimestre}_{RUN_DATE}.zip"
    out_path = RAW_PGFN_DIR / filename

    # idempotência simples (se já existe, não baixa de novo)
    if out_path.exists():
        size_mb = out_path.stat().st_size / (1024 * 1024)
        h16 = sha256_file(out_path)[:16]
        append_manifest_row(
            manifest_path,
            dict(
                run_date=RUN_DATE,
                run_ts=RUN_TS,
                dataset=dataset,
                ano=periodo.ano,
                trimestre=periodo.trimestre,
                url=periodo.url_zip,
                final_url=periodo.url_zip,
                output_path=str(out_path),
                status="SKIPPED_EXISTS",
                http_status=200,
                size_mb=f"{size_mb:.2f}",
                sha256_16=h16,
                message="Arquivo já existia; download não executado.",
            ),
        )
        print(f"[INFO] {periodo.ano} T{periodo.trimestre}: já existe (pulando)")
        return

    print(f"[INFO] Baixando {periodo.ano} T{periodo.trimestre} ...")
    try:
        http_status, final_url = download_file(periodo.url_zip, out_path)
        if http_status != 200:
            # remove arquivo parcial, se existir
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            append_manifest_row(
                manifest_path,
                dict(
                    run_date=RUN_DATE,
                    run_ts=RUN_TS,
                    dataset=dataset,
                    ano=periodo.ano,
                    trimestre=periodo.trimestre,
                    url=periodo.url_zip,
                    final_url=final_url,
                    output_path="",
                    status="FAILED_HTTP",
                    http_status=http_status,
                    size_mb="",
                    sha256_16="",
                    message="HTTP diferente de 200 (provável período indisponível).",
                ),
            )
            print(f"[WARN] {periodo.ano} T{periodo.trimestre}: HTTP {http_status} (pulando)")
            return

        size_mb = out_path.stat().st_size / (1024 * 1024)
        h16 = sha256_file(out_path)[:16]
        append_manifest_row(
            manifest_path,
            dict(
                run_date=RUN_DATE,
                run_ts=RUN_TS,
                dataset=dataset,
                ano=periodo.ano,
                trimestre=periodo.trimestre,
                url=periodo.url_zip,
                final_url=final_url,
                output_path=str(out_path),
                status="OK",
                http_status=200,
                size_mb=f"{size_mb:.2f}",
                sha256_16=h16,
                message="Download concluído.",
            ),
        )
        print(f"[OK] {periodo.ano} T{periodo.trimestre}: {size_mb:.2f} MB")

    except requests.RequestException as e:
        # remove arquivo parcial, se existir
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        append_manifest_row(
            manifest_path,
            dict(
                run_date=RUN_DATE,
                run_ts=RUN_TS,
                dataset=dataset,
                ano=periodo.ano,
                trimestre=periodo.trimestre,
                url=periodo.url_zip,
                final_url="",
                output_path="",
                status="FAILED_REQUEST",
                http_status="",
                size_mb="",
                sha256_16="",
                message=f"Erro de rede/timeout: {type(e).__name__}",
            ),
        )
        print(f"[ERROR] {periodo.ano} T{periodo.trimestre}: erro de rede/timeout ({type(e).__name__})")


def download_dictionary(manifest_path: Path) -> None:
    dataset = "pgfn_dicionario_campos"
    filename = f"{dataset}_{RUN_DATE}.xlsx"
    out_path = RAW_PGFN_DIR / filename

    if out_path.exists():
        size_mb = out_path.stat().st_size / (1024 * 1024)
        h16 = sha256_file(out_path)[:16]
        append_manifest_row(
            manifest_path,
            dict(
                run_date=RUN_DATE,
                run_ts=RUN_TS,
                dataset=dataset,
                ano="",
                trimestre="",
                url=DICIONARIO_CAMPOS_URL,
                final_url=DICIONARIO_CAMPOS_URL,
                output_path=str(out_path),
                status="SKIPPED_EXISTS",
                http_status=200,
                size_mb=f"{size_mb:.2f}",
                sha256_16=h16,
                message="Dicionário já existia; download não executado.",
            ),
        )
        print("[INFO] Dicionário: já existe (pulando)")
        return

    print("[INFO] Baixando dicionário de campos ...")
    try:
        http_status, final_url = download_file(DICIONARIO_CAMPOS_URL, out_path, timeout=120)
        if http_status != 200:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            append_manifest_row(
                manifest_path,
                dict(
                    run_date=RUN_DATE,
                    run_ts=RUN_TS,
                    dataset=dataset,
                    ano="",
                    trimestre="",
                    url=DICIONARIO_CAMPOS_URL,
                    final_url=final_url,
                    output_path="",
                    status="FAILED_HTTP",
                    http_status=http_status,
                    size_mb="",
                    sha256_16="",
                    message="Falha ao baixar dicionário (HTTP != 200).",
                ),
            )
            print(f"[WARN] Dicionário: HTTP {http_status}")
            return

        size_mb = out_path.stat().st_size / (1024 * 1024)
        h16 = sha256_file(out_path)[:16]
        append_manifest_row(
            manifest_path,
            dict(
                run_date=RUN_DATE,
                run_ts=RUN_TS,
                dataset=dataset,
                ano="",
                trimestre="",
                url=DICIONARIO_CAMPOS_URL,
                final_url=final_url,
                output_path=str(out_path),
                status="OK",
                http_status=200,
                size_mb=f"{size_mb:.2f}",
                sha256_16=h16,
                message="Download do dicionário concluído.",
            ),
        )
        print(f"[OK] Dicionário: {size_mb:.2f} MB")

    except requests.RequestException as e:
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        append_manifest_row(
            manifest_path,
            dict(
                run_date=RUN_DATE,
                run_ts=RUN_TS,
                dataset=dataset,
                ano="",
                trimestre="",
                url=DICIONARIO_CAMPOS_URL,
                final_url="",
                output_path="",
                status="FAILED_REQUEST",
                http_status="",
                size_mb="",
                sha256_16="",
                message=f"Erro de rede/timeout: {type(e).__name__}",
            ),
        )
        print(f"[ERROR] Dicionário: erro de rede/timeout ({type(e).__name__})")


def run() -> None:
    periodos_path = CONFIG_DIR / "pgfn_periodos.csv"
    manifest_path = REPORTS_DIR / f"manifest_pgfn_{RUN_DATE}.csv"

    print(f"[INFO] Projeto: {PROJECT_ROOT}")
    print(f"[INFO] Config: {periodos_path}")
    print(f"[INFO] Raw PGFN: {RAW_PGFN_DIR}")
    print(f"[INFO] Manifest: {manifest_path}")
    print("-" * 70)

    periodos = read_periodos_csv(periodos_path)
    print(f"[INFO] Períodos carregados do CSV: {len(periodos)}")
    print("-" * 70)

    # dicionário 1x por execução
    download_dictionary(manifest_path)
    print("-" * 70)

    for p in periodos:
        download_periodo_zip(p, manifest_path)

    print("-" * 70)
    print("[DONE] Ingestão em lote PGFN concluída.")


if __name__ == "__main__":
    run()
