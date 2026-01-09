"""
02_ingest_cnpj.py
Ingestão (download) dos Dados Abertos do CNPJ — snapshot 2025-12 (ou qualquer outro),
controlado por config/cnpj_arquivos.csv.

Entrada:
- config/cnpj_arquivos.csv com colunas: grupo,nome,url

Saídas:
- data/raw/cnpj/<nome>.zip
- reports/manifest_cnpj_YYYYMMDD.csv

Características:
- Idempotente: se o arquivo já existe, pula (SKIPPED_EXISTS)
- Manifest: registra OK/FAILED/SKIPPED com sha256 parcial e tamanho
- Retry HTTP (429/5xx) e backoff
- Não extrai nem transforma (isso é etapa seguinte)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================
# Paths do projeto
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
REPORTS_DIR = PROJECT_ROOT / "reports"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cnpj"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

RUN_DATE = datetime.now().strftime("%Y%m%d")
RUN_TS = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True)
class Recurso:
    grupo: str
    nome: str
    url: str


# =========================
# Sessão HTTP com retry
# =========================

def build_session() -> requests.Session:
    retries = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)

    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


SESSION = build_session()


# =========================
# Utilitários
# =========================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_recursos_csv(path: Path) -> list[Recurso]:
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            f"Crie o CSV com colunas: grupo,nome,url em {CONFIG_DIR / 'cnpj_arquivos.csv'}"
        )

    recursos: list[Recurso] = []
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"grupo", "nome", "url"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV inválido. Esperado cabeçalho com: {sorted(required)}. "
                f"Encontrado: {reader.fieldnames}"
            )

        for row in reader:
            grupo = (row.get("grupo") or "").strip()
            nome = (row.get("nome") or "").strip()
            url = (row.get("url") or "").strip()
            if not (grupo and nome and url):
                continue
            recursos.append(Recurso(grupo=grupo, nome=nome, url=url))

    recursos.sort(key=lambda r: (r.grupo.lower(), r.nome.lower()))
    return recursos


def download_file(url: str, dest_path: Path, timeout: int = 900) -> tuple[int, str]:
    headers = {"User-Agent": "Mozilla/5.0 (PGFN-CNPJ Data Challenge)"}

    # 3 tentativas "externas" (além dos retries do adapter)
    for attempt in range(1, 4):
        try:
            with SESSION.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True) as r:
                status = r.status_code
                final_url = str(r.url)

                if status != 200:
                    return status, final_url

                tmp = dest_path.with_suffix(dest_path.suffix + ".tmp")
                if tmp.exists():
                    tmp.unlink(missing_ok=True)

                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

                tmp.replace(dest_path)
                return 200, final_url

        except requests.RequestException:
            # limpa parcial
            tmp = dest_path.with_suffix(dest_path.suffix + ".tmp")
            if tmp.exists():
                tmp.unlink(missing_ok=True)

            if attempt < 3:
                time.sleep(5 * attempt)
            else:
                raise

    return 0, url


# =========================
# Manifest
# =========================

MANIFEST_HEADERS = [
    "run_date",
    "run_ts",
    "dataset",
    "grupo",
    "nome",
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


def ingest_one(recurso: Recurso, manifest_path: Path, timeout: int) -> None:
    dataset = "cnpj_dados_abertos_snapshot"

    out_path = RAW_DIR / recurso.nome

    if out_path.exists():
        size_mb = out_path.stat().st_size / (1024 * 1024)
        h16 = sha256_file(out_path)[:16]
        append_manifest_row(
            manifest_path,
            dict(
                run_date=RUN_DATE,
                run_ts=RUN_TS,
                dataset=dataset,
                grupo=recurso.grupo,
                nome=recurso.nome,
                url=recurso.url,
                final_url=recurso.url,
                output_path=str(out_path),
                status="SKIPPED_EXISTS",
                http_status=200,
                size_mb=f"{size_mb:.2f}",
                sha256_16=h16,
                message="Arquivo já existia; download não executado.",
            ),
        )
        print(f"[SKIP] {recurso.nome} (já existe)")
        return

    print(f"[DOWN] {recurso.nome} ({recurso.grupo})")
    try:
        http_status, final_url = download_file(recurso.url, out_path, timeout=timeout)

        if http_status != 200:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            append_manifest_row(
                manifest_path,
                dict(
                    run_date=RUN_DATE,
                    run_ts=RUN_TS,
                    dataset=dataset,
                    grupo=recurso.grupo,
                    nome=recurso.nome,
                    url=recurso.url,
                    final_url=final_url,
                    output_path="",
                    status="FAILED_HTTP",
                    http_status=http_status,
                    size_mb="",
                    sha256_16="",
                    message="HTTP diferente de 200 (arquivo indisponível ou URL/nome divergente).",
                ),
            )
            print(f"[WARN] {recurso.nome}: HTTP {http_status}")
            return

        size_mb = out_path.stat().st_size / (1024 * 1024)
        h16 = sha256_file(out_path)[:16]
        append_manifest_row(
            manifest_path,
            dict(
                run_date=RUN_DATE,
                run_ts=RUN_TS,
                dataset=dataset,
                grupo=recurso.grupo,
                nome=recurso.nome,
                url=recurso.url,
                final_url=final_url,
                output_path=str(out_path),
                status="OK",
                http_status=200,
                size_mb=f"{size_mb:.2f}",
                sha256_16=h16,
                message="Download concluído.",
            ),
        )
        print(f"[OK] {recurso.nome}: {size_mb:.2f} MB")

    except requests.RequestException as e:
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        append_manifest_row(
            manifest_path,
            dict(
                run_date=RUN_DATE,
                run_ts=RUN_TS,
                dataset=dataset,
                grupo=recurso.grupo,
                nome=recurso.nome,
                url=recurso.url,
                final_url="",
                output_path="",
                status="FAILED_REQUEST",
                http_status="",
                size_mb="",
                sha256_16="",
                message=f"Erro de rede/timeout: {type(e).__name__}",
            ),
        )
        print(f"[ERR] {recurso.nome}: {type(e).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(CONFIG_DIR / "cnpj_arquivos.csv"),
        help="Caminho do CSV (grupo,nome,url). Padrão: config/cnpj_arquivos.csv",
    )
    parser.add_argument(
        "--grupo",
        default="empresas,estabelecimentos,aux",
        help="Quais grupos baixar (csv). Ex.: empresas,estabelecimentos (padrão: empresas,estabelecimentos,aux)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Timeout por request (segundos). Padrão: 900",
    )
    args = parser.parse_args()

    grupos = {g.strip().lower() for g in args.grupo.split(",") if g.strip()}
    cfg_path = Path(args.config)
    manifest_path = REPORTS_DIR / f"manifest_cnpj_{RUN_DATE}.csv"

    print(f"[INFO] Projeto:   {PROJECT_ROOT}")
    print(f"[INFO] Config:    {cfg_path}")
    print(f"[INFO] Raw dir:   {RAW_DIR}")
    print(f"[INFO] Manifest:  {manifest_path}")
    print(f"[INFO] Grupos:    {sorted(grupos)}")
    print("-" * 70)

    recursos = [r for r in read_recursos_csv(cfg_path) if r.grupo.lower() in grupos]
    print(f"[INFO] Recursos selecionados: {len(recursos)}")
    print("-" * 70)

    for r in recursos:
        ingest_one(r, manifest_path, timeout=args.timeout)

    print("-" * 70)
    print("[DONE] Ingestão CNPJ concluída.")


if __name__ == "__main__":
    main()
