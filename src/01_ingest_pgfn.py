"""
01_ingest_pgfn.py
Ingestão de dados públicos da PGFN (Dívida Ativa)

Objetivo:
- Baixar arquivos públicos da PGFN e salvar em data/raw
- Não transformar dados nesta etapa (apenas ingestão + versionamento)
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests


# =========================
# Configurações
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

TODAY = datetime.now().strftime("%Y%m%d")


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    # sugira extensão: "csv", "zip", "xlsx" etc. (opcional)
    ext: Optional[str] = None


# TODO: Ajuste as URLs reais da PGFN.
# Dica: se a PGFN publicar mais de um arquivo (por UF, por período, etc.),
# cadastre vários itens aqui.
SOURCES = [
    Source(
        name="pgfn_sida_nao_previdenciario_2025_t3",
        url="https://dadosabertos.pgfn.gov.br/2025_trimestre_03/Dados_abertos_Nao_Previdenciario.zip",
        ext="zip",
    ),
    Source(
        name="pgfn_dicionario_campos",
        url="https://www.gov.br/pgfn/pt-br/assuntos/divida-ativa-da-uniao/transparencia-fiscal-1/arquivos-dados-abertos/dicionario_de_campos.xlsx",
        ext="xlsx",
    ),
]



# =========================
# Utilitários
# =========================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def guess_extension_from_headers(response: requests.Response) -> Optional[str]:
    ctype = (response.headers.get("Content-Type") or "").lower()
    # heurística simples
    if "text/csv" in ctype:
        return "csv"
    if "application/zip" in ctype:
        return "zip"
    if "application/json" in ctype:
        return "json"
    if "excel" in ctype or "spreadsheet" in ctype:
        return "xlsx"
    return None


def download_file(url: str, dest_path: Path, timeout: int = 120) -> None:
    headers = {"User-Agent": "Mozilla/5.0 (Data Science Challenge)"}
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
        r.raise_for_status()
        with dest_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


# =========================
# Pipeline de ingestão
# =========================

def run() -> None:
    print(f"[INFO] Projeto: {PROJECT_ROOT}")
    print(f"[INFO] Pasta raw: {DATA_RAW}")
    print(f"[INFO] Data de execução: {TODAY}")
    print("-" * 60)

    for src in SOURCES:
        if "COLE_AQUI" in src.url or not src.url.strip():
            raise ValueError(
                f"URL não configurada para a fonte '{src.name}'. "
                f"Atualize o bloco SOURCES no script."
            )

        print(f"[INFO] Baixando: {src.name}")
        print(f"[INFO] URL: {src.url}")

        # Faz um HEAD simples para tentar inferir extensão (se permitido pelo servidor)
        ext = src.ext
        try:
            head = requests.head(src.url, allow_redirects=True, timeout=30)
            if head.ok and ext is None:
                guessed = guess_extension_from_headers(head)
                ext = guessed or ext
        except Exception:
            # se HEAD falhar, seguimos sem travar
            pass

        # Nome final do arquivo: <fonte>_<YYYYMMDD>.<ext>
        if ext is None:
            # fallback: salva sem extensão
            filename = f"{src.name}_{TODAY}"
        else:
            filename = f"{src.name}_{TODAY}.{ext}"

        out_path = DATA_RAW / filename

        # Evita rebaixar se arquivo já existir (idempotência simples)
        if out_path.exists():
            print(f"[INFO] Já existe: {out_path.name} (pulando download)")
            continue

        download_file(src.url, out_path)

        size_mb = out_path.stat().st_size / (1024 * 1024)
        file_hash = sha256_file(out_path)

        print(f"[OK] Salvo em: {out_path}")
        print(f"[OK] Tamanho: {size_mb:.2f} MB")
        print(f"[OK] SHA256: {file_hash[:16]}...")

        print("-" * 60)

    print("[DONE] Ingestão PGFN concluída.")


if __name__ == "__main__":
    run()
