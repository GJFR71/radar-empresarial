"""
03_extract_pgfn.py
Extração (descompactação) dos ZIPs da PGFN (Não Previdenciário).

Entrada:
- data/raw/pgfn/pgfn_nao_previdenciario_ano=YYYY_tri=T.zip

Saídas:
- data/staging/pgfn/ano=YYYY/trimestre=T/  (conteúdo descompactado)
- reports/manifest_extract_pgfn_YYYYMMDD.csv (registro do que foi extraído/erros)

Observação:
- Este script só extrai. Não transforma.
"""

from __future__ import annotations

import csv
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# =========================
# Paths do projeto
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PGFN_DIR = PROJECT_ROOT / "data" / "raw" / "pgfn"
STAGING_DIR = PROJECT_ROOT / "data" / "staging" / "pgfn"
REPORTS_DIR = PROJECT_ROOT / "reports"

STAGING_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_DATE = datetime.now().strftime("%Y%m%d")
RUN_TS = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =========================
# Modelos / regex
# =========================

ZIP_PATTERN = re.compile(r"pgfn_nao_previdenciario_ano=(\d{4})_tri=(\d)\.zip$", re.I)


@dataclass(frozen=True)
class PeriodoZip:
    ano: int
    trimestre: int
    zip_path: Path


# =========================
# Manifest
# =========================

MANIFEST_HEADERS = [
    "run_date",
    "run_ts",
    "ano",
    "trimestre",
    "zip_path",
    "output_dir",
    "status",
    "files_extracted",
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
# Utilitários
# =========================

def list_periodo_zips(raw_dir: Path) -> list[PeriodoZip]:
    zips: list[PeriodoZip] = []
    for p in raw_dir.glob("pgfn_nao_previdenciario_ano=*_tri=*.zip"):
        m = ZIP_PATTERN.search(p.name)
        if not m:
            continue
        ano = int(m.group(1))
        tri = int(m.group(2))
        zips.append(PeriodoZip(ano=ano, trimestre=tri, zip_path=p))

    zips.sort(key=lambda x: (x.ano, x.trimestre))
    return zips


def output_dir_for(ano: int, trimestre: int) -> Path:
    return STAGING_DIR / f"ano={ano}" / f"trimestre={trimestre}"


def dir_has_content(d: Path) -> bool:
    if not d.exists():
        return False
    # Se tem ao menos 1 arquivo dentro (recursivo), consideramos extraído
    return any(x.is_file() for x in d.rglob("*"))


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> int:
    """
    Extrai ZIP para dest_dir com validação simples contra Zip Slip.
    Retorna quantidade de arquivos extraídos.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # ignora diretórios
            if member.is_dir():
                continue

            member_path = Path(member.filename)

            # normaliza e bloqueia caminhos suspeitos
            if member_path.is_absolute() or ".." in member_path.parts:
                continue

            target_path = dest_dir / member_path
            target_path.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(member, "r") as src, target_path.open("wb") as dst:
                dst.write(src.read())

            extracted += 1

    return extracted


# =========================
# Pipeline
# =========================

def extract_one(pz: PeriodoZip, manifest_path: Path) -> None:
    out_dir = output_dir_for(pz.ano, pz.trimestre)

    # idempotência: se já extraímos e tem conteúdo, pula
    if dir_has_content(out_dir):
        append_manifest_row(
            manifest_path,
            dict(
                run_date=RUN_DATE,
                run_ts=RUN_TS,
                ano=pz.ano,
                trimestre=pz.trimestre,
                zip_path=str(pz.zip_path),
                output_dir=str(out_dir),
                status="SKIPPED_EXISTS",
                files_extracted="",
                message="Diretório já tinha conteúdo; extração não executada.",
            ),
        )
        print(f"[INFO] {pz.ano} T{pz.trimestre}: staging já existe (pulando)")
        return

    print(f"[INFO] Extraindo {pz.ano} T{pz.trimestre} ...")
    try:
        n = safe_extract_zip(pz.zip_path, out_dir)
        append_manifest_row(
            manifest_path,
            dict(
                run_date=RUN_DATE,
                run_ts=RUN_TS,
                ano=pz.ano,
                trimestre=pz.trimestre,
                zip_path=str(pz.zip_path),
                output_dir=str(out_dir),
                status="OK",
                files_extracted=n,
                message="Extração concluída.",
            ),
        )
        print(f"[OK] {pz.ano} T{pz.trimestre}: {n} arquivos extraídos")

    except zipfile.BadZipFile:
        append_manifest_row(
            manifest_path,
            dict(
                run_date=RUN_DATE,
                run_ts=RUN_TS,
                ano=pz.ano,
                trimestre=pz.trimestre,
                zip_path=str(pz.zip_path),
                output_dir=str(out_dir),
                status="FAILED_BAD_ZIP",
                files_extracted="",
                message="Arquivo ZIP corrompido ou inválido.",
            ),
        )
        print(f"[ERROR] {pz.ano} T{pz.trimestre}: ZIP inválido/corrompido")

    except Exception as e:
        append_manifest_row(
            manifest_path,
            dict(
                run_date=RUN_DATE,
                run_ts=RUN_TS,
                ano=pz.ano,
                trimestre=pz.trimestre,
                zip_path=str(pz.zip_path),
                output_dir=str(out_dir),
                status="FAILED_EXCEPTION",
                files_extracted="",
                message=f"{type(e).__name__}: {e}",
            ),
        )
        print(f"[ERROR] {pz.ano} T{pz.trimestre}: falha ({type(e).__name__})")


def run() -> None:
    manifest_path = REPORTS_DIR / f"manifest_extract_pgfn_{RUN_DATE}.csv"

    print(f"[INFO] Projeto: {PROJECT_ROOT}")
    print(f"[INFO] Raw PGFN: {RAW_PGFN_DIR}")
    print(f"[INFO] Staging: {STAGING_DIR}")
    print(f"[INFO] Manifest: {manifest_path}")
    print("-" * 70)

    periodo_zips = list_periodo_zips(RAW_PGFN_DIR)
    print(f"[INFO] ZIPs encontrados: {len(periodo_zips)}")
    print("-" * 70)

    if not periodo_zips:
        print("[WARN] Nenhum ZIP encontrado em data/raw/pgfn. Rode a ingestão primeiro.")
        return

    for pz in periodo_zips:
        extract_one(pz, manifest_path)

    print("-" * 70)
    print("[DONE] Extração PGFN concluída.")


if __name__ == "__main__":
    run()
