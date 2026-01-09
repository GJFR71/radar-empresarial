from __future__ import annotations

from pathlib import Path
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw" / "cnpj"
OUT_EMP = PROJECT / "data" / "staging" / "cnpj" / "2025-12" / "empresas"
OUT_EST = PROJECT / "data" / "staging" / "cnpj" / "2025-12" / "estabelecimentos"

OUT_EMP.mkdir(parents=True, exist_ok=True)
OUT_EST.mkdir(parents=True, exist_ok=True)

def extract(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

emp_zips = sorted(RAW.glob("Empresas*.zip"))
est_zips = sorted(RAW.glob("Estabelecimentos*.zip"))

if not emp_zips:
    raise FileNotFoundError(f"Nenhum zip encontrado: {RAW} / Empresas*.zip")
if not est_zips:
    raise FileNotFoundError(f"Nenhum zip encontrado: {RAW} / Estabelecimentos*.zip")

for zp in emp_zips:
    print("Extract:", zp.name)
    extract(zp, OUT_EMP)

for zp in est_zips:
    print("Extract:", zp.name)
    extract(zp, OUT_EST)

print("DONE")
