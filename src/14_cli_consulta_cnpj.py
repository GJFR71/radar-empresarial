# 14_cli_consulta_cnpj.py
"""
CLI: digitar um CNPJ (14 dígitos) ou raiz (8 dígitos) e obter:
- Score do Modelo 1 (saúde fiscal) + top fatores
- Score do Modelo 2 (descontinuidade) + top fatores

Ele busca automaticamente os CSVs mais recentes em /reports:
- model1_scores_YYYYMMDD.csv
- model1_top_factors_YYYYMMDD.csv
- model2_scores_YYYYMMDD.csv
- model2_top_factors_YYYYMMDD.csv

Uso (PowerShell):
  python -u .\src\14_cli_consulta_cnpj.py --cnpj 12.345.678/0001-99
  python -u .\src\14_cli_consulta_cnpj.py --cnpj 12345678
  python -u .\src\14_cli_consulta_cnpj.py --cnpj 12345678000199 --save_txt

Opcional:
  --reports .\reports
  --date 20260110   (força usar um dia específico)
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ModelFiles:
    scores: Path
    factors: Path


def only_digits(s: str) -> str:
    return re.sub(r"\D+", "", s or "")


def cnpj_raiz_from_input(s: str) -> str:
    d = only_digits(s)
    if len(d) == 14:
        return d[:8]
    if len(d) == 8:
        return d
    raise ValueError("CNPJ inválido. Informe 14 dígitos (CNPJ completo) ou 8 dígitos (raiz).")


def find_latest(pattern: str, reports_dir: Path) -> Optional[Path]:
    files = sorted(reports_dir.glob(pattern))
    if not files:
        return None
    # escolhe o mais recente por mtime
    return max(files, key=lambda p: p.stat().st_mtime)


def find_by_date(prefix: str, date: str, reports_dir: Path) -> Optional[Path]:
    p = reports_dir / f"{prefix}_{date}.csv"
    return p if p.exists() else None


def resolve_model_files(model_prefix: str, reports_dir: Path, date: Optional[str]) -> Optional[ModelFiles]:
    """
    model_prefix = "model1" ou "model2"
    """
    scores_prefix = f"{model_prefix}_scores"
    factors_prefix = f"{model_prefix}_top_factors"

    if date:
        s = find_by_date(scores_prefix, date, reports_dir)
        f = find_by_date(factors_prefix, date, reports_dir)
    else:
        s = find_latest(f"{scores_prefix}_*.csv", reports_dir)
        f = find_latest(f"{factors_prefix}_*.csv", reports_dir)

    if not s or not f:
        return None
    return ModelFiles(scores=s, factors=f)


def load_one(scores_path: Path, factors_path: Path, cnpj_raiz: str, score_col: str) -> tuple[Optional[float], Optional[int], Optional[str]]:
    # scores
    scores = pd.read_csv(scores_path, dtype={"cnpj_raiz": str})
    row = scores.loc[scores["cnpj_raiz"] == cnpj_raiz]
    if row.empty:
        return None, None, None

    score = float(row.iloc[0][score_col])
    ycol = "y_risco_fiscal" if "y_risco_fiscal" in scores.columns else ("y_descontinuidade" if "y_descontinuidade" in scores.columns else None)
    y = int(row.iloc[0][ycol]) if ycol else None

    # factors
    fac = pd.read_csv(factors_path, dtype={"cnpj_raiz": str})
    rowf = fac.loc[fac["cnpj_raiz"] == cnpj_raiz]
    top = None
    if not rowf.empty:
        # coluna padrão do seu script
        if "top_features" in fac.columns:
            top = str(rowf.iloc[0]["top_features"])
        elif "top_factors" in fac.columns:
            top = str(rowf.iloc[0]["top_factors"])

    return score, y, top


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def main(cnpj_input: str, reports_dir: Path, date: Optional[str], save_txt: bool) -> None:
    reports_dir = reports_dir.resolve()
    if not reports_dir.exists():
        raise FileNotFoundError(f"Pasta reports não encontrada: {reports_dir}")

    raiz = cnpj_raiz_from_input(cnpj_input)

    m1 = resolve_model_files("model1", reports_dir, date)
    m2 = resolve_model_files("model2", reports_dir, date)

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"Consulta por CNPJ")
    lines.append(f"Entrada: {cnpj_input}")
    lines.append(f"CNPJ raiz (8 dígitos): {raiz}")
    lines.append("=" * 72)

    # ---------------- Modelo 1 ----------------
    lines.append("\n[MODELO 1] Saúde fiscal / risco de inadimplência (score)")
    if not m1:
        lines.append(" - Arquivos não encontrados (gere com o script 10_score_model1_explain.py).")
    else:
        score, y, top = load_one(m1.scores, m1.factors, raiz, score_col="score_risco_fiscal")
        lines.append(f" - Fonte scores:  {m1.scores.name}")
        lines.append(f" - Fonte fatores: {m1.factors.name}")
        if score is None:
            lines.append(" - Resultado: CNPJ raiz não encontrado no score do Modelo 1.")
        else:
            lines.append(f" - Score (probabilidade estimada): {score:.6f}  ({pct(score)})")
            if y is not None:
                lines.append(f" - Rótulo do dataset (y): {y}  (1 = risco fiscal alto; 0 = baixo)")
            if top:
                lines.append(f" - Top fatores (aprox.): {top}")
            else:
                lines.append(" - Top fatores: não encontrado para este CNPJ.")

    # ---------------- Modelo 2 ----------------
    lines.append("\n[MODELO 2] Descontinuidade (proxy: qtd_ativos == 0)")
    if not m2:
        lines.append(" - Arquivos não encontrados (gere com o script 13_score_model2_explain.py).")
    else:
        score, y, top = load_one(m2.scores, m2.factors, raiz, score_col="score_descontinuidade")
        lines.append(f" - Fonte scores:  {m2.scores.name}")
        lines.append(f" - Fonte fatores: {m2.factors.name}")
        if score is None:
            lines.append(" - Resultado: CNPJ raiz não encontrado no score do Modelo 2.")
        else:
            lines.append(f" - Score (probabilidade estimada): {score:.6f}  ({pct(score)})")
            if y is not None:
                lines.append(f" - Rótulo do dataset (y): {y}  (1 = proxy de descontinuidade; 0 = ativo)")
            if top:
                lines.append(f" - Top fatores (aprox.): {top}")
            else:
                lines.append(" - Top fatores: não encontrado para este CNPJ.")

    lines.append("\nObservação:")
    lines.append("- O score é uma probabilidade estimada (quanto maior, maior o risco no proxy definido).")
    lines.append("- 'Top fatores' é uma explicação leve: impacto aproximado (z-score × coef/importance).")
    lines.append("- Para uso comercial: priorize Top N por score e use os fatores como “argumento de ação”.")

    text = "\n".join(lines)
    print(text)

    if save_txt:
        out = reports_dir / f"consulta_cnpj_{raiz}{'_' + date if date else ''}.txt"
        out.write_text(text, encoding="utf-8")
        print(f"\n[DONE] Relatório salvo: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnpj", required=True, help="CNPJ 14 dígitos (com/sem máscara) ou raiz 8 dígitos")
    ap.add_argument("--reports", default="reports", help="Pasta onde estão os CSVs de score/fatores")
    ap.add_argument("--date", default=None, help="Força usar YYYYMMDD específico (ex.: 20260110)")
    ap.add_argument("--save_txt", action="store_true", help="Salva um .txt com a resposta em reports/")
    args = ap.parse_args()

    main(
        cnpj_input=args.cnpj,
        reports_dir=Path(args.reports),
        date=args.date,
        save_txt=args.save_txt,
    )
