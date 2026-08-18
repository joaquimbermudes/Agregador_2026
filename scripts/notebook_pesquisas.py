# =============================================================================
# NOTEBOOK UNIFICADO – Pesquisas com Lula e Flávio Bolsonaro
# =============================================================================
# Extrai todos os cenários de primeiro e segundo turno da página da Wikipédia
# em que Lula e Flávio aparecem simultaneamente, sem restringir institutos.
# Os demais candidatos são consolidados na coluna "Outros %".
# =============================================================================


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CÉLULA 1 — Imports e configurações
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import json
import re
import sys
import unicodedata
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

URL = (
    "https://pt.wikipedia.org/wiki/"
    "Pesquisas_de_opini%C3%A3o_para_a_elei%C3%A7%C3%A3o_presidencial_no_Brasil_em_2026"
)

# Deixe None para buscar online; informe um HTML local para trabalhar offline.
LOCAL_FILE = None
SNAPSHOT_FILE = "snapshot_pesquisas.json"

OUTPUT_COLUMNS = [
    "Ano",
    "Turno",
    "Contratante",
    "Data(s) de Pesquisa",
    "Tamanho da Amostra",
    "Margem de Erro (pp)",
    "Lula (PT) %",
    "Flávio (PL) %",
    "Outros %",
    "Indecisos e Abstenções %",
    "Cenário",
]

COLUNAS_EXIBICAO = OUTPUT_COLUMNS + [
    "Respostas Válidas %",
    "Lula entre Válidos %",
    "Flávio entre Válidos %",
    "Outros entre Válidos %",
    "Desvio Padrão Flávio",
]

_MONTH_PATTERN = re.compile(
    r"\b(?:jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\b",
    flags=re.IGNORECASE,
)

print("✅  Configurações carregadas.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CÉLULA 2 — Funções de aquisição e interpretação das tabelas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _normalize(text) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip()


def _ascii_key(text) -> str:
    normalized = unicodedata.normalize("NFKD", _normalize(text))
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def _clean_ref(text) -> str:
    text = re.sub(r"\[\s*[\w\-]+\s*\]", "", _normalize(text))
    return re.sub(r"\s+", " ", text).strip()


def _fetch_url(url):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ElectionPollScraper/2.0)"}
    response = requests.get(url, headers=headers, timeout=45)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def _fetch_file(path):
    return BeautifulSoup(Path(path).read_text(encoding="utf-8"), "html.parser")


def _previous_heading(table, name: str):
    return table.find_previous(name)


def _table_context(table) -> tuple[str | None, str | None]:
    """Obtém turno e ano pelos cabeçalhos que antecedem a tabela."""
    h2 = _previous_heading(table, "h2")
    h2_key = _ascii_key(h2.get_text(" ", strip=True)) if h2 else ""
    if "primeiro turno" in h2_key:
        turno = "Primeiro turno"
    elif "segundo turno" in h2_key:
        turno = "Segundo turno"
    else:
        return None, None

    year = None
    for heading in table.find_all_previous(["h3", "h4", "h5"]):
        match = re.fullmatch(r"\s*(20\d{2})(?:\s+\d+)?\s*", _normalize(heading.get_text(" ")))
        if match:
            year = match.group(1)
            break
    return turno, year


def _read_table(table) -> pd.DataFrame:
    """
    Lê a grade sem promover cabeçalhos automaticamente.

    ``read_html`` expande rowspans; assim, uma pesquisa com vários cenários
    conserva contratante, datas e amostra em todas as linhas.
    """
    frames = pd.read_html(
        StringIO(str(table)),
        header=None,
        keep_default_na=False,
        displayed_only=False,
        decimal=",",
        thousands=None,
    )
    return frames[0] if frames else pd.DataFrame()


def _column_label(column) -> str:
    """Reduz colunas simples/MultiIndex ao nível semanticamente informativo."""
    parts = column if isinstance(column, tuple) else (column,)
    useful = []
    for part in parts:
        text = _normalize(part)
        key = _ascii_key(text)
        if not text or key.startswith("unnamed") or key == "nan":
            continue
        if text not in useful:
            useful.append(text)
    return useful[-1] if useful else ""


def _columns_contain_candidates(grid: pd.DataFrame) -> bool:
    labels = [_ascii_key(_column_label(column)) for column in grid.columns]
    return (
        any(re.search(r"\blula\b", label) for label in labels)
        and any(re.search(r"\bflavio\b", label) for label in labels)
    )


def _find_candidate_header(grid: pd.DataFrame) -> int | None:
    for index in range(min(len(grid), 8)):
        values = [_ascii_key(value) for value in grid.iloc[index].tolist()]
        has_lula = any(re.search(r"\blula\b", value) for value in values)
        has_flavio = any(re.search(r"\bflavio\b", value) for value in values)
        if has_lula and has_flavio:
            return index
    return None


def _classify_header(text: str) -> str:
    key = _ascii_key(text)
    if "contratante" in key or key == "pesquisa":
        return "contractor"
    if "data" in key and "pesquis" in key:
        return "date"
    if "tamanho" in key and "amostra" in key:
        return "sample"
    if "margem" in key and "erro" in key:
        return "margin"
    if re.search(r"\blula\b", key):
        return "lula"
    if re.search(r"\bflavio\b", key):
        return "flavio"
    if "indecis" in key or "abst" in key:
        return "undecided"
    if re.search(r"\boutros?\b", key):
        return "reported_other"
    if "vantagem" in key:
        return "advantage"
    if not key or key.startswith("unnamed") or key == "nan":
        return "empty"
    return "candidate"


def _header_map(grid: pd.DataFrame, candidate_header: int | None) -> dict:
    """Mapeia semanticamente cada coluna, mesmo com cabeçalho multinível."""
    if candidate_header is None:
        return {
            column: {
                "kind": _classify_header(_column_label(column)),
                "label": _clean_ref(_column_label(column)),
            }
            for column in grid.columns
        }

    generic_row = grid.iloc[0]
    candidate_row = grid.iloc[candidate_header]
    mapping = {}
    for column in grid.columns:
        candidate_text = _normalize(candidate_row[column])
        generic_text = _normalize(generic_row[column])
        candidate_kind = _classify_header(candidate_text)
        generic_kind = _classify_header(generic_text)

        if candidate_kind not in {"empty", "candidate"}:
            kind, label = candidate_kind, candidate_text
        elif candidate_kind == "candidate":
            kind, label = "candidate", candidate_text
        else:
            kind, label = generic_kind, generic_text
        mapping[column] = {"kind": kind, "label": _clean_ref(label)}
    return mapping


def _column_by_kind(mapping: dict, kind: str):
    return next((column for column, info in mapping.items() if info["kind"] == kind), None)


def _parse_number(value) -> float:
    if value is None or isinstance(value, bool):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if np.isfinite(value) else np.nan
    text = _clean_ref(value).replace("−", "-").replace("—", "-").replace("–", "-")
    if text.strip() in {"", "-", "nan", "NaN"}:
        return np.nan
    match = re.search(r"-?\d+(?:[.,]\d+)?", text.replace("\xa0", " "))
    return float(match.group(0).replace(",", ".")) if match else np.nan


def _parse_sample(value) -> int | None:
    if isinstance(value, (int, np.integer)):
        return int(value) if value > 0 else None
    if isinstance(value, (float, np.floating)):
        return int(value) if np.isfinite(value) and value > 0 else None
    digits = re.sub(r"\D", "", _clean_ref(value))
    return int(digits) if digits else None


def _valid_date_label(value) -> bool:
    return bool(_MONTH_PATTERN.search(_ascii_key(value)))


def _scenario_label(row, mapping: dict) -> str:
    labels = []
    for column, info in mapping.items():
        if info["kind"] not in {"lula", "flavio", "candidate"}:
            continue
        if np.isfinite(_parse_number(row[column])):
            labels.append(info["label"])
    return " × ".join(dict.fromkeys(labels))


def parse_poll_table(table, table_index: int) -> list[dict]:
    """Extrai somente cenários que contenham Lula e Flávio simultaneamente."""
    turno, year = _table_context(table)
    if turno is None or year is None:
        return []

    grid = _read_table(table)
    if grid.empty:
        return []
    headers_in_columns = _columns_contain_candidates(grid)
    header_index = None if headers_in_columns else _find_candidate_header(grid)
    if not headers_in_columns and header_index is None:
        return []
    mapping = _header_map(grid, header_index)

    required = {
        kind: _column_by_kind(mapping, kind)
        for kind in ("contractor", "date", "sample", "margin", "lula", "flavio")
    }
    if any(column is None for column in required.values()):
        return []

    undecided_column = _column_by_kind(mapping, "undecided")
    reported_other_column = _column_by_kind(mapping, "reported_other")
    other_candidate_columns = [
        column for column, info in mapping.items() if info["kind"] == "candidate"
    ]

    records = []
    body_start = 0 if headers_in_columns else header_index + 1
    for _, row in grid.iloc[body_start:].iterrows():
        contractor = _clean_ref(row[required["contractor"]])
        date_label = _clean_ref(row[required["date"]])
        sample = _parse_sample(row[required["sample"]])
        lula = _parse_number(row[required["lula"]])
        flavio = _parse_number(row[required["flavio"]])

        # Elimina cabeçalhos repetidos, efemérides e cenários sem um dos dois.
        if (
            not contractor
            or "contratante" in _ascii_key(contractor)
            or _valid_date_label(contractor)
            or not _valid_date_label(date_label)
            or sample is None or sample < 100
            or not np.isfinite(lula)
            or not np.isfinite(flavio)
            or not (0.0 <= lula <= 100.0)
            or not (0.0 <= flavio <= 100.0)
        ):
            continue

        reported_other = (
            _parse_number(row[reported_other_column])
            if reported_other_column is not None else np.nan
        )
        other_values = [
            _parse_number(row[column]) for column in other_candidate_columns
        ]
        outros = sum(value for value in other_values if np.isfinite(value))
        if np.isfinite(reported_other):
            outros += reported_other
        undecided = (
            _parse_number(row[undecided_column])
            if undecided_column is not None else np.nan
        )
        declared_total = lula + flavio + outros + (undecided if np.isfinite(undecided) else 0.0)
        if not (70.0 <= declared_total <= 130.0):
            continue

        records.append({
            "Ano": year,
            "Turno": turno,
            "Contratante": contractor,
            "Data(s) de Pesquisa": date_label,
            "Tamanho da Amostra": sample,
            "Margem de Erro (pp)": _parse_number(row[required["margin"]]),
            "Lula (PT) %": lula,
            "Flávio (PL) %": flavio,
            "Outros %": round(float(outros), 4),
            "Indecisos e Abstenções %": undecided,
            "Cenário": _scenario_label(row, mapping),
            "_Tabela fonte": table_index,
        })
    return records


def extract_all_polls(soup) -> pd.DataFrame:
    """Percorre todas as wikitables, sem lista prévia de anos ou institutos."""
    records = []
    tables = soup.select("table.wikitable")
    for table_index, table in enumerate(tables):
        records.extend(parse_poll_table(table, table_index))
    if not records:
        raise RuntimeError("Nenhuma pesquisa contendo simultaneamente Lula e Flávio foi encontrada.")

    frame = pd.DataFrame(records)
    # Tabelas podem repetir exatamente uma linha na fronteira entre subseções.
    dedupe_columns = [column for column in OUTPUT_COLUMNS if column != "Cenário"] + ["Cenário"]
    frame = frame.drop_duplicates(subset=dedupe_columns).reset_index(drop=True)
    return frame


print("✅  Funções de aquisição carregadas.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CÉLULA 3 — Snapshot e relatório de alterações
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _serialize_value(value) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}"
    return str(value)


def _df_to_records(df):
    records = {}
    occurrences = {}
    for _, row in df.iterrows():
        base = " | ".join(
            _serialize_value(row[column])
            for column in ("Ano", "Turno", "Contratante", "Data(s) de Pesquisa", "Cenário")
        )
        occurrences[base] = occurrences.get(base, 0) + 1
        key = base if occurrences[base] == 1 else f"{base} | cenário {occurrences[base]}"
        records[key] = {
            column: _serialize_value(row[column]) for column in OUTPUT_COLUMNS
        }
    return records


def _load_snapshot():
    path = Path(SNAPSHOT_FILE)
    if path.exists():
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    return {}


def _save_snapshot(records, timestamp):
    with Path(SNAPSHOT_FILE).open("w", encoding="utf-8") as file:
        json.dump({"timestamp": timestamp, "records": records}, file,
                  ensure_ascii=False, indent=2)
    print(f"\n✅  Snapshot salvo em '{SNAPSHOT_FILE}' ({len(records)} registros).")


def _compare_snapshots(old, new):
    added = set(new) - set(old)
    removed = set(old) - set(new)
    changed = [key for key in set(new) & set(old) if new[key] != old[key]]
    print("\n" + "═" * 72)
    print("📊  RELATÓRIO DE MUDANÇAS")
    print("═" * 72)
    if not added and not removed and not changed:
        print("✔ Nenhuma mudança detectada.")
        return
    print(f"  Novos: {len(added)} | Alterados: {len(changed)} | Removidos: {len(removed)}")
    for key in sorted(added)[:20]:
        row = new[key]
        print(
            f"  + {row['Turno']} | {row['Contratante']} | {row['Data(s) de Pesquisa']} | "
            f"Lula {row['Lula (PT) %']}% | Flávio {row['Flávio (PL) %']}%"
        )
    if len(added) > 20:
        print(f"  ... e mais {len(added) - 20} novo(s) registro(s).")


def scrape(url=URL, local_file=None):
    timestamp = datetime.now().isoformat(timespec="seconds")
    source = f"arquivo: {local_file}" if local_file else url
    print("\n" + "═" * 72)
    print("  🗳️  SCRAPER – PRIMEIRO E SEGUNDO TURNO, TODAS AS CASAS")
    print("═" * 72)
    print(f"  Execução: {timestamp}")
    print(f"  Fonte: {source}\n")

    soup = _fetch_file(local_file) if local_file else _fetch_url(url)
    frame = extract_all_polls(soup)
    counts = frame.groupby(["Turno", "Ano"]).size()
    print(f"✔ {len(frame)} cenários extraídos de {frame['Contratante'].nunique()} casas.")
    for (turno, year), count in counts.items():
        print(f"  - {turno}, {year}: {count}")

    new_records = _df_to_records(frame)
    old_snapshot = _load_snapshot()
    old_records = old_snapshot.get("records", {})
    if old_records:
        print(f"\n📂  Snapshot anterior: {old_snapshot.get('timestamp', '?')}")
        _compare_snapshots(old_records, new_records)
    else:
        print("\n📂  Primeiro snapshot — sem comparação anterior.")
    _save_snapshot(new_records, timestamp)
    return frame.drop(columns=["_Tabela fonte"], errors="ignore")


print("✅  Funções de snapshot carregadas.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CÉLULA 4 — Consolidação de válidos e incerteza amostral
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calcular_percentuais_validos(df):
    """
    Preserva Lula, Flávio, Outros e indecisos/abstenções em separado.

    As colunas "entre válidos" removem apenas indecisos e abstenções; portanto,
    os outros candidatos continuam no denominador no primeiro turno.
    """
    frame = df.copy()
    frame["Respostas Válidas %"] = (
        frame["Lula (PT) %"] + frame["Flávio (PL) %"] + frame["Outros %"]
    )
    valid = frame["Respostas Válidas %"].replace(0, np.nan)
    frame["Lula entre Válidos %"] = (frame["Lula (PT) %"] / valid * 100).round(4)
    frame["Flávio entre Válidos %"] = (frame["Flávio (PL) %"] / valid * 100).round(4)
    frame["Outros entre Válidos %"] = (frame["Outros %"] / valid * 100).round(4)
    return frame


def calcular_desvio_padrao(df):
    """Delta/binomial: sqrt(p(1-p)/(n*q)), com q = fração válida."""
    frame = df.copy()
    p = frame["Flávio entre Válidos %"] / 100.0
    q = frame["Respostas Válidas %"] / 100.0
    n = frame["Tamanho da Amostra"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(n * q > 0.0, np.sqrt(p * (1.0 - p) / (n * q)), np.nan)
    frame["Desvio Padrão Flávio"] = np.round(sigma, 6)
    return frame


def processar(df):
    print("\n" + "═" * 72)
    print("  🔬  CONSOLIDAÇÃO: LULA, FLÁVIO, OUTROS E NÃO RESPOSTAS")
    print("═" * 72)
    frame = calcular_percentuais_validos(df)
    frame = calcular_desvio_padrao(frame)
    print(
        f"✔ {len(frame)} cenários | {frame['Contratante'].nunique()} casas | "
        f"{frame['Turno'].nunique()} turnos"
    )
    return frame


def exibir(df):
    print("\n" + "═" * 72)
    print("  📋  RESULTADOS PROCESSADOS")
    print("═" * 72)
    for (turno, year), subset in df.groupby(["Turno", "Ano"], sort=False):
        display = subset[COLUNAS_EXIBICAO].copy()
        print(f"\n{'─' * 72}\n  {turno} — {year} ({len(display)} cenários)\n{'─' * 72}")
        with pd.option_context(
            "display.max_columns", None, "display.width", 220,
            "display.max_rows", None, "display.max_colwidth", 45,
        ):
            print(display.to_string(index=False))

    print(f"\n{'─' * 72}\n  📊  RESUMO POR TURNO, ANO E CASA\n{'─' * 72}")
    summary = (
        df.groupby(["Turno", "Ano", "Contratante"], sort=False)
        .agg(
            Cenários=("Flávio (PL) %", "count"),
            Lula_Médio=("Lula (PT) %", "mean"),
            Flávio_Médio=("Flávio (PL) %", "mean"),
            Outros_Médio=("Outros %", "mean"),
            Indecisos_Médio=("Indecisos e Abstenções %", "mean"),
        )
        .round(2)
        .reset_index()
    )
    with pd.option_context("display.max_rows", None, "display.width", 180):
        print(summary.to_string(index=False))


print("✅  Funções de processamento carregadas.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CÉLULA 5 — Execução
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    df_raw = scrape(local_file=LOCAL_FILE)
    df_processed = processar(df_raw)
    exibir(df_processed)
