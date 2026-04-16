# =============================================================================
# NOTEBOOK UNIFICADO – Pesquisas 2º Turno Lula × Flávio Bolsonaro
# =============================================================================
# Cole cada bloco em uma célula separada do Jupyter e execute em ordem.
# =============================================================================


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CÉLULA 1 — Imports e configurações
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import json
import re
import unicodedata
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── Configurações do Bloco 1 (scraper) ──────────────────────────────────────

URL = (
    "https://pt.wikipedia.org/wiki/"
    "Pesquisas_de_opini%C3%A3o_para_a_elei%C3%A7%C3%A3o_presidencial_no_Brasil_em_2026"
)

# Deixe None para buscar online; informe o caminho para usar arquivo local:
LOCAL_FILE = None
# LOCAL_FILE = r"C:\Users\joaqu\Downloads\Wikipedia.txt"

TARGET_H3   = "Lula e Flávio Bolsonaro"
TARGET_YEARS = ["2026", "2025"]
SNAPSHOT_FILE = "snapshot_pesquisas.json"

COLUMN_NAMES = [
    "Contratante", "Data(s) de Pesquisa", "Tamanho da Amostra",
    "Margem de Erro (pp)", "Lula (PT) %", "Flávio (PL) %",
    "Indecisos e Absentos %", "Vantagem",
]
KEY_COLS = ["Ano", "Contratante", "Data(s) de Pesquisa"]

# ── Configurações do Bloco 2 (normalização) ──────────────────────────────────

# Institutos a manter (Futura/Apex e Apex/Futura são o mesmo instituto)
INSTITUTOS_ALVO = [
    "Datafolha",
    "Paraná Pesquisas",
    "Genial/Quaest",
    "AtlasIntel",
    "Futura/Apex",
    "Apex/Futura",
]

COLUNAS_EXIBICAO = [
    "Ano", "Contratante", "Data(s) de Pesquisa", "Tamanho da Amostra",
    "Lula (PT) %", "Flávio (PL) %", "Indecisos e Absentos %",
    "Lula Norm %", "Flávio Norm %", "Desvio Padrão Flávio",
]

print("✅  Configurações carregadas.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CÉLULA 2 — Funções do Bloco 1 (scraper)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()

def _cell_text(td):
    return _normalize(td.get_text(separator=" "))

def _clean_ref(text):
    return re.sub(r"\[\s*\d+\s*\]", "", text).strip()

def _fetch_url(url):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ElectionPollScraper/1.0)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")

def _fetch_file(path):
    return BeautifulSoup(Path(path).read_text(encoding="utf-8"), "html.parser")

def find_target_section(soup, h3_text):
    for h3 in soup.find_all("h3"):
        if _normalize(h3.get_text()) == _normalize(h3_text):
            section = h3.find_parent("section")
            if section:
                return section
    return None

def find_year_tables(section, years):
    result = {}
    for h4 in section.find_all("h4"):
        yt = _normalize(h4.get_text())
        for y in years:
            if re.fullmatch(rf"{y}(?:\s+\d+)?", yt) and y not in result:
                sub = h4.find_parent("section") or h4.parent
                tbl = sub.find("table", class_="wikitable")
                if tbl:
                    result[y] = tbl
    return result

def parse_table(table, year):
    rows_data = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if all(c.name == "th" for c in cells):
            continue
        if len(cells) == 1 and cells[0].get("colspan"):
            continue
        texts = [_cell_text(c) for c in cells]
        if not texts or texts[0].lower().startswith("contratante"):
            continue
        n = len(COLUMN_NAMES)
        if len(texts) < n:
            texts.extend([""] * (n - len(texts)))
        texts = texts[:n]
        if all(t == "" for t in texts):
            continue
        rows_data.append(texts)

    df = pd.DataFrame(rows_data, columns=COLUMN_NAMES)
    df.insert(0, "Ano", year)
    df["Contratante"] = df["Contratante"].apply(_clean_ref)

    for col in ["Lula (PT) %", "Flávio (PL) %", "Indecisos e Absentos %"]:
        df[col] = pd.to_numeric(
            df[col].str.replace("%", "", regex=False)
                   .str.replace(",", ".", regex=False).str.strip(),
            errors="coerce")

    df["Margem de Erro (pp)"] = pd.to_numeric(
        df["Margem de Erro (pp)"].str.replace(",", ".", regex=False).str.strip(),
        errors="coerce")

    df["Tamanho da Amostra"] = pd.to_numeric(
        df["Tamanho da Amostra"].str.replace(r"\s", "", regex=True)
                                .str.replace(".", "", regex=False),
        errors="coerce").astype("Int64")

    return df

def _df_to_records(df):
    records = {}
    for _, row in df.iterrows():
        key = " | ".join(str(row[c]) for c in KEY_COLS)
        records[key] = {col: str(row[col]) for col in df.columns}
    return records

def _load_snapshot():
    p = Path(SNAPSHOT_FILE)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_snapshot(records, timestamp):
    with open(SNAPSHOT_FILE, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "records": records},
                  f, ensure_ascii=False, indent=2)
    print(f"\n✅  Snapshot salvo em '{SNAPSHOT_FILE}' ({len(records)} registros).")

def _compare_snapshots(old, new):
    added   = set(new) - set(old)
    removed = set(old) - set(new)
    changed = [k for k in set(new) & set(old) if new[k] != old[k]]

    print("\n" + "═"*68)
    print("📊  RELATÓRIO DE MUDANÇAS")
    print("═"*68)

    if not added and not removed and not changed:
        print("✔   Nenhuma mudança detectada.\n")
        return

    if added:
        print(f"\n🆕  {len(added)} NOVO(S) REGISTRO(S):")
        for k in sorted(added):
            r = new[k]
            print(f"   • {k}\n"
                  f"     Lula: {r.get('Lula (PT) %')}%  |  "
                  f"Flávio: {r.get('Flávio (PL) %')}%  |  "
                  f"Vantagem: {r.get('Vantagem')}")
    if changed:
        print(f"\n✏️   {len(changed)} REGISTRO(S) ALTERADO(S):")
        for k in sorted(changed):
            print(f"   • {k}")
            for col in new[k]:
                if new[k][col] != old[k][col]:
                    print(f"       {col}: '{old[k][col]}' → '{new[k][col]}'")
    if removed:
        print(f"\n❌  {len(removed)} REGISTRO(S) REMOVIDO(S):")
        for k in sorted(removed):
            print(f"   • {k}")
    print()

def scrape(url=URL, local_file=None):
    """Extrai as tabelas de pesquisas da Wikipedia e retorna um DataFrame."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    fonte = f"arquivo: {local_file}" if local_file else url

    print(f"\n{'═'*68}")
    print(f"  🗳️  SCRAPER – 2º Turno Lula × Flávio Bolsonaro")
    print(f"{'═'*68}")
    print(f"  Execução : {timestamp}")
    print(f"  Fonte    : {fonte}\n")

    soup = _fetch_file(local_file) if local_file else _fetch_url(url)

    print(f"🔎  Localizando seção '{TARGET_H3}'...")
    section = find_target_section(soup, TARGET_H3)
    if not section:
        raise RuntimeError(f"Seção '{TARGET_H3}' não encontrada. Verifique TARGET_H3.")
    print("    ✔ Encontrada.")

    print(f"📅  Extraindo subseções {TARGET_YEARS}...")
    year_tables = find_year_tables(section, TARGET_YEARS)
    if not year_tables:
        raise RuntimeError("Nenhuma tabela encontrada. Verifique TARGET_YEARS.")

    frames = []
    for year in TARGET_YEARS:
        if year not in year_tables:
            print(f"⚠️   Subseção {year} não encontrada — pulando.")
            continue
        print(f"📊  Processando {year}...")
        df_year = parse_table(year_tables[year], year)
        frames.append(df_year)
        print(f"    ✔ {len(df_year)} registros extraídos.")

    if not frames:
        raise RuntimeError("Nenhum dado foi extraído.")

    df_all = pd.concat(frames, ignore_index=True)

    new_records = _df_to_records(df_all)
    old_snap    = _load_snapshot()
    old_records = old_snap.get("records", {})

    if old_records:
        print(f"\n📂  Snapshot anterior: {old_snap.get('timestamp', '?')}")
        _compare_snapshots(old_records, new_records)
    else:
        print("\n📂  Primeiro snapshot — sem comparação anterior.")

    _save_snapshot(new_records, timestamp)
    return df_all

print("✅  Funções do Bloco 1 carregadas.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CÉLULA 3 — Funções do Bloco 2 (normalização e desvio padrão)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def filtrar_institutos(df):
    """Mantém apenas os institutos em INSTITUTOS_ALVO (case-insensitive)."""
    alvo_lower = {i.lower() for i in INSTITUTOS_ALVO}
    mask = df["Contratante"].str.lower().isin(alvo_lower)
    df_filt = df[mask].reset_index(drop=True)
    removidos = sorted(set(df["Contratante"]) - set(df_filt["Contratante"]))
    print(f"  ✔ {len(df_filt)} mantidos | {len(df)-len(df_filt)} removidos "
          f"({', '.join(removidos) if removidos else 'nenhum'})")
    return df_filt

def normalizar_percentuais(df):
    """
    Lula Norm % = Lula% / (Lula% + Flávio%) × 100
    Flávio Norm % = Flávio% / (Lula% + Flávio%) × 100
    """
    df = df.copy()
    total = df["Lula (PT) %"] + df["Flávio (PL) %"]
    df["Lula Norm %"]   = (df["Lula (PT) %"]   / total * 100).round(4)
    df["Flávio Norm %"] = (df["Flávio (PL) %"] / total * 100).round(4)
    soma = (df["Lula Norm %"] + df["Flávio Norm %"]).round(2)
    invalidos = (soma != 100).sum()
    if invalidos:
        print(f"  ⚠️  {invalidos} linha(s) com soma ≠ 100%")
    else:
        print("  ✔ Todas as linhas somam 100%")
    return df

def calcular_desvio_padrao(df):
    """
    σ = sqrt( p · (1−p) / (h · n) )
      p = Flávio Norm % / 100
      h = Indecisos e Absentos % / 100
      n = Tamanho da Amostra
    """
    df = df.copy()
    p = df["Flávio Norm %"] / 100
    h = 1-df["Indecisos e Absentos %"] / 100
    n = df["Tamanho da Amostra"].astype(float)
    denominador = h * n
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(
            denominador > 0,
            np.sqrt(p * (1 - p) / denominador),
            np.nan,
        )
    df["Desvio Padrão Flávio"] = np.round(sigma, 6)
    nans = df["Desvio Padrão Flávio"].isna().sum()
    if nans:
        print(f"  ⚠️  {nans} linha(s) com σ = NaN (h=0 ou amostra ausente)")
    else:
        print(f"  ✔ Desvio padrão calculado para todas as {len(df)} linhas")
    return df

def processar(df):
    """Aplica filtro → normalização → desvio padrão em sequência."""
    print("\n" + "═"*68)
    print("  🔬  BLOCO 2 – Normalização e Desvio Padrão")
    print("═"*68)
    print("\n📌  Etapa 1 – Filtro de institutos")
    df = filtrar_institutos(df)
    print("\n📐  Etapa 2 – Normalização (Lula + Flávio = 100%)")
    df = normalizar_percentuais(df)
    print("\n📏  Etapa 3 – Desvio padrão: σ = √(p·(1-p) / (h·n))")
    df = calcular_desvio_padrao(df)
    print(f"\n✅  Concluído — {len(df)} registros, {df['Ano'].nunique()} ano(s).\n")
    return df

def exibir(df):
    """Imprime os resultados organizados por ano com estatísticas resumidas."""
    print("\n" + "═"*68)
    print("  📋  RESULTADOS PROCESSADOS")
    print("═"*68)

    for ano in sorted(df["Ano"].unique(), reverse=True):
        sub = df[df["Ano"] == ano][COLUNAS_EXIBICAO].copy()
        sub["Desvio Padrão Flávio"] = sub["Desvio Padrão Flávio"].map(
            lambda x: f"{x:.4f}" if pd.notna(x) else "—")
        sub["Lula Norm %"]   = sub["Lula Norm %"].map(lambda x: f"{x:.2f}%")
        sub["Flávio Norm %"] = sub["Flávio Norm %"].map(lambda x: f"{x:.2f}%")

        print(f"\n{'─'*68}\n  {ano}  ({len(sub)} pesquisas)\n{'─'*68}")
        with pd.option_context("display.max_columns", None,
                               "display.width", 140, "display.max_rows", None):
            print(sub.drop(columns="Ano").to_string(index=False))

    print(f"\n{'─'*68}")
    print("  📊  ESTATÍSTICAS RESUMIDAS (por instituto e ano)")
    print(f"{'─'*68}")
    resumo = (
        df.groupby(["Ano", "Contratante"], sort=False)
        .agg(N=("Flávio Norm %", "count"),
             Flávio_Norm_Médio=("Flávio Norm %", "mean"),
             Lula_Norm_Médio=("Lula Norm %", "mean"),
             Desvio_Médio=("Desvio Padrão Flávio", "mean"))
        .round(2).reset_index()
    )
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(resumo.to_string(index=False))
    print()

print("✅  Funções do Bloco 2 carregadas.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CÉLULA 4 — Execução
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Para usar arquivo local, edite LOCAL_FILE na CÉLULA 1, por exemplo:
#   LOCAL_FILE = r"C:\Users\joaqu\Downloads\Wikipedia.txt"
#
# Depois execute esta célula:

df_raw       = scrape(local_file=LOCAL_FILE)
df_processed = processar(df_raw)
exibir(df_processed)
