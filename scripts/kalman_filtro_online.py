"""
BLOCO 4 – Filtro de Kalman Online (Seção 2 do PDF)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Carrega os parâmetros estimados pelo EM (kalman_parametros.json),
detecta novas pesquisas no snapshot, aplica o filtro de Kalman
aumentado na série completa e gera dois gráficos:

  Gráfico 1 – Escala logit  : pontos observados (scatter por instituto)
                               + estado filtrado (linha) + IC 95%
  Gráfico 2 – Escala prob.  : mesmos elementos convertidos via logística

Ao final imprime:
  P(Flávio > 50%) e P(Lula > 50%) com base no estado filtrado final.

Uso no Jupyter:
    from kalman_filtro_online import run
    df, fig = run()
"""

import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless – required for GitHub Actions
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
from scipy.stats import norm

# ─────────────────────────────────────────────────────────────────────────────
# ⚙️  CONFIGURAÇÕES
# ─────────────────────────────────────────────────────────────────────────────

KALMAN_FILE   = "kalman_parametros.json"
SNAPSHOT_FILE = "snapshot_pesquisas.json"
OUTPUT_PNG    = "kalman_filtro_online.png"
OUTPUT_JSON   = "kalman_filtro_resultados.json"

# Nomes canônicos dos institutos (mesmo do Bloco 3)
INSTITUTOS_CANONICOS = {
    "Datafolha", "Paraná Pesquisas", "Genial/Quaest",
    "AtlasIntel", "Futura/Apex",
}
NOME_NORMALIZADO = {"Apex/Futura": "Futura/Apex"}

# Paleta de cores por instituto (consistente entre os dois gráficos)
CORES = {
    "Datafolha":        "#E63946",   # vermelho
    "Paraná Pesquisas": "#2A9D8F",   # verde-azulado
    "Genial/Quaest":    "#E9C46A",   # amarelo
    "AtlasIntel":       "#457B9D",   # azul
    "Futura/Apex":      "#9B5DE5",   # roxo
}

# Prior difuso (mesmo utilizado no EM)
X0_MEAN = 0.0
X0_VAR  = 10.0

_MESES = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
}


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO A – PARSING DE DATAS (usa o ÚLTIMO dia do intervalo)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(day: str, month_str: str, year: str) -> datetime | None:
    m = _MESES.get(month_str.strip().lower()[:3])
    if m is None:
        return None
    try:
        return datetime(int(year), m, int(day))
    except ValueError:
        return None


def _last_date(date_range: str, year: str) -> datetime | None:
    """
    Extrai o ÚLTIMO dia do intervalo de coleta.
    Ex.: '08 Abr - 12 Abr' → 2026-04-12
         '07 Abr - 09 Abr' → 2026-04-09
    """
    parts = re.split(r"\s*[–-]\s*", date_range.strip())
    dates = []
    for part in parts:
        m = re.match(r"(\d+)\s+(\w+)", part.strip())
        if m:
            d = _parse_date(m.group(1), m.group(2), year)
            if d:
                dates.append(d)
    if dates:
        return max(dates)   # retorna o último dia
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO B – CARREGAMENTO DOS DADOS
# ─────────────────────────────────────────────────────────────────────────────

def load_kalman_params(kalman_file: str = KALMAN_FILE) -> dict:
    """Lê kalman_parametros.json e devolve dict com sigma2, b_full, estados."""
    with open(kalman_file, encoding="utf-8") as f:
        kp = json.load(f)

    sigma2  = kp["parametros"]["sigma2_por_dia"]
    vieses  = kp["parametros"]["vieses"]          # {instituto: b_i}
    estados = kp["estados"]                        # list of {data, instituto, …}

    # Conjunto de (data, instituto) já suavizados
    # Chave robusta: (instituto, y_obs arredondado) — independe da convenção de data
    y_suavizados = {
        (e["instituto"], round(e["y_obs"], 4)) for e in estados
    }
    datas_suavizadas = {
        (e["data"], e["instituto"]) for e in estados
    }

    print(f"  Parâmetros carregados de '{kalman_file}'")
    print(f"    σ²  = {sigma2:.10f}  |  σ/dia = {np.sqrt(sigma2):.6f}")
    for inst, b in vieses.items():
        print(f"    b[{inst}] = {b:+.6f}")
    print(f"  Estados suavizados no arquivo: {len(estados)}")

    return dict(sigma2=sigma2, vieses=vieses, y_suavizados=y_suavizados,
                datas_suavizadas=y_suavizados, estados=estados,
                raw_kp=kp)


def load_snapshot_observations(
        snapshot_file: str,
        vieses: dict,
        datas_suavizadas: set) -> list[dict]:
    """
    Lê snapshot_pesquisas.json, filtra institutos canônicos e constrói
    y_t, R_t para cada pesquisa. Data = ÚLTIMO dia do intervalo.

    Identifica observações novas (não presentes em datas_suavizadas).
    """
    with open(snapshot_file, encoding="utf-8") as f:
        snap = json.load(f)

    rows = []
    for rec in snap.get("records", {}).values():
        nome = NOME_NORMALIZADO.get(rec["Contratante"], rec["Contratante"])
        if nome not in INSTITUTOS_CANONICOS:
            continue

        date = _last_date(rec["Data(s) de Pesquisa"], rec["Ano"])
        if date is None:
            continue

        try:
            lula_pct   = float(rec["Lula (PT) %"])
            flavio_pct = float(rec["Flávio (PL) %"])
            n          = float(rec["Tamanho da Amostra"])
        except (ValueError, KeyError, TypeError):
            continue

        q_t = (lula_pct + flavio_pct) / 100.0
        if not (0 < q_t < 1):
            continue

        p_t = (flavio_pct / 100.0) / q_t
        if not (0 < p_t < 1):
            continue

        y_t = np.log(p_t / (1.0 - p_t))
        R_t = 1.0 / (n * q_t * p_t * (1.0 - p_t))

        date_str = date.strftime("%Y-%m-%d")
        is_nova  = (nome, round(y_t, 4)) not in datas_suavizadas

        rows.append(dict(
            date=date, date_str=date_str,
            instituto=nome,
            y=y_t, R=R_t,
            p_obs=p_t, q=q_t, n=n,
            lula_pct=lula_pct, flavio_pct=flavio_pct,
            nova=is_nova,
        ))

    rows.sort(key=lambda r: (r["date"], r["instituto"]))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO C – FILTRO DE KALMAN AUMENTADO (Seção 2 do PDF)
# ─────────────────────────────────────────────────────────────────────────────

def _build_H(inst_idx: int, K: int, institutos: list) -> np.ndarray:
    """
    Constrói H_t ∈ R^K conforme Eq. (23)-(24):
      Estado aumentado: [x, b_1, …, b_{K-1}]
      H[0] = 1 (sempre)
      Se i_t = j < K-1: H[1+j] = +1
      Se i_t = K-1    : H[1:] = -1  (b_K = -Σb)
    """
    H = np.zeros(K)
    H[0] = 1.0
    if inst_idx < K - 1:           # Eq. 23
        H[1 + inst_idx] = 1.0
    else:                          # Eq. 24
        H[1:] = -1.0
    return H


def kalman_filter_online(
        rows: list[dict],
        sigma2: float,
        vieses: dict,
        institutos: list[str]) -> list[dict]:
    """
    Filtro de Kalman aumentado na série completa.

    Estado augmentado: x_t = [x_latente, b_1, …, b_{K-1}]^T ∈ R^K
    Dinâmica:  x_t = x_{t-1} + w_t,   Q_t = diag(Δt·σ², 0,…,0)
    Observação: y_t = H_t · x_t + v_t, v_t ~ N(0, R_t)

    Regra especial: NÃO se aplica o passo de predição para a ÚLTIMA
    observação (delays de publicação — estado final = x_{T|T}).

    Retorna lista de dicts com estados filtrados por observação.
    """
    K         = len(institutos)
    inst2idx  = {inst: i for i, inst in enumerate(institutos)}

    # ── Estado inicial: prior difuso em x, vieses fixos ──────────────────
    dim   = K
    x_aug = np.zeros(dim)
    for i, inst in enumerate(institutos[:-1]):   # b_1 … b_{K-1}
        x_aug[1 + i] = vieses[inst]
    P_aug = np.zeros((dim, dim))
    P_aug[0, 0] = X0_VAR                         # incerteza apenas no estado latente

    T = len(rows)
    resultados = []

    for t, row in enumerate(rows):
        is_last = (t == T - 1)

        # ── Predição [Eq. 25-26] ─────────────────────────────────────────
        # Aplica-se para todas as observações exceto a primeira
        # (e não se extrapola APÓS a última)
        if t > 0:
            delta_t = max((row["date"] - rows[t - 1]["date"]).days, 0)
            Q = np.zeros((dim, dim))
            Q[0, 0] = delta_t * sigma2
            x_pred = x_aug.copy()              # Eq. 25: F=I
            P_pred = P_aug + Q                 # Eq. 26
        else:
            x_pred = x_aug.copy()
            P_pred = P_aug.copy()

        # ── Vetor H_t ────────────────────────────────────────────────────
        i_t = inst2idx[row["instituto"]]
        H   = _build_H(i_t, K, institutos)

        # ── Inovação [Eq. 27-28] ─────────────────────────────────────────
        nu  = row["y"] - H @ x_pred           # escalar
        S   = float(H @ P_pred @ H) + row["R"]  # escalar

        # ── Ganho de Kalman [Eq. 29] ─────────────────────────────────────
        Kg  = (P_pred @ H) / S               # (K,)

        # ── Atualização [Eq. 30-31 — forma de Joseph] ────────────────────
        x_filt = x_pred + Kg * nu
        I_KH   = np.eye(dim) - np.outer(Kg, H)
        P_filt = I_KH @ P_pred @ I_KH.T + np.outer(Kg, Kg) * row["R"]

        # Salva resultado (apenas o estado latente x_t)
        x_f = float(x_filt[0])
        P_f = float(P_filt[0, 0])
        x_p = float(x_pred[0])
        P_p = float(P_pred[0, 0])

        resultados.append(dict(
            t=t, **row,
            x_pred=x_p, P_pred=P_p,
            x_filt=x_f, P_filt=P_f,
            p_filt=_logistic(x_f),
            ic95_lo=_logistic(x_f - 1.96 * np.sqrt(P_f)),
            ic95_hi=_logistic(x_f + 1.96 * np.sqrt(P_f)),
        ))

        # Prepara próximo passo (exceto após a última observação)
        if not is_last:
            x_aug = x_filt
            P_aug = P_filt

    return resultados, x_filt, P_filt


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO D – PROBABILIDADES FINAIS
# ─────────────────────────────────────────────────────────────────────────────

def calcular_probabilidades(x_final: float, P_final: float) -> dict:
    """
    Estado filtrado final x_{T|T} ~ N(μ, σ²) na escala logit.
    P(Flávio > 50%) = P(x > 0) = 1 - Φ(-μ/σ) = Φ(μ/σ)
    P(Lula > 50%)   = P(x < 0) = Φ(-μ/σ)
    """
    mu    = x_final
    sigma = np.sqrt(P_final)
    z     = mu / sigma

    p_flavio_vence = float(norm.cdf(z))    # P(x > 0)
    p_lula_vence   = float(norm.cdf(-z))   # P(x < 0)

    return dict(
        x_final=mu, P_final=P_final, sigma_final=sigma, z=z,
        p_flavio_acima_50=p_flavio_vence,
        p_lula_acima_50=p_lula_vence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO E – PLOTAGEM
# ─────────────────────────────────────────────────────────────────────────────

def _formatar_eixo_x(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
    ax.tick_params(axis="x", which="major", labelsize=9)


def plotar(resultados: list[dict], probs: dict) -> plt.Figure:
    """
    Gera figura com dois painéis:
      Painel 1 – Escala logit (y_t observado + estado filtrado + IC 95%)
      Painel 2 – Escala probabilidade (p_obs + p_filt + IC 95%)
    """
    dates    = [r["date"] for r in resultados]
    y_obs    = [r["y"]    for r in resultados]
    p_obs    = [r["p_obs"] for r in resultados]
    x_filt   = [r["x_filt"] for r in resultados]
    p_filt   = [r["p_filt"] for r in resultados]
    ic_lo    = [r["ic95_lo"] for r in resultados]
    ic_hi    = [r["ic95_hi"] for r in resultados]
    ic_lo_logit = [r["x_filt"] - 1.96 * np.sqrt(r["P_filt"]) for r in resultados]
    ic_hi_logit = [r["x_filt"] + 1.96 * np.sqrt(r["P_filt"]) for r in resultados]

    institutos = sorted(set(r["instituto"] for r in resultados))
    fig, axes  = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    fig.patch.set_facecolor("#F8F9FA")

    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4, color="#CCCCCC")
        ax.grid(axis="x", linestyle=":",  alpha=0.25, color="#CCCCCC")

    # ── PAINEL 1 – Logit ─────────────────────────────────────────────────
    ax1 = axes[0]

    # IC 95% (banda)
    ax1.fill_between(dates, ic_lo_logit, ic_hi_logit,
                     color="#457B9D", alpha=0.15, label="IC 95% (filtrado)")

    # Linha filtrada
    ax1.plot(dates, x_filt, color="#1D3557", lw=2.0,
             label="Estado filtrado $x_{t|t}$", zorder=3)

    # Linha y = 0 (50%)
    ax1.axhline(0, color="#E63946", lw=1.0, ls="--", alpha=0.6, label="0  (50% logit)")

    # Scatter por instituto
    for inst in institutos:
        mask  = [r["instituto"] == inst for r in resultados]
        d_i   = [d for d, m in zip(dates, mask) if m]
        y_i   = [y for y, m in zip(y_obs, mask) if m]
        novas = [r["nova"] for r, m in zip(resultados, mask) if m]

        scatter = ax1.scatter(d_i, y_i,
                              color=CORES.get(inst, "#888888"),
                              s=70, zorder=5, label=inst,
                              edgecolors="white", linewidths=0.7)

        # Novas observações: marcador especial (estrela)
        d_nov = [d for d, n in zip(d_i, novas) if n]
        y_nov = [y for y, n in zip(y_i, novas) if n]
        if d_nov:
            ax1.scatter(d_nov, y_nov, marker="*",
                        color=CORES.get(inst, "#888888"),
                        s=200, zorder=6, edgecolors="#1D3557", linewidths=0.8)

    ax1.set_ylabel("Logit da intenção de voto (Flávio)", fontsize=10)
    ax1.set_title("Filtro de Kalman – Segundo Turno Lula × Flávio Bolsonaro\n"
                  "Escala logit", fontsize=12, fontweight="bold", pad=8)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.85,
               ncol=2, columnspacing=1.0)

    # ── PAINEL 2 – Probabilidade ─────────────────────────────────────────
    ax2 = axes[1]

    # IC 95% (banda)
    ax2.fill_between(dates, ic_lo, ic_hi,
                     color="#457B9D", alpha=0.15, label="IC 95% (filtrado)")

    # Linha filtrada
    ax2.plot(dates, p_filt, color="#1D3557", lw=2.0,
             label="Estado filtrado $p_{t|t}$", zorder=3)

    # Linha 50%
    ax2.axhline(0.5, color="#E63946", lw=1.0, ls="--", alpha=0.6, label="50%")

    # Scatter por instituto
    for inst in institutos:
        mask  = [r["instituto"] == inst for r in resultados]
        d_i   = [d for d, m in zip(dates, mask) if m]
        p_i   = [p for p, m in zip(p_obs, mask) if m]
        novas = [r["nova"] for r, m in zip(resultados, mask) if m]

        ax2.scatter(d_i, p_i,
                    color=CORES.get(inst, "#888888"),
                    s=70, zorder=5, label=inst,
                    edgecolors="white", linewidths=0.7)

        d_nov = [d for d, n in zip(d_i, novas) if n]
        p_nov = [p for p, n in zip(p_i, novas) if n]
        if d_nov:
            ax2.scatter(d_nov, p_nov, marker="*",
                        color=CORES.get(inst, "#888888"),
                        s=200, zorder=6, edgecolors="#1D3557", linewidths=0.8)

    # Anotação com probabilidades finais
    p_flavio = probs["p_flavio_acima_50"]
    p_lula   = probs["p_lula_acima_50"]
    p_final  = probs["x_final"]
    data_fim = dates[-1]

    ax2.annotate(
        f"P(Flávio > 50%) = {p_flavio*100:.1f}%\n"
        f"P(Lula > 50%)   = {p_lula*100:.1f}%",
        xy=(data_fim, _logistic(p_final)),
        xytext=(-120, -40), textcoords="offset points",
        fontsize=9, color="#1D3557",
        bbox=dict(boxstyle="round,pad=0.4", fc="#EDF2F4",
                  ec="#1D3557", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#1D3557",
                        connectionstyle="arc3,rad=0.2"),
    )

    ax2.set_ylabel("Intenção de voto normalizada (Flávio)", fontsize=10)
    ax2.set_title("Escala probabilidade", fontsize=12,
                  fontweight="bold", pad=8)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.85,
               ncol=2, columnspacing=1.0)

    _formatar_eixo_x(ax2)

    # Nota de rodapé
    fig.text(0.5, 0.01,
             "★ = observação nova (não incluída no suavizador EM)  |  "
             "Banda = IC 95%  |  "
             "Data da observação = último dia do campo",
             ha="center", fontsize=7.5, color="#555555")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run(kalman_file:   str = KALMAN_FILE,
        snapshot_file: str = SNAPSHOT_FILE,
        output_png:    str = OUTPUT_PNG,
        output_json:   str = OUTPUT_JSON) -> tuple:
    """
    Pipeline completo:
      1. Carrega parâmetros EM de kalman_file
      2. Carrega e prepara observações de snapshot_file
      3. Detecta novas pesquisas
      4. Aplica filtro de Kalman aumentado (série completa)
      5. Calcula P(Flávio>50%) e P(Lula>50%)
      6. Gera e salva os dois gráficos

    Retorna: (resultados: list[dict], probs: dict, fig: Figure)
    """
    print("\n" + "═" * 64)
    print("  📡  FILTRO DE KALMAN ONLINE – Seção 2 do PDF")
    print("═" * 64)

    # 1. Parâmetros
    print(f"\n📂  Carregando parâmetros de '{kalman_file}'...")
    kp = load_kalman_params(kalman_file)

    sigma2           = kp["sigma2"]
    vieses           = kp["vieses"]
    datas_suavizadas = kp["datas_suavizadas"]

    institutos = sorted(vieses.keys())        # ordem canônica
    K          = len(institutos)

    # b_full: inclui b_K por restrição Σb=0
    b_full = np.array([vieses[inst] for inst in institutos])

    # 2. Observações (data = último dia)
    print(f"\n📊  Carregando pesquisas de '{snapshot_file}'...")
    rows = load_snapshot_observations(snapshot_file, vieses, datas_suavizadas)
    T    = len(rows)

    n_novas = sum(r["nova"] for r in rows)
    print(f"  Total de observações: {T}")
    if n_novas > 0:
        print(f"  🆕 {n_novas} nova(s) pesquisa(s) não incluída(s) no suavizador:")
        for r in rows:
            if r["nova"]:
                print(f"     {r['date'].strftime('%d/%m/%Y')}  "
                      f"{r['instituto']}  "
                      f"Flávio={r['flavio_pct']:.1f}%  Lula={r['lula_pct']:.1f}%")
    else:
        print("  ✔ Nenhuma observação nova detectada.")

    # 3. Filtro de Kalman
    print(f"\n🔄  Aplicando filtro de Kalman aumentado (K={K}, T={T})...")
    print(f"     Sem passo de predição após a última observação "
          f"({rows[-1]['date'].strftime('%d/%m/%Y')}).")

    resultados, x_final_vec, P_final_mat = kalman_filter_online(
        rows, sigma2, vieses, institutos)

    x_final = float(x_final_vec[0])
    P_final = float(P_final_mat[0, 0])

    print(f"\n  Estado filtrado final:")
    print(f"    x_{{T|T}}  = {x_final:.6f}")
    print(f"    P_{{T|T}}  = {P_final:.8f}")
    print(f"    p_{{T|T}}  = {_logistic(x_final)*100:.2f}%  "
          f"IC95%: [{_logistic(x_final - 1.96*np.sqrt(P_final))*100:.2f}%, "
          f"{_logistic(x_final + 1.96*np.sqrt(P_final))*100:.2f}%]")

    # 4. Probabilidades
    probs = calcular_probabilidades(x_final, P_final)

    # 5. Salva JSON com intenções de voto e probabilidades
    p_f    = _logistic(x_final)
    sig    = np.sqrt(P_final)
    saida  = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_ultima_pesquisa": rows[-1]["date"].strftime("%Y-%m-%d"),
        "instituto_ultima_pesquisa": rows[-1]["instituto"],
        "intencao_voto_flavio": {
            "estimativa": round(p_f, 4),
            "ic95_lo":    round(_logistic(x_final - 1.96 * sig), 4),
            "ic95_hi":    round(_logistic(x_final + 1.96 * sig), 4),
        },
        "intencao_voto_lula": {
            "estimativa": round(1.0 - p_f, 4),
            "ic95_lo":    round(1.0 - _logistic(x_final + 1.96 * sig), 4),
            "ic95_hi":    round(1.0 - _logistic(x_final - 1.96 * sig), 4),
        },
        "probabilidade_vitoria": {
            "flavio_acima_50pct": round(probs["p_flavio_acima_50"], 4),
            "lula_acima_50pct":   round(probs["p_lula_acima_50"],   4),
        },
        "estado_filtrado_logit": {
            "x_final": round(x_final, 6),
            "P_final": round(P_final, 8),
            "sigma":   round(float(sig), 8),
        },
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)
    print(f"  ✔ Resultados salvos em '{output_json}'.")

    # 6. Gráfico
    print(f"\n📈  Gerando gráficos...")
    fig = plotar(resultados, probs)
    fig.savefig(output_png, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  ✔ Gráfico salvo em '{output_png}'.")

    return resultados, probs, fig


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DIRETA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    resultados, probs, fig = run()