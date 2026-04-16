"""
BLOCO 3 – Agregador de Pesquisas via Filtro de Kalman + Suavizador RTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implementa a Nota Técnica de Joaquim Antônio Costa Bermudes (15/04/2026).

Pipeline:
  1. Carrega snapshot_pesquisas.json e prepara (y_t, R_t, Δt)
  2. Estima (σ², b_1…b_{K-1}) via EM:
       E-step: filtro de Kalman escalar + suavizador RTS
       M-step: atualização fechada de σ² e regressão ponderada para vieses
  3. Aplica filtro de Kalman aumentado (estado + vieses) para inferência online
  4. Grava kalman_parametros.json com parâmetros e estados suavizados

Uso no Jupyter:
    from kalman_agregador import run
    resultado = run()      # usa snapshot_pesquisas.json
"""

import json
import re
import numpy as np
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ⚙️  CONFIGURAÇÕES
# ─────────────────────────────────────────────────────────────────────────────

SNAPSHOT_FILE  = "snapshot_pesquisas.json"
OUTPUT_FILE    = "kalman_parametros.json"

# Mesmos institutos do Bloco 2 (Apex/Futura → Futura/Apex)
INSTITUTOS_CANONICOS = {
    "Datafolha", "Paraná Pesquisas", "Genial/Quaest",
    "AtlasIntel", "Futura/Apex",
}
NOME_NORMALIZADO = {"Apex/Futura": "Futura/Apex"}

# Algoritmo EM
EM_MAX_ITER  = 500
EM_TOL       = 1e-9        # tolerância no log-verossimilhança marginal
SIGMA2_INIT  = 0.005       # variância inicial por dia (escala logit)
SIGMA2_MIN   = 1e-12       # piso numérico

# Prior difuso para x_0
X0_MEAN = 0.0
X0_VAR  = 10.0

# Meses em PT → número
_MESES = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
}


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 1 – PREPARAÇÃO DOS DADOS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(day: str, month_str: str, year: str) -> datetime | None:
    m = _MESES.get(month_str.strip().lower()[:3])
    if m is None:
        return None
    try:
        return datetime(int(year), m, int(day))
    except ValueError:
        return None


def _midpoint(date_range: str, year: str) -> datetime | None:
    """
    Converte '07 Abr - 09 Abr' na data média do intervalo.
    Aceita traço simples '-' e travessão '–'.
    """
    parts = re.split(r"\s*[–-]\s*", date_range.strip())
    dates = []
    for part in parts:
        m = re.match(r"(\d+)\s+(\w+)", part.strip())
        if m:
            d = _parse_date(m.group(1), m.group(2), year)
            if d:
                dates.append(d)
    if len(dates) == 2:
        return dates[0] + timedelta(days=(dates[1] - dates[0]).days / 2)
    if len(dates) == 1:
        return dates[0]
    return None


def load_and_prepare(snapshot_file: str = SNAPSHOT_FILE) -> dict:
    """
    Lê o snapshot e constrói os vetores de observações para o modelo.

    Para cada pesquisa t:
        q_t = (Lula% + Flávio%) / 100          ← fração de respostas válidas  [Eq. 1]
        p_t = (Flávio% / 100) / q_t             ← intenção ajustada            [Eq. 2]
        y_t = log(p_t / (1 - p_t))              ← transformação logit           [Eq. 3]
        R_t = 1 / (n_t · q_t · p_t · (1-p_t))  ← variância de observação      [Eq. 9]

    Retorna dict com:
        y, R, delta_t  : arrays numpy (T,)
        dates          : list[datetime]
        inst_idx       : array int (T,) — índice do instituto em 0..K-1
        institutos     : list[str] — ordem canônica dos K institutos
        raw            : list[dict] — metadados por observação
    """
    with open(snapshot_file, encoding="utf-8") as f:
        snap = json.load(f)

    rows = []
    for rec in snap.get("records", {}).values():
        nome = NOME_NORMALIZADO.get(rec["Contratante"], rec["Contratante"])
        if nome not in INSTITUTOS_CANONICOS:
            continue

        date = _midpoint(rec["Data(s) de Pesquisa"], rec["Ano"])
        if date is None:
            continue

        try:
            lula_pct   = float(rec["Lula (PT) %"])
            flavio_pct = float(rec["Flávio (PL) %"])
            n          = float(rec["Tamanho da Amostra"])
        except (ValueError, KeyError, TypeError):
            continue

        q_t = (lula_pct + flavio_pct) / 100.0  # fração de respostas válidas
        if not (0 < q_t < 1):
            continue

        p_t = (flavio_pct / 100.0) / q_t        # prob. ajustada de Flávio
        if not (0 < p_t < 1):
            continue

        y_t = np.log(p_t / (1.0 - p_t))         # logit
        R_t = 1.0 / (n * q_t * p_t * (1.0 - p_t))

        rows.append({
            "date":       date,
            "instituto":  nome,
            "y":          y_t,
            "R":          R_t,
            "p":          p_t,
            "q":          q_t,
            "n":          n,
            "lula_pct":   lula_pct,
            "flavio_pct": flavio_pct,
        })

    if not rows:
        raise ValueError("Nenhum dado válido no snapshot.")

    # Ordena por data
    rows.sort(key=lambda r: r["date"])
    T = len(rows)

    # Δt em dias entre observações consecutivas (Δt[0] não é usado no filtro)
    delta_t = np.zeros(T)
    for t in range(1, T):
        delta_t[t] = (rows[t]["date"] - rows[t - 1]["date"]).days

    # Codificação de institutos
    institutos = sorted({r["instituto"] for r in rows})
    inst2idx   = {inst: i for i, inst in enumerate(institutos)}
    inst_idx   = np.array([inst2idx[r["instituto"]] for r in rows], dtype=int)
    K = len(institutos)

    y = np.array([r["y"] for r in rows])
    R = np.array([r["R"] for r in rows])

    print(f"\n{'─'*62}")
    print(f"  📦  Dados carregados: {T} observações | {K} institutos")
    print(f"       Período: {rows[0]['date'].date()} → {rows[-1]['date'].date()}")
    for inst in institutos:
        n_obs = int((inst_idx == inst2idx[inst]).sum())
        print(f"       – {inst}: {n_obs} pesquisas")

    return dict(y=y, R=R, delta_t=delta_t, dates=[r["date"] for r in rows],
                inst_idx=inst_idx, institutos=institutos, K=K, raw=rows)


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 2 – FILTRO DE KALMAN (estado escalar, para o EM)
# ─────────────────────────────────────────────────────────────────────────────

def kalman_filter(y, R, delta_t, sigma2, b_full, inst_idx):
    """
    Filtro de Kalman para estado escalar x_t (passeio aleatório).

    Modelo:
        x_t = x_{t-1} + ε_t,   ε_t ~ N(0, Δt · σ²)
        y_t = x_t + b_{i_t} + v_t,  v_t ~ N(0, R_t)

    Equivalente a filtrar y_adj_t = y_t - b_{i_t}.

    Retorna: x_pred, P_pred, x_filt, P_filt (T,) e log-verossimilhança marginal.
    """
    T = len(y)
    x_pred = np.empty(T)
    P_pred = np.empty(T)
    x_filt = np.empty(T)
    P_filt = np.empty(T)
    log_lik = 0.0

    # Prior difuso
    x_curr = X0_MEAN
    P_curr = X0_VAR

    for t in range(T):
        # ── Predição [Eq. 25-26] ────────────────────────────────────────────
        if t == 0:
            x_p = x_curr
            P_p = P_curr
        else:
            x_p = x_filt[t - 1]
            P_p = P_filt[t - 1] + delta_t[t] * sigma2

        x_pred[t] = x_p
        P_pred[t] = P_p

        # Observação ajustada
        y_adj = y[t] - b_full[inst_idx[t]]

        # ── Inovação [Eq. 27-28] ────────────────────────────────────────────
        nu = y_adj - x_p
        S  = P_p + R[t]

        # Contribuição ao log-verossimilhança (decomposição da inovação)
        log_lik -= 0.5 * (np.log(2.0 * np.pi * S) + nu ** 2 / S)

        # ── Ganho e atualização [Eq. 29-31] ─────────────────────────────────
        Kg = P_p / S
        x_filt[t] = x_p + Kg * nu
        # Forma de Joseph: numericamente estável
        P_filt[t] = (1.0 - Kg) ** 2 * P_p + Kg ** 2 * R[t]

    return x_pred, P_pred, x_filt, P_filt, log_lik


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 2 – SUAVIZADOR RTS (Rauch–Tung–Striebel)
# ─────────────────────────────────────────────────────────────────────────────

def rts_smoother(x_pred, P_pred, x_filt, P_filt):
    """
    Suavizador RTS backward pass.

    Retorna:
        x_smooth  (T,) : E[x_t | y_{1:T}]
        P_smooth  (T,) : Var[x_t | y_{1:T}]
        P_cross   (T,) : Cov[x_t, x_{t-1} | y_{1:T}]  (P_cross[0] não usado)
    """
    T = len(x_filt)
    x_smooth = np.empty(T)
    P_smooth = np.empty(T)
    P_cross  = np.zeros(T)

    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for t in range(T - 2, -1, -1):
        # Ganho do suavizador: G_t = P_{t|t} / P_{t+1|t}  (F=1 para passeio aleatório)
        G = P_filt[t] / P_pred[t + 1]

        x_smooth[t] = x_filt[t] + G * (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + G ** 2 * (P_smooth[t + 1] - P_pred[t + 1])

        # Covariância cruzada P_{t+1, t | T}  [Eq. 47]
        P_cross[t + 1] = G * P_smooth[t + 1]

    return x_smooth, P_smooth, P_cross


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 3 – PASSO M DO EM
# ─────────────────────────────────────────────────────────────────────────────

def m_step(y, R, delta_t, x_smooth, P_smooth, P_cross, inst_idx, K):
    """
    Atualização fechada dos parâmetros (passo M do EM).

    σ²:  [Eq. 48]
        σ² = Σ E[(x_t - x_{t-1})²] / Σ Δt
        E[(x_t - x_{t-1})²] = P_t + P_{t-1} - 2 P_{t,t-1} + (x̂_t - x̂_{t-1})²  [Eq. 51]

    Vieses b_{1..K-1}:  [Eq. 57]
        Regressão ponderada: r_t = y_t - x̂_t ≈ b_{i_t}

    Retorna: sigma2_new (float), biases_new (K-1,)
    """
    T = len(y)

    # ── Atualização de σ² ─────────────────────────────────────────────────
    num_s2 = 0.0
    den_s2 = 0.0
    for t in range(1, T):
        E_diff2 = (P_smooth[t] + P_smooth[t - 1]
                   - 2.0 * P_cross[t]
                   + (x_smooth[t] - x_smooth[t - 1]) ** 2)
        num_s2 += E_diff2
        den_s2 += delta_t[t]

    sigma2_new = max(num_s2 / den_s2, SIGMA2_MIN)

    # ── Atualização dos vieses [Eq. 56-57] ───────────────────────────────
    if K == 1:
        return sigma2_new, np.zeros(0)

    r = y - x_smooth        # resíduos do estado suavizado
    W = 1.0 / R             # pesos

    # Matriz H: T × (K-1)
    # h_t = e_j  se i_t = j < K
    # h_t = (-1, …, -1)  se i_t = K-1 (último instituto)  [Eq. 55]
    H = np.zeros((T, K - 1))
    for t in range(T):
        i = inst_idx[t]
        if i < K - 1:
            H[t, i] = 1.0
        else:
            H[t, :] = -1.0

    # (H^T W H)^{-1} H^T W r
    HWH = H.T @ (W[:, None] * H)
    HWr = H.T @ (W * r)

    # Regularização leve para evitar singularidade
    HWH += np.eye(K - 1) * 1e-12
    biases_new = np.linalg.solve(HWH, HWr)

    return sigma2_new, biases_new


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 3 – LOOP EM
# ─────────────────────────────────────────────────────────────────────────────

def run_em(data: dict) -> tuple:
    """
    Algoritmo EM completo para estimar σ² e b_{1..K-1}.

    Retorna:
        sigma2     : float — variância por dia estimada
        b_full     : (K,) — vieses completos (inclui b_K por restrição Σb=0)
        institutos : list[str]
        x_smooth   : (T,) — estados suavizados
        P_smooth   : (T,) — variâncias suavizadas
        log_lik    : float — log-verossimilhança final
        n_iter     : int
    """
    y        = data["y"]
    R        = data["R"]
    delta_t  = data["delta_t"]
    inst_idx = data["inst_idx"]
    K        = data["K"]
    T        = len(y)

    # Inicialização
    sigma2  = SIGMA2_INIT
    biases  = np.zeros(K - 1)   # b_1, ..., b_{K-1}

    prev_ll = -np.inf

    print(f"\n{'─'*62}")
    print(f"  🔄  EM — T={T} obs. | K={K} institutos | máx. {EM_MAX_ITER} iter.")
    print(f"{'─'*62}")
    print(f"  {'Iter':>5}  {'Log-Lik':>14}  {'Δ Log-Lik':>13}  {'σ (por dia)':>12}")
    print(f"  {'─'*5}  {'─'*14}  {'─'*13}  {'─'*12}")

    n_iter = 0
    for iteration in range(EM_MAX_ITER):
        # Vetor completo de vieses (inclui b_K = -Σb)
        b_full = np.zeros(K)
        b_full[: K - 1] = biases
        b_full[K - 1]   = -np.sum(biases)

        # ── E-step ──────────────────────────────────────────────────────────
        x_pred, P_pred, x_filt, P_filt, log_lik = kalman_filter(
            y, R, delta_t, sigma2, b_full, inst_idx)
        x_smooth, P_smooth, P_cross = rts_smoother(
            x_pred, P_pred, x_filt, P_filt)

        # ── M-step ──────────────────────────────────────────────────────────
        sigma2, biases = m_step(
            y, R, delta_t, x_smooth, P_smooth, P_cross, inst_idx, K)

        delta_ll = log_lik - prev_ll
        n_iter   = iteration + 1

        # Log a cada 25 iterações ou nas primeiras 5
        if n_iter <= 5 or n_iter % 25 == 0:
            print(f"  {n_iter:>5}  {log_lik:>14.4f}  {delta_ll:>13.6f}  "
                  f"{np.sqrt(sigma2):>12.6f}")

        if abs(delta_ll) < EM_TOL and iteration > 0:
            print(f"  {n_iter:>5}  {log_lik:>14.4f}  {delta_ll:>13.2e}  "
                  f"{np.sqrt(sigma2):>12.6f}")
            print(f"\n  ✔ Convergência em {n_iter} iterações "
                  f"(|ΔLL| = {abs(delta_ll):.2e} < {EM_TOL:.0e})")
            break
        prev_ll = log_lik
    else:
        print(f"\n  ⚠️  Máximo de {EM_MAX_ITER} iterações atingido.")

    # Suavizador final com parâmetros convergidos
    b_full = np.zeros(K)
    b_full[: K - 1] = biases
    b_full[K - 1]   = -np.sum(biases)

    _, _, _, _, log_lik = kalman_filter(y, R, delta_t, sigma2, b_full, inst_idx)
    x_pred, P_pred, x_filt, P_filt, _ = kalman_filter(
        y, R, delta_t, sigma2, b_full, inst_idx)
    x_smooth, P_smooth, _ = rts_smoother(x_pred, P_pred, x_filt, P_filt)

    return sigma2, b_full, x_smooth, P_smooth, x_filt, P_filt, log_lik, n_iter


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 2 – FILTRO DE KALMAN AUMENTADO (online, parâmetros fixos)
# ─────────────────────────────────────────────────────────────────────────────

def kalman_augmented(data: dict, sigma2: float, b_full: np.ndarray):
    """
    Filtro de Kalman aumentado conforme Seção 2 do PDF.

    Estado aumentado: x_t = [x_t, b_1, ..., b_{K-1}]^T  ∈ R^K
    Dinâmica: x_t = x_{t-1} + w_t,  w_t ~ N(0, Q_t)
    Q_t = Δt · diag(σ², 0, ..., 0)

    Observação: y_t = H_t x_t + v_t,  v_t ~ N(0, R_t)
    H_t: linha com 1 na posição x e ±1 na posição do viés.

    Retorna arrays de médias e variâncias filtradas (apenas x_t, 1º componente).
    """
    y        = data["y"]
    R        = data["R"]
    delta_t  = data["delta_t"]
    inst_idx = data["inst_idx"]
    K        = data["K"]
    T        = len(y)

    # Inicializa estado e covariância aumentados
    dim = K
    x_aug = np.zeros(dim)
    x_aug[1:] = b_full[: K - 1]          # vieses fixos (b_1…b_{K-1})
    P_aug = np.zeros((dim, dim))
    P_aug[0, 0] = X0_VAR                  # incerteza apenas no estado latente

    x_filt_aug = np.empty(T)
    P_filt_aug = np.empty(T)

    for t in range(T):
        # ── Predição ────────────────────────────────────────────────────────
        Q = np.zeros((dim, dim))
        if t > 0:
            Q[0, 0] = delta_t[t] * sigma2
        P_pred = P_aug + Q                 # Eq. 26

        # ── Vetor H_t ───────────────────────────────────────────────────────
        H = np.zeros(dim)
        H[0] = 1.0
        i = inst_idx[t]
        if i < K - 1:                      # Eq. 23
            H[1 + i] = 1.0
        else:                              # Eq. 24  (último instituto)
            H[1:] = -1.0

        # ── Inovação [Eq. 27-28] ────────────────────────────────────────────
        nu = y[t] - H @ x_aug             # usa predição x_aug (= x_{t|t-1})
        S  = H @ P_pred @ H + R[t]        # escalar

        # ── Ganho e atualização [Eq. 29-31] ─────────────────────────────────
        Kg = (P_pred @ H) / S             # (K,)
        x_aug = x_aug + Kg * nu
        I_KH  = np.eye(dim) - np.outer(Kg, H)
        P_aug = I_KH @ P_pred @ I_KH.T + np.outer(Kg, Kg) * R[t]  # Joseph

        x_filt_aug[t] = x_aug[0]
        P_filt_aug[t] = P_aug[0, 0]

    return x_filt_aug, P_filt_aug


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def _logit_to_prob(x):
    return 1.0 / (1.0 + np.exp(-x))


def _ic95(x, P):
    """Intervalo de credibilidade 95% na escala de probabilidade."""
    z = 1.959964
    lo = _logit_to_prob(x - z * np.sqrt(P))
    hi = _logit_to_prob(x + z * np.sqrt(P))
    return lo, hi


def run(snapshot_file: str = SNAPSHOT_FILE,
        output_file:   str = OUTPUT_FILE) -> dict:
    """
    Executa o pipeline completo e salva os resultados em JSON.

    Retorna o dicionário de resultados.
    """
    print("\n" + "═" * 62)
    print("  📡  AGREGADOR DE PESQUISAS – Filtro de Kalman + RTS")
    print("═" * 62)
    print(f"  Nota Técnica: Joaquim Antônio Costa Bermudes (15/04/2026)")

    # 1. Preparação dos dados
    print(f"\n📥  Carregando dados de '{snapshot_file}'...")
    data = load_and_prepare(snapshot_file)

    # 2. EM
    print("\n🔄  Estimando parâmetros via EM...")
    (sigma2, b_full, x_smooth, P_smooth,
     x_filt, P_filt, log_lik, n_iter) = run_em(data)

    institutos = data["institutos"]
    K          = data["K"]

    # 3. Filtro aumentado (online)
    print("\n📊  Aplicando filtro de Kalman aumentado (parâmetros fixos)...")
    x_filt_aug, P_filt_aug = kalman_augmented(data, sigma2, b_full)
    print("    ✔ Concluído.")

    # 4. Resumo dos parâmetros
    print(f"\n{'─'*62}")
    print("  📋  PARÂMETROS ESTIMADOS")
    print(f"{'─'*62}")
    print(f"  σ² (por dia, escala logit) : {sigma2:.8f}")
    print(f"  σ  (por dia, escala logit) : {np.sqrt(sigma2):.6f}")
    print(f"  σ  (por semana)            : {np.sqrt(sigma2 * 7):.6f}")
    print(f"  Log-verossimilhança        : {log_lik:.4f}")
    print(f"  Iterações EM               : {n_iter}")
    print(f"\n  Vieses b_i (escala logit) — Σ b_i = 0 por construção:")
    for i, inst in enumerate(institutos):
        print(f"    [{i}] {inst:<22s}: {b_full[i]:+.6f}")

    # Estado suavizado final (última observação)
    p_final  = _logit_to_prob(x_smooth[-1])
    p_lo, p_hi = _ic95(x_smooth[-1], P_smooth[-1])
    print(f"\n  Estimativa suavizada (última obs., {data['dates'][-1].date()}):")
    print(f"    Flávio (PL): {p_final*100:.2f}%  IC95%: "
          f"[{p_lo*100:.2f}%, {p_hi*100:.2f}%]")

    # 5. Serialização
    timestamp = datetime.now().isoformat(timespec="seconds")

    estados = []
    for t in range(len(data["y"])):
        p_s   = _logit_to_prob(x_smooth[t])
        p_f   = _logit_to_prob(x_filt_aug[t])
        lo_s, hi_s = _ic95(x_smooth[t],   P_smooth[t])
        lo_f, hi_f = _ic95(x_filt_aug[t], P_filt_aug[t])

        estados.append({
            "t":                     t,
            "data":                  data["dates"][t].strftime("%Y-%m-%d"),
            "instituto":             data["institutos"][data["inst_idx"][t]],
            "y_obs":                 round(float(data["y"][t]), 6),
            "R_obs":                 round(float(data["R"][t]), 6),
            # Suavizador RTS
            "x_smooth":              round(float(x_smooth[t]), 6),
            "P_smooth":              round(float(P_smooth[t]), 6),
            "p_smooth":              round(float(p_s), 4),
            "p_smooth_ic95_lo":      round(float(lo_s), 4),
            "p_smooth_ic95_hi":      round(float(hi_s), 4),
            # Filtro aumentado (online)
            "x_filt":                round(float(x_filt_aug[t]), 6),
            "P_filt":                round(float(P_filt_aug[t]), 6),
            "p_filt":                round(float(p_f), 4),
            "p_filt_ic95_lo":        round(float(lo_f), 4),
            "p_filt_ic95_hi":        round(float(hi_f), 4),
        })

    resultado = {
        "timestamp":   timestamp,
        "fonte":       snapshot_file,
        "nota_tecnica": "Bermudes, J.A.C. (2026-04-15)",
        "parametros": {
            "sigma2_por_dia":    round(float(sigma2), 10),
            "sigma_por_dia":     round(float(np.sqrt(sigma2)), 8),
            "sigma_por_semana":  round(float(np.sqrt(sigma2 * 7)), 8),
            "log_verossimilhanca": round(float(log_lik), 6),
            "n_iteracoes_em":    n_iter,
            "convergiu":         n_iter < EM_MAX_ITER,
            "vieses": {
                inst: round(float(b_full[i]), 8)
                for i, inst in enumerate(institutos)
            },
        },
        "estados": estados,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    print(f"\n✅  Parâmetros e estados salvos em '{output_file}'.")
    print(f"    Total de {len(estados)} estados ({len(estados)} observações).")

    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DIRETA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    resultado = run()