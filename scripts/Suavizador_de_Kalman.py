"""
Estimação offline do agregador eleitoral com reversão à média.

Implementa a nota técnica ``Agregador_de_Pesquisas_Eleitorais_Reversao_Media.tex``:

* estado dinâmico s_t = [x_t, mu_t], em escala logit;
* x_t reverte ao componente persistente mu_t por uma transição OU exata;
* mu_t segue passeio aleatório;
* vieses dos institutos somam zero;
* a matriz de covariância do processo e os vieses são estimados por EM + RTS.

O JSON produzido é consumido por ``kalman_filtro_online.py``.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


SNAPSHOT_FILE = "snapshot_pesquisas.json"
OUTPUT_FILE = "kalman_parametros.json"

# Segundo turno das eleições gerais de 2026 (calendário do TSE).
ELECTION_DATE = datetime(2026, 10, 25)
# Fração do desvio inicial x-mu que permanece até a eleição (99% dissipado).
REVERSAO_RESTANTE = 0.01

INSTITUTOS_CANONICOS = {
    "Datafolha", "Paraná Pesquisas", "Genial/Quaest",
    "AtlasIntel", "Futura/Apex",
}
NOME_NORMALIZADO = {"Apex/Futura": "Futura/Apex"}

EM_MAX_ITER = 2000
# Tolerância relativa da log-verossimilhança. Com poucas pesquisas, exigir
# variação absoluta quase nula prolonga o EM sem mudança material no ajuste.
EM_TOL = 1e-6
SIGMA_X2_INIT = 0.005
Q_MU_INIT = 0.0001
VARIANCE_MIN = 1e-12
OBSERVATION_VARIANCE_MULTIPLIER = 2.0
PROCESS_COV_INIT = np.diag([SIGMA_X2_INIT, Q_MU_INIT])

S0_MEAN = np.array([0.0, 0.0])
S0_COV = np.diag([10.0, 10.0])

_MESES = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
}


def _parse_date(day: str, month_str: str, year: str) -> datetime | None:
    month = _MESES.get(month_str.strip().lower()[:3])
    if month is None:
        return None
    try:
        return datetime(int(year), month, int(day))
    except (TypeError, ValueError):
        return None


def _midpoint(date_range: str, year: str) -> datetime | None:
    """Representa o período de campo por seu ponto médio."""
    dates = []
    for part in re.split(r"\s*[–-]\s*", str(date_range).strip()):
        match = re.match(r"(\d+)\s+(\w+)", part.strip())
        if match:
            parsed = _parse_date(match.group(1), match.group(2), year)
            if parsed is not None:
                dates.append(parsed)
    if len(dates) == 2:
        return dates[0] + timedelta(days=(dates[1] - dates[0]).days / 2)
    return dates[0] if dates else None


def _nonresponse_fraction(rec: dict) -> float | None:
    """Obtém w+u+a quando o snapshot oferece uma categoria agregada."""
    for key in (
        "Brancos, Nulos, Indecisos e Abstenções %",
        "Brancos, Nulos, Indecisos e Absentos %",
        "Indecisos e Abstenções %",
        "Indecisos e Absentos %",
    ):
        try:
            return float(rec[key]) / 100.0
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _prepare_record(rec: dict) -> dict | None:
    nome = NOME_NORMALIZADO.get(rec.get("Contratante"), rec.get("Contratante"))
    if nome not in INSTITUTOS_CANONICOS:
        return None

    date = _midpoint(rec.get("Data(s) de Pesquisa", ""), rec.get("Ano", ""))
    if date is None:
        return None

    try:
        lula_pct = float(rec["Lula (PT) %"])
        flavio_pct = float(rec["Flávio (PL) %"])
        n = float(rec["Tamanho da Amostra"])
    except (KeyError, TypeError, ValueError):
        return None

    nonresponse = _nonresponse_fraction(rec)
    q_t = 1.0 - nonresponse if nonresponse is not None else (lula_pct + flavio_pct) / 100.0
    # Em disputas de dois candidatos, a soma observada é a alternativa mais
    # consistente quando categorias agregadas têm arredondamento diferente.
    q_candidates = (lula_pct + flavio_pct) / 100.0
    if abs(q_t - q_candidates) > 0.02:
        q_t = q_candidates

    if n <= 0 or not (0.0 < q_t <= 1.0):
        return None
    p_t = (flavio_pct / 100.0) / q_t
    if not (0.0 < p_t < 1.0):
        return None

    y_t = float(np.log(p_t / (1.0 - p_t)))
    R_t = float(
        OBSERVATION_VARIANCE_MULTIPLIER
        / (n * q_t * p_t * (1.0 - p_t))
    )
    return {
        "date": date,
        "instituto": nome,
        "y": y_t,
        "R": R_t,
        "p": p_t,
        "q": q_t,
        "n": n,
        "lula_pct": lula_pct,
        "flavio_pct": flavio_pct,
    }


def load_and_prepare(snapshot_file: str = SNAPSHOT_FILE) -> dict:
    """Constrói (y_t, R_t, Delta_t) segundo a primeira seção da nota."""
    with open(snapshot_file, encoding="utf-8") as file:
        snapshot = json.load(file)

    rows = []
    for rec in snapshot.get("records", {}).values():
        row = _prepare_record(rec)
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError("Nenhuma pesquisa válida encontrada no snapshot.")

    rows.sort(key=lambda row: (row["date"], row["instituto"], row["y"]))
    dates = [row["date"] for row in rows]
    delta_t = np.zeros(len(rows))
    for t in range(1, len(rows)):
        delta_t[t] = max((dates[t] - dates[t - 1]).total_seconds() / 86400.0, 0.0)

    institutos = sorted({row["instituto"] for row in rows})
    inst2idx = {inst: index for index, inst in enumerate(institutos)}
    inst_idx = np.array([inst2idx[row["instituto"]] for row in rows], dtype=int)

    print(f"\n{'─' * 68}")
    print(f"  Dados: {len(rows)} observações | {len(institutos)} institutos")
    print(f"  Período: {dates[0].date()} -> {dates[-1].date()}")
    for inst in institutos:
        print(f"  - {inst}: {int(np.sum(inst_idx == inst2idx[inst]))} pesquisas")

    return {
        "y": np.array([row["y"] for row in rows]),
        "R": np.array([row["R"] for row in rows]),
        "delta_t": delta_t,
        "dates": dates,
        "inst_idx": inst_idx,
        "institutos": institutos,
        "K": len(institutos),
        "raw": rows,
    }


def calibrate_lambda(
    start_date: datetime,
    election_date: datetime = ELECTION_DATE,
    residual_fraction: float = REVERSAO_RESTANTE,
) -> tuple[float, float]:
    """Calcula lambda=-log(r)/D0 e devolve também D0 em dias."""
    if not (0.0 < residual_fraction < 1.0):
        raise ValueError("residual_fraction deve pertencer a (0, 1).")
    D0 = (election_date - start_date).total_seconds() / 86400.0
    if D0 <= 0.0:
        raise ValueError("A data da eleição deve ser posterior à primeira pesquisa.")
    return float(-np.log(residual_fraction) / D0), float(D0)


def transition_matrices(delta: float, lambda_: float, process_cov: np.ndarray):
    """Matrizes A_t e G_t, incluindo a covariância entre x_t e mu_t."""
    if delta < 0.0:
        raise ValueError("Delta_t não pode ser negativo.")
    phi = float(np.exp(-lambda_ * delta))
    if lambda_ > 1e-12:
        a_t = float(-np.expm1(-2.0 * lambda_ * delta) / (2.0 * lambda_))
    else:
        a_t = float(delta)
    A_t = np.array([[phi, 1.0 - phi], [0.0, 1.0]])
    scales = np.sqrt(np.array([a_t, delta], dtype=float))
    G_t = process_cov * np.outer(scales, scales)
    G_t = 0.5 * (G_t + G_t.T)
    return A_t, G_t, scales


def _nearest_process_cov(matrix: np.ndarray) -> np.ndarray:
    """Projeta uma estimativa simétrica no cone positivo definido."""
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, VARIANCE_MIN)
    projected = (eigenvectors * eigenvalues) @ eigenvectors.T
    return 0.5 * (projected + projected.T)


def _full_biases(free_biases: np.ndarray, K: int) -> np.ndarray:
    full = np.zeros(K)
    if K > 1:
        full[:-1] = free_biases
        full[-1] = -float(np.sum(free_biases))
    return full


def kalman_filter(
    y: np.ndarray,
    R: np.ndarray,
    delta_t: np.ndarray,
    lambda_: float,
    process_cov: np.ndarray,
    b_full: np.ndarray,
    inst_idx: np.ndarray,
):
    """Filtro bidimensional usado no E-step; vieses entram como offset."""
    T = len(y)
    means_pred = np.empty((T, 2))
    covs_pred = np.empty((T, 2, 2))
    means_filt = np.empty((T, 2))
    covs_filt = np.empty((T, 2, 2))
    transitions = np.repeat(np.eye(2)[None, :, :], T, axis=0)
    log_lik = 0.0
    H = np.array([1.0, 0.0])
    identity = np.eye(2)

    for t in range(T):
        if t == 0:
            mean_pred = S0_MEAN.copy()
            cov_pred = S0_COV.copy()
        else:
            A_t, G_t, _ = transition_matrices(delta_t[t], lambda_, process_cov)
            transitions[t] = A_t
            mean_pred = A_t @ means_filt[t - 1]
            cov_pred = A_t @ covs_filt[t - 1] @ A_t.T + G_t

        innovation = y[t] - b_full[inst_idx[t]] - H @ mean_pred
        innovation_var = float(H @ cov_pred @ H + R[t])
        if not np.isfinite(innovation_var) or innovation_var <= 0.0:
            raise FloatingPointError("Variância de inovação inválida no filtro.")
        gain = cov_pred @ H / innovation_var
        mean_filt = mean_pred + gain * innovation
        I_KH = identity - np.outer(gain, H)
        cov_filt = I_KH @ cov_pred @ I_KH.T + np.outer(gain, gain) * R[t]
        cov_filt = 0.5 * (cov_filt + cov_filt.T)

        means_pred[t], covs_pred[t] = mean_pred, cov_pred
        means_filt[t], covs_filt[t] = mean_filt, cov_filt
        log_lik -= 0.5 * (
            np.log(2.0 * np.pi * innovation_var) + innovation**2 / innovation_var
        )

    return means_pred, covs_pred, means_filt, covs_filt, transitions, float(log_lik)


def rts_smoother(means_pred, covs_pred, means_filt, covs_filt, transitions):
    """Suavizador RTS e covariâncias cruzadas P_{t,t-1|T}."""
    T = len(means_filt)
    means_smooth = means_filt.copy()
    covs_smooth = covs_filt.copy()
    cross_covs = np.zeros((T, 2, 2))
    smoother_gains = np.zeros((max(T - 1, 0), 2, 2))

    for t in range(T - 2, -1, -1):
        A_next = transitions[t + 1]
        gain = np.linalg.solve(covs_pred[t + 1], A_next @ covs_filt[t]).T
        smoother_gains[t] = gain
        means_smooth[t] = means_filt[t] + gain @ (
            means_smooth[t + 1] - means_pred[t + 1]
        )
        covs_smooth[t] = covs_filt[t] + gain @ (
            covs_smooth[t + 1] - covs_pred[t + 1]
        ) @ gain.T
        covs_smooth[t] = 0.5 * (covs_smooth[t] + covs_smooth[t].T)

    for t in range(1, T):
        cross_covs[t] = covs_smooth[t] @ smoother_gains[t - 1].T
    return means_smooth, covs_smooth, cross_covs


def m_step(
    y, R, delta_t, lambda_, means_smooth, covs_smooth, cross_covs, inst_idx, K
):
    """Atualiza a matriz de covariância do processo e os vieses."""
    normalized_covs = []
    for t in range(1, len(y)):
        if delta_t[t] <= 0.0:
            # Pesquisas no mesmo instante são atualizações observacionais,
            # não transições informativas sobre variância de processo.
            continue
        A_t, _, scales = transition_matrices(
            delta_t[t], lambda_, np.zeros((2, 2))
        )
        residual_mean = means_smooth[t] - A_t @ means_smooth[t - 1]
        omega_t = (
            covs_smooth[t]
            + A_t @ covs_smooth[t - 1] @ A_t.T
            - cross_covs[t] @ A_t.T
            - A_t @ cross_covs[t].T
            + np.outer(residual_mean, residual_mean)
        )
        inverse_scales = np.diag(1.0 / scales)
        normalized_covs.append(inverse_scales @ omega_t @ inverse_scales)

    if not normalized_covs:
        raise ValueError("São necessárias pesquisas em pelo menos duas datas distintas.")
    process_cov = _nearest_process_cov(np.mean(normalized_covs, axis=0))

    if K == 1:
        return process_cov, np.zeros(0)

    residual = y - means_smooth[:, 0]
    weights = 1.0 / R
    design = np.zeros((len(y), K - 1))
    for t, index in enumerate(inst_idx):
        if index < K - 1:
            design[t, index] = 1.0
        else:
            design[t, :] = -1.0
    normal = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * residual)
    free_biases = np.linalg.solve(normal + np.eye(K - 1) * 1e-12, rhs)
    return process_cov, free_biases


def run_em(data: dict, lambda_: float):
    """Executa EM até estabilização da verossimilhança marginal."""
    y, R = data["y"], data["R"]
    delta_t, inst_idx, K = data["delta_t"], data["inst_idx"], data["K"]
    process_cov = PROCESS_COV_INIT.copy()
    free_biases = np.zeros(max(K - 1, 0))
    previous_ll = -np.inf
    converged = False

    print(f"\n  EM bidimensional: T={len(y)}, K={K}, lambda={lambda_:.8f}/dia")
    print(
        f"  {'Iter':>5} {'Log-Lik':>14} {'Delta':>13} "
        f"{'sigma_x':>11} {'sigma_mu':>11} {'rho':>9}"
    )
    for iteration in range(1, EM_MAX_ITER + 1):
        b_full = _full_biases(free_biases, K)
        filtered = kalman_filter(
            y, R, delta_t, lambda_, process_cov, b_full, inst_idx
        )
        means_pred, covs_pred, means_filt, covs_filt, transitions, log_lik = filtered
        means_smooth, covs_smooth, cross_covs = rts_smoother(
            means_pred, covs_pred, means_filt, covs_filt, transitions
        )
        new_process_cov, new_biases = m_step(
            y, R, delta_t, lambda_, means_smooth, covs_smooth,
            cross_covs, inst_idx, K,
        )
        delta_ll = log_lik - previous_ll
        new_rho = new_process_cov[0, 1] / np.sqrt(
            new_process_cov[0, 0] * new_process_cov[1, 1]
        )
        if iteration <= 5 or iteration % 25 == 0:
            print(
                f"  {iteration:5d} {log_lik:14.5f} {delta_ll:13.5g} "
                f"{np.sqrt(new_process_cov[0, 0]):11.6f} "
                f"{np.sqrt(new_process_cov[1, 1]):11.6f} {new_rho:9.5f}"
            )

        process_cov, free_biases = new_process_cov, new_biases
        ll_scale = 1.0 + abs(previous_ll)
        if iteration > 1 and abs(delta_ll) <= EM_TOL * ll_scale:
            converged = True
            print(
                f"  Convergência em {iteration} iterações "
                f"(|Delta LL|/(1+|LL|)={abs(delta_ll) / ll_scale:.2e})."
            )
            break
        previous_ll = log_lik

    b_full = _full_biases(free_biases, K)
    filtered = kalman_filter(y, R, delta_t, lambda_, process_cov, b_full, inst_idx)
    means_pred, covs_pred, means_filt, covs_filt, transitions, log_lik = filtered
    means_smooth, covs_smooth, _ = rts_smoother(
        means_pred, covs_pred, means_filt, covs_filt, transitions
    )
    return {
        "process_cov": process_cov,
        "b_full": b_full,
        "means_smooth": means_smooth,
        "covs_smooth": covs_smooth,
        "means_filt": means_filt,
        "covs_filt": covs_filt,
        "log_lik": log_lik,
        "n_iter": iteration,
        "converged": converged,
    }


def _build_H(inst_index: int, K: int) -> np.ndarray:
    H = np.zeros(K + 1)
    H[0] = 1.0
    if K > 1:
        if inst_index < K - 1:
            H[2 + inst_index] = 1.0
        else:
            H[2:] = -1.0
    return H


def kalman_augmented(data: dict, lambda_: float, process_cov: np.ndarray, b_full):
    """Filtro do vetor [x, mu, b_1, ..., b_{K-1}] com forma de Joseph."""
    K, T = data["K"], len(data["y"])
    dim = K + 1
    mean = np.zeros(dim)
    mean[:2] = S0_MEAN
    if K > 1:
        mean[2:] = b_full[:-1]
    cov = np.zeros((dim, dim))
    cov[:2, :2] = S0_COV
    means = np.empty((T, dim))
    covs = np.empty((T, dim, dim))

    for t in range(T):
        if t == 0:
            mean_pred, cov_pred = mean.copy(), cov.copy()
        else:
            A_t, G_t, _ = transition_matrices(
                data["delta_t"][t], lambda_, process_cov
            )
            F_t = np.eye(dim)
            F_t[:2, :2] = A_t
            Q_t = np.zeros((dim, dim))
            Q_t[:2, :2] = G_t
            mean_pred = F_t @ mean
            cov_pred = F_t @ cov @ F_t.T + Q_t

        H_t = _build_H(data["inst_idx"][t], K)
        innovation = data["y"][t] - H_t @ mean_pred
        S_t = float(H_t @ cov_pred @ H_t + data["R"][t])
        gain = cov_pred @ H_t / S_t
        mean = mean_pred + gain * innovation
        I_KH = np.eye(dim) - np.outer(gain, H_t)
        cov = I_KH @ cov_pred @ I_KH.T + np.outer(gain, gain) * data["R"][t]
        cov = 0.5 * (cov + cov.T)
        means[t], covs[t] = mean, cov
    return means, covs


def _logistic(value):
    value = np.asarray(value)
    return np.where(value >= 0, 1.0 / (1.0 + np.exp(-value)), np.exp(value) / (1.0 + np.exp(value)))


def _probability_interval(mean: float, variance: float) -> tuple[float, float]:
    radius = 1.959964 * np.sqrt(max(float(variance), 0.0))
    return float(_logistic(mean - radius)), float(_logistic(mean + radius))


def run(
    snapshot_file: str = SNAPSHOT_FILE,
    output_file: str = OUTPUT_FILE,
    election_date: datetime = ELECTION_DATE,
    residual_fraction: float = REVERSAO_RESTANTE,
) -> dict:
    """Estima o modelo e grava parâmetros, estados curtos e estados persistentes."""
    print("\n" + "=" * 68)
    print("  AGREGADOR DE PESQUISAS - REVERSÃO À MÉDIA + RTS")
    print("=" * 68)
    data = load_and_prepare(snapshot_file)
    lambda_, D0 = calibrate_lambda(data["dates"][0], election_date, residual_fraction)
    print(
        f"\n  Calibração: eleição={election_date.date()}, D0={D0:.1f} dias, "
        f"r={residual_fraction:g}, lambda={lambda_:.8f}/dia, "
        f"meia-vida={np.log(2.0) / lambda_:.1f} dias"
    )

    estimates = run_em(data, lambda_)
    online_means, online_covs = kalman_augmented(
        data, lambda_, estimates["process_cov"], estimates["b_full"]
    )

    process_cov = estimates["process_cov"]
    process_corr = process_cov[0, 1] / np.sqrt(process_cov[0, 0] * process_cov[1, 1])
    print("\n  Parâmetros estimados:")
    print(f"  Q_processo=\n{process_cov}")
    print(f"  corr(x, mu)={process_corr:+.6f}")
    for inst, bias in zip(data["institutos"], estimates["b_full"]):
        print(f"  b[{inst}]={bias:+.6f}")

    states = []
    for t, row in enumerate(data["raw"]):
        x_s, mu_s = estimates["means_smooth"][t]
        P_s = estimates["covs_smooth"][t]
        x_f, mu_f = online_means[t, :2]
        P_f = online_covs[t, :2, :2]
        x_s_lo, x_s_hi = _probability_interval(x_s, P_s[0, 0])
        mu_s_lo, mu_s_hi = _probability_interval(mu_s, P_s[1, 1])
        x_f_lo, x_f_hi = _probability_interval(x_f, P_f[0, 0])
        mu_f_lo, mu_f_hi = _probability_interval(mu_f, P_f[1, 1])
        states.append({
            "t": t,
            "data": row["date"].strftime("%Y-%m-%d"),
            "instituto": row["instituto"],
            "y_obs": round(float(data["y"][t]), 8),
            "R_obs": round(float(data["R"][t]), 8),
            "x_smooth": round(float(x_s), 8),
            "mu_smooth": round(float(mu_s), 8),
            "P_smooth": [[round(float(v), 8) for v in line] for line in P_s],
            "p_curto_smooth": round(float(_logistic(x_s)), 6),
            "p_curto_smooth_ic95_lo": round(x_s_lo, 6),
            "p_curto_smooth_ic95_hi": round(x_s_hi, 6),
            "p_longo_smooth": round(float(_logistic(mu_s)), 6),
            "p_longo_smooth_ic95_lo": round(mu_s_lo, 6),
            "p_longo_smooth_ic95_hi": round(mu_s_hi, 6),
            "x_filt": round(float(x_f), 8),
            "mu_filt": round(float(mu_f), 8),
            "P_filt": [[round(float(v), 8) for v in line] for line in P_f],
            "p_curto_filt": round(float(_logistic(x_f)), 6),
            "p_curto_filt_ic95_lo": round(x_f_lo, 6),
            "p_curto_filt_ic95_hi": round(x_f_hi, 6),
            "p_longo_filt": round(float(_logistic(mu_f)), 6),
            "p_longo_filt_ic95_lo": round(mu_f_lo, 6),
            "p_longo_filt_ic95_hi": round(mu_f_hi, 6),
        })

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "fonte": str(snapshot_file),
        "nota_tecnica": "Agregador_de_Pesquisas_Eleitorais_Reversao_Media.tex",
        "modelo": "reversao_media_ou_componente_persistente_covariancia_v2",
        "parametros": {
            "data_eleicao": election_date.strftime("%Y-%m-%d"),
            "data_inicial_calibracao": data["dates"][0].strftime("%Y-%m-%d"),
            "D0_dias": round(D0, 6),
            "fracao_desvio_restante": residual_fraction,
            "lambda_por_dia": round(lambda_, 12),
            "meia_vida_dias": round(float(np.log(2.0) / lambda_), 6),
            "matriz_covariancia_processo": [
                [round(float(value), 12) for value in row]
                for row in process_cov
            ],
            "sigma_x2": round(float(process_cov[0, 0]), 12),
            "sigma_x": round(float(np.sqrt(process_cov[0, 0])), 10),
            "q_mu": round(float(process_cov[1, 1]), 12),
            "sigma_mu": round(float(np.sqrt(process_cov[1, 1])), 10),
            "cov_x_mu": round(float(process_cov[0, 1]), 12),
            "corr_x_mu": round(float(process_corr), 10),
            "multiplicador_variancia_observacional": OBSERVATION_VARIANCE_MULTIPLIER,
            "log_verossimilhanca": round(float(estimates["log_lik"]), 8),
            "n_iteracoes_em": estimates["n_iter"],
            "convergiu": estimates["converged"],
            "vieses": {
                inst: round(float(bias), 10)
                for inst, bias in zip(data["institutos"], estimates["b_full"])
            },
        },
        "estados": states,
    }
    output_path = Path(output_file)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(f"\n  Resultado salvo em '{output_path}'.")
    return result


if __name__ == "__main__":
    resultado = run()
