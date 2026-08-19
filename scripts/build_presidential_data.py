"""Gera dados e gráficos presidenciais estáticos para o site.

O modelo segue a nota ``Agregador_de_Pesquisas_Eleitorais_Reversao_Media.tex``:
estado acoplado ``[x_t, mu_t]``, transição OU exata, EM com suavizador RTS para
estimar a matriz de covariância do processo e nova passagem do filtro causal com os
parâmetros finais. Não são usados vieses por instituto.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import Suavizador_de_Kalman as kalman


PROJECT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_FILE = PROJECT_DIR / "snapshot_pesquisas.json"
DATA_FILE = PROJECT_DIR / "app" / "data" / "presidente.json"
PLOTS_DIR = PROJECT_DIR / "public" / "plots"

ELECTION_DATES = {
    "Primeiro turno": datetime(2026, 10, 4),
    "Segundo turno": datetime(2026, 10, 25),
}
CATEGORIES = {
    "Primeiro turno": (
        ("lula", "Lula", "Lula (PT) %"),
        ("flavio", "Flávio Bolsonaro", "Flávio (PL) %"),
        ("outros", "Outros", "Outros %"),
    ),
    "Segundo turno": (
        ("lula", "Lula", "Lula (PT) %"),
        ("flavio", "Flávio Bolsonaro", "Flávio (PL) %"),
    ),
}
COLORS = {"lula": "#B5423C", "flavio": "#1F5A82", "outros": "#9A8F7A"}
MONTHS = {
    "jan": 1, "janeiro": 1, "fev": 2, "fevereiro": 2,
    "mar": 3, "marco": 3, "abr": 4, "abril": 4,
    "mai": 5, "maio": 5, "jun": 6, "junho": 6,
    "jul": 7, "julho": 7, "ago": 8, "agosto": 8,
    "set": 9, "setembro": 9, "out": 10, "outubro": 10,
    "nov": 11, "novembro": 11, "dez": 12, "dezembro": 12,
}
RESIDUAL_FRACTION = 0.01
EM_MAX_ITER = 2000
EM_TOL = 1e-5
MC_DRAWS = 1000
EPSILON = 1e-9


def _ascii(text: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def _field_midpoint(period: str, year: int) -> datetime | None:
    text = _ascii(period).replace("º", "").replace("°", "")
    tokens = []
    for match in re.finditer(r"(?<!\d)(\d{1,2})\s*(?:de\s+)?([a-z]+)?", text):
        day = int(match.group(1))
        month_text = (match.group(2) or "").strip(". ")
        month = MONTHS.get(month_text) or MONTHS.get(month_text[:3])
        if 1 <= day <= 31:
            tokens.append([day, month])
    if not tokens:
        return None
    for index, token in enumerate(tokens):
        if token[1] is not None:
            continue
        right = next((item[1] for item in tokens[index + 1:] if item[1] is not None), None)
        left = next((item[1] for item in reversed(tokens[:index]) if item[1] is not None), None)
        token[1] = right or left
    dates = []
    for day, month in tokens[:2]:
        if month is None:
            continue
        try:
            dates.append(datetime(year, int(month), int(day)))
        except ValueError:
            continue
    if not dates:
        return None
    if len(dates) == 1:
        return dates[0]
    return dates[0] + (dates[1] - dates[0]) / 2


def _float(value: Any) -> float | None:
    if value is None:
        return None
    text = re.sub(r"[^0-9,.-]", "", str(value)).replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _load_records() -> list[dict[str, Any]]:
    payload = json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
    records = list(payload.get("records", {}).values())
    if not records:
        raise ValueError("O snapshot não contém pesquisas presidenciais.")
    return records


def _prepare_turn(records: list[dict[str, Any]], turn: str) -> dict[str, list[dict[str, Any]]]:
    prepared = {category_id: [] for category_id, _, _ in CATEGORIES[turn]}
    columns = {category_id: source for category_id, _, source in CATEGORIES[turn]}
    for record in records:
        if record.get("Turno") != turn or str(record.get("Ano")) != "2026":
            continue
        date = _field_midpoint(str(record.get("Data(s) de Pesquisa", "")), 2026)
        sample = _float(record.get("Tamanho da Amostra"))
        values = {category_id: _float(record.get(source)) for category_id, source in columns.items()}
        if date is None or sample is None or sample <= 1 or any(value is None for value in values.values()):
            continue
        valid_total = sum(max(float(value), 0.0) for value in values.values())
        if valid_total <= 0:
            continue
        valid_fraction = min(max(valid_total / 100.0, EPSILON), 1.0)
        effective_n = sample * valid_fraction
        institute = str(record.get("Contratante") or "Instituto não informado")
        for category_id, value in values.items():
            raw_p = max(float(value), 0.0) / valid_total
            p = (raw_p * effective_n + 0.5) / (effective_n + 1.0)
            p = min(max(p, EPSILON), 1.0 - EPSILON)
            prepared[category_id].append({
                "date": date,
                "date_str": date.strftime("%Y-%m-%d"),
                "instituto": institute,
                "cenario": str(record.get("Cenário") or ""),
                "valor_fonte_pct": float(value),
                "percentual_validos": raw_p * 100.0,
                "y": float(np.log(p / (1.0 - p))),
                "R": float(
                    kalman.OBSERVATION_VARIANCE_MULTIPLIER
                    / (effective_n * p * (1.0 - p))
                ),
            })
    for rows in prepared.values():
        rows.sort(key=lambda row: (row["date"], row["instituto"], row["cenario"], row["y"]))
    return prepared


def _kalman_data(rows: list[dict[str, Any]]) -> dict[str, Any]:
    delta_t = np.zeros(len(rows))
    for index in range(1, len(rows)):
        delta_t[index] = max((rows[index]["date"] - rows[index - 1]["date"]).total_seconds() / 86400.0, 0.0)
    return {
        "y": np.array([row["y"] for row in rows], dtype=float),
        "R": np.array([row["R"] for row in rows], dtype=float),
        "delta_t": delta_t,
        "dates": [row["date"] for row in rows],
        "inst_idx": np.zeros(len(rows), dtype=int),
    }


def _estimate(data: dict[str, Any], election_date: datetime) -> dict[str, Any]:
    if len({date.date() for date in data["dates"]}) < 2:
        raise ValueError("São necessárias pesquisas em pelo menos duas datas.")
    lambda_, calibration_days = kalman.calibrate_lambda(
        data["dates"][0], election_date, RESIDUAL_FRACTION
    )
    process_cov = kalman.PROCESS_COV_INIT.copy()
    zero_offset = np.zeros(1)
    previous_ll = -np.inf
    converged = False
    for iteration in range(1, EM_MAX_ITER + 1):
        filtered = kalman.kalman_filter(
            data["y"], data["R"], data["delta_t"], lambda_, process_cov,
            zero_offset, data["inst_idx"],
        )
        means_pred, covs_pred, means_filt, covs_filt, transitions, log_lik = filtered
        means_smooth, covs_smooth, cross_covs = kalman.rts_smoother(
            means_pred, covs_pred, means_filt, covs_filt, transitions
        )
        process_cov_new, _ = kalman.m_step(
            data["y"], data["R"], data["delta_t"], lambda_, means_smooth,
            covs_smooth, cross_covs, data["inst_idx"], 1,
        )
        delta_ll = log_lik - previous_ll
        scale = 1.0 + abs(previous_ll)
        process_cov = process_cov_new
        if iteration > 1 and abs(delta_ll) <= EM_TOL * scale:
            converged = True
            break
        previous_ll = log_lik

    filtered = kalman.kalman_filter(
        data["y"], data["R"], data["delta_t"], lambda_, process_cov,
        zero_offset, data["inst_idx"],
    )
    means_pred, covs_pred, means_filt, covs_filt, transitions, log_lik = filtered
    means_smooth, covs_smooth, _ = kalman.rts_smoother(
        means_pred, covs_pred, means_filt, covs_filt, transitions
    )
    return {
        "lambda": lambda_, "calibration_days": calibration_days,
        "process_cov": process_cov,
        "means_filt": means_filt, "covs_filt": covs_filt,
        "means_smooth": means_smooth, "covs_smooth": covs_smooth,
        "log_lik": log_lik, "iteration": iteration, "converged": converged,
    }


def _logistic(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "big")


def _summaries(
    turn: str, category_ids: list[str], series: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    dates = sorted({row["date"] for item in series.values() for row in item["rows"]})
    positions = {category_id: 0 for category_id in category_ids}
    trajectory = []
    for target in dates:
        entry: dict[str, Any] = {"data": target.strftime("%Y-%m-%d")}
        for horizon_name, horizon_index in (("curto", 0), ("longo", 1)):
            entry[horizon_name] = {}
            for method_name, estimate_key, covariance_key in (
                ("filtrado", "means_filt", "covs_filt"),
                ("suavizado", "means_smooth", "covs_smooth"),
            ):
                means, variances = [], []
                for category_id in category_ids:
                    item = series[category_id]
                    row_dates = [row["date"] for row in item["rows"]]
                    index = max(np.searchsorted(row_dates, target, side="right") - 1, 0)
                    positions[category_id] = int(index)
                    estimates = item["estimate"]
                    means.append(float(estimates[estimate_key][index, horizon_index]))
                    variances.append(max(float(estimates[covariance_key][index, horizon_index, horizon_index]), 0.0))
                raw = _logistic(np.array(means))
                point = raw / raw.sum()
                rng = np.random.default_rng(_seed(turn, entry["data"], horizon_name, method_name))
                samples = rng.normal(np.array(means), np.sqrt(np.array(variances)), size=(MC_DRAWS, len(category_ids)))
                probabilities = _logistic(samples)
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                low, high = np.quantile(probabilities, [0.025, 0.975], axis=0)
                entry[horizon_name][method_name] = {
                    category_id: {
                        "estimativa_pct": round(float(point[index] * 100), 3),
                        "ic95_lo_pct": round(float(low[index] * 100), 3),
                        "ic95_hi_pct": round(float(high[index] * 100), 3),
                    }
                    for index, category_id in enumerate(category_ids)
                }
        trajectory.append(entry)
    return trajectory


def _plot_turn(turn: str, result: dict[str, Any], output: Path) -> None:
    trajectory = result["trajetoria"]
    dates = [datetime.strptime(item["data"], "%Y-%m-%d") for item in trajectory]
    categories = result["categorias"]
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9.5), sharex=True)
    fig.patch.set_facecolor("#F4F0E7")
    for ax, key, title in (
        (axes[0], "curto", "Curto prazo · intenção corrente xₜ"),
        (axes[1], "longo", "Longo prazo · componente persistente μₜ"),
    ):
        ax.set_facecolor("#FFFCF6")
        for category in categories:
            category_id = category["id"]
            color = COLORS[category_id]
            filtered = [item[key]["filtrado"][category_id] for item in trajectory]
            smoothed = [item[key]["suavizado"][category_id] for item in trajectory]
            ax.fill_between(
                dates,
                [item["ic95_lo_pct"] for item in filtered],
                [item["ic95_hi_pct"] for item in filtered],
                color=color, alpha=0.10,
            )
            ax.plot(dates, [item["estimativa_pct"] for item in filtered], color=color, lw=2.4)
            ax.plot(
                dates, [item["estimativa_pct"] for item in smoothed],
                color=color, lw=1.0, ls=":", alpha=0.7,
            )
            if key == "curto":
                ax.scatter(
                    [datetime.strptime(item["data"], "%Y-%m-%d") for item in category["observacoes"]],
                    [item["percentual_validos"] for item in category["observacoes"]],
                    color=color, s=18, alpha=0.38, edgecolors="white", linewidths=0.3,
                )
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("Votos válidos (%)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#D9D2C3", ls="--", alpha=0.55)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
    fig.suptitle(f"Presidente · {turn}", fontsize=17, fontweight="bold", y=0.985)
    handles = [
        Line2D([0], [0], color=COLORS[item["id"]], lw=3, label=item["nome"])
        for item in categories
    ] + [
        Line2D([0], [0], color="#262626", lw=2.4, label="Filtro causal pós-EM"),
        Line2D([0], [0], color="#262626", lw=1.2, ls=":", label="RTS suavizado"),
        Line2D([0], [0], color="#777", marker="o", lw=0, label="Pesquisa"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), frameon=False, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", dir=output.parent, delete=False) as file:
            temporary = Path(file.name)
        fig.savefig(temporary, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
        temporary.replace(output)
    finally:
        plt.close(fig)
        if temporary and temporary.exists():
            temporary.unlink()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False
        ) as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
            temporary = Path(file.name)
        temporary.replace(path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def build() -> dict[str, Any]:
    records = _load_records()
    turns = {}
    for turn, categories in CATEGORIES.items():
        prepared = _prepare_turn(records, turn)
        series = {}
        category_output = []
        for category_id, name, _ in categories:
            rows = prepared[category_id]
            data = _kalman_data(rows)
            estimate = _estimate(data, ELECTION_DATES[turn])
            series[category_id] = {"rows": rows, "estimate": estimate}
            category_output.append({
                "id": category_id,
                "nome": name,
                "cor": COLORS[category_id],
                "observacoes": [{
                    "data": row["date_str"],
                    "instituto": row["instituto"],
                    "percentual_validos": round(row["percentual_validos"], 3),
                } for row in rows],
                "parametros": {
                    "lambda_por_dia": round(float(estimate["lambda"]), 12),
                    "matriz_covariancia_processo": [
                        [round(float(value), 12) for value in row]
                        for row in estimate["process_cov"]
                    ],
                    "sigma_x2": round(float(estimate["process_cov"][0, 0]), 12),
                    "q_mu": round(float(estimate["process_cov"][1, 1]), 12),
                    "cov_x_mu": round(float(estimate["process_cov"][0, 1]), 12),
                    "corr_x_mu": round(float(
                        estimate["process_cov"][0, 1]
                        / np.sqrt(
                            estimate["process_cov"][0, 0]
                            * estimate["process_cov"][1, 1]
                        )
                    ), 10),
                    "multiplicador_variancia_observacional": (
                        kalman.OBSERVATION_VARIANCE_MULTIPLIER
                    ),
                    "log_verossimilhanca": round(float(estimate["log_lik"]), 8),
                    "iteracoes_em": int(estimate["iteration"]),
                    "convergiu": bool(estimate["converged"]),
                    "efeitos_instituto": "não utilizados",
                },
            })
        category_ids = [category_id for category_id, _, _ in categories]
        trajectory = _summaries(turn, category_ids, series)
        turn_id = "primeiro_turno" if turn == "Primeiro turno" else "segundo_turno"
        turns[turn_id] = {
            "titulo": turn,
            "data_eleicao": ELECTION_DATES[turn].strftime("%Y-%m-%d"),
            "quantidade_pesquisas": len({
                (row["date_str"], row["instituto"], row["cenario"])
                for rows in prepared.values() for row in rows
            }),
            "data_ultima_pesquisa": trajectory[-1]["data"],
            "categorias": category_output,
            "estimativa_atual": trajectory[-1],
            "trajetoria": trajectory,
            "grafico": f"/plots/presidente-{turn_id.replace('_', '-')}.png",
        }
        _plot_turn(
            turn, turns[turn_id],
            PLOTS_DIR / f"presidente-{turn_id.replace('_', '-')}.png",
        )

    payload = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": "estado_acoplado_covariancia_completa_em_rts_sem_vies_instituto",
        "fonte": "Pesquisas presidenciais de 2026 na Wikipédia",
        "turnos": turns,
    }
    _atomic_json(payload, DATA_FILE)
    print(
        f"Dados gerados: {DATA_FILE}\n"
        f"Primeiro turno: {turns['primeiro_turno']['quantidade_pesquisas']} pesquisas\n"
        f"Segundo turno: {turns['segundo_turno']['quantidade_pesquisas']} pesquisas"
    )
    return payload


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    build()
