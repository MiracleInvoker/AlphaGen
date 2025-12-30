import os
import pickle
from datetime import datetime
from itertools import groupby

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from rich.console import Console

import ace_lib as ace

console = Console()


def get_stored_session():
    brain_session = ace.SingleSession()

    if os.path.exists("session.pkl"):
        try:
            with open("session.pkl", "rb") as f:
                session_data = pickle.load(f)

                brain_session.cookies.update(session_data["cookies"])
                brain_session.headers.update(session_data["headers"])

            console.print("Loaded stored session from disk.", style="yellow")

        except Exception as e:
            console.print(f"Failed to load session: {e}", style="red")

    time_to_live = ace.check_session_timeout(brain_session)

    if time_to_live > 2000:
        console.print(
            f"Session is valid. Expires in {time_to_live} seconds.", style="yellow"
        )
        return brain_session

    console.print("Session expired or missing. Logging in...", style="yellow")
    brain_session = ace.start_session()

    with open("session.pkl", "wb") as f:
        pickle.dump(
            {"cookies": brain_session.cookies, "headers": brain_session.headers}, f
        )
    console.print("New session saved to disk.", style="yellow")

    return brain_session


def fix_fastexpr(alpha_expression):
    alpha_expression = (
        alpha_expression.replace("\n", "").replace("; ", ";").replace(";", ";\n")
    )

    alpha_expression = alpha_expression.strip()
    return alpha_expression


def get_check(checks, name):

    for check in checks:
        if check["name"] == name:
            return check

    return None


def sub_universe_robustness(simulation_result):
    insample = simulation_result["is"]
    checks = insample["checks"]

    low_sub_universe_sharpe = get_check(checks, "LOW_SUB_UNIVERSE_SHARPE")

    if low_sub_universe_sharpe is None:
        return None

    else:
        is_sub_universe_sharpe = low_sub_universe_sharpe["value"]
        is_sub_universe_sharpe_limit = low_sub_universe_sharpe["limit"]

    if is_sub_universe_sharpe_limit <= 0 or is_sub_universe_sharpe <= 0:
        return 0

    return round(0.75 * is_sub_universe_sharpe / is_sub_universe_sharpe_limit, 2)


def check_submission(simulation_result):
    insample = simulation_result["is"]
    checks = insample["checks"]

    to_check = [
        "LOW_SHARPE",
        "LOW_FITNESS",
        "LOW_TURNOVER",
        "HIGH_TURNOVER",
        "CONCENTRATED_WEIGHT",
        "LOW_SUB_UNIVERSE_SHARPE",
        "IS_LADDER_SHARPE",
        "LOW_2Y_SHARPE",
    ]

    for check in checks:
        if check["name"] in to_check and check["result"] != "PASS":
            return False

    return True


def get_metrics(simulation_result):
    insample = simulation_result["is"]
    checks = insample["checks"]

    warnings = []

    sharpe = insample["sharpe"]
    sharpe_lb = get_check(checks, "LOW_SHARPE")["limit"]

    ladder_sharpe_check = get_check(checks, "IS_LADDER_SHARPE")

    if ladder_sharpe_check is None:
        ladder_sharpe_check = get_check(checks, "LOW_2Y_SHARPE")

    ladder_sharpe = ladder_sharpe_check["value"]
    ladder_sharpe_lb = ladder_sharpe_check["limit"]

    fitness = insample["fitness"]
    fitness_lb = get_check(checks, "LOW_FITNESS")["limit"]

    turnover = insample["turnover"]
    turnover_lb = get_check(checks, "LOW_TURNOVER")["limit"]
    turnover_ub = get_check(checks, "HIGH_TURNOVER")["limit"]

    sur = sub_universe_robustness(simulation_result)
    sur_lb = 0.75

    weight_concentration = get_check(checks, "CONCENTRATED_WEIGHT")
    if weight_concentration["result"] == "FAIL":
        if weight_concentration.get("value"):
            warnings.append(
                f"Weight Concentration {round(weight_concentration['value'] * 100, 2)}% is above cutoff of {round(weight_concentration['limit'] * 100, 2)}%."
            )
        else:
            warnings.append(
                "Weight is too strongly concentrated or too few instruments are assigned weight."
            )
    else:
        warnings.append("Weight is well distributed over instruments.")

    metrics = {
        "Sharpe": [sharpe, sharpe_lb],
        "Fitness": [fitness, fitness_lb],
        "Turnover": [turnover, turnover_lb, turnover_ub],
        "Sub Universe Robustness": [sur, sur_lb],
        "Ladder Sharpe": [ladder_sharpe, ladder_sharpe_lb],
        "WARNINGS": warnings,
    }

    if sur is None:
        metrics.pop("Sub Universe Robustness")

    return metrics


def get_user_context(simulation_result):
    metrics = get_metrics(simulation_result)
    is_submittable = check_submission(simulation_result)

    user_context = []

    for metric in metrics.keys():
        value = metrics[metric]

        if metric == "WARNINGS":
            user_context += value
            break

        txt = f"{metric}: {value[0]}"

        if len(value) == 2:
            if value[0] < value[1]:
                txt += f" is less than {value[1]}"

        if len(value) == 3:
            if value[0] < value[1]:
                txt += f" is less than {value[1]}"

            if value[0] > value[2]:
                txt += f" is more than {value[2]}"

        user_context.append(txt)

    if is_submittable:
        user_context.append("Alpha Expression is Submittable.")
    else:
        user_context.append("Alpha Expression is NOT Submittable.")

    return is_submittable, "\n".join(user_context)


def pnl_chart(pnl_config, pnl_df):

    def format_y(value, _):
        """Render large values with K/M suffix and preserve sign."""
        v = float(value)
        sign = "-" if v < 0 else ""
        abs_val = abs(v)
        if abs_val >= 1e7:
            val, suffix = abs_val / 1e6, "M"
        elif abs_val >= 1e4:
            val, suffix = abs_val / 1e3, "K"
        else:
            return f"{v:,.0f}" if v.is_integer() else f"{v:,.1f}"
        return (
            f"{sign}{val:,.0f}{suffix}"
            if val.is_integer()
            else f"{sign}{val:,.1f}{suffix}"
        )

    plot_df = pnl_df.select_dtypes(include=["number"])
    dates = plot_df.index.to_pydatetime().tolist()

    series_values = [plot_df[col].values.tolist() for col in plot_df.columns]

    n_points = len(dates)
    test_len = pnl_config["test"]
    highlight_start = max(0, n_points - test_len)

    tick_candidates = []
    for _, group in groupby(dates, key=lambda d: d.year):
        year_dates = list(group)
        tick_candidates.append(year_dates[0])
        tick_candidates.append(year_dates[len(year_dates) // 2])

    endpoints = {dates[0], dates[-1]}
    ticks = sorted({dt for dt in tick_candidates if dt not in endpoints})

    filtered_labels = []
    prev_label = None
    for dt in ticks:
        label = dt.strftime("%b '%y")
        if label != prev_label:
            filtered_labels.append((dt, label))
            prev_label = label

    idx_by_date = {dt: i for i, dt in enumerate(dates)}
    positions = [idx_by_date[dt] for dt, _ in filtered_labels]
    labels = [lbl for _, lbl in filtered_labels]

    train_color = pnl_config.get("train_color", "#808080")
    test_color = pnl_config.get("test_color", "#0000FF")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=500, facecolor="white")
    x_full = range(n_points)
    x_train = range(highlight_start)

    lw = 0.7
    ls = "-"

    for idx, svals in enumerate(series_values):
        if idx == 0:
            ax.plot(
                x_full, svals, linewidth=lw, linestyle=ls, color=test_color, alpha=1.0
            )
            if highlight_start > 0:
                ax.plot(
                    x_train,
                    svals[:highlight_start],
                    linewidth=lw,
                    linestyle=ls,
                    color=train_color,
                    alpha=1.0,
                )

        elif idx == 1:
            col = pnl_config.get("secondary_color", "#FF0000")
            ax.plot(x_full, svals, linewidth=lw, linestyle=ls, color=col, alpha=1.0)

        elif idx == 2:
            col = pnl_config.get("tertiary_color", "#00FF00")
            ax.plot(x_full, svals, linewidth=lw, linestyle=ls, color=col, alpha=1.0)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha="right", color="#7b8292", fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(format_y))
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.5)

    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#ccd6eb")

    ax.tick_params(axis="y", colors="#7b8292", labelsize=8)
    ax.set_xlim(left=0)

    file_name = pnl_config["file_name"]

    plt.tight_layout()
    plt.savefig(file_name, dpi=500, bbox_inches="tight")
    plt.close(fig)
