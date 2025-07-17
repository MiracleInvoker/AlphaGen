import config
from datetime import datetime
from itertools import groupby
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def score(simulation_result):
    insample = simulation_result['is']
    train = simulation_result['train']
    test = simulation_result['test']
    checks = insample['checks']

    is_sub_universe_sharpe = [check['value'] for check in checks if check['name'] == 'LOW_SUB_UNIVERSE_SHARPE']

    is_pnl = insample['pnl']
    is_sharpe = insample['sharpe']
    is_turnover = insample['turnover']
    is_returns = insample['returns']
    is_drawdown = insample['drawdown']
    is_fitness = insample['fitness']
    is_margin = insample['margin']

    if (is_sub_universe_sharpe == []):
        is_sub_universe_sharpe = is_sharpe
    else:
        is_sub_universe_sharpe = is_sub_universe_sharpe[0]

    test_pnl = test['pnl']
    test_sharpe = test['sharpe']
    test_turnover = test['turnover']
    test_returns = test['returns']
    test_drawdown = test['drawdown']
    test_fitness = test['fitness']
    test_margin = test['margin']

    train_pnl = train['pnl']
    train_sharpe = train['sharpe']
    train_turnover = train['turnover']
    train_returns = train['returns']
    train_drawdown = train['drawdown']
    train_fitness = train['fitness']
    train_margin = train['margin']

    if (test_sharpe < 0 or train_sharpe < 0 or is_sharpe < 0 or is_sub_universe_sharpe < 0):
        return 0

    core_performance = test_fitness
    stability_multiplier = min(1, test_sharpe / train_sharpe)
    robustness_multiplier = is_sub_universe_sharpe / is_sharpe
    drawdown_penalty = (1 - test_drawdown)

    score = core_performance * stability_multiplier * robustness_multiplier * drawdown_penalty

    return round(score, 2)


def pnl_chart(pnl_data):

    def format_y(value, _):
        """Render large values with K/M suffix and preserve sign."""
        sign = '-' if value < 0 else ''
        abs_val = abs(value)
        if abs_val >= 1e7:
            val, suffix = abs_val / 1e6, 'M'
        elif abs_val >= 1e4:
            val, suffix = abs_val / 1e3, 'K'
        else:
            return f"{value:,.0f}" if value.is_integer() else f"{value:,.1f}"
        return f"{sign}{val:,.0f}{suffix}" if val.is_integer() else f"{sign}{val:,.1f}{suffix}"

    date_strs, values = zip(*pnl_data)
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in date_strs]
    n_points = len(values)

    _, steep_idx = min(
        ((values[i+1] - values[i], i) for i in range(n_points - 1)), 
        key=lambda item: item[0]
    )

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

    positions = [dates.index(dt) for dt, _ in filtered_labels]
    labels = [lbl for _, lbl in filtered_labels]
    highlight_start = max(0, n_points - 252)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=500, facecolor='white')

    if highlight_start > 0:
        ax.plot(range(highlight_start), values[:highlight_start],
                linewidth=0.7, linestyle='-', color=config.Graph.train_color)

    ax.plot(range(highlight_start, n_points), values[highlight_start:],
            linewidth=0.7, linestyle='-', color=config.Graph.test_color)

    x0, x1 = steep_idx, steep_idx + 1
    y0, y1 = values[x0], values[x1]
    ax.plot([x0, x1], [y0, y1], linewidth=0.7, linestyle='-', color=config.Graph.drawdown_color)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha='right', color='#7b8292', fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(format_y))
    ax.grid(axis='y', color='#e6e6e6', linewidth=0.5)

    for spine in ['top', 'left', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#ccd6eb')

    ax.tick_params(axis='y', colors='#7b8292', labelsize=8)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig('pnl_chart.png', dpi=500, bbox_inches='tight')
    plt.close(fig)