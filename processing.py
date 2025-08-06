import config
from datetime import datetime
from itertools import groupby
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def alpha_quality_factor(simulation_result):
    train = simulation_result['train']
    test = simulation_result['test']

    test_sharpe = test['sharpe']
    test_fitness = test['fitness']

    train_sharpe = train['sharpe']
    train_fitness = train['fitness']

    if (train_sharpe <= 0 or train_fitness <= 0):
        return 0

    else:
        sign = abs(test_sharpe) / test_sharpe
        
        sharpe_stability = test_sharpe / train_sharpe
        fitness_stability = test_fitness / train_fitness

        if (sharpe_stability < 1 or fitness_stability < 1):
            return round(sign * (min(1, sharpe_stability) * min(1, fitness_stability)) ** (1 / 2), 2)

        return round(sign * ((sharpe_stability * fitness_stability) ** (1 / 2)), 2)
            

def turnover_stability(simulation_result):
    train = simulation_result['train']
    test = simulation_result['test']

    test_turnover = test['turnover']
    train_turnover = train['turnover']

    turnover_stability = min(test_turnover, train_turnover) / max(test_turnover, train_turnover)

    return turnover_stability


def sub_universe_robustness(simulation_result):
    insample = simulation_result['is']
    checks = insample['checks']

    is_sub_universe_sharpe = [check['value'] for check in checks if check['name'] == 'LOW_SUB_UNIVERSE_SHARPE']

    if (is_sub_universe_sharpe == []):
        return None
    else:
        is_sub_universe_sharpe = is_sub_universe_sharpe[0]
    
    is_sub_universe_sharpe_limit = [check['limit'] for check in checks if check['name'] == 'LOW_SUB_UNIVERSE_SHARPE'][0]

    return round(0.75 * is_sub_universe_sharpe / is_sub_universe_sharpe_limit, 2)


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
                linewidth=0.7, linestyle='-', color=config.pnl_chart['train_color'])

    ax.plot(range(highlight_start, n_points), values[highlight_start:],
            linewidth=0.7, linestyle='-', color=config.pnl_chart['test_color'])

    x0, x1 = steep_idx, steep_idx + 1
    y0, y1 = values[x0], values[x1]
    ax.plot([x0, x1], [y0, y1], linewidth=0.7, linestyle='-', color=config.pnl_chart['drawdown_color'])

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