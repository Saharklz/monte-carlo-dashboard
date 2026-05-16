import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict

# ── Design tokens ─────────────────────────────────────────────────────────────
BG = "#060614"
ACCENT_BLUE = "#4d8eff"
ACCENT_GREEN = "#00e676"
ACCENT_RED = "#ff5252"
GRID = "rgba(100,120,255,0.04)"
FONT_MONO = "Space Mono"
FONT_SANS = "DM Sans"
LABEL_COLOR = "#7a7a9a"


# ── Data generation ───────────────────────────────────────────────────────────

def generate_demo_data(n_trades: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    win_rate = 0.54
    n_wins = int(n_trades * win_rate)
    n_losses = n_trades - n_wins

    wins = rng.normal(loc=0.013, scale=0.009, size=n_wins)
    losses = rng.normal(loc=-0.009, scale=0.006, size=n_losses)

    # Occasional larger moves for realism
    n_big = max(1, int(n_trades * 0.04))
    big_wins = rng.uniform(0.03, 0.07, size=n_big)
    big_losses = rng.uniform(-0.05, -0.03, size=n_big)

    returns = np.concatenate([wins[n_big:], losses[n_big:], big_wins, big_losses])
    rng.shuffle(returns)
    returns = np.clip(returns, -0.08, 0.12)

    dates = pd.bdate_range(start="2022-01-03", periods=len(returns))
    return pd.DataFrame({"date": dates, "trade_return": returns})


# ── Core simulation math ──────────────────────────────────────────────────────

def calculate_equity_curve(returns: np.ndarray, initial_capital: float) -> np.ndarray:
    equity = initial_capital * np.cumprod(1.0 + returns)
    return np.insert(equity, 0, initial_capital)


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peaks) / peaks
    return float(drawdowns.min())


def run_simulations(
    returns: np.ndarray,
    initial_capital: float = 100_000,
    n_simulations: int = 1000,
    noise_std: float = 0.003,
    seed: int = None,
) -> Dict:
    rng = np.random.default_rng(seed)
    n_trades = len(returns)

    # Vectorised: shuffle rows independently and add uniform noise
    base = np.tile(returns, (n_simulations, 1))          # (n_sim, n_trades)
    shuffled = rng.permuted(base, axis=1)
    noise = rng.uniform(-noise_std, noise_std, (n_simulations, n_trades))
    sim_returns = shuffled + noise

    # Equity curves — prepend starting capital column
    equity = initial_capital * np.cumprod(1.0 + sim_returns, axis=1)
    init_col = np.full((n_simulations, 1), initial_capital)
    all_curves = np.concatenate([init_col, equity], axis=1)  # (n_sim, n_trades+1)

    final_values = all_curves[:, -1]

    # Vectorised max drawdown
    peaks = np.maximum.accumulate(all_curves, axis=1)
    max_drawdowns = ((all_curves - peaks) / peaks).min(axis=1)

    # Original (un-shuffled, noiseless) backtest
    original_curve = calculate_equity_curve(returns, initial_capital)
    original_final = float(original_curve[-1])
    original_mdd = calculate_max_drawdown(original_curve)

    original_rank = float(np.mean(final_values < original_final))

    return {
        "all_curves": all_curves,
        "final_values": final_values,
        "max_drawdowns": max_drawdowns,
        "original_curve": original_curve,
        "original_final": original_final,
        "original_mdd": original_mdd,
        "p5_curve":  np.percentile(all_curves, 5,  axis=0),
        "p25_curve": np.percentile(all_curves, 25, axis=0),
        "p50_curve": np.percentile(all_curves, 50, axis=0),
        "p75_curve": np.percentile(all_curves, 75, axis=0),
        "p95_curve": np.percentile(all_curves, 95, axis=0),
        "prob_loss":  float(np.mean(final_values < initial_capital)),
        "prob_dd_20": float(np.mean(max_drawdowns <= -0.20)),
        "prob_dd_30": float(np.mean(max_drawdowns <= -0.30)),
        "median_final": float(np.median(final_values)),
        "p5_final":    float(np.percentile(final_values, 5)),
        "p95_final":   float(np.percentile(final_values, 95)),
        "original_rank": original_rank,
        "overfitting": original_rank > 0.90,
        "initial_capital": initial_capital,
        "n_simulations": n_simulations,
        "n_trades": n_trades,
    }


# ── Color helpers ─────────────────────────────────────────────────────────────

def _lerp_color(c1: str, c2: str, t: float):
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    return int(r1 + (r2 - r1) * t), int(g1 + (g2 - g1) * t), int(b1 + (b2 - b1) * t)


def _curve_color(rank: float, alpha: float = 0.014) -> str:
    """Map a 0-1 percentile rank to red → blue → green rgba string."""
    if rank < 0.5:
        r, g, b = _lerp_color(ACCENT_RED, ACCENT_BLUE, rank * 2)
    else:
        r, g, b = _lerp_color(ACCENT_BLUE, ACCENT_GREEN, (rank - 0.5) * 2)
    return f"rgba({r},{g},{b},{alpha})"


# ── Chart builders ────────────────────────────────────────────────────────────

def build_fan_chart(results: Dict) -> go.Figure:
    all_curves = results["all_curves"]
    n_sims, n_steps = all_curves.shape
    x = np.arange(n_steps)

    # Sort simulations by final value so colour buckets map to outcome rank
    sort_idx = np.argsort(all_curves[:, -1])
    sorted_curves = all_curves[sort_idx]

    fig = go.Figure()

    # ── 10 colour buckets: each is a single Scattergl trace with NaN breaks ──
    N_BUCKETS = 10
    bucket_sz = n_sims // N_BUCKETS

    for b in range(N_BUCKETS):
        rank = (b + 0.5) / N_BUCKETS
        color = _curve_color(rank, alpha=0.014)
        lo = b * bucket_sz
        hi = (b + 1) * bucket_sz if b < N_BUCKETS - 1 else n_sims
        bucket = sorted_curves[lo:hi]
        n_c = len(bucket)
        pts = n_steps + 1  # data points + NaN separator

        cx = np.empty(pts * n_c)
        cy = np.empty(pts * n_c)
        for j in range(n_c):
            s = j * pts
            cx[s:s + n_steps] = x
            cx[s + n_steps] = np.nan
            cy[s:s + n_steps] = bucket[j]
            cy[s + n_steps] = np.nan

        fig.add_trace(go.Scattergl(
            x=cx, y=cy,
            mode="lines",
            line=dict(color=color, width=0.7),
            showlegend=False,
            hoverinfo="skip",
        ))

    # ── Percentile bands ──────────────────────────────────────────────────────
    p5  = results["p5_curve"]
    p25 = results["p25_curve"]
    p50 = results["p50_curve"]
    p75 = results["p75_curve"]
    p95 = results["p95_curve"]
    x_band = np.concatenate([x, x[::-1]])

    fig.add_trace(go.Scatter(
        x=x_band, y=np.concatenate([p95, p5[::-1]]),
        fill="toself", fillcolor="rgba(77,142,255,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=True, name="5–95th %ile", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x_band, y=np.concatenate([p75, p25[::-1]]),
        fill="toself", fillcolor="rgba(77,142,255,0.13)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=True, name="25–75th %ile", hoverinfo="skip",
    ))

    # ── Median ────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=p50,
        mode="lines",
        line=dict(color=ACCENT_BLUE, width=2.5),
        name="Median",
    ))

    # ── Original backtest (glow layer + crisp layer) ──────────────────────────
    orig = results["original_curve"]
    fig.add_trace(go.Scatter(
        x=x, y=orig,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.10)", width=12),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=orig,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.90)", width=2),
        name="Original Backtest",
    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    axis_base = dict(
        gridcolor=GRID, showgrid=True, zeroline=False,
        tickfont=dict(family=FONT_MONO, color=LABEL_COLOR, size=10),
        title_font=dict(family=FONT_SANS, color=LABEL_COLOR, size=12),
        color=LABEL_COLOR,
    )
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        height=600,
        margin=dict(l=80, r=30, t=20, b=60),
        xaxis=dict(**axis_base, title="Trade #"),
        yaxis=dict(**axis_base, title="Portfolio Value ($)", tickformat="$,.0f"),
        legend=dict(
            bgcolor="rgba(8,8,28,0.80)",
            bordercolor="rgba(100,120,255,0.15)",
            borderwidth=1,
            font=dict(family=FONT_SANS, color="#c8c8d8", size=11),
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ),
        hovermode="x unified",
    )
    return fig


def build_histograms(results: Dict) -> go.Figure:
    fv   = results["final_values"]
    mdd  = results["max_drawdowns"] * 100
    o_fv = results["original_final"]
    o_dd = results["original_mdd"] * 100
    init = results["initial_capital"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Final Portfolio Value", "Max Drawdown Distribution"],
        horizontal_spacing=0.12,
    )

    # ── Histograms ────────────────────────────────────────────────────────────
    fig.add_trace(go.Histogram(
        x=fv, nbinsx=50,
        marker=dict(color="rgba(13,13,36,0.9)", line=dict(color=ACCENT_BLUE, width=0.9)),
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=mdd, nbinsx=50,
        marker=dict(color="rgba(13,13,36,0.9)", line=dict(color=ACCENT_RED, width=0.9)),
        showlegend=False,
    ), row=1, col=2)

    # ── Vertical reference lines via shapes ───────────────────────────────────
    fig.update_layout(shapes=[
        dict(type="line", xref="x",  yref="paper", x0=o_fv,  x1=o_fv,
             y0=0, y1=1, line=dict(color="rgba(255,255,255,0.7)", width=2, dash="dash")),
        dict(type="line", xref="x",  yref="paper", x0=init,  x1=init,
             y0=0, y1=1, line=dict(color=ACCENT_RED, width=1.5, dash="dot")),
        dict(type="line", xref="x2", yref="paper", x0=o_dd,  x1=o_dd,
             y0=0, y1=1, line=dict(color="rgba(255,255,255,0.7)", width=2, dash="dash")),
    ])

    # Annotations for the vertical lines
    fig.add_annotation(x=o_fv, y=0.97, xref="x",  yref="paper",
                       text="Original", showarrow=False, xanchor="left",
                       font=dict(family=FONT_SANS, color="#c8c8d8", size=10))
    fig.add_annotation(x=init, y=0.87, xref="x",  yref="paper",
                       text="Break-even", showarrow=False, xanchor="left",
                       font=dict(family=FONT_SANS, color=ACCENT_RED, size=10))
    fig.add_annotation(x=o_dd, y=0.97, xref="x2", yref="paper",
                       text="Original", showarrow=False, xanchor="left",
                       font=dict(family=FONT_SANS, color="#c8c8d8", size=10))

    # ── Axis & layout ─────────────────────────────────────────────────────────
    ax = dict(gridcolor=GRID, showgrid=True, zeroline=False,
              tickfont=dict(family=FONT_MONO, color=LABEL_COLOR, size=9),
              title_font=dict(family=FONT_SANS, color=LABEL_COLOR, size=11),
              color=LABEL_COLOR)

    fig.update_xaxes(**ax, tickformat="$,.0f", title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_xaxes(**ax, ticksuffix="%",     title_text="Max Drawdown (%)",    row=1, col=2)
    fig.update_yaxes(**ax, title_text="Simulations", row=1, col=1)
    fig.update_yaxes(**ax, title_text="Simulations", row=1, col=2)

    for ann in fig.layout.annotations:
        if ann.text in ("Final Portfolio Value", "Max Drawdown Distribution"):
            ann.font = dict(family=FONT_SANS, color=LABEL_COLOR, size=12)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        height=360,
        margin=dict(l=70, r=30, t=55, b=60),
        bargap=0.04,
    )
    return fig


# ── Interpretation text ───────────────────────────────────────────────────────

def generate_interpretation(results: Dict) -> str:
    r = results
    init = r["initial_capital"]
    median_ret = (r["median_final"] / init - 1) * 100
    orig_ret   = (r["original_final"] / init - 1) * 100

    sentences = [
        f"Across <b>{r['n_simulations']:,}</b> simulated orderings of {r['n_trades']} trades, "
        f"your strategy has a <b>{r['prob_loss']*100:.1f}%</b> probability of ending below the "
        f"starting capital of <b>${init:,.0f}</b>.",

        f"The median outcome is <b>${r['median_final']:,.0f}</b> ({median_ret:+.1f}% return), "
        f"with the central 90% of paths landing between "
        f"<b>${r['p5_final']:,.0f}</b> and <b>${r['p95_final']:,.0f}</b>.",

        f"Max drawdowns exceed <b>20%</b> in {r['prob_dd_20']*100:.1f}% of simulations "
        f"and exceed <b>30%</b> in {r['prob_dd_30']*100:.1f}%.",
    ]

    if r["overfitting"]:
        sentences.append(
            f"<span style='color:{ACCENT_RED}; font-weight:600;'>Overfitting warning:</span> "
            f"the original backtest ({orig_ret:+.1f}%) outperforms "
            f"{r['original_rank']*100:.0f}% of Monte Carlo paths. "
            f"Performance this far above the simulation cloud suggests the backtest may be "
            f"over-fitted and could disappoint out-of-sample."
        )
    else:
        sentences.append(
            f"The original backtest ({orig_ret:+.1f}%) ranks at the "
            f"<b>{r['original_rank']*100:.0f}th percentile</b> — well within the distribution, "
            f"suggesting performance expectations are realistic rather than inflated."
        )

    return " ".join(sentences)
