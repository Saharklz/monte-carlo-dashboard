import streamlit as st
import numpy as np
import pandas as pd

from utils import (
    generate_demo_data,
    run_simulations,
    build_fan_chart,
    build_histograms,
    generate_interpretation,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monte Carlo · Deep Space",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

/* ── Base background ── */
html, body { background-color: #060614 !important; }

.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, section.main {
    background: radial-gradient(ellipse 180% 90% at 50% 8%, #0c1038 0%, #060614 52%) !important;
    background-color: #060614 !important;
    color: #c8c8d8;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="block-container"],
.block-container,
[data-testid="stMainBlockContainer"] {
    background: transparent !important;
    max-width: 1400px !important;
    padding-top: 1.5rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background-color: #09091e !important;
    border-right: 1px solid rgba(100,120,255,0.08) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #c8c8d8 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1a2a70 0%, #2860d0 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(77,142,255,0.30) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.2rem !important;
    transition: box-shadow 0.25s ease, border-color 0.25s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 0 28px rgba(77,142,255,0.45), 0 0 8px rgba(77,142,255,0.20) !important;
    border-color: rgba(77,142,255,0.65) !important;
}
.stButton > button:active {
    box-shadow: 0 0 12px rgba(77,142,255,0.35) !important;
}

/* ── Number input ── */
[data-testid="stNumberInput"] input {
    background-color: #0d0d24 !important;
    color: #c8c8d8 !important;
    border: 1px solid rgba(100,120,255,0.15) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: rgba(77,142,255,0.50) !important;
    box-shadow: 0 0 0 3px rgba(77,142,255,0.10) !important;
    outline: none !important;
}
[data-testid="stNumberInput"] button {
    background-color: #0d0d24 !important;
    border-color: rgba(100,120,255,0.15) !important;
    color: #7a7a9a !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div {
    background: rgba(77,142,255,0.18) !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: #4d8eff !important;
}
[data-testid="stSlider"] p { color: #7a7a9a !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] section {
    background-color: #0d0d24 !important;
    border: 1px dashed rgba(100,120,255,0.22) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] * { color: #7a7a9a !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #4d8eff !important; }
.stSpinner > div { border-top-color: #4d8eff !important; }

/* ── Status messages ── */
[data-testid="stAlert"] { border-radius: 10px !important; }
.stSuccess {
    background-color: rgba(0,230,118,0.07) !important;
    border: 1px solid rgba(0,230,118,0.20) !important;
    color: #00e676 !important;
}
.stError {
    background-color: rgba(255,82,82,0.07) !important;
    border: 1px solid rgba(255,82,82,0.20) !important;
    color: #ff5252 !important;
}

/* ── Metric cards ── */
.mc-card {
    background: #0d0d24;
    border: 1px solid rgba(100,120,255,0.08);
    border-radius: 16px;
    padding: 20px 22px 16px;
    position: relative;
    overflow: hidden;
    min-height: 130px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.mc-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.mc-card.green::before { background: linear-gradient(90deg, #00e676, #00c853); }
.mc-card.amber::before { background: linear-gradient(90deg, #ffab40, #ff8f00); }
.mc-card.red::before   { background: linear-gradient(90deg, #ff5252, #d50000); }
.mc-card.blue::before  { background: linear-gradient(90deg, #4d8eff, #1565c0); }

.mc-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    color: #7a7a9a;
    margin-bottom: 2px;
}
.mc-value {
    font-family: 'Space Mono', monospace;
    font-size: 40px;
    font-weight: 700;
    color: #4d8eff;
    line-height: 1.05;
    letter-spacing: -0.02em;
}
.mc-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    color: #4a4a6a;
    margin-top: 2px;
}

/* ── Section headers ── */
.sec-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    font-weight: 400;
    color: #4a4a6a;
    letter-spacing: 0.14em;
    margin: 36px 0 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(100,120,255,0.05);
}

/* ── Interpretation card ── */
.interp-card {
    background: linear-gradient(135deg, #0d0d2c 0%, #090920 100%);
    border: 1px solid rgba(77,142,255,0.10);
    border-radius: 16px;
    padding: 24px 28px;
    margin-top: 6px;
}
.interp-card h4 {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    font-weight: 400;
    color: #4a4a6a;
    letter-spacing: 0.14em;
    margin: 0 0 14px;
}
.interp-card p {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    color: #9a9ab8;
    line-height: 1.80;
    margin: 0;
}

/* ── Sidebar labels ── */
.sb-title {
    font-family: 'Space Mono', monospace;
    font-size: 14px;
    font-weight: 700;
    color: #4d8eff;
    letter-spacing: 0.04em;
    margin-bottom: 2px;
}
.sb-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    color: #4a4a6a;
    letter-spacing: 0.08em;
    margin-bottom: 20px;
}
.sb-section {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    color: #4a4a6a;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 4px 0 10px;
}
.data-badge {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    color: #4d8eff;
    background: rgba(77,142,255,0.08);
    border: 1px solid rgba(77,142,255,0.18);
    border-radius: 6px;
    padding: 4px 10px;
    display: inline-block;
    margin-top: 6px;
}

/* ── Page title ── */
.page-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    color: #4a4a6a;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.page-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 26px;
    font-weight: 400;
    color: #e0e0f0;
    margin: 0 0 4px;
}
.page-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    color: #4a4a6a;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 110px 40px;
    color: #4a4a6a;
}
.empty-icon {
    font-size: 56px;
    opacity: 0.25;
    margin-bottom: 20px;
    line-height: 1;
}
.empty-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    color: #5a5a7a;
    margin-bottom: 8px;
}
.empty-body {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    color: #3a3a5a;
    line-height: 1.6;
}

/* ── Plotly container spacing ── */
[data-testid="stPlotlyChart"] { margin-top: -8px; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, .stDeployButton { display: none !important; }
header[data-testid="stHeader"] {
    background: rgba(6,6,20,0.85) !important;
    border-bottom: 1px solid rgba(100,120,255,0.05) !important;
    backdrop-filter: blur(12px);
}
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [("returns", None), ("results", None), ("data_label", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-title">Monte Carlo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Universe of Outcomes</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sb-section">Data Source</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV (date, trade_return)",
        type=["csv"],
        label_visibility="collapsed",
    )

    if st.button("Generate Demo Data", use_container_width=True):
        df = generate_demo_data()
        st.session_state.returns = df["trade_return"].values
        st.session_state.results = None
        st.session_state.data_label = f"Demo data · {len(df)} trades"

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "trade_return" not in df.columns:
                st.error("CSV needs a 'trade_return' column.")
            else:
                st.session_state.returns = df["trade_return"].values
                st.session_state.results = None
                st.session_state.data_label = f"{uploaded_file.name} · {len(df)} trades"
        except Exception as exc:
            st.error(f"Read error: {exc}")

    if st.session_state.data_label:
        st.markdown(
            f'<div class="data-badge">⬤ &nbsp;{st.session_state.data_label}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="sb-section">Parameters</div>', unsafe_allow_html=True)

    initial_capital = st.number_input(
        "Starting Capital ($)",
        value=100_000,
        step=10_000,
        min_value=1_000,
        format="%d",
    )
    n_simulations = st.slider("Simulations", min_value=100, max_value=2000,
                              value=1000, step=100)

    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button("Run Simulation", type="primary", use_container_width=True)


# ── Run logic ─────────────────────────────────────────────────────────────────
if run_clicked:
    if st.session_state.returns is None:
        st.error("Load data first — upload a CSV or click Generate Demo Data.")
    else:
        with st.spinner(f"Running {n_simulations:,} simulations…"):
            st.session_state.results = run_simulations(
                st.session_state.returns,
                initial_capital=float(initial_capital),
                n_simulations=n_simulations,
            )


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 16px 0 24px;">
    <div class="page-eyebrow">Quantitative Risk Analysis</div>
    <div class="page-title">Monte Carlo Simulator</div>
    <div class="page-sub">Stress-test your backtest across 1,000 alternative universes</div>
</div>
""", unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────
r = st.session_state.results

if r is None:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">◎</div>
        <div class="empty-title">No simulation data yet</div>
        <div class="empty-body">
            Upload a CSV with <code style="background:rgba(77,142,255,0.1);
            padding:1px 6px; border-radius:4px; color:#4d8eff;">date</code> and
            <code style="background:rgba(77,142,255,0.1); padding:1px 6px;
            border-radius:4px; color:#4d8eff;">trade_return</code> columns,<br>
            or click <b style="color:#4d8eff;">Generate Demo Data</b> in the sidebar,<br>
            then hit <b style="color:#4d8eff;">Run Simulation</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">KEY METRICS</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")

    prob_loss_pct = r["prob_loss"] * 100
    loss_color = "green" if prob_loss_pct < 15 else "amber" if prob_loss_pct < 35 else "red"

    median_ret_pct = (r["median_final"] / r["initial_capital"] - 1) * 100
    med_color = "green" if median_ret_pct > 8 else "amber" if median_ret_pct >= 0 else "red"

    p5_mdd_pct = abs(float(np.percentile(r["max_drawdowns"], 5)) * 100)
    dd_color = "green" if p5_mdd_pct < 12 else "amber" if p5_mdd_pct < 22 else "red"

    of_color = "red" if r["overfitting"] else "green"

    def _card(col, color, label, value, sub):
        col.markdown(f"""
        <div class="mc-card {color}">
            <div class="mc-label">{label}</div>
            <div class="mc-value">{value}</div>
            <div class="mc-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    _card(c1, loss_color,
          "Probability of Loss",
          f"{prob_loss_pct:.1f}%",
          "of paths end below starting capital")

    _card(c2, med_color,
          "Median Return",
          f"{median_ret_pct:+.1f}%",
          f"${r['median_final']:,.0f} median final value")

    _card(c3, dd_color,
          "Worst 5% Drawdown",
          f"−{p5_mdd_pct:.1f}%",
          "5th-percentile max drawdown across simulations")

    _card(c4, of_color,
          "Overfitting Risk",
          "HIGH" if r["overfitting"] else "LOW",
          f"Original at {r['original_rank']*100:.0f}th percentile of simulations")

    # ── Fan chart ─────────────────────────────────────────────────────────────
    n_s = r["n_simulations"]
    st.markdown(
        f'<div class="sec-header">PROBABILITY CLOUD · {n_s:,} SIMULATED FUTURES</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(build_fan_chart(r), use_container_width=True)

    # ── Histograms ────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">DISTRIBUTION ANALYSIS</div>', unsafe_allow_html=True)
    st.plotly_chart(build_histograms(r), use_container_width=True)

    # ── Interpretation ────────────────────────────────────────────────────────
    interp_html = generate_interpretation(r)
    st.markdown(f"""
    <div class="interp-card">
        <h4>INTERPRETATION</h4>
        <p>{interp_html}</p>
    </div>
    """, unsafe_allow_html=True)

    # Extra padding at bottom
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
