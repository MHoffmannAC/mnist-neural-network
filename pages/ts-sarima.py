import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, pacf

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Classical NFL Playground")

# Custom CSS for the Playground aesthetic
st.markdown(
    """
<style>
    .reportview-container {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    h1, h2, h3 {
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 0.5rem;
        padding: 1.5rem;
        text-align: center;
        border-bottom: 4px solid #3b82f6;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #60a5fa;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
    }
    .summary-box {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 0.5rem;
        padding: 1rem;
        font-family: 'Fira Code', monospace;
        font-size: 0.8rem;
        color: #93c5fd;
        overflow-x: auto;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- Data Loading ---
@st.cache_data
def load_nfl_trends():
    df = pd.read_csv("./data/NFL_daily_trends.csv")
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    return df


# --- Load Data ---
df_raw = load_nfl_trends()

# --- Sidebar ---
with st.sidebar:
    st.title("📈 Model Configuration")

    model_type = st.radio(
        "Select Model Architecture",
        ["AR (AutoRegressive)", "ARIMA (Integrated)", "SARIMA (Seasonal)"],
        index=0,
    )

    st.markdown("---")
    st.subheader("1. Data Resampling")
    resample_freq = st.selectbox(
        "Time Resolution",
        ["Daily", "Weekly", "Monthly", "Yearly"],
        index=0,
        help="Aggregates data. ARIMA often performs better on Weekly or Monthly totals by smoothing daily spikes.",
    )

    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Yearly": "YE"}
    period_map = {"Daily": 365, "Weekly": 52, "Monthly": 12, "Yearly": 1}

    current_freq = freq_map[resample_freq]
    yearly_period = period_map[resample_freq]

    st.markdown("---")
    st.subheader("2. Mode & Horizon")

    validation_mode = st.toggle(
        "Validation Mode (Backtest)",
        value=False,
        help="Uses the last year of data to test accuracy instead of forecasting the future.",
    )

    window_map = {"D": 365, "W": 52, "ME": 12, "YE": 1}
    val_window = window_map[current_freq]

    if validation_mode:
        st.info(
            f"Training on data up to {val_window} {resample_freq} steps ago. Testing on final year.",
        )
        periods = val_window
    else:
        horizon_max = 2 * window_map[current_freq]
        periods = st.slider(
            f"{resample_freq} Steps to Forecast",
            1,
            int(2 * horizon_max),
            int(horizon_max),
        )

    st.markdown("---")
    st.subheader("3. Model Parameters")

    # Lag Order p scale: up to 2 years (730 for daily, 104 for weekly, etc.)
    p = st.slider("p (Lag Order)", 0, yearly_period * 2, 0)

    # Differencing is now available for AR as well to handle non-stationarity
    d = st.slider(
        "d (Degree of Differencing)",
        0,
        3,
        0,
        help="Stabilizes the mean. Essential for AR models when the data has a trend.",
    )

    q = 0
    if model_type in ["ARIMA (Integrated)", "SARIMA (Seasonal)"]:
        q = st.slider("q (Moving Average Order)", 0, yearly_period * 2, 0)

    P, D, Q, s = 0, 0, 0, 0
    if model_type == "SARIMA (Seasonal)":
        st.markdown("**Seasonal Component**")
        P = st.slider("P (Seasonal AR)", 0, 5, 0)
        D = st.slider("D (Seasonal Diff)", 0, 3, 0)
        Q = st.slider("Q (Seasonal MA)", 0, 5, 0)

        # Smart suggestions for 's' based on frequency
        s_default = 7 if current_freq == "D" else yearly_period
        s = st.number_input("s (Seasonal Period)", value=s_default, min_value=1)

        if s > 1 and P > 0 and p >= s:
            st.warning(f"⚠️ **Lag Conflict:** Reduce p below {s} or set P to 0.")

    st.markdown("---")
    st.subheader("4. Diagnostics")
    diag_decomp = st.checkbox("Decomposition", value=False)

    # Selection for Decomposition Mode
    decomp_mode = "additive"
    if diag_decomp:
        decomp_mode = st.radio(
            "Decomp Mode",
            ["additive", "multiplicative"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )

    diag_corr = st.checkbox("Correlation (ACF/PACF)", value=False)

    if diag_corr:
        n_lags_to_show = st.slider(
            "Diagnostic Lags to Analyze",
            min_value=5,
            max_value=yearly_period * 4,
            value=yearly_period * 2,
            help="Higher lags reveal longer-term seasonal cycles.",
        )

    diag_adf = st.checkbox("ADF (Stationarity Test)", value=False)

    if diag_adf:
        adf_diff_order = st.slider(
            "Differencing Order for ADF Test",
            0,
            3,
            0,
            help="Check if first or second order differencing makes the series stationary.",
        )

    st.markdown("---")
    st.subheader("5. Visualization View")
    view_mode = st.radio(
        "Display Window",
        ["Full History", "Last 3 Years", "Last 12 Months"],
    )

# --- Processing Resampled Data ---
df_processed = df_raw.set_index("ds").resample(current_freq).mean().reset_index()


# --- Model Execution ---
@st.cache_resource
def run_timeseries_engine(data, m_type, p, d, q, P, D, Q, s, steps, is_validation):
    if is_validation:
        train_series = data["y"].values[:-steps]
        test_series = data["y"].values[-steps:]
    else:
        train_series = data["y"].values
        test_series = None

    try:
        if m_type == "AR (AutoRegressive)":
            if d == 0:
                # Standard AR on levels
                model = AutoReg(train_series, lags=p)
                res = model.fit()
                forecast = res.predict(
                    start=len(train_series),
                    end=len(train_series) + steps - 1,
                )
                aic, summary = res.aic, str(res.summary())
            else:
                # Differenced AR (ARI model) handled via SARIMAX for automatic integration back to original scale
                model = SARIMAX(
                    train_series,
                    order=(p, d, 0),
                    seasonal_order=(0, 0, 0, 0),
                )
                res = model.fit(disp=False)
                forecast = res.get_forecast(steps=steps).predicted_mean
                aic, summary = res.aic, str(res.summary())

        elif m_type == "ARIMA (Integrated)":
            model = SARIMAX(train_series, order=(p, d, q), seasonal_order=(0, 0, 0, 0))
            res = model.fit(disp=False)
            forecast = res.get_forecast(steps=steps).predicted_mean
            aic, summary = res.aic, str(res.summary())

        elif m_type == "SARIMA (Seasonal)":
            if s > 1 and P > 0 and p >= s:
                return None, None, f"Invalid Model: p={p} >= s={s} with P={P}.", None
            if s > len(train_series) / 2:
                return (
                    None,
                    None,
                    f"Seasonal period {s} too large for training size.",
                    None,
                )

            model = SARIMAX(train_series, order=(p, d, q), seasonal_order=(P, D, Q, s))
            res = model.fit(disp=False)
            forecast = res.get_forecast(steps=steps).predicted_mean
            aic, summary = res.aic, str(res.summary())

    except Exception as e:
        return None, None, None, f"System Error: {e!s}"

    return forecast, aic, summary, test_series


# Run engine
with st.spinner("Calculating statistical projections..."):
    forecast_vals, model_aic, model_summary, actual_vals = run_timeseries_engine(
        df_processed,
        model_type,
        p,
        d,
        q,
        P,
        D,
        Q,
        s,
        periods,
        validation_mode,
    )

# --- Data Preparation for Visualization ---
if forecast_vals is not None:
    # 1. Create the Forecast/Prediction Dataframe
    last_train_date = (
        df_processed["ds"].iloc[-periods - 1]
        if validation_mode
        else df_processed["ds"].max()
    )
    future_dates = pd.date_range(
        start=last_train_date,
        periods=periods + 1,
        freq=current_freq,
    )[1:]

    df_forecast = pd.DataFrame(
        {
            "ds": future_dates,
            "y": forecast_vals,
            "Type": "Validation Prediction" if validation_mode else "Forecast",
        },
    )

    if validation_mode:
        # Data used for training
        df_historical = df_processed.iloc[:-periods].copy()
        df_historical["Type"] = "Actual (Training)"

        # Data used for ground truth check
        df_ground_truth = df_processed.tail(periods).copy()
        df_ground_truth["Type"] = "Actual (Ground Truth)"

        plot_df = pd.concat([df_historical, df_forecast, df_ground_truth])
    else:
        df_actual = df_processed.copy()
        df_actual["Type"] = "Actual"
        plot_df = pd.concat([df_actual, df_forecast])

    # 3. View Windows
    last_date = df_processed["ds"].max()
    if view_mode == "Last 3 Years":
        plot_df = plot_df[plot_df["ds"] >= (last_date - pd.DateOffset(years=3))]
    elif view_mode == "Last 12 Months":
        plot_df = plot_df[plot_df["ds"] >= (last_date - pd.DateOffset(months=12))]

# --- Main UI ---
st.title(f"📈 {model_type} Playground")
st.markdown(f"### Historical analysis at **{resample_freq}** resolution")

if forecast_vals is None:
    st.error(model_summary)
    st.stop()

# Metrics
m1, m2, m3 = st.columns(3)

if validation_mode:
    mae = mean_absolute_error(actual_vals, forecast_vals)
    rmse = np.sqrt(mean_squared_error(actual_vals, forecast_vals))
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Mean Absolute Error (MAE)</div><div class="metric-value">{mae:.2f}</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Root Mean Sq. Error (RMSE)</div><div class="metric-value">{rmse:.2f}</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">Validation Period</div><div class="metric-value">Last Year</div></div>',
            unsafe_allow_html=True,
        )
else:
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Model AIC</div><div class="metric-value">{model_aic:.1f}</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        peak_f = np.max(forecast_vals)
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Projected Peak</div><div class="metric-value">{peak_f:.1f}</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Observation Points</div><div class="metric-value">{len(df_processed)}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# Main Forecast Plot
st.subheader(
    "📅 Accuracy Check: Actual vs. Predicted"
    if validation_mode
    else f"📅 {resample_freq} Interest Projection",
)

color_map = {
    "Actual": "#94a3b8",
    "Actual (Training)": "#94a3b8",
    "Actual (Ground Truth)": "#f8fafc",
    "Forecast": "#60a5fa",
    "Validation Prediction": "#f87171",
}

present_types = [t for t in color_map if t in plot_df["Type"].unique()]
color_domain = present_types
color_range = [color_map[t] for t in present_types]

base = alt.Chart(plot_df).encode(
    x=alt.X("ds:T", title="Date"),
    y=alt.Y("y:Q", title="Interest Score (Mean)"),
    color=alt.Color(
        "Type:N",
        scale=alt.Scale(domain=color_domain, range=color_range),
        legend=alt.Legend(title="Data Series"),
    ),
    tooltip=[
        alt.Tooltip("ds:T", title="Date"),
        alt.Tooltip("y:Q", title="Score", format=".1f"),
        alt.Tooltip("Type:N"),
    ],
)

historical_line = base.transform_filter(
    (alt.datum.Type == "Actual") | (alt.datum.Type == "Actual (Training)"),
).mark_line(opacity=0.7)

prediction_line = base.transform_filter(
    (alt.datum.Type == "Forecast") | (alt.datum.Type == "Validation Prediction"),
).mark_line(strokeDash=[5, 5])

ground_truth_points = base.transform_filter(
    alt.datum.Type == "Actual (Ground Truth)",
).mark_point(size=60, filled=True, opacity=1.0)

final_chart = (
    alt.layer(historical_line, prediction_line, ground_truth_points)
    .properties(
        width="container",
        height=500,
    )
    .interactive()
)

st.altair_chart(final_chart, width="stretch")

# --- DIAGNOSTICS SECTION ---
if any([diag_decomp, diag_corr, diag_adf]):
    st.markdown("---")
    st.title("🔍 Model Diagnostics")

    # 1. Decomposition
    if diag_decomp:
        st.subheader("1. Classical Decomposition")
        st.write(
            f"Isolating Trend and Seasonality using **{decomp_mode.capitalize()}** logic (Period: {yearly_period}).",
        )

        try:
            # Multiplicative cannot handle 0 values, we add a tiny offset if needed
            working_series = df_processed["y"].copy()
            if decomp_mode == "multiplicative":
                working_series = working_series.replace(0, 0.0001)

            decomp = seasonal_decompose(
                working_series,
                model=decomp_mode,
                period=yearly_period,
            )

            decomp_df = pd.DataFrame(
                {
                    "ds": df_processed["ds"],
                    "Trend": decomp.trend,
                    "Seasonal": decomp.seasonal,
                    "Residual": decomp.resid,
                },
            ).melt("ds", var_name="Component", value_name="Value")

            # Stacked charts for decomposition with INDEPENDENT Y-scales
            decomp_chart = (
                alt.Chart(decomp_df)
                .mark_line(color="#60a5fa")
                .encode(
                    x=alt.X("ds:T", title=None),
                    y=alt.Y("Value:Q", title=None),
                    row=alt.Row(
                        "Component:N",
                        sort=["Trend", "Seasonal", "Residual"],
                        title=None,
                    ),
                )
                .properties(width="container", height=150)
                .interactive()
                .resolve_scale(
                    y="independent",
                )
            )

            st.altair_chart(decomp_chart, width="stretch")
        except Exception as e:
            st.error(f"Could not perform decomposition: {e}")

    # 2. Correlation Analysis
    if diag_corr:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("2. Correlation Analysis (ACF & PACF)")
        st.write(f"Displaying {n_lags_to_show} lags for {resample_freq} resolution.")

        acf_vals = acf(df_processed["y"], nlags=n_lags_to_show)
        pacf_vals = pacf(df_processed["y"], nlags=n_lags_to_show)

        corr_df = pd.DataFrame(
            {
                "Lag": np.arange(len(acf_vals)),
                "ACF": acf_vals,
                "PACF": pacf_vals,
            },
        )

        conf_interval = 1.96 / np.sqrt(len(df_processed))
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**ACF (Autocorrelation)**")
            acf_chart = (
                alt.Chart(corr_df)
                .mark_bar(size=1 if n_lags_to_show > 200 else 3, color="#60a5fa")
                .encode(
                    x="Lag:O",
                    y="ACF:Q",
                )
            )
            conf_band = (
                alt.Chart(corr_df)
                .mark_area(opacity=0.1, color="#94a3b8")
                .encode(
                    x="Lag:O",
                    y=alt.datum(-conf_interval),
                    y2=alt.datum(conf_interval),
                )
            )
            st.altair_chart(
                (conf_band + acf_chart).properties(height=250),
                width="stretch",
            )

        with c2:
            st.markdown("**PACF (Partial Autocorrelation)**")
            pacf_chart = (
                alt.Chart(corr_df)
                .mark_bar(size=1 if n_lags_to_show > 200 else 3, color="#f87171")
                .encode(
                    x="Lag:O",
                    y="PACF:Q",
                )
            )
            st.altair_chart(
                (conf_band + pacf_chart).properties(height=250),
                width="stretch",
            )

    # 3. ADF Test
    if diag_adf:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("3. Stationarity Test (ADF)")
        st.write("Checking if the series mean and variance are constant over time.")

        test_series = df_processed["y"].copy()
        for _ in range(adf_diff_order):
            test_series = test_series.diff().dropna()

        try:
            adf_res = adfuller(test_series)
            adf_stat, p_val = adf_res[0], adf_res[1]

            stat_col, p_col, res_col = st.columns(3)
            stat_col.metric("ADF Statistic", f"{adf_stat:.3f}")
            p_col.metric("p-value", f"{p_val:.4f}")

            if p_val < 0.05:
                res_col.success("✅ Stationary (Series is stable)")
            else:
                res_col.error("❌ Non-Stationary (Needs Differencing)")

            st.caption(f"Testing series with differencing order: {adf_diff_order}")
        except Exception as e:
            st.error(f"Could not calculate ADF: {e}")

# Summary Section
st.markdown("---")
col_info, col_sum = st.columns([1, 1.5])

with col_info:
    st.subheader("Statistical Interpretation")
    if validation_mode:
        st.write("""
        **Validation Window:**
        - **Red Dashed Line**: Model projection.
        - **White Points**: Real data.
        - **Grey Line**: History.
        """)
    else:
        st.write(
            "Tune model parameters to minimize AIC and improve forecasting behavior.",
        )

    st.info(
        "💡 **Diagnostics Tip:** If the AR model shows a poor fit on non-stationary data, try increasing the 'd' parameter to perform an ARI model.",
    )

with col_sum:
    st.subheader("📝 Statistical Summary")
    st.markdown(
        f'<div class="summary-box">{model_summary}</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "Validation mode uses the final year of data for accuracy checking. Future forecasts are hidden in this mode.",
)
