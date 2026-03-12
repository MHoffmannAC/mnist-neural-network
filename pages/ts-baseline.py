import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Baseline NFL Playground")

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
        border-bottom: 4px solid #10b981;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #34d399;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
    }
    .explanation-box {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 0.5rem;
        padding: 1rem;
        font-size: 0.9rem;
        color: #94a3b8;
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
    st.title("📉 Baseline Configuration")

    model_type = st.radio(
        "Select Baseline Model",
        [
            "Naive",
            "Seasonal Naive",
            "Mean",
            "Recent Mean",
            "Trend Naive (Drift)",
            "Trend Seasonal",
        ],
        index=1,
    )

    st.markdown("---")
    st.subheader("1. Data Resampling")
    resample_freq = st.selectbox(
        "Time Resolution",
        ["Daily", "Weekly", "Monthly"],
        index=1,
    )

    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
    period_map = {"Daily": 365, "Weekly": 52, "Monthly": 12}

    current_freq = freq_map[resample_freq]
    yearly_period = period_map[resample_freq]

    st.markdown("---")
    st.subheader("2. Mode & Horizon")

    validation_mode = st.toggle(
        "Validation Mode (Backtest)",
        value=True,
        help="Uses the last year of data to test accuracy.",
    )

    val_window = yearly_period

    if validation_mode:
        st.info(f"Testing on the final {val_window} {resample_freq} steps.")
        periods = val_window
    else:
        horizon_max = yearly_period * 2
        periods = st.slider(
            f"{resample_freq} Steps to Forecast",
            1,
            int(horizon_max),
            int(yearly_period),
        )

    st.markdown("---")
    st.subheader("3. Model Settings")

    seasonal_m = yearly_period
    recent_k = 10
    trend_type = "Additive"

    if model_type == "Seasonal Naive" or model_type == "Trend Seasonal":
        seasonal_m = st.number_input(
            "Seasonal Period (m)",
            value=yearly_period,
            min_value=1,
            help="How many steps back to look for the matching seasonal value.",
        )
        if model_type == "Trend Seasonal":
            trend_type = st.radio(
                "Trend Type",
                ["Additive", "Multiplicative"],
                index=0,
                help="Additive adds a linear drift. Multiplicative scales the cycle by a growth factor.",
            )

    elif model_type == "Recent Mean":
        recent_k = st.slider(
            "Window Size (k)",
            2,
            yearly_period * 2,
            yearly_period,
            help="Average of the last k observations.",
        )

    st.markdown("---")
    st.subheader("4. Visualization View")
    view_mode = st.radio(
        "Display Window",
        ["Full History", "Last 3 Years", "Last 12 Months"],
    )

# --- Processing Resampled Data ---
df_processed = df_raw.set_index("ds").resample(current_freq).mean().reset_index()


# --- Model Execution ---
def run_baseline_engine(data, m_type, steps, is_validation, m_val, k_val, t_type):
    if is_validation:
        train_series = data["y"].values[:-steps]
        test_series = data["y"].values[-steps:]
    else:
        train_series = data["y"].values
        test_series = None

    last_val = train_series[-1]
    first_val = train_series[0]
    n_train = len(train_series)
    h_vals = np.arange(1, steps + 1)

    if m_type == "Naive":
        forecast = np.repeat(last_val, steps)

    elif m_type == "Seasonal Naive":
        m = int(m_val)
        source_pattern = train_series[-m:]
        reps = int(np.ceil(steps / m))
        forecast = np.tile(source_pattern, reps)[:steps]

    elif m_type == "Mean":
        global_mean = np.mean(train_series)
        forecast = np.repeat(global_mean, steps)

    elif m_type == "Recent Mean":
        k = int(k_val)
        local_mean = np.mean(train_series[-k:])
        forecast = np.repeat(local_mean, steps)

    elif m_type == "Trend Naive (Drift)":
        # Calculate slope between first and last point
        drift = (last_val - first_val) / (n_train - 1)
        forecast = last_val + (h_vals * drift)

    elif m_type == "Trend Seasonal":
        m = int(m_val)
        source_pattern = train_series[-m:]
        reps = int(np.ceil(steps / m))
        base_seasonal = np.tile(source_pattern, reps)[:steps]

        if t_type == "Additive":
            # Additive: Apply the linear drift to the seasonal pattern
            drift = (last_val - first_val) / (n_train - 1)
            forecast = base_seasonal + (h_vals * drift)
        else:
            # Multiplicative: Scale by compound growth factor
            # We use a small epsilon to avoid division by zero
            safe_first = max(0.1, first_val)
            safe_last = max(0.1, last_val)
            growth_rate = (safe_last / safe_first) ** (1 / (n_train - 1))
            forecast = base_seasonal * (growth_rate**h_vals)

    return forecast, test_series


# Run engine
forecast_vals, actual_vals = run_baseline_engine(
    df_processed, model_type, periods, validation_mode, seasonal_m, recent_k, trend_type,
)

# --- Data Preparation for Visualization ---
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
    df_historical = df_processed.iloc[:-periods].copy()
    df_historical["Type"] = "Actual (Training)"
    df_ground_truth = df_processed.tail(periods).copy()
    df_ground_truth["Type"] = "Actual (Ground Truth)"
    plot_df = pd.concat([df_historical, df_forecast, df_ground_truth])
else:
    df_actual = df_processed.copy()
    df_actual["Type"] = "Actual"
    plot_df = pd.concat([df_actual, df_forecast])

# View Windows
last_date = df_processed["ds"].max()
if view_mode == "Last 3 Years":
    plot_df = plot_df[plot_df["ds"] >= (last_date - pd.DateOffset(years=3))]
elif view_mode == "Last 12 Months":
    plot_df = plot_df[plot_df["ds"] >= (last_date - pd.DateOffset(months=12))]

# --- Main UI ---
st.title(f"📊 {model_type} Baseline")
st.markdown(f"### Establishing a performance floor at **{resample_freq}** resolution")

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
            f'<div class="metric-card"><div class="metric-label">Forecast Horizon</div><div class="metric-value">{periods} {current_freq}</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Avg Predicted Value</div><div class="metric-value">{np.mean(forecast_vals):.1f}</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Model Logic</div><div class="metric-value">{model_type}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# Main Plot
st.subheader(
    "📅 Baseline Comparison"
    if validation_mode
    else f"📅 {resample_freq} Baseline Forecast",
)

color_map = {
    "Actual": "#94a3b8",
    "Actual (Training)": "#94a3b8",
    "Actual (Ground Truth)": "#f8fafc",
    "Forecast": "#10b981",
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

st.altair_chart(
    alt.layer(historical_line, prediction_line, ground_truth_points)
    .properties(
        width="container",
        height=500,
    )
    .interactive(),
    width="stretch",
)

# --- Details Section ---
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.subheader("Model Definition")
    st.info(
        "💡 **Notation Guide:** $y$ = Interest Score, $t$ = Current Time, $h$ = Forecast Steps, $m$ = Seasonal Period.",
    )
    if model_type == "Naive":
        st.write("The future is exactly like the most recent observation:")
        st.write(r"$$\hat{y}_{t+h} = y_t$$")
    elif model_type == "Seasonal Naive":
        st.write(
            f"The future follows the exact pattern of the last cycle ($m={seasonal_m}$):",
        )
        st.write(r"$$\hat{y}_{t+h} = y_{t+h-m}$$")
    elif model_type == "Trend Naive (Drift)":
        st.write(
            "Allows the naive forecast to increase or decrease over time based on the average historical change (Drift):",
        )
        st.write(r"$$\hat{y}_{t+h} = y_t + h \left( \frac{y_t - y_1}{t-1} \right)$$")
    elif model_type == "Trend Seasonal":
        if trend_type == "Additive":
            st.write(
                f"Combines the seasonal pattern ($m={seasonal_m}$) with a linear drift component:",
            )
            st.write(
                r"$$\hat{y}_{t+h} = y_{t+h-m} + h \left( \frac{y_t - y_1}{t-1} \right)$$",
            )
        else:
            st.write(
                f"Scales the seasonal pattern ($m={seasonal_m}$) by a multiplicative growth factor ($r$):",
            )
            st.write(r"$$\hat{y}_{t+h} = y_{t+h-m} \times r^h$$")
    elif model_type == "Mean":
        st.write("Assumes the future will regress to the long-term historical mean:")
        st.write("$$\\hat{y}_{t+h} = \\bar{y}$$")
    else:
        st.write(
            f"Averages the last **{recent_k}** periods to smooth out noise while respecting the current level.",
        )

with c2:
    st.subheader("The Importance of Baselines")
    st.markdown(
        """
    <div class="explanation-box">
    <b>Drift and Trend baselines</b> are essential for data that shows clear growth. 
    <br><br>
    NFL interest has grown significantly since 2009. A standard Naive model will always under-predict because it ignores the multi-year growth trend. 
    <br><br>
    If your SARIMA model cannot outperform the <b>Trend Seasonal</b> baseline, the model is likely failing to capture the underlying growth correctly.
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "Baseline Playground: Naive, Seasonal Naive, Mean, Recent Mean, Drift, and Trend-Seasonal implementations.",
)
