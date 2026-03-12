import datetime
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Prophet NFL Playground")

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
        border-bottom: 4px solid #f59e0b;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #fbbf24;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Season Schedule Data ---
# Structure: (Preseason Start, Regular Season Start, Super Bowl)
NFL_SCHEDULE = [
    ("2009-08-13", "2009-09-10", "2010-02-07"),
    ("2010-08-12", "2010-09-09", "2011-02-06"),
    ("2011-08-11", "2011-09-08", "2012-02-05"),
    ("2012-08-09", "2012-09-05", "2013-02-03"),
    ("2013-08-08", "2013-09-05", "2014-02-02"),
    ("2014-08-07", "2014-09-04", "2015-02-01"),
    ("2015-08-06", "2015-09-10", "2016-02-07"),
    ("2016-08-04", "2016-09-08", "2017-02-05"),
    ("2017-08-03", "2017-09-07", "2018-02-04"),
    ("2018-08-02", "2018-09-06", "2019-02-03"),
    ("2019-08-01", "2019-09-05", "2020-02-02"),
    ("2020-07-30", "2020-09-10", "2021-02-07"),
    ("2021-07-29", "2021-09-09", "2022-02-13"),
    ("2022-07-28", "2022-09-08", "2023-02-12"),
    ("2023-07-27", "2023-09-07", "2024-02-11"),
    ("2024-07-25", "2024-09-05", "2025-02-09"),
    ("2025-07-31", "2025-09-04", "2026-02-08"),
    ("2026-08-06", "2026-09-10", "2027-02-14"),
]

# Mapping years to Super Bowl names for descriptive UI
SB_ROMANS = {
    2010: "XLIV",
    2011: "XLV",
    2012: "XLVI",
    2013: "XLVII",
    2014: "XLVIII",
    2015: "XLIX",
    2016: "50",
    2017: "LI",
    2018: "LII",
    2019: "LIII",
    2020: "LIV",
    2021: "LV",
    2022: "LVI",
    2023: "LVII",
    2024: "LVIII",
    2025: "LIX",
    2026: "LX",
    2027: "LXI",
}

# Pre-convert seasons to timestamps for faster comparison
SCHEDULE_RANGES = [
    (pd.Timestamp(pre).date(), pd.Timestamp(reg).date(), pd.Timestamp(sb).date())
    for pre, reg, sb in NFL_SCHEDULE
]
sb_dates = [s[2] for s in NFL_SCHEDULE]


def get_detailed_season_flags(dates):
    """Vectorized calculation of NFL Pre, Reg, and Off states"""
    date_objs = pd.to_datetime(dates).dt.date
    is_reg = np.zeros(len(dates), dtype=bool)
    is_pre = np.zeros(len(dates), dtype=bool)

    for pre_start, reg_start, sb_end in SCHEDULE_RANGES:
        is_reg |= ((date_objs >= reg_start) & (date_objs <= sb_end)).values
        is_pre |= ((date_objs >= pre_start) & (date_objs < reg_start)).values

    # Handle future fallback beyond 2027
    mask_future = (pd.to_datetime(dates).dt.year > 2026).values
    months = pd.to_datetime(dates).dt.month.values

    # Simple logic for deep future: Aug is pre, Sept-Feb is regular
    is_pre[mask_future] = np.where(months[mask_future] == 8, True, False)
    is_reg[mask_future] = np.where(
        (months[mask_future] >= 9) | (months[mask_future] <= 2),
        True,
        False,
    )

    return is_pre, is_reg


# --- Data Loading ---
@st.cache_data
def load_nfl_trends():
    df = pd.read_csv("./data/NFL_daily_trends.csv")
    df["ds"] = pd.to_datetime(df["ds"])

    is_pre, is_reg = get_detailed_season_flags(df["ds"])
    df["pre_season"] = is_pre
    df["on_season"] = (
        is_reg  # Note: 'on_season' represents regular season for backward compatibility
    )
    df["off_season"] = ~(is_pre | is_reg)
    df["nfl_season_regressor"] = is_reg.astype(float)
    return df


# --- Load Data ---
df_raw = load_nfl_trends()

# --- Sidebar ---
with st.sidebar:
    st.title("🔮 Prophet Settings")

    st.subheader("1. Mode & Horizon")
    
    validation_mode = st.toggle(
        "Validation Mode (Backtest)", 
        value=False, 
        help="Withholds the last year(s) of data to test the model accuracy."
    )
    
    if validation_mode:
        periods = st.slider("Validation Days (withheld)", 30, 3 * 365, 365)
        st.info(f"Training on data until {df_raw['ds'].iloc[-periods-1].strftime('%Y-%m-%d')}.")
    else:
        periods = st.slider("Days to Forecast (Future)", 1, 5 * 365, 730)

    st.markdown("---")
    st.subheader("2. Seasonal Complexity")

    season_mode = st.radio("Seasonality Mode", ["additive", "multiplicative"], index=0)

    include_yearly = st.checkbox("Yearly (NFL Season Cycle)", value=True)
    fourier_order_year = st.slider("Yearly Fourier Order", 2, 30, 10)

    include_weekly = st.checkbox("Weekly (Game Cycles)", value=False)
    fourier_order_week = st.slider(
        "Weekly Fourier Order",
        1,
        15,
        5,
        disabled=not include_weekly,
    )

    st.markdown("---")
    st.subheader("3. Season Logic (Impact Type)")

    use_regressor = st.checkbox(
        "Global Level Shift",
        value=False,
        help="Adds a flat 'bonus' score to the entire regular season window.",
    )

    use_conditional = st.checkbox(
        "Conditional Weekly Patterns",
        value=False,
        help="Separates Sunday spikes into 'In-Season' and 'Off-Season' versions.",
        disabled=not include_weekly,
    )

    # Visual indentation for Preseason checkbox
    col_indent, col_checkbox = st.columns([0.15, 0.85])
    with col_checkbox:
        use_pre_conditional = st.checkbox(
            "Include Preseason Weekly",
            value=False,
            help="Adds a third weekly pattern specifically for the Preseason (August).",
            disabled=not (include_weekly and use_conditional),
        )

    regressor_prior = st.slider("Regressor Prior Scale", 0.1, 100.0, 10.0)

    st.markdown("---")
    st.subheader("4. Holiday Settings")
    include_superbowl = st.checkbox("Model Super Bowls", value=False)

    lower_win, upper_win, holiday_prior = 1, 1, 10.0
    if include_superbowl:
        lower_win = st.slider("Lower Window (Before)", 0, 14, 0)
        upper_win = st.slider("Upper Window (After)", 0, 14, 0)
        holiday_prior = st.slider("Holiday Prior Scale", 0.01, 100.0, 10.0)

    st.markdown("---")
    st.subheader("5. Model Flexibility")
    cp_flex = st.slider("Trend Flexibility", 0.001, 0.5, 0.05, format="%.3f")
    cp_range = st.slider("Changepoint Range", 0.5, 1.0, 0.8)
    season_prior = st.slider("Seasonality Prior Scale", 0.1, 100.0, 10.0)

    st.markdown("---")
    st.subheader("6. Visualization View")
    view_mode = st.radio(
        "Display Window",
        ["Full History", "Last 3 Years", "Last 6 Months", "Custom Range"],
    )
    if view_mode == "Custom Range":
        min_selectable = df_raw["ds"].min().to_pydatetime()
        max_selectable = (
            df_raw["ds"].max() + datetime.timedelta(days=(0 if validation_mode else periods))
        ).to_pydatetime()
        date_range = st.date_input(
            "Select Date Range",
            value=(min_selectable, max_selectable),
            min_value=min_selectable,
            max_value=max_selectable,
        )

    st.markdown("---")
    st.subheader("7. Growth Constraints")
    growth_type = st.radio(
        "Growth Model",
        ["linear", "logistic"],
        help="Logistic allows setting a mathematical floor (0) and a saturation cap.",
    )

    cap_val = 150.0
    if growth_type == "logistic":
        cap_val = st.slider(
            "Interest Cap (Max Potential)",
            float(df_raw["y"].max()),
            500.0,
            200.0,
            help="The theoretical maximum interest score the trend can reach.",
        )
        st.caption("✨ Floor is strictly set to 0.0 in Logistic mode.")
    else:
        st.caption(
            "⚠️ Linear mode allows values to drop below zero if seasonal pressure is high.",
        )


# --- Optimized Model Execution ---
@st.cache_resource
def run_prophet_engine(
    _df,
    periods,
    is_validation,
    include_weekly,
    conditional_weekly,
    use_pre_cond,
    include_yearly,
    fourier_year,
    fourier_week,
    mode,
    use_reg,
    reg_prior,
    include_sb,
    l_win,
    u_win,
    h_prior,
    flex,
    r_range,
    s_prior,
    growth,
    cap,
):
    # Slice data if in validation mode
    if is_validation:
        train_df = _df.iloc[:-periods].copy()
    else:
        train_df = _df.copy()

    if growth == "logistic":
        train_df["cap"] = cap
        train_df["floor"] = 0.0

    holidays = None
    if include_sb:
        holidays = pd.DataFrame(
            {
                "holiday": "superbowl",
                "ds": pd.to_datetime(sb_dates),
                "lower_window": -l_win,
                "upper_window": u_win,
            },
        )

    m = Prophet(
        growth=growth,
        weekly_seasonality=False,
        yearly_seasonality=False,
        holidays=holidays,
        changepoint_prior_scale=flex,
        changepoint_range=r_range,
        seasonality_mode=mode,
        holidays_prior_scale=h_prior,
    )

    if use_reg:
        m.add_regressor("nfl_season_regressor", mode=mode, prior_scale=reg_prior)

    if include_weekly:
        if conditional_weekly:
            m.add_seasonality(
                name="weekly_on_season",
                period=7,
                fourier_order=fourier_week,
                prior_scale=s_prior,
                condition_name="on_season",
            )
            if use_pre_cond:
                m.add_seasonality(
                    name="weekly_pre_season",
                    period=7,
                    fourier_order=fourier_week,
                    prior_scale=s_prior,
                    condition_name="pre_season",
                )

            # Off-season remains its own flatter pattern
            m.add_seasonality(
                name="weekly_off_season",
                period=7,
                fourier_order=3,
                prior_scale=s_prior,
                condition_name="off_season",
            )
        else:
            m.add_seasonality(
                name="weekly",
                period=7,
                fourier_order=fourier_week,
                prior_scale=s_prior,
            )

    if include_yearly:
        m.add_seasonality(
            name="yearly",
            period=365.25,
            fourier_order=fourier_year,
            prior_scale=s_prior,
        )

    m.fit(train_df)

    fut = m.make_future_dataframe(periods=periods)
    fut_pre, fut_reg = get_detailed_season_flags(fut["ds"])
    fut["pre_season"] = fut_pre
    fut["on_season"] = fut_reg
    fut["off_season"] = ~(fut_pre | fut_reg)
    fut["nfl_season_regressor"] = fut["on_season"].astype(float)

    if growth == "logistic":
        fut["cap"] = cap
        fut["floor"] = 0.0

    fcst = m.predict(fut)

    reg_coef = 0.0
    if use_reg:
        try:
            from prophet.utilities import regressor_coefficients

            coeffs = regressor_coefficients(m)
            reg_coef = coeffs[coeffs["regressor"] == "nfl_season_regressor"][
                "coef"
            ].values[0]
        except:
            pass

    return fcst, reg_coef


# Run the cached engine
forecast, reg_impact = run_prophet_engine(
    df_raw,
    periods,
    validation_mode,
    include_weekly,
    use_conditional,
    use_pre_conditional,
    include_yearly,
    fourier_order_year,
    fourier_order_week,
    season_mode,
    use_regressor,
    regressor_prior,
    include_superbowl,
    lower_win,
    upper_win,
    holiday_prior,
    cp_flex,
    cp_range,
    season_prior,
    growth_type,
    cap_val,
)

# --- Visualization Filtering ---
if validation_mode:
    # Split historical data into training and testing
    historical_train = df_raw.iloc[:-periods].copy()
    historical_train["Type"] = "Actual (Training)"
    
    historical_test = df_raw.iloc[-periods:].copy()
    historical_test["Type"] = "Actual (Ground Truth)"
    
    # Prediction on the test window
    forecast_window = forecast.tail(periods).copy()
    forecast_window["Type"] = "Validation Prediction"
    
    plot_df = pd.concat([
        historical_train[["ds", "y", "Type"]],
        historical_test[["ds", "y", "Type"]],
        forecast_window[["ds", "yhat", "Type"]].rename(columns={"yhat": "y"})
    ])
    
    # Accuracy Metrics Calculation
    y_true = historical_test['y'].values
    y_pred = forecast_window['yhat'].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
else:
    historical = df_raw.copy()
    historical["Type"] = "Actual"
    forecast_only = forecast[forecast["ds"] > df_raw["ds"].max()].copy()
    forecast_only["Type"] = "Forecast"
    plot_df = pd.concat(
        [
            historical[["ds", "y", "Type"]],
            forecast_only[["ds", "yhat", "Type"]].rename(columns={"yhat": "y"}),
        ],
    )

# Filtering view
if view_mode == "Last 6 Months":
    filter_start = df_raw["ds"].max() - datetime.timedelta(days=180)
    plot_df = plot_df[plot_df["ds"] >= filter_start]
    forecast_display = forecast[forecast["ds"] >= filter_start]
elif view_mode == "Last 3 Years":
    filter_start = df_raw["ds"].max() - pd.DateOffset(years=3)
    plot_df = plot_df[plot_df["ds"] >= filter_start]
    forecast_display = forecast[forecast["ds"] >= filter_start]
elif view_mode == "Custom Range" and len(date_range) == 2:
    plot_df = plot_df[
        (plot_df["ds"] >= pd.to_datetime(date_range[0]))
        & (plot_df["ds"] <= pd.to_datetime(date_range[1]))
    ]
    forecast_display = forecast[
        (forecast["ds"] >= pd.to_datetime(date_range[0]))
        & (forecast["ds"] <= pd.to_datetime(date_range[1]))
    ]
else:
    forecast_display = forecast

# --- UI Render ---
st.title("🔮 Prophet NFL Trends Playground")

m1, m2, m3 = st.columns(3)

if validation_mode:
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Mean Absolute Error (MAE)</div><div class="metric-value">{mae:.2f}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Root Mean Sq. Error (RMSE)</div><div class="metric-value">{rmse:.2f}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Validation Period</div><div class="metric-value">{periods} days</div></div>', unsafe_allow_html=True)
else:
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Predicted Peak Interest</div><div class="metric-value">{forecast["yhat"].max():.1f}</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        growth_label = (
            "Saturation Headroom" if growth_type == "logistic" else "Base Trend Growth"
        )
        if growth_type == "logistic":
            headroom = cap_val - forecast["trend"].iloc[-1]
            growth_val = f"{headroom:.1f}"
        else:
            growth_pct = (forecast["trend"].iloc[-1] / forecast["trend"].iloc[0]) - 1
            growth_val = f"{growth_pct:+.1%}"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{growth_label}</div><div class="metric-value">{growth_val}</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        reg_label = (
            "Season Bonus (Shift)" if season_mode == "multiplicative" else "Season Offset"
        )
        reg_val = (
            f"+{reg_impact:.1%}"
            if season_mode == "multiplicative"
            else f"+{reg_impact:.1f}"
        )
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{reg_label}</div><div class="metric-value">{reg_val if use_regressor else "N/A"}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# --- Forecast Plot ---
st.subheader("📅 Accuracy Check: Actual vs. Predicted" if validation_mode else "📈 Interest Forecast")

y_axis_scale = alt.Scale(domainMin=0) if growth_type == "logistic" else alt.Scale()

# Legend Map
color_map = {
    "Actual": "#94a3b8",
    "Actual (Training)": "#94a3b8",
    "Actual (Ground Truth)": "#f8fafc",
    "Forecast": "#fbbf24",
    "Validation Prediction": "#fbbf24",
}
color_domain = [t for t in color_map.keys() if t in plot_df["Type"].unique()]
color_range = [color_map[t] for t in color_domain]

# 1. Uncertainty Band
band = (
    alt.Chart(forecast_display)
    .mark_area(opacity=0.3, color="#fbbf24")
    .encode(
        x="ds:T",
        y="yhat_lower:Q",
        y2="yhat_upper:Q",
        tooltip=[
            alt.Tooltip("ds:T", title="Date"),
            alt.Tooltip("yhat:Q", title="Predicted", format=".1f"),
        ],
    )
)

# 2. Base Chart Encoding
base = alt.Chart(plot_df).encode(
    x=alt.X("ds:T", title="Date"),
    y=alt.Y("y:Q", title="Interest Score", scale=y_axis_scale),
    color=alt.Color(
        "Type:N",
        scale=alt.Scale(domain=color_domain, range=color_range),
        legend=alt.Legend(title="Data Series"),
    ),
    tooltip=[
        alt.Tooltip("ds:T", title="Date"),
        alt.Tooltip("y:Q", title="Value", format=".1f"),
        alt.Tooltip("Type:N"),
    ],
)

# 3. Layer: Historical Line (Actual/Training)
historical_line = base.transform_filter(
    (alt.datum.Type == "Actual") | (alt.datum.Type == "Actual (Training)")
).mark_line(opacity=0.7)

# 4. Layer: Ground Truth Dots
ground_truth_points = base.transform_filter(
    alt.datum.Type == "Actual (Ground Truth)"
).mark_point(size=60, filled=True, opacity=1.0)

# 5. Layer: Prediction Line (Dashed) - Placed last to be on top
prediction_line = base.transform_filter(
    (alt.datum.Type == "Forecast") | (alt.datum.Type == "Validation Prediction")
).mark_line(strokeDash=[5, 5])

# 6. Combined Chart
final_chart = alt.layer(
    band, historical_line, ground_truth_points, prediction_line
).properties(width="container", height=450).interactive()

st.altair_chart(final_chart, width="stretch")

# --- Components ---
st.markdown("---")
st.subheader("🧩 Decomposing the NFL Cycle")

col_trend, col_season = st.columns([1, 1])

with col_trend:
    st.markdown("**Underlying Growth Trend**")
    st.altair_chart(
        alt.Chart(forecast)
        .mark_line(color="#fbbf24", size=3)
        .encode(x="ds:T", y="trend:Q")
        .properties(width="container", height=250)
        .interactive(),
        width="stretch",
    )

with col_season:
    st.markdown("**Yearly Pattern (Fixed Calendar Cycle)**")
    if include_yearly:
        yearly_comp = forecast[["ds", "yearly"]].iloc[:365].copy()
        st.altair_chart(
            alt.Chart(yearly_comp)
            .mark_line(color="#fbbf24", strokeWidth=3)
            .encode(x=alt.X("ds:T", axis=alt.Axis(format="%b")), y="yearly:Q")
            .properties(width="container", height=250),
            width="stretch",
        )
    else:
        st.info("Yearly seasonality is disabled.")

st.markdown("**Weekly Dynamics & Impact**")
comp_week, comp_impact = st.columns([1.5, 1])

with comp_week:
    st.markdown("**Weekly Pattern Comparison**")
    if include_weekly:
        days_map = {
            "Monday": "Mon.",
            "Tuesday": "Tue.",
            "Wednesday": "Wed.",
            "Thursday": "Thu.",
            "Friday": "Fri.",
            "Saturday": "Sat.",
            "Sunday": "Sun.",
        }

        if use_conditional:
            all_week_data = []

            # 1. In-Season
            on_data = forecast[["ds", "weekly_on_season"]].copy()
            on_data["Day"] = on_data["ds"].dt.day_name().map(days_map)
            on_summary = on_data.groupby("Day")["weekly_on_season"].mean().reset_index()
            on_summary["Phase"] = "Regular Season"
            on_summary = on_summary.rename(columns={"weekly_on_season": "Adjustment"})
            all_week_data.append(on_summary)

            # 2. Preseason (Conditional)
            if use_pre_conditional and "weekly_pre_season" in forecast.columns:
                pre_data = forecast[["ds", "weekly_pre_season"]].copy()
                pre_data["Day"] = pre_data["ds"].dt.day_name().map(days_map)
                pre_summary = (
                    pre_data.groupby("Day")["weekly_pre_season"].mean().reset_index()
                )
                pre_summary["Phase"] = "Preseason"
                pre_summary = pre_summary.rename(
                    columns={"weekly_pre_season": "Adjustment"},
                )
                all_week_data.append(pre_summary)

            # 3. Off-Season
            off_data = forecast[["ds", "weekly_off_season"]].copy()
            off_data["Day"] = off_data["ds"].dt.day_name().map(days_map)
            off_summary = (
                off_data.groupby("Day")["weekly_off_season"].mean().reset_index()
            )
            off_summary["Phase"] = "Off-Season"
            off_summary = off_summary.rename(
                columns={"weekly_off_season": "Adjustment"},
            )
            all_week_data.append(off_summary)

            week_summary = pd.concat(all_week_data)

            week_chart = (
                alt.Chart(week_summary)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "Day:N",
                        sort=["Mon.", "Tue.", "Wed.", "Thu.", "Fri.", "Sat.", "Sun."],
                        title="Day of Week",
                        axis=alt.Axis(labelAngle=0),
                    ),
                    y=alt.Y("Adjustment:Q", title="Multiplier / Adjustment"),
                    xOffset="Phase:N",
                    color=alt.Color(
                        "Phase:N",
                        scale=alt.Scale(
                            domain=["Regular Season", "Preseason", "Off-Season"],
                            range=["#fbbf24", "#f59e0b", "#475569"],
                        ),
                    ),
                )
                .properties(height=250)
            )

            st.altair_chart(week_chart, width="stretch")
        else:
            weekly_comp = forecast[["ds", "weekly"]].copy()
            weekly_comp["Day"] = weekly_comp["ds"].dt.day_name().map(days_map)
            weekly_summary = (
                weekly_comp.groupby("Day")["weekly"]
                .mean()
                .reindex(["Mon.", "Tue.", "Wed.", "Thu.", "Fri.", "Sat.", "Sun."])
                .reset_index()
            )
            st.altair_chart(
                alt.Chart(weekly_summary)
                .mark_bar(color="#fbbf24")
                .encode(
                    x=alt.X("Day:N", sort=None, axis=alt.Axis(labelAngle=0)),
                    y="weekly:Q",
                )
                .properties(width="container", height=250),
                width="stretch",
            )
    else:
        st.info("Weekly seasonality is disabled.")

with comp_impact:
    st.markdown("**Season Regressor Impact**")
    if use_regressor:
        reg_display_col = (
            "extra_regressors_multiplicative"
            if (
                season_mode == "multiplicative"
                and "extra_regressors_multiplicative" in forecast.columns
            )
            else "nfl_season_regressor"
        )
        if reg_display_col in forecast.columns:
            reg_comp = forecast[["ds", reg_display_col]].iloc[: 365 * 2]
            st.altair_chart(
                alt.Chart(reg_comp)
                .mark_line(color="#fbbf24")
                .encode(x="ds:T", y=f"{reg_display_col}:Q")
                .properties(width="container", height=250)
                .interactive(),
                width="stretch",
            )
            st.caption("Binary boost applied when season is active.")
        else:
            st.info("Computing impact...")
    else:
        st.info("Global Regressor is disabled.")

if include_superbowl and "superbowl" in forecast.columns:
    st.markdown("**Special Events**")
    last_sb_dt = pd.to_datetime(sb_dates[-2])
    sb_roman = SB_ROMANS.get(last_sb_dt.year, "")
    sb_title_text = (
        f"Days relative to Super Bowl {sb_roman} ({last_sb_dt.strftime('%b %Y')})"
    )
    st.markdown(f"**{sb_title_text}**")
    sb_sample = forecast[
        (forecast["ds"] >= last_sb_dt - datetime.timedelta(days=lower_win + 5))
        & (forecast["ds"] <= last_sb_dt + datetime.timedelta(days=upper_win + 5))
    ]
    sb_chart = (
        alt.Chart(sb_sample)
        .mark_area(color="#fbbf24", opacity=0.6)
        .encode(
            x=alt.X("ds:T", title=sb_title_text),
            y=alt.Y("superbowl:Q", title="Score Adjustment"),
            tooltip=[
                alt.Tooltip("ds:T", title="Date"),
                alt.Tooltip("superbowl:Q", title="Effect", format=".2f"),
            ],
        )
    )
    st.altair_chart(
        sb_chart.properties(width="container", height=250).interactive(),
        width="stretch",
    )