import datetime
import gc
import threading

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Prophet NFL Playground")


# Thread-safe lock for Prophet model training
if "model_lock" not in st.session_state:
    st.session_state.model_lock = threading.Lock()

# Custom CSS for the Playground aesthetic (Updated to support modern Streamlit selectors)
st.markdown(
    """
<style>
    .stApp, [data-testid="stAppViewContainer"], .reportview-container {
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

# Chronological Structure: (Draft Date, Preseason Start, Regular Season Start, Postseason Start, Super Bowl Date)
NFL_SCHEDULE = [
    ("2009-04-25", "2009-08-13", "2009-09-10", "2010-01-09", "2010-02-07"),
    ("2010-04-22", "2010-08-12", "2010-09-09", "2011-01-08", "2011-02-06"),
    ("2011-04-28", "2011-08-11", "2011-09-08", "2012-01-07", "2012-02-05"),
    ("2012-04-26", "2012-08-09", "2012-09-05", "2013-01-05", "2013-02-03"),
    ("2013-04-25", "2013-08-08", "2013-09-05", "2014-01-04", "2014-02-02"),
    ("2014-05-08", "2014-08-07", "2014-09-04", "2015-01-03", "2015-02-01"),
    ("2015-04-30", "2015-08-06", "2015-09-10", "2016-01-09", "2016-02-07"),
    ("2016-04-28", "2016-08-04", "2016-09-08", "2017-01-07", "2017-02-05"),
    ("2017-04-27", "2017-08-03", "2017-09-07", "2018-01-06", "2018-02-04"),
    ("2018-04-26", "2018-08-02", "2018-09-06", "2019-01-05", "2019-02-03"),
    (
        "2019-04-25",
        "2019-08-01",
        "2019-09-05",
        "2020-02-02",
        "2020-02-02",
    ),  # Fallback SB alignment
    ("2020-04-23", "2020-07-30", "2020-09-10", "2021-01-09", "2021-02-07"),
    ("2021-04-29", "2021-07-29", "2021-09-09", "2022-01-15", "2022-02-13"),
    ("2022-04-28", "2022-07-28", "2022-09-08", "2023-01-14", "2023-02-12"),
    ("2023-04-27", "2023-07-27", "2023-09-07", "2024-01-13", "2024-02-11"),
    ("2024-04-25", "2024-07-25", "2024-09-05", "2025-01-11", "2025-02-09"),
    ("2025-04-24", "2025-07-31", "2025-09-04", "2026-01-10", "2026-02-08"),
    ("2026-04-23", "2026-08-06", "2026-09-10", "2027-01-16", "2027-02-14"),
    ("2027-04-29", "2027-08-05", "2027-09-09", "2028-01-15", "2028-02-13"),
]

# Extract derived components dynamically
DRAFT_DATES = [s[0] for s in NFL_SCHEDULE]
sb_dates = [s[4] for s in NFL_SCHEDULE]

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
    2028: "LXII",
}

# Pre-convert seasons to timestamps for faster comparison
SCHEDULE_RANGES = [
    (
        pd.Timestamp(pre).date(),
        pd.Timestamp(reg).date(),
        pd.Timestamp(post).date(),
        pd.Timestamp(sb).date(),
    )
    for draft, pre, reg, post, sb in NFL_SCHEDULE
]


def get_detailed_season_flags(dates):
    """Vectorized calculation of NFL Pre, Reg, Post, and Off states"""
    date_series = pd.to_datetime(dates)
    date_objs = pd.to_datetime(dates).dt.date

    is_pre = np.zeros(len(dates), dtype=bool)
    is_reg = np.zeros(len(dates), dtype=bool)
    is_post = np.zeros(len(dates), dtype=bool)

    for pre_start, reg_start, post_start, sb_end in SCHEDULE_RANGES:
        is_pre |= ((date_objs >= pre_start) & (date_objs < reg_start)).values
        is_reg |= ((date_objs >= reg_start) & (date_objs < post_start)).values
        is_post |= ((date_objs >= post_start) & (date_objs <= sb_end)).values

    # Handle future fallback beyond schedule range
    max_scheduled_date = SCHEDULE_RANGES[-1][3]  # e.g., 2028-02-13
    mask_future = (date_objs > max_scheduled_date).values

    if np.any(mask_future):
        months = date_series.dt.month.values
        is_pre[mask_future] = months[mask_future] == 8
        is_reg[mask_future] = (months[mask_future] >= 9) & (months[mask_future] <= 12)
        is_post[mask_future] = (months[mask_future] >= 1) & (months[mask_future] <= 2)

    return is_pre, is_reg, is_post


# --- Data Loading ---
@st.cache_data
def load_nfl_trends():
    df = pd.read_csv("./data/NFL_daily_trends.csv")
    df["ds"] = pd.to_datetime(df["ds"])

    is_pre, is_reg, is_post = get_detailed_season_flags(df["ds"])

    df["pre_season"] = is_pre.astype(bool)
    df["reg_season"] = is_reg.astype(bool)
    df["post_season"] = is_post.astype(bool)
    df["on_season"] = (is_reg | is_post).astype(bool)
    df["off_season"] = (~(is_pre | is_reg | is_post)).astype(bool)
    df["y"] = df["y"].astype(np.float32)
    return df


# --- Load Data ---
df_raw = load_nfl_trends()

# --- Sidebar ---
with st.sidebar:
    st.markdown(" ")
    st.markdown(" ")
    st.markdown("---")
    st.markdown(" ")
    st.title("🔮 Prophet Settings")

    st.subheader("1. Mode & Horizon")

    validation_mode = st.toggle(
        "Validation Mode (Backtest)",
        value=True,
        help="Withholds the last year(s) of data to test the model accuracy.",
    )

    if validation_mode:
        periods = st.slider("Validation Days (withheld)", 30, 3 * 365, 365)
        st.info(
            f"Training on data until {df_raw['ds'].iloc[-periods - 1].strftime('%Y-%m-%d')}.",
        )
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
        "Seasonal Regressor",
        value=False,
        help="Adds a flat 'bonus' score to the NFL season window.",
    )

    use_different_post_offset = False
    post_offset_mult = 1.0
    use_pre_offset = False
    pre_offset_mult = 0.0

    if use_regressor:
        col_reg_indent, col_reg_widgets = st.columns([0.15, 0.85])
        with col_reg_widgets:
            use_different_post_offset = st.checkbox(
                "Different Postseason Offset",
                value=False,
                help="Tune the postseason regressor offset level separate from regular season.",
            )
            if use_different_post_offset:
                post_offset_mult = st.slider(
                    "Postseason Offset Multiplier",
                    min_value=0.0,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="Offset strength multiplier during postseason (relative to standard regular season offset).",
                )

            use_pre_offset = st.checkbox(
                "Include Preseason Offset",
                value=False,
                help="Add a flat bonus offset to the preseason window.",
            )
            if use_pre_offset:
                pre_offset_mult = st.slider(
                    "Preseason Offset Multiplier",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.05,
                    help="Offset strength multiplier during preseason (relative to standard regular season offset).",
                )

    use_conditional = st.checkbox(
        "Conditional Weekly Patterns",
        value=False,
        help="Separates Sunday spikes into 'In-Season' and 'Off-Season' versions.",
        disabled=not include_weekly,
    )

    # Visual indentation for Preseason & Postseason checkboxes
    use_pre_conditional = False
    use_post_conditional = False
    if include_weekly and use_conditional:
        col_indent, col_checkbox = st.columns([0.15, 0.85])
        with col_checkbox:
            use_pre_conditional = st.checkbox(
                "Include Preseason Weekly",
                value=False,
                help="Adds a third weekly pattern specifically for the Preseason (August).",
            )
            use_post_conditional = st.checkbox(
                "Include Postseason Weekly",
                value=False,
                help="Adds a separate fourth weekly pattern specifically for the Postseason (January-February).",
            )

    st.markdown("---")
    st.subheader("4. Holiday Settings")

    # Inbuilt Country Holidays Option (Normal Bank Holidays)
    add_country_holidays = st.checkbox(
        "Enable Inbuilt Bank Holidays",
        value=False,
        help="Model country-specific calendar events automatically using Prophet's library.",
    )
    country_code = "US"
    if add_country_holidays:
        country_code = st.selectbox(
            "Select Country Calendar",
            options=["US", "CA", "GB", "DE", "FR", "JP", "AU"],
            index=0,
            help="US is highly recommended for NFL search interest modelling.",
        )

    st.markdown("##### Special NFL Events")

    # NFL Draft custom holiday configuration
    include_draft = st.checkbox("Model NFL Draft", value=False)
    draft_lower_win, draft_upper_win = 1, 2
    if include_draft:
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            draft_lower_win = st.slider(
                "Draft Lower Window (Before)",
                0,
                7,
                1,
                help="Include days leading up to first round.",
            )
        with d_col2:
            draft_upper_win = st.slider(
                "Draft Upper Window (After)",
                0,
                7,
                2,
                help="Include draft weekend rounds 2-7.",
            )

    # Super Bowl custom holiday configuration
    include_superbowl = st.checkbox("Model Super Bowls", value=False)
    lower_win, upper_win = 1, 1
    if include_superbowl:
        sb_col1, sb_col2 = st.columns(2)
        with sb_col1:
            lower_win = st.slider("Super Bowl Lower Window (Before)", 0, 14, 1)
        with sb_col2:
            upper_win = st.slider("Super Bowl Upper Window (After)", 0, 14, 1)

    st.markdown("---")
    st.subheader("5. Model Flexibility")
    cp_flex = st.slider("Trend Flexibility", 0.001, 0.5, 0.05, format="%.3f")
    cp_range = st.slider("Changepoint Range", 0.5, 1.0, 0.8)

    # Dynamic consolidated prior scales
    # Seasonality Prior Scale (Visible if either Yearly or Weekly is active)
    if include_yearly or include_weekly:
        season_prior = st.slider("Seasonality Prior Scale", 0.1, 100.0, 10.0)
    else:
        season_prior = 10.0

    # Regressor Prior Scale (Visible if Seasonal Regressor is active)
    if use_regressor:
        regressor_prior = st.slider("Regressor Prior Scale", 0.1, 100.0, 10.0)
    else:
        regressor_prior = 10.0

    # Holidays/Events Prior Scale (Visible if any Custom or Bank Holiday option is active)
    if add_country_holidays or include_draft or include_superbowl:
        holiday_prior = st.slider("Holidays/Events Prior Scale", 0.01, 100.0, 10.0)
    else:
        holiday_prior = 10.0

    st.markdown("---")
    st.subheader("6. Visualization View")
    view_mode = st.radio(
        "Display Window",
        ["Full History", "Last 3 Years", "Last 6 Months", "Custom Range"],
    )
    if view_mode == "Custom Range":
        min_selectable = df_raw["ds"].min().to_pydatetime()
        max_selectable = (
            df_raw["ds"].max()
            + datetime.timedelta(days=(0 if validation_mode else periods))
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

@st.cache_resource(max_entries=1)
def run_prophet_engine(
    _df,
    periods,
    is_validation,
    include_weekly,
    conditional_weekly,
    use_pre_cond,
    use_post_cond,
    include_yearly,
    fourier_year,
    fourier_week,
    mode,
    use_reg,
    use_diff_post,
    post_mult,
    use_pre_off,
    pre_mult,
    reg_prior,
    add_country_hols,
    country_name,
    include_drft,
    drft_l_win,
    drft_u_win,
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
    train_df = _df.iloc[:-periods].copy() if is_validation else _df.copy()

    if growth == "logistic":
        train_df["cap"] = cap
        train_df["floor"] = 0.0

    # Dynamically build nfl_season_regressor based on postseason and preseason multipliers
    p_mult = pre_mult if use_pre_off else 0.0
    po_mult = post_mult if use_diff_post else 1.0

    train_df["nfl_season_regressor"] = np.where(
        train_df["reg_season"],
        1.0,
        np.where(
            train_df["post_season"],
            po_mult,
            np.where(train_df["pre_season"], p_mult, 0.0),
        ),
    )

    # Build custom holiday dataframes
    holiday_dfs = []
    if include_sb:
        sb_df = pd.DataFrame(
            {
                "holiday": "superbowl",
                "ds": pd.to_datetime(sb_dates),
                "lower_window": -l_win,
                "upper_window": u_win,
            },
        )
        holiday_dfs.append(sb_df)

    if include_drft:
        drft_df = pd.DataFrame(
            {
                "holiday": "nfl_draft",
                "ds": pd.to_datetime(DRAFT_DATES),
                "lower_window": -drft_l_win,
                "upper_window": drft_u_win,
            },
        )
        holiday_dfs.append(drft_df)

    holidays = pd.concat(holiday_dfs, ignore_index=True) if holiday_dfs else None

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

    # Inbuilt country holidays added directly to Prophet model instance
    if add_country_hols:
        m.add_country_holidays(country_name=country_name)

    if use_reg:
        m.add_regressor("nfl_season_regressor", mode=mode, prior_scale=reg_prior)

    if include_weekly:
        if conditional_weekly:
            # If postseason has its own pattern, regular on_season seasonality applies only to reg_season dates
            on_season_cond = "reg_season" if use_post_cond else "on_season"

            m.add_seasonality(
                name="weekly_on_season",
                period=7,
                fourier_order=fourier_week,
                prior_scale=s_prior,
                condition_name=on_season_cond,
            )
            if use_pre_cond:
                m.add_seasonality(
                    name="weekly_pre_season",
                    period=7,
                    fourier_order=fourier_week,
                    prior_scale=s_prior,
                    condition_name="pre_season",
                )
            if use_post_cond:
                m.add_seasonality(
                    name="weekly_post_season",
                    period=7,
                    fourier_order=fourier_week,
                    prior_scale=s_prior,
                    condition_name="post_season",
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
    fut_pre, fut_reg, fut_post = get_detailed_season_flags(fut["ds"])
    fut["pre_season"] = fut_pre
    fut["reg_season"] = fut_reg
    fut["post_season"] = fut_post
    fut["on_season"] = fut_reg | fut_post
    fut["off_season"] = ~(fut_pre | fut_reg | fut_post)

    fut["nfl_season_regressor"] = np.where(
        fut["reg_season"],
        1.0,
        np.where(
            fut["post_season"],
            po_mult,
            np.where(fut["pre_season"], p_mult, 0.0),
        ),
    )

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
            ].to_numpy()[0]
        except:
            pass

    # Safely retrieve list of trained holidays (custom + country)
    try:
        train_holidays = list(m.train_holiday_names)
    except:
        train_holidays = []

    del m
    gc.collect()

    return fcst, reg_coef, train_holidays


# Run the engine
with st.session_state.model_lock:
    forecast, reg_impact, trained_holidays = run_prophet_engine(
        df_raw,
        periods,
        validation_mode,
        include_weekly,
        use_conditional,
        use_pre_conditional,
        use_post_conditional,
        include_yearly,
        fourier_order_year,
        fourier_order_week,
        season_mode,
        use_regressor,
        use_different_post_offset,
        post_offset_mult,
        use_pre_offset,
        pre_offset_mult,
        regressor_prior,
        add_country_holidays,
        country_code,
        include_draft,
        draft_lower_win,
        draft_upper_win,
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

# Extract individual bank holiday effects dynamically for robust tooltip descriptions
bank_holiday_columns = [
    h for h in trained_holidays if h not in ["superbowl", "nfl_draft"]
]
bank_holiday_data = []

if bank_holiday_columns:
    cols_present = [col for col in bank_holiday_columns if col in forecast.columns]
    for col in cols_present:
        holiday_rows = forecast[forecast[col].abs() > 0.001][["ds", col]].copy()
        if not holiday_rows.empty:
            holiday_rows = holiday_rows.rename(columns={col: "effect"})
            holiday_rows["holiday_name"] = col
            bank_holiday_data.append(holiday_rows)

if bank_holiday_data:
    bank_holidays_df = pd.concat(bank_holiday_data, ignore_index=True)
    cols_present = [col for col in bank_holiday_columns if col in forecast.columns]
    forecast["bank_holidays"] = forecast[cols_present].sum(axis=1)
else:
    bank_holidays_df = pd.DataFrame(columns=["ds", "effect", "holiday_name"])
    forecast["bank_holidays"] = 0.0

# --- Visualization Filtering ---
if validation_mode:
    historical_train = df_raw.iloc[:-periods].copy()
    historical_train["Type"] = "Actual (Training)"

    historical_test = df_raw.iloc[-periods:].copy()
    historical_test["Type"] = "Actual (Ground Truth)"

    forecast_window = forecast.tail(periods).copy()
    forecast_window["Type"] = "Validation Prediction"

    plot_df = pd.concat(
        [
            historical_train[["ds", "y", "Type"]],
            historical_test[["ds", "y", "Type"]],
            forecast_window[["ds", "yhat", "Type"]].rename(columns={"yhat": "y"}),
        ],
    )

    y_true = historical_test["y"].values
    y_pred = forecast_window["yhat"].values
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

# Enrich plot_df with predictions and metadata to enable rich tooltips directly on all lines/dots
plot_df = pd.merge(plot_df, forecast[["ds", "yhat"]], on="ds", how="left")
plot_df = plot_df.rename(columns={"yhat": "Predicted"})
plot_df["Actual"] = np.where(
    plot_df["Type"].isin(["Actual", "Actual (Training)", "Actual (Ground Truth)"]),
    plot_df["y"],
    np.nan,
)
plot_df["Delta"] = plot_df["Actual"] - plot_df["Predicted"]
plot_df["Pct_Error"] = (plot_df["Delta"].abs() / plot_df["Actual"]) * 100

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
            f'<div class="metric-card"><div class="metric-label">Validation Period</div><div class="metric-value">{periods} days</div></div>',
            unsafe_allow_html=True,
        )
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
            "Season Bonus (Shift)"
            if season_mode == "multiplicative"
            else "Season Offset"
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
st.subheader(
    "📅 Accuracy Check: Actual vs. Predicted"
    if validation_mode
    else "📈 Interest Forecast",
)

y_axis_scale = alt.Scale(domainMin=0) if growth_type == "logistic" else alt.Scale()

color_map = {
    "Actual": "#94a3b8",
    "Actual (Training)": "#94a3b8",
    "Actual (Ground Truth)": "#f8fafc",
    "Forecast": "#fbbf24",
    "Validation Prediction": "#fbbf24",
}
color_domain = [t for t in color_map if t in plot_df["Type"].unique()]
color_range = [color_map[t] for t in color_domain]

# Detailed common tooltips shared across base lines and dots
detailed_tooltips = [
    alt.Tooltip("ds:T", title="Date", format="%b %d, %Y"),
    alt.Tooltip("Actual:Q", title="Actual Value", format=".1f"),
    alt.Tooltip("Predicted:Q", title="Predicted Value", format=".1f"),
    alt.Tooltip("Delta:Q", title="Delta (Actual - Pred)", format="+.1f"),
    alt.Tooltip("Pct_Error:Q", title="Percentage Error (%)", format=".1f"),
]

# 1. Uncertainty Band
band = (
    alt.Chart(forecast_display)
    .mark_area(opacity=0.3, color="#fbbf24")
    .encode(
        x="ds:T",
        y="yhat_lower:Q",
        y2="yhat_upper:Q",
        tooltip=alt.value(None),  # suppress single series default tooltips
    )
)

# 2. Base Chart Encoding with integrated default tooltips for direct hover on visual line coordinates
base = alt.Chart(plot_df).encode(
    x=alt.X("ds:T", title="Date"),
    y=alt.Y("y:Q", title="Interest Score", scale=y_axis_scale),
    color=alt.Color(
        "Type:N",
        scale=alt.Scale(domain=color_domain, range=color_range),
        legend=alt.Legend(title="Data Series"),
    ),
    tooltip=detailed_tooltips,
)

# 3. Layer: Historical Line
historical_line = base.transform_filter(
    (alt.datum.Type == "Actual") | (alt.datum.Type == "Actual (Training)"),
).mark_line(opacity=0.7)

# 4. Layer: Ground Truth Dots - explicitly equipped with detailed tooltip properties
ground_truth_points = (
    base.transform_filter(
        alt.datum.Type == "Actual (Ground Truth)",
    )
    .mark_point(size=60, filled=True, opacity=1.0)
    .encode(tooltip=detailed_tooltips)
)

# 5. Layer: Prediction Line (Dashed)
prediction_line = base.transform_filter(
    (alt.datum.Type == "Forecast") | (alt.datum.Type == "Validation Prediction"),
).mark_line(strokeDash=[5, 5])

# Combined chart layer without transparent selector overlays, lines, rules or points
# This completely removes the selection-fading / gray dimming layer artifact
final_chart = (
    alt.layer(
        band,
        historical_line,
        ground_truth_points,
        prediction_line,
    )
    .properties(width="container", height=450, background="transparent")
    .interactive()
)

st.altair_chart(final_chart, width="stretch")

# --- Decomposition Components & Equation Formulator ---
st.markdown("---")
st.subheader("🧩 Decomposing the NFL Cycle")

# Display formula details directly to the user in a styled, left-aligned code block
st.markdown("**Exact Mathematical Model Equation (Active Parameters)**")

# Format the formula as a clean, multi-line mathematical text representation
formula_lines = []
if season_mode == "additive":
    formula_lines.append("y(t) = trend(t)")
    if include_yearly:
        formula_lines.append("       + yearly(t)")
    if include_weekly:
        w_terms_txt = []
        if use_conditional:
            w_terms_txt.append("weekly_off(t)")
            if use_post_conditional:
                w_terms_txt.append("weekly_reg(t)")
                w_terms_txt.append("weekly_post(t)")
            else:
                w_terms_txt.append("weekly_on(t)")
            if use_pre_conditional:
                w_terms_txt.append("weekly_pre(t)")
        else:
            w_terms_txt.append("weekly(t)")
        formula_lines.append("       + " + " + ".join(w_terms_txt))
    if use_regressor:
        formula_lines.append("       + season_reg(t)")

    h_terms_txt = []
    if include_superbowl:
        h_terms_txt.append("superbowl(t)")
    if include_draft:
        h_terms_txt.append("draft(t)")
    if add_country_holidays:
        h_terms_txt.append("bank_holidays(t)")
    if h_terms_txt:
        formula_lines.append("       + " + " + ".join(h_terms_txt))
    formula_lines.append("       + error")
else:
    # Multiplicative formula structure
    has_other_terms = (
        include_yearly
        or include_weekly
        or use_regressor
        or include_superbowl
        or include_draft
        or add_country_holidays
    )
    if not has_other_terms:
        formula_lines.append("y(t) = trend(t) + error")
    else:
        formula_lines.append("y(t) = trend(t) * (")
        formula_lines.append("         1")
        if include_yearly:
            formula_lines.append("         + yearly(t)")
        if include_weekly:
            w_terms_txt = []
            if use_conditional:
                w_terms_txt.append("weekly_off(t)")
                if use_post_conditional:
                    w_terms_txt.append("weekly_reg(t)")
                    w_terms_txt.append("weekly_post(t)")
                else:
                    w_terms_txt.append("weekly_on(t)")
                if use_pre_conditional:
                    w_terms_txt.append("weekly_pre(t)")
            else:
                w_terms_txt.append("weekly(t)")
            formula_lines.append("         + " + " + ".join(w_terms_txt))
        if use_regressor:
            formula_lines.append("         + season_reg(t)")

        h_terms_txt = []
        if include_superbowl:
            h_terms_txt.append("superbowl(t)")
        if include_draft:
            h_terms_txt.append("draft(t)")
        if add_country_holidays:
            h_terms_txt.append("bank_holidays(t)")
        if h_terms_txt:
            formula_lines.append("         + " + " + ".join(h_terms_txt))
        formula_lines.append("       ) + error")

plain_formula_text = "\n".join(formula_lines)
st.code(plain_formula_text, language="text")

col_trend, col_season = st.columns([1, 1])

with col_trend:
    st.markdown("**Underlying Growth Trend**")
    st.altair_chart(
        alt.Chart(forecast)
        .mark_line(color="#fbbf24", size=3)
        .encode(x="ds:T", y="trend:Q")
        .properties(width="container", height=250, background="transparent")
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
            .properties(width="container", height=250, background="transparent"),
            width="stretch",
        )
    else:
        st.info("Yearly seasonality is disabled.")

st.markdown("**Weekly Dynamics & Impact**")
comp_week, comp_impact = st.columns([1, 1])

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

            # 1. Regular Season (or standard "In-Season" if postseason conditional is inactive)
            on_data = forecast[["ds", "weekly_on_season"]].copy()
            on_data["Day"] = on_data["ds"].dt.day_name().map(days_map)
            on_summary = on_data.groupby("Day")["weekly_on_season"].mean().reset_index()
            on_summary["Phase"] = (
                "Regular Season" if use_post_conditional else "In-Season"
            )
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

            # 3. Postseason (Conditional)
            if use_post_conditional and "weekly_post_season" in forecast.columns:
                post_data = forecast[["ds", "weekly_post_season"]].copy()
                post_data["Day"] = post_data["ds"].dt.day_name().map(days_map)
                post_summary = (
                    post_data.groupby("Day")["weekly_post_season"].mean().reset_index()
                )
                post_summary["Phase"] = "Postseason"
                post_summary = post_summary.rename(
                    columns={"weekly_post_season": "Adjustment"},
                )
                all_week_data.append(post_summary)

            # 4. Off-Season
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

            # Determine dynamically active phases & color sequences to show only active components in the legend
            active_phases = ["Regular Season" if use_post_conditional else "In-Season"]
            phase_colors = ["#fbbf24"]
            if use_pre_conditional and "weekly_pre_season" in forecast.columns:
                active_phases.append("Preseason")
                phase_colors.append("#f59e0b")
            if use_post_conditional and "weekly_post_season" in forecast.columns:
                active_phases.append("Postseason")
                phase_colors.append("#ef4444")
            active_phases.append("Off-Season")
            phase_colors.append("#475569")

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
                            domain=active_phases,
                            range=phase_colors,
                        ),
                        legend=alt.Legend(
                            orient="bottom",
                            direction="horizontal",
                            title="Phase",
                        ),
                    ),
                )
                .properties(height=250, background="transparent")
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
                .properties(width="container", height=250, background="transparent"),
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
                .properties(width="container", height=250, background="transparent")
                .interactive(),
                width="stretch",
            )
        else:
            st.info("Computing impact...")
    else:
        st.info("Global Regressor is disabled.")

# --- Special Events Integration (Integrated inside decomposition sequence) ---
active_views = []
if include_superbowl and "superbowl" in forecast.columns:
    active_views.append("superbowl")
if include_draft and "nfl_draft" in forecast.columns:
    active_views.append("nfl_draft")
if (
    add_country_holidays
    and "bank_holidays" in forecast.columns
    and forecast["bank_holidays"].abs().sum() > 0
):
    active_views.append("bank_holidays")

if active_views:
    st.markdown(
        "<br><b>Detailed Event & Holiday Components Analysis</b>",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(active_views))

    col_idx = 0
    if "superbowl" in active_views:
        with cols[col_idx]:
            st.markdown("**Super Bowl Impact Component**")

            if validation_mode:
                window_start = df_raw["ds"].iloc[-periods - 1]
                window_end = df_raw["ds"].max()
                window_name = "validation window"
            else:
                window_start = df_raw["ds"].max()
                window_end = forecast["ds"].max()
                window_name = "forecast horizon"

            valid_sb_dates = [
                pd.Timestamp(d)
                for d in sb_dates
                if window_start < pd.Timestamp(d) <= window_end
            ]

            if valid_sb_dates:
                target_sb_dt = pd.to_datetime(valid_sb_dates[0])
                sb_roman = SB_ROMANS.get(target_sb_dt.year, "")
                sb_title_text = f"Days relative to Super Bowl {sb_roman} ({target_sb_dt.strftime('%b %d, %Y')})"

                st.markdown(f"*{sb_title_text}*")

                sb_sample = forecast[
                    (
                        forecast["ds"]
                        >= target_sb_dt - datetime.timedelta(days=lower_win + 5)
                    )
                    & (
                        forecast["ds"]
                        <= target_sb_dt + datetime.timedelta(days=upper_win + 5)
                    )
                ]

                sb_area = (
                    alt.Chart(sb_sample)
                    .mark_area(color="#fbbf24", opacity=0.6)
                    .encode(
                        x=alt.X(
                            "ds:T",
                            title=sb_title_text,
                            axis=alt.Axis(
                                format="%b %d",
                                labelAngle=0,
                                tickCount="day",
                            ),
                        ),
                        y=alt.Y("superbowl:Q", title="Score Adjustment"),
                        tooltip=[
                            alt.Tooltip("ds:T", title="Date"),
                            alt.Tooltip("superbowl:Q", title="Effect", format=".2f"),
                        ],
                    )
                )
                rule_df = pd.DataFrame({"ds": [target_sb_dt]})
                sb_rule = (
                    alt.Chart(rule_df)
                    .mark_rule(
                        color="#f8fafc",
                        strokeWidth=2,
                        strokeDash=[8, 8],
                    )
                    .encode(x="ds:T")
                )

                combined_sb_chart = (
                    alt.layer(sb_area, sb_rule)
                    .properties(width="container", height=250, background="transparent")
                    .interactive()
                )

                st.altair_chart(combined_sb_chart, width="stretch")
            else:
                st.info(
                    f"No Super Bowl falls within the current {window_name} ({periods} days). "
                    f"Increase the period slider to reach February.",
                )
        col_idx += 1

    if "nfl_draft" in active_views:
        with cols[col_idx]:
            st.markdown("**NFL Draft Impact Component**")

            if validation_mode:
                window_start = df_raw["ds"].iloc[-periods - 1]
                window_end = df_raw["ds"].max()
                window_name = "validation window"
            else:
                window_start = df_raw["ds"].max()
                window_end = forecast["ds"].max()
                window_name = "forecast horizon"

            valid_draft_dates = [
                pd.Timestamp(d)
                for d in DRAFT_DATES
                if window_start < pd.Timestamp(d) <= window_end
            ]

            if valid_draft_dates:
                target_draft_dt = pd.to_datetime(valid_draft_dates[0])
                draft_title_text = f"Days relative to NFL Draft ({target_draft_dt.strftime('%b %d, %Y')})"

                st.markdown(f"*{draft_title_text}*")

                draft_sample = forecast[
                    (
                        forecast["ds"]
                        >= target_draft_dt
                        - datetime.timedelta(days=draft_lower_win + 5)
                    )
                    & (
                        forecast["ds"]
                        <= target_draft_dt
                        + datetime.timedelta(days=draft_upper_win + 5)
                    )
                ]

                draft_area = (
                    alt.Chart(draft_sample)
                    .mark_area(color="#fbbf24", opacity=0.6)
                    .encode(
                        x=alt.X(
                            "ds:T",
                            title=draft_title_text,
                            axis=alt.Axis(
                                format="%b %d",
                                labelAngle=0,
                                tickCount="day",
                            ),
                        ),
                        y=alt.Y("nfl_draft:Q", title="Score Adjustment"),
                        tooltip=[
                            alt.Tooltip("ds:T", title="Date"),
                            alt.Tooltip("nfl_draft:Q", title="Effect", format=".2f"),
                        ],
                    )
                )
                rule_df = pd.DataFrame({"ds": [target_draft_dt]})
                draft_rule = (
                    alt.Chart(rule_df)
                    .mark_rule(
                        color="#f8fafc",
                        strokeWidth=2,
                        strokeDash=[8, 8],
                    )
                    .encode(x="ds:T")
                )

                combined_draft_chart = (
                    alt.layer(draft_area, draft_rule)
                    .properties(width="container", height=250, background="transparent")
                    .interactive()
                )

                st.altair_chart(combined_draft_chart, width="stretch")
            else:
                st.info(
                    f"No NFL Draft falls within the current {window_name} ({periods} days). "
                    f"Increase the period slider to reach late April/May.",
                )
        col_idx += 1

    if "bank_holidays" in active_views:
        with cols[col_idx]:
            st.markdown("**Inbuilt Bank Holidays Component**")

            # Constrain window to only the last full year of the horizon
            last_year_limit = forecast["ds"].max() - datetime.timedelta(days=365)
            bank_sample = bank_holidays_df[
                (bank_holidays_df["ds"] >= last_year_limit)
                & (bank_holidays_df["ds"] <= forecast["ds"].max())
            ].copy()

            if not bank_sample.empty:
                st.markdown(
                    f"*Showing last full year: {last_year_limit.strftime('%b %Y')} to {forecast['ds'].max().strftime('%b %Y')}*",
                )

                bank_chart = (
                    alt.Chart(bank_sample)
                    .mark_bar(color="#f59e0b", size=5)
                    .encode(
                        x=alt.X(
                            "ds:T",
                            title="Holiday Date",
                            axis=alt.Axis(format="%b %d, %Y", labelAngle=30),
                        ),
                        y=alt.Y("effect:Q", title="Score Adjustment"),
                        tooltip=[
                            alt.Tooltip("ds:T", title="Date", format="%B %d, %Y"),
                            alt.Tooltip("holiday_name:N", title="Bank Holiday"),
                            alt.Tooltip("effect:Q", title="Effect", format=".2f"),
                        ],
                    )
                    .properties(width="container", height=250, background="transparent")
                    .interactive()
                )
                st.altair_chart(bank_chart, width="stretch")
            else:
                st.info(
                    "No active bank holidays found inside the final year of the horizon.",
                )
        col_idx += 1


# --- Code Snippet Generator ---
st.markdown("---")
with st.expander("💻 View Generated Python Code for Prophet Model"):
    st.markdown("Here is the Python code to reproduce this exact Prophet model setup:")

    code_lines = [
        "import numpy as np",
        "import pandas as pd",
        "from prophet import Prophet",
        "",
        "# --- Training DataFrame Schema Example ---",
        "# Your df_train DataFrame must be formatted with the following columns & data types:",
        "#",
        "# Column Name           | DataType       | Description",
        "# ----------------------|----------------|----------------------------------------------",
        "# ds                    | datetime64[ns] | Complete timestamp series (e.g., '2023-09-07')",
        "# y                     | float64        | Search/Trend metric value",
        "# nfl_season_regressor  | float64        | Extra regressor value",
        "# on_season             | bool           | True if regular or postseason is active, else False",
        "# reg_season            | bool           | True if regular season is active, else False",
        "# post_season           | bool           | True if postseason is active, else False",
        "# pre_season            | bool           | True if preseason is active, else False",
        "# off_season            | bool           | True if off-season is active, else False",
        "",
        "# Example of dummy DataFrame creation:",
        "df_train = pd.DataFrame({",
        "    'ds': pd.to_datetime(['2023-05-15', '2023-08-10', '2023-09-15', '2024-01-20']),",
        "    'y': [12.4, 38.2, 95.1, 88.3],",
        "    'nfl_season_regressor': [0.0, 0.0, 1.0, 1.0],",
        "    'on_season': [False, False, True, True],",
        "    'reg_season': [False, False, True, False],",
        "    'post_season': [False, False, False, True],",
        "    'pre_season': [False, True, False, False],",
        "    'off_season': [True, False, False, False]",
        "})",
        "",
        "# 1. Define custom holiday matrices",
    ]

    # Clean custom holiday setup avoiding list concatenation if only 1 is selected
    if include_superbowl and include_draft:
        code_lines.extend(
            [
                f"sb_df = pd.DataFrame({{\n"
                f"    'holiday': 'superbowl',\n"
                f"    'ds': pd.to_datetime({sb_dates}),\n"
                f"    'lower_window': -{lower_win},\n"
                f"    'upper_window': {upper_win}\n"
                f"}})",
                f"draft_df = pd.DataFrame({{\n"
                f"    'holiday': 'nfl_draft',\n"
                f"    'ds': pd.to_datetime({DRAFT_DATES}),\n"
                f"    'lower_window': -{draft_lower_win},\n"
                f"    'upper_window': {draft_upper_win}\n"
                f"}})",
                "holidays = pd.concat([sb_df, draft_df], ignore_index=True)",
            ],
        )
    elif include_superbowl:
        code_lines.append(
            f"holidays = pd.DataFrame({{\n"
            f"    'holiday': 'superbowl',\n"
            f"    'ds': pd.to_datetime({sb_dates}),\n"
            f"    'lower_window': -{lower_win},\n"
            f"    'upper_window': {upper_win}\n"
            f"}})",
        )
    elif include_draft:
        code_lines.append(
            f"holidays = pd.DataFrame({{\n"
            f"    'holiday': 'nfl_draft',\n"
            f"    'ds': pd.to_datetime({DRAFT_DATES}),\n"
            f"    'lower_window': -{draft_lower_win},\n"
            f"    'upper_window': {draft_upper_win}\n"
            f"}})",
        )
    else:
        code_lines.append("holidays = None")

    code_lines.extend(
        [
            "",
            "# 2. Initialize Prophet model",
            "m = Prophet(",
            f"    growth='{growth_type}',",
            "    weekly_seasonality=False,",
            "    yearly_seasonality=False,",
            "    holidays=holidays,"
            if (include_superbowl or include_draft)
            else "    holidays=None,",
            f"    changepoint_prior_scale={cp_flex},",
            f"    changepoint_range={cp_range},",
            f"    seasonality_mode='{season_mode}',",
            f"    holidays_prior_scale={holiday_prior},",
            ")",
        ],
    )

    if add_country_holidays:
        code_lines.append(f"m.add_country_holidays(country_name='{country_code}')")

    # Custom Seasonalities & Regressors
    code_lines.append("")
    code_lines.append("# 3. Add Custom Seasonalities & Regressors")

    if use_regressor:
        code_lines.append(
            f"m.add_regressor('nfl_season_regressor', mode='{season_mode}', prior_scale={regressor_prior})",
        )
        if use_different_post_offset or use_pre_offset:
            post_val = post_offset_mult if use_different_post_offset else 1.0
            pre_val = pre_offset_mult if use_pre_offset else 0.0
            code_lines.append(
                f"# Custom regressor offsets: Preseason={pre_val}, Regular Season=1.0, Postseason={post_val}\n"
                f"df_train['nfl_season_regressor'] = np.where(\n"
                f"    df_train['reg_season'], 1.0,\n"
                f"    np.where(\n"
                f"        df_train['post_season'], {post_val},\n"
                f"        np.where(df_train['pre_season'], {pre_val}, 0.0)\n"
                f"    )\n"
                f")",
            )

    if include_yearly:
        code_lines.append(
            f"m.add_seasonality(name='yearly', period=365.25, fourier_order={fourier_order_year}, prior_scale={season_prior})",
        )

    if include_weekly:
        if use_conditional:
            reg_season_cond = "reg_season" if use_post_conditional else "on_season"
            code_lines.append(
                f"m.add_seasonality(name='weekly_on_season', period=7, fourier_order={fourier_order_week}, prior_scale={season_prior}, condition_name='{reg_season_cond}')",
            )
            if use_pre_conditional:
                code_lines.append(
                    f"m.add_seasonality(name='weekly_pre_season', period=7, fourier_order={fourier_order_week}, prior_scale={season_prior}, condition_name='pre_season')",
                )
            if use_post_conditional:
                code_lines.append(
                    f"m.add_seasonality(name='weekly_post_season', period=7, fourier_order={fourier_order_week}, prior_scale={season_prior}, condition_name='post_season')",
                )
            code_lines.append(
                f"m.add_seasonality(name='weekly_off_season', period=7, fourier_order=3, prior_scale={season_prior}, condition_name='off_season')",
            )
        else:
            code_lines.append(
                f"m.add_seasonality(name='weekly', period=7, fourier_order={fourier_order_week}, prior_scale={season_prior})",
            )

    code_lines.extend(
        [
            "",
            "# 4. Fit Model",
            "m.fit(df_train)",
        ],
    )

    if growth_type == "logistic":
        code_lines.extend(
            [
                f"df_train['cap'] = {cap_val}",
                "df_train['floor'] = 0.0",
                f"future['cap'] = {cap_val}",
                "future['floor'] = 0.0",
            ],
        )

    # Render formatted Python script block
    generated_code = "\n".join(code_lines)
    st.code(generated_code, language="python")

if "plot_df" in locals():
    del plot_df
if "forecast_display" in locals():
    del forecast_display
gc.collect()
