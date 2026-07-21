"""
Streamlit application for analyzing and forecasting NFL Google Trends data
using Prophet.

This application serves as an interactive playground for time series forecasting.
It allows users to explore various configurations of the Prophet model and
observe their impact on forecasting NFL-related search interest data. Key features
include validation mode, customizable seasonality, holiday effects,
and regressor tuning.
"""

import datetime
import gc
import threading
from dataclasses import dataclass, field

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet
from prophet.utilities import regressor_coefficients
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Constants and Page Configuration ---
# --------------------------------------------------------------------------

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Prophet NFL Playground")

# --- Date Constants ---
AUGUST = 8
SEPTEMBER = 9
DECEMBER = 12
JANUARY = 1
FEBRUARY = 2

# --- Model Hyperparameter Defaults ---
DEFAULT_FOURIER_YEAR = 10
DEFAULT_FOURIER_WEEK = 5
DEFAULT_CP_FLEX = 0.05
DEFAULT_CP_RANGE = 0.8
DEFAULT_SEASON_PRIOR = 10.0
DEFAULT_REG_PRIOR = 10.0
DEFAULT_HOLIDAY_PRIOR = 10.0

# --- Miscellaneous Constants ---
CUSTOM_RANGE_DATE_COUNT = 2


# --------------------------------------------------------------------------
# --- NFL Schedule and Holiday Data ---
# --------------------------------------------------------------------------

# Hardcoded NFL schedule data. For a production system, this might be sourced
# from a separate file (e.g., CSV, JSON) or a database.
# fmt: off
_NFL_SCHEDULE = [
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
    ("2019-04-25", "2019-08-01", "2019-09-05", "2020-02-02", "2020-02-02"),
    ("2020-04-23", "2020-07-30", "2020-09-10", "2021-01-09", "2021-02-07"),
    ("2021-04-29", "2021-07-29", "2021-09-09", "2022-01-15", "2022-02-13"),
    ("2022-04-28", "2022-07-28", "2022-09-08", "2023-01-14", "2023-02-12"),
    ("2023-04-27", "2023-07-27", "2023-09-07", "2024-01-13", "2024-02-11"),
    ("2024-04-25", "2024-07-25", "2024-09-05", "2025-01-11", "2025-02-09"),
    ("2025-04-24", "2025-07-31", "2025-09-04", "2026-01-10", "2026-02-08"),
    ("2026-04-23", "2026-08-06", "2026-09-10", "2027-01-16", "2027-02-14"),
    ("2027-04-29", "2027-08-05", "2027-09-09", "2028-01-15", "2028-02-13"),
]
# fmt: on

@st.cache_data(show_spinner="Processing NFL schedule data...")
def _process_nfl_schedule(schedule_data: list[tuple[str, ...]]):
    """Processes raw NFL schedule data into usable date lists."""
    draft_dates = [s[0] for s in schedule_data]
    sb_dates = [s[4] for s in schedule_data]
    schedule_ranges = [
        (
            pd.Timestamp(pre).date(),
            pd.Timestamp(reg).date(),
            pd.Timestamp(post).date(),
            pd.Timestamp(sb).date(),
        )
        for _, pre, reg, post, sb in schedule_data
    ]
    return draft_dates, sb_dates, schedule_ranges

DRAFT_DATES, SB_DATES, SCHEDULE_RANGES = _process_nfl_schedule(_NFL_SCHEDULE)

# --------------------------------------------------------------------------
# --- Data Configuration Classes ---
# --------------------------------------------------------------------------


@dataclass
class SeasonalityConfig:
    """Configuration for seasonalities."""

    include_yearly: bool = True
    fourier_year: int = 10
    include_weekly: bool = False
    fourier_week: int = 5
    conditional_weekly: bool = False
    use_pre_cond: bool = False
    use_post_cond: bool = False
    mode: str = "additive"


@dataclass
class RegressorConfig:
    """Configuration for regressors."""

    use_reg: bool = False
    use_diff_post: bool = False
    post_mult: float = 1.0
    use_pre_off: bool = False
    pre_mult: float = 0.0
    prior_scale: float = 10.0


@dataclass
class HolidayConfig:
    """Configuration for holidays."""

    add_country_hols: bool = False
    country_name: str = "US"
    include_drft: bool = False
    drft_l_win: int = 1
    drft_u_win: int = 2
    include_sb: bool = False
    l_win: int = 1
    u_win: int = 1
    prior_scale: float = 10.0


@dataclass
class ProphetConfig:
    """Configuration settings for the Prophet model."""

    periods: int
    is_validation: bool
    seasonality: SeasonalityConfig = field(default_factory=SeasonalityConfig)
    regressor: RegressorConfig = field(default_factory=RegressorConfig)
    holidays: HolidayConfig = field(default_factory=HolidayConfig)
    flex: float = 0.05
    r_range: float = 0.8
    s_prior: float = 10.0
    growth: str = "linear"
    cap: float = 150.0
    view_mode: str = "Full History"
    date_range: tuple[datetime.date, datetime.date] | None = None


# --------------------------------------------------------------------------
# --- Data Loading and Preparation ---
# --------------------------------------------------------------------------


def get_detailed_season_flags(
    dates: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized calculation of NFL season flags for a series of dates.

    Flags dates as pre-season, regular season, or post-season based on the
    official NFL schedule. For dates beyond the known schedule, it approximates
    seasons by month.

    Args:
        dates: A pandas Series of dates.

    Returns:
        A tuple of boolean numpy arrays: (is_pre, is_reg, is_post).
    """
    date_series = pd.to_datetime(dates)
    date_objs = date_series.dt.date
    is_pre = np.zeros(len(dates), dtype=bool)
    is_reg = np.zeros(len(dates), dtype=bool)
    is_post = np.zeros(len(dates), dtype=bool)

    # Use known NFL schedule ranges for historical data
    for pre_start, reg_start, post_start, sb_end in SCHEDULE_RANGES:
        is_pre |= (date_objs >= pre_start) & (date_objs < reg_start)
        is_reg |= (date_objs >= reg_start) & (date_objs < post_start)
        is_post |= (date_objs >= post_start) & (date_objs <= sb_end)

    # For future dates, approximate seasons by month
    max_scheduled_date = SCHEDULE_RANGES[-1][3]
    future_mask = date_objs > max_scheduled_date
    if np.any(future_mask):
        future_dates = date_series[future_mask]
        months = future_dates.dt.month
        is_pre[future_mask] = months == AUGUST
        is_reg[future_mask] = months.isin([SEPTEMBER, 10, 11, DECEMBER])
        is_post[future_mask] = months.isin([JANUARY, FEBRUARY])

    return is_pre, is_reg, is_post


def add_season_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Adds detailed NFL season flags to a DataFrame with a 'ds' column."""
    df_copy = df.copy()
    is_pre, is_reg, is_post = get_detailed_season_flags(df_copy["ds"])
    df_copy["pre_season"] = is_pre
    df_copy["reg_season"] = is_reg
    df_copy["post_season"] = is_post
    df_copy["on_season"] = is_reg | is_post
    df_copy["off_season"] = ~(is_pre | is_reg | is_post)
    return df_copy


@st.cache_data(show_spinner="Loading and processing NFL trend data...")
def load_nfl_trends() -> pd.DataFrame:
    """Loads NFL daily trends data, parses dates, and adds season flags."""
    df = pd.read_csv("./data/NFL_daily_trends.csv")
    df["ds"] = pd.to_datetime(df["ds"])
    df = add_season_flags(df)
    df["y"] = df["y"].astype(np.float32)
    return df


# --------------------------------------------------------------------------
# --- Prophet Modeling ---
# --------------------------------------------------------------------------


class ProphetModelRunner:
    """A class to manage the Prophet modeling pipeline."""

    def __init__(self, config: ProphetConfig, df_raw: pd.DataFrame):
        self.config = config
        self.df_raw = df_raw
        self.model: Prophet | None = None
        self.forecast: pd.DataFrame | None = None
        self.reg_impact = 0.0
        self.trained_holidays: list[str] = []

    def run(self):
        """Runs the full Prophet modeling pipeline."""
        train_df = self._get_training_data()
        train_df = self._prepare_features(train_df)
        holidays_df = self._build_holidays_df()

        self._initialize_and_fit_model(train_df, holidays_df)

        future_df = self._make_future_dataframe()
        self.forecast = self.model.predict(future_df)

        self._calculate_regressor_impact()
        self._capture_trained_holidays()
        self._cleanup()

    def _get_training_data(self) -> pd.DataFrame:
        """Returns the training dataset based on validation mode."""
        if self.config.is_validation:
            return self.df_raw.iloc[: -self.config.periods].copy()
        return self.df_raw.copy()

    def _make_future_dataframe(self) -> pd.DataFrame:
        """Creates and prepares the future dataframe for prediction."""
        future_df = self.model.make_future_dataframe(periods=self.config.periods)
        future_df = add_season_flags(future_df)
        return self._prepare_features(future_df)

    def _initialize_and_fit_model(
        self, train_df: pd.DataFrame, holidays_df: pd.DataFrame | None,
    ):
        """Initializes, configures, and fits the Prophet model."""
        self.model = Prophet(
            growth=self.config.growth,
            weekly_seasonality=False,
            yearly_seasonality=False,
            holidays=holidays_df,
            changepoint_prior_scale=self.config.flex,
            changepoint_range=self.config.r_range,
            seasonality_mode=self.config.seasonality.mode,
            holidays_prior_scale=self.config.holidays.prior_scale,
        )
        self._add_country_holidays()
        self._add_regressors()
        self._add_seasonalities()
        self.model.fit(train_df)

    def _build_holidays_df(self) -> pd.DataFrame | None:
        """Constructs a DataFrame of custom holidays for Prophet."""
        holiday_dfs = []
        holidays_config = self.config.holidays
        if holidays_config.include_sb:
            holiday_dfs.append(
                pd.DataFrame(
                    {
                        "holiday": "superbowl",
                        "ds": pd.to_datetime(SB_DATES),
                        "lower_window": -holidays_config.l_win,
                        "upper_window": holidays_config.u_win,
                    },
                ),
            )
        if holidays_config.include_drft:
            holiday_dfs.append(
                pd.DataFrame(
                    {
                        "holiday": "nfl_draft",
                        "ds": pd.to_datetime(DRAFT_DATES),
                        "lower_window": -holidays_config.drft_l_win,
                        "upper_window": holidays_config.drft_u_win,
                    },
                ),
            )
        return pd.concat(holiday_dfs, ignore_index=True) if holiday_dfs else None

    def _add_seasonalities(self):
        """Adds seasonal components to the Prophet model."""
        seasonality_config = self.config.seasonality
        if seasonality_config.include_yearly:
            self.model.add_seasonality(
                name="yearly",
                period=365.25,
                fourier_order=seasonality_config.fourier_year,
                prior_scale=self.config.s_prior,
            )
        if seasonality_config.include_weekly:
            self._add_weekly_seasonality(seasonality_config)

    def _add_weekly_seasonality(self, config: SeasonalityConfig):
        """Helper to add standard or conditional weekly seasonality."""
        if not config.conditional_weekly:
            self.model.add_seasonality(
                name="weekly",
                period=7,
                fourier_order=config.fourier_week,
                prior_scale=self.config.s_prior,
            )
            return

        on_season_cond = "reg_season" if config.use_post_cond else "on_season"
        self.model.add_seasonality(
            name="weekly_on_season",
            period=7,
            fourier_order=config.fourier_week,
            prior_scale=self.config.s_prior,
            condition_name=on_season_cond,
        )
        if config.use_pre_cond:
            self.model.add_seasonality(
                name="weekly_pre_season",
                period=7,
                fourier_order=config.fourier_week,
                prior_scale=self.config.s_prior,
                condition_name="pre_season",
            )
        if config.use_post_cond:
            self.model.add_seasonality(
                name="weekly_post_season",
                period=7,
                fourier_order=config.fourier_week,
                prior_scale=self.config.s_prior,
                condition_name="post_season",
            )
        self.model.add_seasonality(
            name="weekly_off_season",
            period=7,
            fourier_order=3,
            prior_scale=self.config.s_prior,
            condition_name="off_season",
        )

    def _add_country_holidays(self):
        """Adds country-specific holidays if configured."""
        if self.config.holidays.add_country_hols:
            self.model.add_country_holidays(
                country_name=self.config.holidays.country_name,
            )

    def _add_regressors(self):
        """Adds regressors to the model if configured."""
        if self.config.regressor.use_reg:
            self.model.add_regressor(
                name="nfl_season_regressor",
                mode=self.config.seasonality.mode,
                prior_scale=self.config.regressor.prior_scale,
            )

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds features required for the model to a dataframe."""
        df = df.copy()
        if self.config.regressor.use_reg:
            df = self._add_regressor_features(df)
        if self.config.growth == "logistic":
            df = self._add_logistic_growth_features(df)
        return df

    def _add_regressor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds the seasonal regressor feature."""
        reg_config = self.config.regressor
        pre_mult = reg_config.pre_mult if reg_config.use_pre_off else 0.0
        post_mult = reg_config.post_mult if reg_config.use_diff_post else 1.0

        df["nfl_season_regressor"] = np.where(
            df["reg_season"],
            1.0,
            np.where(
                df["post_season"], post_mult, np.where(df["pre_season"], pre_mult, 0.0),
            ),
        )
        return df

    def _add_logistic_growth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds cap and floor columns for logistic growth model."""
        df["cap"] = self.config.cap
        df["floor"] = 0.0
        return df

    def _calculate_regressor_impact(self):
        """Calculates the coefficient of the main regressor if used."""
        if not self.config.regressor.use_reg:
            self.reg_impact = 0.0
            return
        try:
            coeffs = regressor_coefficients(self.model)
            reg_coeffs = coeffs[coeffs["regressor"] == "nfl_season_regressor"]
            self.reg_impact = (
                reg_coeffs["coef"].iloc[0] if not reg_coeffs.empty else 0.0
            )
        except (IndexError, KeyError):
            # Handles cases where the regressor is not found in the coefficients
            self.reg_impact = 0.0

    def _capture_trained_holidays(self):
        """Stores a list of holiday names the model was trained on."""
        if self.model and self.model.train_holiday_names is not None:
            self.trained_holidays = list(self.model.train_holiday_names)
        else:
            self.trained_holidays = []

    def _cleanup(self):
        """Removes the model object to free up memory."""
        del self.model
        self.model = None
        gc.collect()


# --------------------------------------------------------------------------
# --- UI Styling and Components ---
# --------------------------------------------------------------------------


def apply_custom_css():
    """Applies custom CSS for the app's dark-themed aesthetic."""
    st.markdown(
        """
<style>
    .stApp, [data-testid="stAppViewContainer"], .main .block-container {
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
        text-transform: uppercase;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
    }
</style>
""",
        unsafe_allow_html=True,
    )


def display_metric_card(
    column: "st.delta_generator.DeltaGenerator",
    label: str,
    value: str,
    help_text: str = "",
):
    """Displays a stylized metric card in a given Streamlit column."""
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
            help=help_text,
        )


# --------------------------------------------------------------------------
# --- Sidebar UI ---
# --------------------------------------------------------------------------


class Sidebar:
    """Manages the rendering of the sidebar UI and captures user configuration."""

    def __init__(self, df_raw: pd.DataFrame):
        self.df_raw = df_raw
        self.config = ProphetConfig(periods=365, is_validation=True)

    def render(self) -> ProphetConfig:
        """Renders the sidebar and returns the collected configuration."""
        with st.sidebar:
            st.title("🔮 Prophet Settings")
            self._render_section_1_mode_horizon()
            self._render_section_2_seasonal_complexity()
            self._render_section_3_season_logic()
            self._render_section_4_holiday_settings()
            self._render_section_5_model_flexibility()
            self._render_section_6_visualization()
            self._render_section_7_growth_constraints()
        return self.config

    def _render_section_1_mode_horizon(self):
        st.subheader("1. Mode & Horizon")
        self.config.is_validation = st.toggle("Validation Mode", value=True)
        if self.config.is_validation:
            self.config.periods = st.slider("Validation Days", 30, 3 * 365, 365)
        else:
            self.config.periods = st.slider("Forecast Days", 1, 5 * 365, 730)

    def _render_section_2_seasonal_complexity(self):
        st.subheader("2. Seasonal Complexity")
        seasonality = self.config.seasonality
        seasonality.mode = st.radio("Seasonality Mode", ["additive", "multiplicative"])
        seasonality.include_yearly = st.checkbox("Yearly", value=True)
        seasonality.fourier_year = st.slider(
            "Yearly Fourier Order", 2, 30, DEFAULT_FOURIER_YEAR,
        )
        seasonality.include_weekly = st.checkbox("Weekly", value=False)
        seasonality.fourier_week = st.slider(
            "Weekly Fourier Order",
            1,
            15,
            DEFAULT_FOURIER_WEEK,
            disabled=not seasonality.include_weekly,
        )

    def _render_section_3_season_logic(self):
        st.subheader("3. Season Logic")
        regressor = self.config.regressor
        seasonality = self.config.seasonality
        regressor.use_reg = st.checkbox("Seasonal Regressor", value=False)
        if regressor.use_reg:
            _, col_reg_widgets = st.columns([0.07, 0.93])
            with col_reg_widgets:
                regressor.use_diff_post = st.checkbox(
                    "Different Postseason Offset", value=False,
                )
                if regressor.use_diff_post:
                    regressor.post_mult = st.slider(
                        "Postseason Multiplier", 0.0, 2.0, 1.0, 0.1,
                    )
                regressor.use_pre_off = st.checkbox(
                    "Include Preseason Offset", value=False,
                )
                if regressor.use_pre_off:
                    regressor.pre_mult = st.slider(
                        "Preseason Multiplier", 0.0, 1.0, 0.1, 0.05,
                    )

        if seasonality.include_weekly:
            seasonality.conditional_weekly = st.checkbox(
                "Conditional Weekly Patterns",
                value=False,
                disabled=not seasonality.include_weekly,
            )
            if seasonality.conditional_weekly:
                _, col_checkbox = st.columns([0.07, 0.93])
                with col_checkbox:
                    seasonality.use_pre_cond = st.checkbox(
                        "Include Preseason Weekly", value=False,
                    )
                    seasonality.use_post_cond = st.checkbox(
                        "Include Postseason Weekly", value=False,
                    )

    def _render_section_4_holiday_settings(self):
        st.subheader("4. Holiday Settings")
        holidays = self.config.holidays
        holidays.add_country_hols = st.checkbox("Enable Bank Holidays", value=False)
        if holidays.add_country_hols:
            _, col_country = st.columns([0.07, 0.93])
            with col_country:
                holidays.country_name = st.selectbox(
                    "Country Calendar", ["US", "CA", "GB", "DE", "FR", "JP", "AU"],
                )

        st.markdown("##### Special NFL Events")
        holidays.include_drft = st.checkbox("Model NFL Draft", value=False)
        if holidays.include_drft:
            _, d_col1, d_col2 = st.columns([0.07, 0.465, 0.465])
            with d_col1:
                holidays.drft_l_win = st.slider("Draft Lower Window", 0, 7, 1)
            with d_col2:
                holidays.drft_u_win = st.slider("Draft Upper Window", 0, 7, 2)

        holidays.include_sb = st.checkbox("Model Super Bowls", value=False)
        if holidays.include_sb:
            _, sb_col1, sb_col2 = st.columns([0.07, 0.465, 0.465])
            with sb_col1:
                holidays.l_win = st.slider("SB Lower Window", 0, 14, 1)
            with sb_col2:
                holidays.u_win = st.slider("SB Upper Window", 0, 14, 1)

    def _render_section_5_model_flexibility(self):
        st.subheader("5. Model Flexibility")
        self.config.flex = st.slider(
            "Trend Flexibility", 0.001, 0.5, DEFAULT_CP_FLEX, format="%.3f",
        )
        self.config.r_range = st.slider("Changepoint Range", 0.5, 1.0, DEFAULT_CP_RANGE)
        self.config.s_prior = st.slider(
            "Seasonality Prior", 0.1, 100.0, DEFAULT_SEASON_PRIOR,
        )
        if self.config.regressor.use_reg:
            self.config.regressor.prior_scale = st.slider(
                "Regressor Prior", 0.1, 100.0, DEFAULT_REG_PRIOR,
            )
        self.config.holidays.prior_scale = st.slider(
            "Holidays Prior", 0.01, 100.0, DEFAULT_HOLIDAY_PRIOR,
        )

    def _render_section_6_visualization(self):
        st.subheader("6. Visualization View")
        self.config.view_mode = st.radio(
            "Display Window",
            ["Full History", "Last 3 Years", "Last 6 Months", "Custom Range"],
        )
        if self.config.view_mode == "Custom Range":
            min_dt = self.df_raw["ds"].min().to_pydatetime()
            max_dt = (
                self.df_raw["ds"].max()
                + datetime.timedelta(
                    days=(0 if self.config.is_validation else self.config.periods),
                )
            ).to_pydatetime()
            self.config.date_range = st.date_input(
                "Select Date Range", (min_dt, max_dt), min_dt, max_dt,
            )

    def _render_section_7_growth_constraints(self):
        st.subheader("7. Growth Constraints")
        self.config.growth = st.radio("Growth Model", ["linear", "logistic"])
        if self.config.growth == "logistic":
            self.config.cap = st.slider(
                "Interest Cap", float(self.df_raw["y"].max()), 500.0, 200.0,
            )


# --------------------------------------------------------------------------
# --- Main Content UI and Logic ---
# --------------------------------------------------------------------------


def render_metrics(
    config: ProphetConfig,
    forecast: pd.DataFrame,
    reg_impact: float,
    mae: float = 0.0,
    rmse: float = 0.0,
):
    """Renders the top-level metric cards."""
    m1, m2, m3 = st.columns(3)
    if config.is_validation:
        display_metric_card(m1, "Mean Absolute Error (MAE)", f"{mae:.2f}")
        display_metric_card(m2, "Root Mean Sq. Error (RMSE)", f"{rmse:.2f}")
        display_metric_card(m3, "Validation Period", f"{config.periods} days")
    else:
        display_metric_card(
            m1, "Predicted Peak Interest", f"{forecast['yhat'].max():.1f}",
        )
        _render_forecast_metrics(m2, m3, config, forecast, reg_impact)


def _render_forecast_metrics(
    m2: "st.delta_generator.DeltaGenerator",
    m3: "st.delta_generator.DeltaGenerator",
    config: ProphetConfig,
    forecast: pd.DataFrame,
    reg_impact: float,
):
    """Renders metrics specific to the forecast mode."""
    if config.growth == "logistic":
        headroom = config.cap - forecast["trend"].iloc[-1]
        growth_val = f"{headroom:.1f}"
        growth_label = "Saturation Headroom"
    else:
        growth_pct = (forecast["trend"].iloc[-1] / forecast["trend"].iloc[-366]) - 1
        growth_val = f"{growth_pct:+.1%}"
        growth_label = "YoY Trend Growth"
    display_metric_card(m2, growth_label, growth_val)

    mode = config.seasonality.mode
    reg_label = f"Season Bonus ({'Mult' if mode == 'multiplicative' else 'Add'})"
    reg_val = f"+{reg_impact:.1%}" if mode == "multiplicative" else f"+{reg_impact:.1f}"
    display_metric_card(m3, reg_label, reg_val if config.regressor.use_reg else "N/A")


def calculate_validation_metrics(
    y_true: pd.Series, y_pred: pd.Series,
) -> tuple[float, float]:
    """Calculates MAE and RMSE for validation."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def prepare_plot_data(
    config: ProphetConfig, forecast: pd.DataFrame, df_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    """
    Prepares a merged DataFrame for plotting by combining historical and
    forecasted data. Also calculates validation metrics if applicable.
    """
    if config.is_validation:
        plot_df, mae, rmse = _prepare_validation_plot_data(config, forecast, df_raw)
    else:
        plot_df, mae, rmse = _prepare_forecast_plot_data(forecast, df_raw)

    plot_df = plot_df.merge(forecast[["ds", "yhat"]], on="ds", how="left").rename(
        columns={"yhat": "Predicted"},
    )
    plot_df["Actual"] = np.where(
        plot_df["Type"].str.contains("Actual"), plot_df["y"], np.nan,
    )
    plot_df["Delta"] = plot_df["Actual"] - plot_df["Predicted"]
    plot_df["Pct_Error"] = (plot_df["Delta"].abs() / plot_df["Actual"]).replace(
        np.inf, 0,
    ) * 100
    return plot_df, mae, rmse


def _prepare_validation_plot_data(
    config: ProphetConfig, forecast: pd.DataFrame, df_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    """Prepares plot data for validation mode."""
    train = df_raw.iloc[: -config.periods].copy()
    train["Type"] = "Actual (Training)"
    test = df_raw.iloc[-config.periods :].copy()
    test["Type"] = "Actual (Ground Truth)"
    pred = forecast.tail(config.periods).copy()
    pred["Type"] = "Validation Prediction"

    plot_df = pd.concat(
        [
            train[["ds", "y", "Type"]],
            test[["ds", "y", "Type"]],
            pred[["ds", "yhat", "Type"]].rename(columns={"yhat": "y"}),
        ],
    )
    mae, rmse = calculate_validation_metrics(test["y"], pred["yhat"])
    return plot_df, mae, rmse


def _prepare_forecast_plot_data(
    forecast: pd.DataFrame, df_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    """Prepares plot data for forecast mode."""
    historical = df_raw.copy()
    historical["Type"] = "Actual"
    future = forecast[forecast["ds"] > df_raw["ds"].max()].copy()
    future["Type"] = "Forecast"

    plot_df = pd.concat(
        [
            historical[["ds", "y", "Type"]],
            future[["ds", "yhat", "Type"]].rename(columns={"yhat": "y"}),
        ],
    )
    return plot_df, 0.0, 0.0


def get_display_data(
    config: ProphetConfig,
    plot_df: pd.DataFrame,
    forecast: pd.DataFrame,
    df_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filters data for display based on sidebar view settings."""
    max_date = df_raw["ds"].max()
    plot_df_display = filter_df_by_view(
        plot_df, config.view_mode, config.date_range, max_date,
    )
    forecast_display = filter_df_by_view(
        forecast, config.view_mode, config.date_range, max_date,
    )
    return plot_df_display, forecast_display


def filter_df_by_view(
    df: pd.DataFrame,
    view_mode: str,
    date_range: tuple | None,
    max_date: pd.Timestamp,
) -> pd.DataFrame:
    """Filters a DataFrame based on the selected time window."""
    if view_mode == "Last 6 Months":
        filter_start = max_date - datetime.timedelta(days=180)
        return df[df["ds"] >= filter_start]
    if view_mode == "Last 3 Years":
        filter_start = max_date - pd.DateOffset(years=3)
        return df[df["ds"] >= filter_start]
    if (
        view_mode == "Custom Range"
        and date_range
        and len(date_range) == CUSTOM_RANGE_DATE_COUNT
    ):
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        return df[(df["ds"] >= start) & (df["ds"] <= end)]
    return df


def render_forecast_chart(
    plot_df: pd.DataFrame, forecast_display: pd.DataFrame, config: ProphetConfig,
):
    """Renders the main forecast chart using Altair."""
    st.subheader(
        "📅 Accuracy Check: Actual vs. Predicted"
        if config.is_validation
        else "📈 Interest Forecast",
    )

    y_scale = alt.Scale(zero=True)
    color_map = {
        "Actual": "#94a3b8",
        "Actual (Training)": "#94a3b8",
        "Actual (Ground Truth)": "#f8fafc",
        "Forecast": "#fbbf24",
        "Validation Prediction": "#fbbf24",
    }
    color_domain = [t for t in color_map if t in plot_df["Type"].unique()]
    color_range = [color_map[t] for t in color_domain]

    tooltips = [
        alt.Tooltip("ds:T", title="Date", format="%b %d, %Y"),
        alt.Tooltip("Actual:Q", title="Actual", format=".1f"),
        alt.Tooltip("Predicted:Q", title="Predicted", format=".1f"),
        alt.Tooltip("Delta:Q", title="Delta", format="+.1f"),
        alt.Tooltip("Pct_Error:Q", title="Error (%)", format=".1f"),
    ]

    band = (
        alt.Chart(forecast_display, title="Uncertainty Interval")
        .mark_area(opacity=0.3, color="#fbbf24")
        .encode(x="ds:T", y="yhat_lower:Q", y2="yhat_upper:Q")
    )

    base = alt.Chart(plot_df).encode(
        x=alt.X("ds:T", title="Date"),
        y=alt.Y("y:Q", title="Interest Score", scale=y_scale),
        color=alt.Color(
            "Type:N",
            scale=alt.Scale(domain=color_domain, range=color_range),
            legend=alt.Legend(title="Data Series"),
        ),
        tooltip=tooltips,
    )

    hist_line = base.transform_filter(
        (alt.datum.Type == "Actual") | (alt.datum.Type == "Actual (Training)"),
    ).mark_line(opacity=0.7)
    truth_points = base.transform_filter(
        alt.datum.Type == "Actual (Ground Truth)",
    ).mark_point(size=60, filled=True, opacity=1.0)
    pred_line = base.transform_filter(
        (alt.datum.Type == "Forecast") | (alt.datum.Type == "Validation Prediction"),
    ).mark_line(strokeDash=[5, 5])

    chart = (
        alt.layer(band, hist_line, truth_points, pred_line)
        .properties(height=450, background="transparent")
        .interactive()
    )
    st.altair_chart(chart, width="stretch")


# --------------------------------------------------------------------------
# --- Model Caching ---
# --------------------------------------------------------------------------


@st.cache_resource(max_entries=1)
def get_model_runner(
    _config_hash: int, config: ProphetConfig, df_raw: pd.DataFrame,
) -> ProphetModelRunner:
    """
    Gets or creates a cached ProphetModelRunner instance.

    The _config_hash parameter is a simple but effective way to bust the cache
    when the model configuration changes. Streamlit reruns the script on every
    interaction, and this ensures we only retrain the model when necessary.

    Args:
        _config_hash: Hash of the configuration object, used for caching.
        config: The Prophet configuration object.
        df_raw: The raw NFL trends DataFrame.

    Returns:
        A trained ProphetModelRunner instance.
    """
    runner = ProphetModelRunner(config, df_raw)
    runner.run()
    return runner


# --------------------------------------------------------------------------
# --- Main Application ---
# --------------------------------------------------------------------------


def main():
    """Main function to run the Streamlit application."""
    # --- 1. App Setup ---
    apply_custom_css()
    if "model_lock" not in st.session_state:
        st.session_state.model_lock = threading.Lock()
    st.title("🔮 Prophet NFL Trends Playground")

    # --- 2. Data Loading ---
    df_raw = load_nfl_trends()

    # --- 3. Sidebar Configuration ---
    sidebar = Sidebar(df_raw)
    config = sidebar.render()

    # --- 4. Model Execution ---
    # Use a lock to prevent race conditions when multiple sessions might
    # access the cached model simultaneously.
    with st.session_state.model_lock:
        config_hash = hash(str(config))
        model_runner = get_model_runner(config_hash, config, df_raw)

    forecast = model_runner.forecast

    # --- 5. Data Preparation for Plotting ---
    plot_df, mae, rmse = prepare_plot_data(config, forecast, df_raw)
    plot_df_display, forecast_display = get_display_data(
        config, plot_df, forecast, df_raw,
    )

    # --- 6. Render Main Content ---
    render_metrics(config, forecast, model_runner.reg_impact, mae, rmse)
    st.markdown("<br>", unsafe_allow_html=True)
    render_forecast_chart(plot_df_display, forecast_display, config)

    # --- 7. Decomposition Components & Equation Formulator ---
    st.markdown("---")
    st.subheader("🧩 Decomposing the NFL Cycle")

    col_trend, col_season = st.columns([1, 1])

    with col_trend:
        st.markdown("**Underlying Growth Trend**")
        st.altair_chart(
            alt.Chart(forecast)
            .mark_line(color="#fbbf24", size=3)
            .encode(x="ds:T", y="trend:Q")
            .properties(height=250, background="transparent")
            .interactive(),
            width="stretch",
        )

    with col_season:
        st.markdown("**Yearly Pattern (Fixed Calendar Cycle)**")
        if config.seasonality.include_yearly:
            yearly_comp = forecast[["ds", "yearly"]].iloc[:365].copy()
            st.altair_chart(
                alt.Chart(yearly_comp)
                .mark_line(color="#fbbf24", strokeWidth=3)
                .encode(x=alt.X("ds:T", axis=alt.Axis(format="%b")), y="yearly:Q")
                .properties(height=250, background="transparent"),
                width="stretch",
            )
        else:
            st.info("Yearly seasonality is disabled.")

    st.markdown("**Weekly Dynamics & Impact**")
    comp_week, comp_impact = st.columns([1, 1])

    with comp_week:
        st.markdown("**Weekly Pattern Comparison**")
        if config.seasonality.include_weekly:
            days_map = {
                "Monday": "Mon.",
                "Tuesday": "Tue.",
                "Wednesday": "Wed.",
                "Thursday": "Thu.",
                "Friday": "Fri.",
                "Saturday": "Sat.",
                "Sunday": "Sun.",
            }

            if config.seasonality.conditional_weekly:
                all_week_data = []

                # 1. Regular Season (or "In-Season" if postseason conditional is off)
                on_data = forecast[["ds", "weekly_on_season"]].copy()
                on_data["Day"] = on_data["ds"].dt.day_name().map(days_map)
                on_summary = (
                    on_data.groupby("Day")["weekly_on_season"].mean().reset_index()
                )
                on_summary["Phase"] = (
                    "Regular Season"
                    if config.seasonality.use_post_cond
                    else "In-Season"
                )
                on_summary = on_summary.rename(
                    columns={"weekly_on_season": "Adjustment"},
                )
                all_week_data.append(on_summary)

                # 2. Preseason (Conditional)
                if (
                    config.seasonality.use_pre_cond
                    and "weekly_pre_season" in forecast.columns
                ):
                    pre_data = forecast[["ds", "weekly_pre_season"]].copy()
                    pre_data["Day"] = pre_data["ds"].dt.day_name().map(days_map)
                    pre_summary = (
                        pre_data.groupby("Day")["weekly_pre_season"]
                        .mean()
                        .reset_index()
                    )
                    pre_summary["Phase"] = "Preseason"
                    pre_summary = pre_summary.rename(
                        columns={"weekly_pre_season": "Adjustment"},
                    )
                    all_week_data.append(pre_summary)

                # 3. Postseason (Conditional)
                if (
                    config.seasonality.use_post_cond
                    and "weekly_post_season" in forecast.columns
                ):
                    post_data = forecast[["ds", "weekly_post_season"]].copy()
                    post_data["Day"] = post_data["ds"].dt.day_name().map(days_map)
                    post_summary = (
                        post_data.groupby("Day")["weekly_post_season"]
                        .mean()
                        .reset_index()
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

                # Determine dynamically active phases & color sequences for the legend
                active_phases = [
                    "Regular Season"
                    if config.seasonality.use_post_cond
                    else "In-Season",
                ]
                phase_colors = ["#fbbf24"]
                if (
                    config.seasonality.use_pre_cond
                    and "weekly_pre_season" in forecast.columns
                ):
                    active_phases.append("Preseason")
                    phase_colors.append("#f59e0b")
                if (
                    config.seasonality.use_post_cond
                    and "weekly_post_season" in forecast.columns
                ):
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
                            sort=[
                                "Mon.",
                                "Tue.",
                                "Wed.",
                                "Thu.",
                                "Fri.",
                                "Sat.",
                                "Sun.",
                            ],
                            title="Day of Week",
                            axis=alt.Axis(labelAngle=0),
                        ),
                        y=alt.Y("Adjustment:Q", title="Multiplier / Adjustment"),
                        xOffset="Phase:N",
                        color=alt.Color(
                            "Phase:N",
                            scale=alt.Scale(domain=active_phases, range=phase_colors),
                            legend=alt.Legend(
                                orient="bottom", direction="horizontal", title="Phase",
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
                    .reindex(
                        ["Mon.", "Tue.", "Wed.", "Thu.", "Fri.", "Sat.", "Sun."],
                    )
                    .reset_index()
                )
                st.altair_chart(
                    alt.Chart(weekly_summary)
                    .mark_bar(color="#fbbf24")
                    .encode(
                        x=alt.X("Day:N", sort=None, axis=alt.Axis(labelAngle=0)),
                        y="weekly:Q",
                    )
                    .properties(height=250, background="transparent"),
                    width="stretch",
                )
        else:
            st.info("Weekly seasonality is disabled.")

    with comp_impact:
        st.markdown("**Season Regressor Impact**")
        if config.regressor.use_reg:
            reg_display_col = (
                "extra_regressors_multiplicative"
                if (
                    config.seasonality.mode == "multiplicative"
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
                    .properties(height=250, background="transparent")
                    .interactive(),
                    width="stretch",
                )
            else:
                st.info("Computing impact...")
        else:
            st.info("Global Regressor is disabled.")

    # --- Special Events Integration ---
    active_views = []
    if config.holidays.include_sb and "superbowl" in forecast.columns:
        active_views.append("superbowl")
    if config.holidays.include_drft and "nfl_draft" in forecast.columns:
        active_views.append("nfl_draft")
    if (
        config.holidays.add_country_hols
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

                if config.is_validation:
                    window_start = df_raw["ds"].iloc[-config.periods - 1]
                    window_end = df_raw["ds"].max()
                    window_name = "validation window"
                else:
                    window_start = df_raw["ds"].max()
                    window_end = forecast["ds"].max()
                    window_name = "forecast horizon"

                valid_sb_dates = [
                    pd.Timestamp(d)
                    for d in SB_DATES
                    if window_start < pd.Timestamp(d) <= window_end
                ]

                if valid_sb_dates:
                    target_sb_dt = pd.to_datetime(valid_sb_dates[0])
                    sb_roman = {
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
                    }.get(target_sb_dt.year, "")
                    sb_title_text = (
                        f"Days relative to Super Bowl {sb_roman} "
                        f"({target_sb_dt.strftime('%b %d, %Y')})"
                    )

                    st.markdown(f"*{sb_title_text}*")

                    sb_sample = forecast[
                        (
                            forecast["ds"]
                            >= target_sb_dt
                            - datetime.timedelta(days=config.holidays.l_win + 5)
                        )
                        & (
                            forecast["ds"]
                            <= target_sb_dt
                            + datetime.timedelta(days=config.holidays.u_win + 5)
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
                                    format="%b %d", labelAngle=0, tickCount="day",
                                ),
                            ),
                            y=alt.Y("superbowl:Q", title="Score Adjustment"),
                            tooltip=[
                                alt.Tooltip("ds:T", title="Date"),
                                alt.Tooltip(
                                    "superbowl:Q", title="Effect", format=".2f",
                                ),
                            ],
                        )
                    )
                    rule_df = pd.DataFrame({"ds": [target_sb_dt]})
                    sb_rule = (
                        alt.Chart(rule_df)
                        .mark_rule(color="#f8fafc", strokeWidth=2, strokeDash=[8, 8])
                        .encode(x="ds:T")
                    )

                    combined_sb_chart = (
                        alt.layer(sb_area, sb_rule)
                        .properties(height=250, background="transparent")
                        .interactive()
                    )

                    st.altair_chart(combined_sb_chart, width="stretch")
                else:
                    st.info(
                        f"No Super Bowl falls within the current {window_name} "
                        f"({config.periods} days). Increase the period slider to reach February.",
                    )
            col_idx += 1

        if "nfl_draft" in active_views:
            with cols[col_idx]:
                st.markdown("**NFL Draft Impact Component**")

                if config.is_validation:
                    window_start = df_raw["ds"].iloc[-config.periods - 1]
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
                    draft_title_text = (
                        f"Days relative to NFL Draft "
                        f"({target_draft_dt.strftime('%b %d, %Y')})"
                    )

                    st.markdown(f"*{draft_title_text}*")

                    draft_sample = forecast[
                        (
                            forecast["ds"]
                            >= target_draft_dt
                            - datetime.timedelta(
                                days=config.holidays.drft_l_win + 5,
                            )
                        )
                        & (
                            forecast["ds"]
                            <= target_draft_dt
                            + datetime.timedelta(
                                days=config.holidays.drft_u_win + 5,
                            )
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
                                    format="%b %d", labelAngle=0, tickCount="day",
                                ),
                            ),
                            y=alt.Y("nfl_draft:Q", title="Score Adjustment"),
                            tooltip=[
                                alt.Tooltip("ds:T", title="Date"),
                                alt.Tooltip(
                                    "nfl_draft:Q", title="Effect", format=".2f",
                                ),
                            ],
                        )
                    )
                    rule_df = pd.DataFrame({"ds": [target_draft_dt]})
                    draft_rule = (
                        alt.Chart(rule_df)
                        .mark_rule(color="#f8fafc", strokeWidth=2, strokeDash=[8, 8])
                        .encode(x="ds:T")
                    )

                    combined_draft_chart = (
                        alt.layer(draft_area, draft_rule)
                        .properties(height=250, background="transparent")
                        .interactive()
                    )

                    st.altair_chart(combined_draft_chart, width="stretch")
                else:
                    st.info(
                        f"No NFL Draft falls within the current {window_name} "
                        f"({config.periods} days). Increase the period slider to reach "
                        "late April/May.",
                    )
            col_idx += 1


    # --- 8. Memory Management ---
    # In a long-running Streamlit app, it's good practice to explicitly
    # delete large objects that are no longer needed to free up memory.
    del plot_df, forecast, df_raw, model_runner, plot_df_display, forecast_display
    gc.collect()


if __name__ == "__main__":
    main()
