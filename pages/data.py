import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title="Sports Dataset Explorer")

st.markdown(
    """
<style>
    .stApp, [data-testid="stAppViewContainer"], .reportview-container {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    h1, h2, h3, h4 {
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .info-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 0.5rem;
        padding: 1.2rem;
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
</style>
""",
    unsafe_allow_html=True,
)

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

DRAFT_DATES = [s[0] for s in NFL_SCHEDULE]
sb_dates = [s[4] for s in NFL_SCHEDULE]

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
    date_objs = pd.to_datetime(dates).dt.date

    is_pre = np.zeros(len(dates), dtype=bool)
    is_reg = np.zeros(len(dates), dtype=bool)
    is_post = np.zeros(len(dates), dtype=bool)

    for pre_start, reg_start, post_start, sb_end in SCHEDULE_RANGES:
        is_pre |= ((date_objs >= pre_start) & (date_objs < reg_start)).values
        is_reg |= ((date_objs >= reg_start) & (date_objs < post_start)).values
        is_post |= ((date_objs >= post_start) & (date_objs <= sb_end)).values

    return is_pre, is_reg, is_post


def get_nfl_season_year(dates):
    date_objs = pd.to_datetime(dates).dt.date
    season_years = np.zeros(len(dates), dtype=int)

    draft_dates = [pd.Timestamp(s[0]).date() for s in NFL_SCHEDULE]

    for i in range(len(draft_dates)):
        start_date = draft_dates[i]
        if i < len(draft_dates) - 1:
            end_date = draft_dates[i + 1]
        else:
            end_date = datetime.date(2035, 1, 1)

        mask = (date_objs >= start_date) & (date_objs < end_date)
        season_years[mask] = start_date.year

    mask_before = date_objs < draft_dates[0]
    season_years[mask_before] = 2008

    return season_years


DATASETS = {
    "🏈 Google Search Interest: NFL": {
        "path": "./data/NFL_daily_trends.csv",
        "description": """
            <h3>🏈 Google Search Interest: NFL (2009 - Present)</h3>
            <p><strong>Source:</strong> Daily Google Search interest for the query "NFL", normalized on a 0 to 100 scale.</p>
            <p><strong>Context & Insights:</strong></p>
            <ul>
                <li><strong>Regular Season (September - December):</strong> Features structured surges matching active play weeks.</li>
                <li><strong>Postseason & Super Bowl (January - February):</strong> Displays the absolute historical search volume peaks.</li>
                <li><strong>NFL Draft (Late April):</strong> Captures a reliable off-season highlight spike.</li>
                <li><strong>Off-Season (May - July):</strong> Represents the yearly baseline minimum search levels.</li>
            </ul>
        """,
    },
}


@st.cache_data
def load_dataset(filepath):
    df = pd.read_csv(filepath)
    df["ds"] = pd.to_datetime(df["ds"])

    is_pre, is_reg, is_post = get_detailed_season_flags(df["ds"])
    df["Phase"] = np.select(
        [is_reg, is_post, is_pre],
        ["Regular Season", "Postseason", "Preseason"],
        default="Off-season",
    )
    df["Season_Year"] = get_nfl_season_year(df["ds"])

    df["Segment_ID"] = (df["Phase"] != df["Phase"].shift()).cumsum()
    return df


st.title("📊 Sports Dataset Explorer")

dataset_choice = st.selectbox(
    "Choose a Dataset to Explore:",
    ["Select a dataset..."] + list(DATASETS.keys()),
    index = 1,
)

if dataset_choice == "Select a dataset...":
    st.markdown(
        """
        <div class="info-card">
            <h2>👋 Welcome to the Sports Dataset Explorer</h2>
            <p>Please select a dataset from the dropdown above to filter, inspect, and analyze historical records.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📚 Available Datasets & Metadata Descriptions")
    for name, info in DATASETS.items():
        st.markdown(
            f"""
            <div class="info-card">
                {info["description"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    df_raw = load_dataset(DATASETS[dataset_choice]["path"])

    st.subheader("🔍 Interactive Data Filters")

    min_selectable_date = df_raw["ds"].min().to_pydatetime()
    max_selectable_date = df_raw["ds"].max().to_pydatetime()

    date_range = st.date_input(
        "Define Display Range:",
        value=(min_selectable_date, max_selectable_date),
        min_value=min_selectable_date,
        max_value=max_selectable_date,
        help="Adjust the start and end dates to focus on a specific era.",
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_selectable_date, max_selectable_date

    df_filtered = df_raw[
        (df_raw["ds"] >= pd.to_datetime(start_date))
        & (df_raw["ds"] <= pd.to_datetime(end_date))
    ].copy()

    col1, col2, col3, col4 = st.columns(4)

    if not df_filtered.empty:
        with col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Filtered Days</div>'
                f'<div class="metric-value">{len(df_filtered)}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Average Search Score</div>'
                f'<div class="metric-value">{df_filtered["y"].mean():.2f}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with col3:
            max_idx = df_filtered["y"].idxmax()
            peak_score = df_filtered.loc[max_idx, "y"]
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Peak Search Score</div>'
                f'<div class="metric-value">{peak_score:.1f}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with col4:
            peak_date = df_filtered.loc[max_idx, "ds"].strftime("%b %d, %Y")
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Peak Score Date</div>'
                f'<div class="metric-value" style="font-size: 1.45rem; padding-top: 0.3rem;">{peak_date}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.warning("Please select a valid date range to view statistical metrics.")

    st.markdown("<br>", unsafe_allow_html=True)

    tab_chart, tab_table, tab_metadata = st.tabs(
        [
            "📊 Interactive Trend Chart",
            "🗂️ Sliced Data Table & Export",
            "📖 Source Metadata",
        ],
    )

    with tab_chart:
        if not df_filtered.empty:
            col_toggle1, col_toggle2 = st.columns(2)
            with col_toggle1:
                highlight_seasons = st.toggle(
                    "Highlight NFL Calendar Phases",
                    value=False,
                    help="Color-code the Preseason, Regular Season, Postseason, and Off-season stages cleanly without crossover line artifacts.",
                )
            with col_toggle2:
                show_events = st.toggle(
                    "Display Special Events (Draft & Super Bowls)",
                    value=False,
                    help="Overlay vertical colored dotted indicator lines corresponding to Super Bowls and Draft days.",
                )

            st.markdown("<br>", unsafe_allow_html=True)

            if highlight_seasons:
                color_scale = alt.Scale(
                    domain=["Regular Season", "Postseason", "Preseason", "Off-season"],
                    range=["#fbbf24", "#ef4444", "#f59e0b", "#475569"],
                )

                chart_df = df_filtered.copy()
                chart_df["Season_Phase"] = (
                    chart_df["Phase"] + "_" + chart_df["Segment_ID"].astype(str)
                )

                base_line = (
                    alt.Chart(chart_df)
                    .mark_line(color="#334155", strokeWidth=1, opacity=0.3)
                    .encode(
                        x=alt.X("ds:T"),
                        y=alt.Y("y:Q"),
                    )
                )

                phase_lines = (
                    alt.Chart(chart_df)
                    .mark_line(strokeWidth=1.8)
                    .encode(
                        x=alt.X("ds:T", title="Date"),
                        y=alt.Y("y:Q", title="Search Interest Score (0 - 100)"),
                        color=alt.Color(
                            "Phase:N",
                            scale=color_scale,
                            legend=alt.Legend(title="NFL Calendar Phase"),
                        ),
                        detail="Season_Phase:N",
                        tooltip=[
                            alt.Tooltip("ds:T", title="Date", format="%B %d, %Y"),
                            alt.Tooltip("y:Q", title="Search Interest"),
                            alt.Tooltip("Phase:N", title="NFL Phase"),
                        ],
                    )
                )
                chart_layers = [base_line, phase_lines]

            else:
                chart_df = df_filtered.copy()
                chart_df["Series"] = "Search Interest"

                trend_line = (
                    alt.Chart(chart_df)
                    .mark_line(strokeWidth=1.8)
                    .encode(
                        x=alt.X("ds:T", title="Date"),
                        y=alt.Y("y:Q", title="Search Interest Score (0 - 100)"),
                        color=alt.Color(
                            "Series:N",
                            scale=alt.Scale(
                                domain=["Search Interest"],
                                range=["#fbbf24"],
                            ),
                            legend=alt.Legend(title="NFL Calendar Phase"),
                        ),
                        tooltip=[
                            alt.Tooltip("ds:T", title="Date", format="%B %d, %Y"),
                            alt.Tooltip("y:Q", title="Search Interest"),
                        ],
                    )
                )
                chart_layers = [trend_line]

            if show_events:
                visible_drafts = [
                    pd.Timestamp(d)
                    for d in DRAFT_DATES
                    if pd.Timestamp(start_date)
                    <= pd.Timestamp(d)
                    <= pd.Timestamp(end_date)
                ]
                visible_sbs = [
                    pd.Timestamp(s)
                    for s in sb_dates
                    if pd.Timestamp(start_date)
                    <= pd.Timestamp(s)
                    <= pd.Timestamp(end_date)
                ]

                event_records = []
                for d in visible_drafts:
                    event_records.append(
                        {
                            "ds": d,
                            "Event": "NFL Draft",
                            "Detail": f"NFL Draft {d.year}",
                        },
                    )
                for s in visible_sbs:
                    roman = SB_ROMANS.get(s.year, "")
                    label = f"Super Bowl {roman}" if roman else "Super Bowl"
                    event_records.append(
                        {
                            "ds": s,
                            "Event": "Super Bowl",
                            "Detail": f"{label} ({s.year})",
                        },
                    )

                events_df = pd.DataFrame(event_records)

                if not events_df.empty:
                    event_rules = (
                        alt.Chart(events_df)
                        .mark_rule(strokeWidth=1.5, strokeDash=[4, 4])
                        .encode(
                            x="ds:T",
                            color=alt.Color(
                                "Event:N",
                                scale=alt.Scale(
                                    domain=["NFL Draft", "Super Bowl"],
                                    range=["#10b981", "#ef4444"],
                                ),
                                legend=alt.Legend(title="Special Events"),
                            ),
                            tooltip=[
                                alt.Tooltip(
                                    "ds:T",
                                    title="Event Date",
                                    format="%B %d, %Y",
                                ),
                                alt.Tooltip("Detail:N", title="Event Description"),
                            ],
                        )
                    )
                    chart_layers.append(event_rules)

            chart = (
                alt.layer(*chart_layers)
                .resolve_scale(
                    color="independent",
                )
                .properties(
                    width="container",
                    height=450,
                    background="transparent",
                )
                .interactive()
            )

            st.altair_chart(chart, width="stretch")

    with tab_table:
        if not df_filtered.empty:
            col_table, col_download = st.columns([2, 1])

            clean_display_df = df_filtered[["ds", "y", "Phase"]].rename(
                columns={"ds": "Date", "y": "Search Score"},
            )

            with col_table:
                st.write("### Sliced Data Table View")
                st.dataframe(clean_display_df, width="stretch", hide_index=True)

            with col_download:
                st.write("### Export Selection")
                st.write(
                    "Generate a CSV of the filtered slice for external sports analytics pipelines.",
                )
                csv_data = clean_display_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Filtered Selection as CSV",
                    data=csv_data,
                    file_name=f"sliced_trends_{start_date}_to_{end_date}.csv",
                    mime="text/csv",
                    width="stretch",
                )

    with tab_metadata:
        st.markdown(
            f"""
            <div class="info-card">
                {DATASETS[dataset_choice]["description"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        clean_raw_df = df_raw[["ds", "y", "Phase"]].rename(
            columns={"ds": "Date", "y": "Search Score"},
        )

        st.write("### Complete Raw Dataset Preview")
        st.dataframe(clean_raw_df, width="stretch", hide_index=True)
