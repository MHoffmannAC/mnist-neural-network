import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Time-Series CV Playground")

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
        border-bottom: 4px solid #8b5cf6;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #a78bfa;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
    }
    .info-box {
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


df_raw = load_nfl_trends()

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("⏱️ CV Configuration")
    
    st.subheader("1. Data Resolution")
    resample_freq = st.selectbox("Resolution", ["Daily", "Weekly", "Monthly"], index=1)
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
    period_map = {"Daily": 365, "Weekly": 52, "Monthly": 12}
    
    current_freq = freq_map[resample_freq]
    yearly_period = period_map[resample_freq]

    # Pre-calculate data size for stability checks
    df_temp = df_raw.set_index("ds").resample(current_freq).mean().reset_index()
    n_available = len(df_temp)
    
    st.markdown("---")
    st.subheader("2. Window Strategy")
    window_type = st.radio("Window Logic", ["Expanding Window", "Fixed/Sliding Window"], 
                           help="Expanding: starts at zero and grows. Fixed: slides a constant-size box.")
    
    # Robust slider bounds checking
    # SARIMA needs more data for seasonal fitting, so we set min to 2 cycles
    min_train = int(yearly_period * 2) 
    max_train = int(max(min_train + 1, n_available - 10))
    
    train_size = st.slider("Train Window Size", 
                           min_value=min_train, 
                           max_value=max_train, 
                           value=int(min(yearly_period * 3, max_train)),
                           help="Minimum history needed. SARIMA requires at least 2 full seasonal cycles to be stable.")
    
    max_horizon = int(max(1, n_available - train_size - 1))
    horizon = st.slider("Forecast Horizon (H)", 1, min(yearly_period, max_horizon), 
                        value=int(min(yearly_period/4, max_horizon)),
                        help="How many steps ahead to predict in each fold.")
    
    step_input = st.number_input("Step Size (Stride)", value=0, min_value=0,
                                 help="If 0, we jump forward by the horizon length. Larger values skip more data.")
    step_size = step_input if step_input > 0 else horizon

    st.markdown("---")
    st.subheader("3. Model for Simulation")
    model_choice = st.selectbox(
        "Test Model", 
        ["SARIMA (1,1,1)x(1,1,1)", "Seasonal Naive", "Naive", "Global Mean"],
        index=0
    )
    if "SARIMA" in model_choice:
        st.info(f"Using SARIMA: $(1, 1, 1) \\times (1, 1, 1)_{{{yearly_period}}}$")
        st.caption("Fitting statistical models over many folds is slow. Please be patient.")

# --- Data Processing ---
df_resampled = df_raw.set_index("ds").resample(current_freq).mean().reset_index()
n_total = len(df_resampled)

# --- Final Stability Check ---
if train_size + horizon > n_total:
    st.error(f"Configuration Impossible: Train Size ({train_size}) + Horizon ({horizon}) exceeds total data points ({n_total}).")
    st.stop()

# --- Cross-Validation Logic ---
folds = []
metrics = []
fold_viz_data = []

# Iteration Logic
start_idx = train_size
iteration_range = list(range(start_idx, n_total - horizon + 1, step_size))
num_folds = len(iteration_range)

if num_folds > 50 and "SARIMA" in model_choice:
    st.warning(f"⚠️ High Computation Load: You have requested {num_folds} folds using a SARIMA model. Consider increasing the 'Step Size' if it takes too long.")

# Setup progress bar
progress_bar = st.progress(0)
status_text = st.empty()

for fold_idx, current_end in enumerate(iteration_range):
    status_text.text(f"Processing Fold {fold_idx + 1} of {num_folds}...")
    
    # 1. Determine Window Bounds
    train_start = 0 if window_type == "Expanding Window" else (current_end - train_size)
    train_end = current_end
    test_start = current_end
    test_end = current_end + horizon
    
    # Safety Check for bounds
    if test_end > n_total:
        break

    # 2. Extract Data
    y_train = df_resampled['y'].values[train_start:train_end]
    y_test = df_resampled['y'].values[test_start:test_end]
    
    # 3. Model Logic
    try:
        if model_choice == "SARIMA (1,1,1)x(1,1,1)":
            model = SARIMAX(
                y_train, 
                order=(1, 1, 1), 
                seasonal_order=(1, 1, 1, yearly_period),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False)
            y_pred = res.get_forecast(steps=horizon).predicted_mean
        
        elif model_choice == "Seasonal Naive":
            m = yearly_period
            if len(y_train) < m:
                y_pred = np.repeat(y_train[-1], horizon)
            else:
                source_pattern = y_train[-m:]
                reps = int(np.ceil(horizon / m))
                y_pred = np.tile(source_pattern, reps)[:horizon]
        
        elif model_choice == "Naive":
            y_pred = np.repeat(y_train[-1], horizon)
            
        else: # Global Mean
            y_pred = np.repeat(np.mean(y_train), horizon)

    except Exception as e:
        # Fallback to naive if optimization fails
        y_pred = np.repeat(y_train[-1], horizon)
        
    # 4. Calculate Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # Avoid division by zero for MAPE
    y_test_safe = np.where(y_test == 0, 1e-5, y_test)
    mape = np.mean(np.abs((y_test - y_pred) / y_test_safe)) * 100
    
    metrics.append({
        'Fold': fold_idx + 1,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Mid_Date': df_resampled['ds'].iloc[test_start]
    })
    
    # 5. Visual Representation Data
    # Use index - 1 for End to prevent out-of-bounds timestamp access
    fold_viz_data.append({
        'Fold': fold_idx + 1, 
        'Start': df_resampled['ds'].iloc[train_start], 
        'End': df_resampled['ds'].iloc[train_end - 1], 
        'Type': 'Train'
    })
    fold_viz_data.append({
        'Fold': fold_idx + 1, 
        'Start': df_resampled['ds'].iloc[test_start], 
        'End': df_resampled['ds'].iloc[test_end - 1], 
        'Type': 'Test'
    })
    
    progress_bar.progress((fold_idx + 1) / num_folds)

status_text.empty()
progress_bar.empty()

if not metrics:
    st.warning("No folds could be generated with the current settings.")
    st.stop()

metrics_df = pd.DataFrame(metrics)
viz_df = pd.DataFrame(fold_viz_data)

# --- Main UI ---
st.title("⏱️ Time-Series Cross-Validation")
st.markdown(f"### Evaluating **{model_choice.split(' ')[0]}** performance across **{len(metrics_df)} folds**")

# Top Level Stats
m1, m2, m3 = st.columns(3)
with m1: st.markdown(f'<div class="metric-card"><div class="metric-label">Avg. MAE</div><div class="metric-value">{metrics_df["MAE"].mean():.2f}</div></div>', unsafe_allow_html=True)
with m2: st.markdown(f'<div class="metric-card"><div class="metric-label">Avg. RMSE</div><div class="metric-value">{metrics_df["RMSE"].mean():.2f}</div></div>', unsafe_allow_html=True)
with m3: st.markdown(f'<div class="metric-card"><div class="metric-label">Avg. MAPE</div><div class="metric-value">{metrics_df["MAPE"].mean():.1f}%</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 1. Visualizing the Folds
st.subheader("📅 Cross-Validation Strategy & Data Coverage")
st.write("The bars show the sliding windows, while the line shows the raw interest data to highlight seasonal coverage.")

# Window Bars
fold_chart = alt.Chart(viz_df).mark_bar(height=10 if len(metrics_df) > 20 else 20, opacity=0.4).encode(
    x=alt.X('Start:T', title="Date Timeline"),
    x2='End:T',
    y=alt.Y('Fold:O', title="Iteration (Fold)", sort='ascending'),
    color=alt.Color('Type:N', scale=alt.Scale(domain=['Train', 'Test'], range=['#475569', '#8b5cf6']), title="Window Role"),
    tooltip=['Fold', 'Type', 'Start', 'End']
)

# Raw Data Line (Layered in background)
raw_line = alt.Chart(df_resampled).mark_line(color='#64748b', strokeWidth=1.5, opacity=0.8).encode(
    x='ds:T',
    y=alt.Y('y:Q', title="Raw Interest Score")
)

# Layering with independent Y-axes and FIXED height
combined_strategy = alt.layer(raw_line, fold_chart).resolve_scale(
    y='independent'
).properties(height=500)

st.altair_chart(combined_strategy, width='stretch')

# 2. Performance Over Time
st.markdown("---")
st.subheader("📈 Error Metrics across Folds")
st.write("Does the model perform worse during certain years or seasons?")

metric_to_plot = st.radio("Select Metric to View", ["MAE", "RMSE", "MAPE"], horizontal=True)

error_chart = alt.Chart(metrics_df).mark_line(point=True, color='#8b5cf6').encode(
    x=alt.X('Mid_Date:T', title="Test Window Period"),
    y=alt.Y(f'{metric_to_plot}:Q', title=metric_to_plot),
    tooltip=['Fold', 'Mid_Date', 'MAE', 'RMSE', 'MAPE']
).properties(height=300).interactive()

st.altair_chart(error_chart, width='stretch')

# --- Educational Deep Dive ---
st.markdown("---")
with st.expander("📚 Deep Dive: Time-Series Cross-Validation Concepts", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 1. Why can't we use K-Fold?")
        st.write("""
        Standard **K-Fold Cross-Validation** randomly assigns data points to folds. In time series, this is illegal because:
        - It violates the **Temporal Order**: You might end up using data from 2024 to predict 2012.
        - **Autocorrelation**: Data points are not independent. Knowing today's interest score makes it very easy to guess tomorrow's score if you have tomorrow's data in your training set.
        
        **Time-Series CV** ensures that every test set is strictly chronologically *after* its corresponding training set.
        """)
        
    with c2:
        st.markdown("#### 2. Strategy Comparison")
        st.write(f"**Current Strategy:** {window_type}")
        if window_type == "Expanding Window":
            st.write("""
            The **Expanding Window** (or Cumulative) approach keeps all history. 
            - **Pros**: The model has maximum information and can learn long-term patterns better.
            - **Cons**: Training can become slow as history grows, and very old, irrelevant data might confuse the model.
            """)
        else:
            st.write("""
            The **Sliding Window** approach only keeps the most recent $N$ observations.
            - **Pros**: Adapts quickly to new trends or structural changes in the market.
            - **Cons**: It "forgets" what happened years ago, which is bad if the data is highly seasonal.
            """)

    st.markdown("---")
    st.markdown("#### 3. Parameter Tuning Guide")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.write("**Forecast Horizon ($H$)**")
        st.info("Larger horizons are harder. Error metrics usually increase significantly as you try to look further into the future.")
    with t2:
        st.write("**Train Window Size**")
        st.info("Since we are fitting SARIMA, ensure your window size covers at least two seasonal periods (e.g., 104 weeks) to allow the model to learn the year-over-year impact.")
    with t3:
        st.write("**Step Size**")
        st.info("Controls how many simulations we run. A small step size (e.g., 1 week) provides a very smooth error plot but takes longer to compute, especially with SARIMA.")

st.markdown("---")
st.caption("Cross-Validation Playground: Visualizing the rigor of time-series backtesting using statistical and baseline engines.")