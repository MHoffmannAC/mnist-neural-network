import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import time
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pages.utils.ml import select_roles, generate_ml_data

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Decision Tree NFL Playground")

# Custom CSS for the Playground aesthetic
st.markdown("""
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
        padding: 1.2rem;
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
    .rules-container {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 0.5rem;
        padding: 1.5rem;
        font-family: 'Fira Code', 'Courier New', monospace;
        font-size: 0.8rem;
        color: #10b981;
        max-height: 500px;
        overflow-y: auto;
        white-space: pre;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialization & Session State ---
if 'dt_is_training' not in st.session_state:
    st.session_state.dt_is_training = False
if 'dt_current_depth' not in st.session_state:
    st.session_state.dt_current_depth = 1
if 'dt_history' not in st.session_state:
    st.session_state.dt_history = []

# --- Sidebar ---
with st.sidebar:
    st.title("🌳 Decision Tree Hyperparameters")
    
    selected_roles = select_roles()

    st.markdown("---")
    st.subheader("2. Execution Mode")
    mode = st.radio("Model Logic", ["Exact (Full Tree)", "Iterative (Growth)"], 
                    help="Exact shows the tree at its final depth. Iterative shows the tree growing level by level.")

    st.markdown("---")
    st.subheader("3. Tree Tuning")
    
    max_depth_target = st.slider("Max Depth", 1, 15, 5, 
                                 help="The maximum depth of the tree. Deeper trees are more complex but can overfit.")
    
    criterion = st.selectbox("Split Criterion", ["gini", "entropy"], index=0,
                             help="Gini measures impurity; Entropy measures information gain.")
    
    min_samples_split = st.slider("Min Samples to Split", 2, 20, 5)

    if mode == "Iterative (Growth)":
        st.markdown("---")
        st.subheader("4. Animation Speed")
        step_delay = st.slider("Step Delay (seconds)", 0.5, 3.0, 1.5)

    st.markdown("---")
    
    if mode == "Iterative (Growth)":
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.dt_current_depth >= max_depth_target:
                btn_label = "✅ Finished"
                btn_disabled = True
            else:
                btn_label = "⏸ Pause" if st.session_state.dt_is_training else "🚀 Grow"
                btn_disabled = False
            if st.button(btn_label, disabled=btn_disabled):
                st.session_state.dt_is_training = not st.session_state.dt_is_training
                st.rerun()
        with col2:
            if st.button("♻️ Reset"):
                st.session_state.dt_is_training = False
                st.session_state.dt_current_depth = 1
                st.session_state.dt_history = []
                st.rerun()
    else:
        st.info("Exact mode computes the tree for the full depth instantly.")

# --- Data Prep ---
if len(selected_roles) < 2:
    st.title("🌳 Decision Tree NFL Playground")
    st.warning("Please select at least **two** roles in the sidebar to perform classification.")
    st.stop()

df = generate_ml_data(selected_roles)
X = df[['Rush', 'Rec']]
y = df['True Role']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- Decision Tree Logic ---
current_depth = max_depth_target if mode == "Exact (Full Tree)" else st.session_state.dt_current_depth

model = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=current_depth,
    min_samples_split=min_samples_split,
    random_state=42
)
model.fit(X, y_encoded)
accuracy = model.score(X, y_encoded)

# --- UI Header ---
st.title("🌳 Decision Tree NFL Playground")
if mode == "Iterative (Growth)":
    # Logic fix: Check if we've reached max depth first to avoid showing "Growing" at the end
    if st.session_state.dt_current_depth >= max_depth_target:
        status_text = "✅ Depth Reached"
    elif st.session_state.dt_is_training:
        status_text = "🟢 Growing..."
    else:
        status_text = "🟡 Idle"
    st.markdown(f"**Current Depth:** `{st.session_state.dt_current_depth} / {max_depth_target}` | Status: {status_text}")
else:
    st.markdown(f"**Status:** ✅ Full Tree Computed (Depth: {max_depth_target})")

# --- Decision Boundary Calculation ---
x_max = df['Rush'].max() + 150
y_max = df['Rec'].max() + 150
h = 20 
xx, yy = np.meshgrid(np.arange(0, x_max, h), np.arange(0, y_max, h))

grid_points = np.c_[xx.ravel(), yy.ravel()]
Z_encoded = model.predict(grid_points)
Z = le.inverse_transform(Z_encoded)

grid_df = pd.DataFrame({
    'Rush': xx.ravel(),
    'Rec': yy.ravel(),
    'Rush2': xx.ravel() + h,
    'Rec2': yy.ravel() + h,
    'Predicted_Role': Z
})

# --- Main Columns ---
col_viz, col_rules = st.columns([2, 1])

with col_viz:
    # --- Metrics ---
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-value">{accuracy:.1%}</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Nodes</div><div class="metric-value">{model.tree_.node_count}</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Training Size</div><div class="metric-value">{len(df)}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Visualization ---
    color_scale = alt.Scale(domain=selected_roles, scheme='tableau10')

    background = alt.Chart(grid_df).mark_rect(opacity=0.3).encode(
        x=alt.X('Rush:Q', title="Rushing Yards", scale=alt.Scale(domain=[0, x_max])),
        x2='Rush2:Q',
        y=alt.Y('Rec:Q', title="Receiving Yards", scale=alt.Scale(domain=[0, y_max])),
        y2='Rec2:Q',
        color=alt.Color('Predicted_Role:N', scale=color_scale, legend=None),
        tooltip=alt.value(None)
    )

    points = alt.Chart(df).mark_circle(size=75, stroke='black', strokeWidth=0.5).encode(
        x='Rush:Q',
        y='Rec:Q',
        color=alt.Color('True Role:N', scale=color_scale, title="Real Role"),
        tooltip=['True Role', 'Rush', 'Rec']
    )

    st.altair_chart((background + points).properties(width='container', height=500).interactive(), width='stretch')

with col_rules:
    st.subheader("📜 Current Decision Rules")
    st.write("This is the 'if-then' logic the computer has generated so far:")
    
    # Export tree rules to text
    raw_rules = export_text(model, feature_names=['Rushing Yards', 'Receiving Yards'])
    
    # Clean up formatting and map class indices back to role names
    formatted_rules = raw_rules
    # Iterate backwards through classes to avoid substring replacement issues (e.g., class 10 vs class 1)
    for i in sorted(range(len(le.classes_)), reverse=True):
        formatted_rules = formatted_rules.replace(f"class: {i}", f"Predict: {le.classes_[i]}")
        
    st.markdown(f"""<div class="rules-container">{formatted_rules}</div>""", unsafe_allow_html=True)
    
    st.info("The tree looks at each variable and makes a binary cut. As depth increases, the rules become more specific.")

# --- Training / Growth Loop ---
if mode == "Iterative (Growth)" and st.session_state.dt_is_training:
    if st.session_state.dt_current_depth < max_depth_target:
        # Use step_delay from sidebar to slow down the process
        time.sleep(step_delay) 
        st.session_state.dt_current_depth += 1
        st.session_state.dt_history.append(accuracy)
        st.rerun()
    else:
        # Stop training when target depth is reached
        st.session_state.dt_is_training = False
        st.rerun()

# --- Explanation Section ---
st.markdown("---")
col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    st.subheader("What are axis-aligned splits?")
    st.write("""
    Unlike SVMs which can draw diagonal or curved boundaries, **Decision Trees** only make 
    one decision at a time based on a single variable (e.g., 'Is Rushing Yards > 500?'). 
    
    This is why the regions look like a collection of perfect rectangles. Every 'split' in the 
    tree is a horizontal or vertical cut on this map.
    """)

with col_exp2:
    st.subheader("Recursive Partitioning")
    st.write("""
    In **Iterative Mode**, you are watching the tree grow by depth. 
    - **Depth 1**: One split, two regions.
    - **Depth 2**: Up to three splits, four regions.
    
    As the tree grows deeper, it becomes more 'opinionated' and can start to capture very 
    specific outliers, which can lead to **overfitting**.
    """)