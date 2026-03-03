import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

from pages.utils.ml import generate_ml_data, select_roles

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Decision Tree NFL Playground")

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
""",
    unsafe_allow_html=True,
)

# --- Initialization & Session State ---
if "dt_is_training" not in st.session_state:
    st.session_state.dt_is_training = False
if "dt_current_depth" not in st.session_state:
    st.session_state.dt_current_depth = 1
if "dt_history" not in st.session_state:
    st.session_state.dt_history = []

# --- Sidebar ---
with st.sidebar:
    st.title("🌳 Decision Tree Hyperparameters")

    selected_roles = select_roles()

    st.markdown("---")
    st.subheader("2. Execution Mode")
    mode = st.radio(
        "Model Logic",
        ["Exact (Full Tree)", "Iterative (Growth)"],
        help="Exact shows the tree at its final depth. Iterative shows the tree growing level by level.",
    )

    st.markdown("---")
    st.subheader("3. Tree Tuning")

    max_depth_target = st.slider(
        "Max Depth",
        1,
        15,
        5,
        help="The maximum depth of the tree. Deeper trees are more complex but can overfit.",
    )

    criterion = st.selectbox(
        "Split Criterion",
        ["gini", "entropy"],
        index=0,
        help="Gini measures impurity; Entropy measures information gain.",
    )

    min_samples_split = st.slider("Min Samples to Split", 2, 20, 5)

    show_lines = st.checkbox(
        "Show Decision Lines",
        value=True,
        help="Draw the lines where the tree makes its cuts.",
    )

    if mode == "Iterative (Growth)":
        st.markdown("---")
        st.subheader("4. Animation Speed")
        step_delay = st.slider("Step Delay (seconds)", 0.1, 3.0, 1.0)

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
    st.warning(
        "Please select at least **two** roles in the sidebar to perform classification.",
    )
    st.stop()

df = generate_ml_data(selected_roles)
X = df[["Rush", "Rec"]]
y = df["True Role"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- Decision Tree Logic ---
current_depth = (
    max_depth_target
    if mode == "Exact (Full Tree)"
    else st.session_state.dt_current_depth
)

model = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=current_depth,
    min_samples_split=min_samples_split,
)
model.fit(X, y_encoded)
accuracy = model.score(X, y_encoded)


# --- Extracting Split Lines for Visualization ---
def get_split_lines(tree, node_id, x_min, x_max, y_min, y_max, lines):
    # If leaf node, stop
    if tree.feature[node_id] == -2:
        return

    feature = tree.feature[node_id]
    threshold = tree.threshold[node_id]

    if feature == 0:  # Rushing Yards Split (Vertical Line)
        lines.append(
            {
                "x1": threshold,
                "x2": threshold,
                "y1": y_min,
                "y2": y_max,
                "depth": node_id,
            },
        )
        # Recurse children with new boundaries
        get_split_lines(
            tree, tree.children_left[node_id], x_min, threshold, y_min, y_max, lines,
        )
        get_split_lines(
            tree, tree.children_right[node_id], threshold, x_max, y_min, y_max, lines,
        )
    else:  # Receiving Yards Split (Horizontal Line)
        lines.append(
            {
                "x1": x_min,
                "x2": x_max,
                "y1": threshold,
                "y2": threshold,
                "depth": node_id,
            },
        )
        # Recurse children with new boundaries
        get_split_lines(
            tree, tree.children_left[node_id], x_min, x_max, y_min, threshold, lines,
        )
        get_split_lines(
            tree, tree.children_right[node_id], x_min, x_max, threshold, y_max, lines,
        )


# --- UI Header ---
st.title("🌳 Decision Tree NFL Playground")
if mode == "Iterative (Growth)":
    if st.session_state.dt_current_depth >= max_depth_target:
        status_text = "✅ Depth Reached"
    elif st.session_state.dt_is_training:
        status_text = "🟢 Growing..."
    else:
        status_text = "🟡 Idle"
    st.markdown(
        f"**Current Depth:** `{st.session_state.dt_current_depth} / {max_depth_target}` | Status: {status_text}",
    )
else:
    st.markdown(f"**Status:** ✅ Full Tree Computed (Depth: {max_depth_target})")

# --- Boundary Boundaries ---
x_max_val = df["Rush"].max() + 150
y_max_val = df["Rec"].max() + 150

# --- Get the Lines ---
split_lines = []
get_split_lines(model.tree_, 0, 0, x_max_val, 0, y_max_val, split_lines)
lines_df = pd.DataFrame(split_lines)

# --- Decision Boundary Background Calculation ---
h = 20
xx, yy = np.meshgrid(np.arange(0, x_max_val, h), np.arange(0, y_max_val, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z_encoded = model.predict(grid_points)
Z = le.inverse_transform(Z_encoded)

grid_df = pd.DataFrame(
    {
        "Rush": xx.ravel(),
        "Rec": yy.ravel(),
        "Rush2": xx.ravel() + h,
        "Rec2": yy.ravel() + h,
        "Predicted_Role": Z,
    },
)

# --- Main Columns ---
col_viz, col_rules = st.columns([2, 1])

with col_viz:
    # --- Metrics ---
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-value">{accuracy:.1%}</div></div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">Total Nodes</div><div class="metric-value">{model.tree_.node_count}</div></div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">Total Cuts</div><div class="metric-value">{len(lines_df)}</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Visualization ---
    color_scale = alt.Scale(domain=selected_roles, scheme="tableau10")

    background = (
        alt.Chart(grid_df)
        .mark_rect(opacity=0.3)
        .encode(
            x=alt.X(
                "Rush:Q", title="Rushing Yards", scale=alt.Scale(domain=[0, x_max_val]),
            ),
            x2="Rush2:Q",
            y=alt.Y(
                "Rec:Q",
                title="Receiving Yards",
                scale=alt.Scale(domain=[0, y_max_val]),
            ),
            y2="Rec2:Q",
            color=alt.Color("Predicted_Role:N", scale=color_scale, legend=None),
            tooltip=alt.value(None),
        )
    )

    points = (
        alt.Chart(df)
        .mark_circle(size=75, stroke="black", strokeWidth=0.5)
        .encode(
            x="Rush:Q",
            y="Rec:Q",
            color=alt.Color("True Role:N", scale=color_scale, title="Real Role"),
            tooltip=["True Role", "Rush", "Rec"],
        )
    )

    # Decision Lines Layer
    layers = [background]

    if show_lines and not lines_df.empty:
        # We draw the lines. Older nodes (lower ID) get more opacity to show importance
        lines_layer = (
            alt.Chart(lines_df)
            .mark_rule(color="white")
            .encode(
                x="x1:Q",
                x2="x2:Q",
                y="y1:Q",
                y2="y2:Q",
                strokeWidth=alt.value(2),
                opacity=alt.value(0.6),
            )
        )
        layers.append(lines_layer)

    layers.append(points)

    st.altair_chart(
        alt.layer(*layers).properties(width="container", height=500).interactive(),
        width="stretch",
    )

with col_rules:
    st.subheader("📜 Current Decision Rules")
    st.write("This is the 'if-then' logic generated so far:")

    # Export tree rules to text
    raw_rules = export_text(model, feature_names=["Rushing Yards", "Receiving Yards"])

    formatted_rules = raw_rules
    for i in sorted(range(len(le.classes_)), reverse=True):
        formatted_rules = formatted_rules.replace(
            f"class: {i}",
            f"Predict: {le.classes_[i]}",
        )

    st.markdown(
        f"""<div class="rules-container">{formatted_rules}</div>""",
        unsafe_allow_html=True,
    )

    st.info(
        """White lines show exactly where the model divides the space.
        Each line is constrained by the lines that came before it.""",
    )

# --- Training / Growth Loop ---
if mode == "Iterative (Growth)" and st.session_state.dt_is_training:
    if st.session_state.dt_current_depth < max_depth_target:
        time.sleep(step_delay)
        st.session_state.dt_current_depth += 1
        st.session_state.dt_history.append(accuracy)
        st.rerun()
    else:
        st.session_state.dt_is_training = False
        st.rerun()

# --- Explanation Section ---
st.markdown("---")
col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    st.subheader("What are axis-aligned splits?")
    st.write("""
    Decision Trees only make one decision at a time based on a single variable.
    This creates the "nested boxes" structure you see.

    - **Vertical Lines**: Splitting based on Rushing Yards.
    - **Horizontal Lines**: Splitting based on Receiving Yards.
    """)

with col_exp2:
    st.subheader("Recursive Partitioning")
    st.write("""
    In **Iterative Mode**, you are watching the tree grow.
    Notice how early lines cross the whole map, while deeper lines are "boxed in" by
    previous decisions. This is the core of how decision trees build complex boundaries
    out of simple rules.
    """)
