import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from pages.utils.ml import generate_ml_data, select_roles

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Logistic Regression NFL Playground")

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
        border-bottom: 4px solid #ef4444;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #f87171;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
    }
    .coef-container {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 0.5rem;
        padding: 1.5rem;
        font-family: 'Fira Code', monospace;
        font-size: 0.75rem;
        color: #f87171;
        max-height: 600px;
        overflow-y: auto;
        white-space: pre;
        line-height: 1.6;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Initialization & Session State ---
if "lr_is_training" not in st.session_state:
    st.session_state.lr_is_training = False
if "lr_current_iteration" not in st.session_state:
    st.session_state.lr_current_iteration = 0
if "lr_history" not in st.session_state:
    st.session_state.lr_history = []
if "lr_converged" not in st.session_state:
    st.session_state.lr_converged = False

# --- Sidebar ---
with st.sidebar:
    st.title("📈 Logistic Regression")

    selected_roles = select_roles()

    st.markdown("---")
    st.subheader("1. Execution Mode")
    mode = st.radio(
        "Model Logic",
        ["Exact (Solver)", "Iterative (Gradient Descent)"],
        help="Exact finds the best lines instantly. Iterative shows the model learning step-by-step.",
    )

    st.markdown("---")
    st.subheader("2. Model Tuning")

    c_param = st.slider(
        "Inverse Regularization (C)",
        0.01,
        10.0,
        1.0,
        help="Lower C = Stronger Regularization (smoother lines).",
    )

    use_balanced = st.checkbox(
        "Balanced Class Weights",
        value=True,
        help="Heavily penalizes misclassifying rare or 'hard' roles like Tight Ends.",
    )

    seed_val = st.slider("Random Seed", 0, 100, 42)
    use_poly = st.checkbox("Use Polynomial Features (Degree 2)", value=False)

    if mode == "Iterative (Gradient Descent)":
        st.markdown("---")
        st.subheader("3. Iteration Settings")
        learning_rate = st.selectbox("Learning Speed", [0.001, 0.01, 0.1, 0.5], index=1)
        max_iter = st.number_input("Max Iterations", 10, 500, 100)
        step_delay = st.slider("Step Delay (seconds)", 0.05, 1.0, 0.1)

    st.markdown("---")

    # Reset logic for config changes
    current_config = {
        "roles": sorted(selected_roles),
        "poly": use_poly,
        "C": c_param,
        "seed": seed_val,
        "balanced": use_balanced,
    }
    if (
        "last_config" not in st.session_state
        or st.session_state.last_config != current_config
    ):
        st.session_state.last_config = current_config
        st.session_state.lr_is_training = False
        st.session_state.lr_current_iteration = 0
        st.session_state.lr_history = []
        st.session_state.lr_converged = False
        if "lr_model" in st.session_state:
            del st.session_state.lr_model

    if mode == "Iterative (Gradient Descent)":
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.lr_current_iteration >= max_iter:
                btn_label = "✅ Finished"
                btn_disabled = True
            else:
                btn_label = "⏸ Pause" if st.session_state.lr_is_training else "🚀 Train"
                btn_disabled = False
            if st.button(btn_label, disabled=btn_disabled):
                st.session_state.lr_is_training = not st.session_state.lr_is_training
                st.rerun()
        with col2:
            if st.button("♻️ Reset"):
                st.session_state.lr_is_training = False
                st.session_state.lr_current_iteration = 0
                st.session_state.lr_history = []
                st.session_state.lr_converged = False
                if "lr_model" in st.session_state:
                    del st.session_state.lr_model
                st.rerun()

# --- Data Prep ---
if len(selected_roles) < 2:
    st.title("📈 Logistic Regression Playground")
    st.warning("Please select at least **two** roles to begin.")
    st.stop()

df = generate_ml_data(selected_roles)
X = df[["Rush", "Rec"]]
y = df["True Role"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = np.unique(y_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_input = poly.fit_transform(X_scaled) if use_poly else X_scaled
feature_names = (
    poly.get_feature_names_out(["Rush", "Rec"]) if use_poly else ["Rush", "Rec"]
)

# --- Model Logic ---
class_weight = "balanced" if use_balanced else None

if mode == "Exact (Solver)":
    model = LogisticRegression(
        C=c_param,
        solver="lbfgs",
        max_iter=1000,
        class_weight=class_weight,
    )
    model.fit(X_input, y_encoded)
    accuracy = model.score(X_input, y_encoded)
else:
    if "lr_model" not in st.session_state:
        model = SGDClassifier(
            loss="log_loss",
            alpha=1 / c_param,
            learning_rate="constant",
            eta0=learning_rate,
            random_state=seed_val,
        )
        np.random.seed(seed_val)
        model.coef_ = (
            np.random.randn(len(classes) if len(classes) > 2 else 1, X_input.shape[1])
            * 2.0
        )
        model.intercept_ = (
            np.random.randn(len(classes) if len(classes) > 2 else 1) * 2.0
        )
        model.classes_ = classes
        st.session_state.lr_model = model

    model = st.session_state.lr_model
    accuracy = model.score(X_input, y_encoded) if hasattr(model, "coef_") else 0.0

# --- UI Header ---
st.title("📈 Logistic Regression NFL Playground")
if mode == "Iterative (Gradient Descent)":
    if st.session_state.lr_current_iteration >= max_iter:
        status_text = "✅ Training Complete"
    elif st.session_state.lr_is_training:
        status_text = "🟢 Training..."
    else:
        status_text = "🟡 Idle"
    st.markdown(
        f"**Iteration:** `{st.session_state.lr_current_iteration} / {max_iter}` | Status: {status_text}",
    )
else:
    st.markdown("**Status:** ✅ Optimal Boundary Found (Exact Mode)")

# --- Calculations & Visualization ---
x_max, y_max = df["Rush"].max() + 150, df["Rec"].max() + 150
h = 15
xx, yy = np.meshgrid(np.arange(0, x_max, h), np.arange(0, y_max, h))
grid_raw = np.c_[xx.ravel(), yy.ravel()]
# Wrap in DataFrame to keep feature names and avoid UserWarning
grid_scaled = scaler.transform(pd.DataFrame(grid_raw, columns=["Rush", "Rec"]))
grid_input = poly.transform(grid_scaled) if use_poly else grid_scaled

Z_raw = model.predict(grid_input)
Z = le.inverse_transform(Z_raw)
grid_df = pd.DataFrame(
    {
        "Rush": xx.ravel(),
        "Rec": yy.ravel(),
        "Rush2": xx.ravel() + h,
        "Rec2": yy.ravel() + h,
        "Predicted_Role": Z,
    },
)

# --- Calculating Decision Lines (Only for Linear Case) ---
lines_data = []
if not use_poly and hasattr(model, "coef_"):
    # Normalize coefs/intercepts for binary case
    if len(classes) == 2:
        W = np.vstack([-model.coef_, model.coef_])
        B = np.array([-model.intercept_[0], model.intercept_[0]])
    else:
        W = model.coef_
        B = model.intercept_

    # Pairwise boundaries
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            dw = W[i] - W[j]
            db = B[i] - B[j]

            if abs(dw[1]) > 1e-5:
                # We sample the line across the visible domain
                x_ends = np.array([0, x_max])
                x_ends_scaled = (x_ends - scaler.mean_[0]) / scaler.scale_[0]
                y_ends_scaled = -(dw[0] * x_ends_scaled + db) / dw[1]
                y_ends = y_ends_scaled * scaler.scale_[1] + scaler.mean_[1]

                # Higher resolution sampling to ensure continuity
                samples = 100
                sample_xs = np.linspace(x_ends[0], x_ends[1], samples)
                sample_ys = np.linspace(y_ends[0], y_ends[1], samples)

                for s in range(samples - 1):
                    x1, x2 = sample_xs[s], sample_xs[s + 1]
                    y1, y2 = sample_ys[s], sample_ys[s + 1]

                    # Only plot segments that fall within the y-bounds of the chart
                    if (y1 < 0 and y2 < 0) or (y1 > y_max and y2 > y_max):
                        continue

                    # Check midpoint of segment to determine active/inactive status
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    # Wrap in DataFrame to keep feature names and avoid UserWarning
                    mid_scaled = scaler.transform(
                        pd.DataFrame([[mid_x, mid_y]], columns=["Rush", "Rec"]),
                    )
                    mid_winner = model.predict(mid_scaled)[0]

                    # Boundary is "Active" if it separates the two current highest probability classes
                    is_active = mid_winner == classes[i] or mid_winner == classes[j]

                    lines_data.append(
                        {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "Pair": f"{le.classes_[i]} vs {le.classes_[j]}",
                            "Active": "Solid" if is_active else "Dotted",
                        },
                    )

lines_df = pd.DataFrame(lines_data)

# --- Layout ---
col_viz, col_rules = st.columns([2, 1])

with col_viz:
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-value">{accuracy:.1%}</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Weights</div><div class="metric-value">{model.coef_.size}</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Training Size</div><div class="metric-value">{len(df)}</div></div>',
            unsafe_allow_html=True,
        )

    color_scale = alt.Scale(domain=selected_roles, scheme="tableau10")
    background = (
        alt.Chart(grid_df)
        .mark_rect(opacity=0.3)
        .encode(
            x=alt.X(
                "Rush:Q",
                title="Rushing Yards",
                scale=alt.Scale(domain=[0, x_max]),
            ),
            x2="Rush2:Q",
            y=alt.Y(
                "Rec:Q",
                title="Receiving Yards",
                scale=alt.Scale(domain=[0, y_max]),
            ),
            y2="Rec2:Q",
            color=alt.Color("Predicted_Role:N", scale=color_scale, legend=None),
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

    layers = [background, points]
    if not lines_df.empty:
        # Plot active lines (solid, thick)
        layers.append(
            alt.Chart(lines_df[lines_df["Active"] == "Solid"])
            .mark_rule(
                color="white",
                opacity=0.8,
                strokeWidth=2,
            )
            .encode(x="x1:Q", y="y1:Q", x2="x2:Q", y2="y2:Q"),
        )

        # Plot inactive lines (dotted, thin)
        layers.append(
            alt.Chart(lines_df[lines_df["Active"] == "Dotted"])
            .mark_rule(
                color="white",
                opacity=0.3,
                strokeWidth=1,
                strokeDash=[4, 4],
            )
            .encode(x="x1:Q", y="y1:Q", x2="x2:Q", y2="y2:Q"),
        )

    st.altair_chart(
        alt.layer(*layers).properties(width="container", height=500).interactive(),
        width="stretch",
    )

with col_rules:
    st.subheader("⚖️ Model Coefficients")
    st.write("Mathematical weights for each role's decision plane.")

    # Extract coefficients correctly for binary vs multinomial
    if len(classes) == 2:
        display_W = np.vstack([-model.coef_, model.coef_])
        display_B = np.array([-model.intercept_[0], model.intercept_[0]])
    else:
        display_W = model.coef_
        display_B = model.intercept_

    # Formatting requested by user
    separator = "----------------\n"
    coef_text = separator
    for i, role in enumerate(le.classes_):
        coef_text += f"[{role}]\n"
        coef_text += f"Intercept: {display_B[i]:.3f}\n"
        for fname, weight in zip(feature_names, display_W[i]):
            coef_text += f" {fname}: {weight:+.3f}\n"
        coef_text += separator

    st.markdown(
        f'<div class="coef-container">{coef_text}</div>',
        unsafe_allow_html=True,
    )

# --- History Plot ---
if mode == "Iterative (Gradient Descent)" and st.session_state.lr_history:
    st.markdown("---")
    st.subheader("📈 Accuracy Over Time")
    hist_df = pd.DataFrame(
        {
            "Iteration": range(len(st.session_state.lr_history)),
            "Accuracy": st.session_state.lr_history,
        },
    )
    st.altair_chart(
        alt.Chart(hist_df)
        .mark_line(color="#f87171")
        .encode(x="Iteration", y="Accuracy")
        .properties(width="container", height=150),
        width="stretch",
    )

# --- Training Loop ---
# This block is at the end to ensure all UI elements (including the history plot) are rendered before st.rerun()
if mode == "Iterative (Gradient Descent)" and st.session_state.lr_is_training:
    if st.session_state.lr_current_iteration < max_iter:
        time.sleep(step_delay)
        sample_weights = (
            compute_sample_weight("balanced", y_encoded) if use_balanced else None
        )
        st.session_state.lr_model.partial_fit(
            X_input,
            y_encoded,
            classes=classes,
            sample_weight=sample_weights,
        )
        st.session_state.lr_current_iteration += 1
        st.session_state.lr_history.append(
            st.session_state.lr_model.score(X_input, y_encoded),
        )
        st.rerun()
    else:
        st.session_state.lr_is_training = False
        st.rerun()
