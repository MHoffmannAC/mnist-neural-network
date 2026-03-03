import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import time
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pages.utils.ml import select_roles, generate_ml_data

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="SVM NFL Playground")

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
</style>
""", unsafe_allow_html=True)

# --- Initialization & Session State ---
if 'is_training' not in st.session_state:
    st.session_state.is_training = False
if 'current_iteration' not in st.session_state:
    st.session_state.current_iteration = 0
if 'history' not in st.session_state:
    st.session_state.history = []
if 'converged' not in st.session_state:
    st.session_state.converged = False

# --- Sidebar ---
with st.sidebar:
    st.title("🛡️ SVM Hyperparameters")
    
    selected_roles = select_roles()

    st.markdown("---")
    st.subheader("1. Execution Mode")
    mode = st.radio("Model Type", ["Exact (SVC)", "Iterative (SGD)"], 
                    help="Exact finds the perfect boundary instantly. Iterative shows the learning process.")

    st.markdown("---")
    st.subheader("2. Model Tuning")
    
    c_param = st.slider("C (Regularization)", 0.01, 10.0, 1.0, 
                        help="Lower C allows for a softer margin. High C forces strict classification.")
    
    gamma = st.slider("Gamma (Complexity)", 0.01, 5.0, 0.5, 
                      help="Higher Gamma creates more complex, localized boundaries.")

    if mode == "Iterative (SGD)":
        learning_rate = st.selectbox("Learning Speed", [0.001, 0.01, 0.1, 0.5], index=1)
        tol = st.slider("Convergence Tolerance", 0.0001, 0.01, 0.001, format="%.4f")
    
    st.markdown("---")
    
    if mode == "Iterative (SGD)":
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.converged:
                btn_label = "✅ Done"
                btn_disabled = True
            else:
                btn_label = "⏸ Pause" if st.session_state.is_training else "🚀 Train"
                btn_disabled = False
            if st.button(btn_label, disabled=btn_disabled):
                st.session_state.is_training = not st.session_state.is_training
                st.rerun()
        with col2:
            if st.button("♻️ Reset"):
                st.session_state.is_training = False
                st.session_state.current_iteration = 0
                st.session_state.history = []
                st.session_state.converged = False
                if 'iter_model' in st.session_state: del st.session_state.iter_model
                st.rerun()
    else:
        st.info("Exact mode computes the optimal boundary instantly.")

# --- Data Prep ---
if len(selected_roles) < 2:
    st.title("🛡️ SVM NFL Playground")
    st.warning("Please select at least **two** roles in the sidebar to perform classification.")
    st.stop()

df = generate_ml_data(selected_roles)
X = df[['Rush', 'Rec']]
y = df['True Role']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = np.unique(y_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Model Logic ---
support_vectors = None

if mode == "Exact (SVC)":
    # The "Real" SVM
    model = SVC(kernel='rbf', C=c_param, gamma=gamma, probability=True)
    model.fit(X_scaled, y_encoded)
    accuracy = model.score(X_scaled, y_encoded)
    # SVC explicitly tells us who the support vectors are
    support_vectors = df.iloc[model.support_]

else:
    # The "Iterative" Approximation
    if 'iter_model' not in st.session_state:
        rbf_feature = RBFSampler(gamma=gamma, n_components=100, random_state=42)
        rbf_feature.fit(X_scaled)
        iter_model = SGDClassifier(loss='hinge', alpha=1/(c_param * len(df)), 
                                   learning_rate='constant', eta0=learning_rate, random_state=42)
        st.session_state.iter_model = iter_model
        st.session_state.rbf_feature = rbf_feature
        st.session_state.current_iteration = 0
        st.session_state.converged = False

    model = st.session_state.iter_model
    rbf_feature = st.session_state.rbf_feature
    
    # Check accuracy if already fitted
    if hasattr(model, "coef_"):
        accuracy = model.score(rbf_feature.transform(X_scaled), y_encoded)
    else:
        accuracy = 0.0

# --- UI Header ---
st.title("🛡️ SVM NFL Playground")
if mode == "Iterative (SGD)":
    status_text = "🟢 Training..." if st.session_state.is_training else ("✅ Converged" if st.session_state.converged else "🟡 Idle")
    st.markdown(f"**Iteration:** `{st.session_state.current_iteration}` | Status: {status_text}")
else:
    st.markdown("**Status:** ✅ Optimal Boundary Found (Exact Solver)")

# --- Decision Boundary Calculation ---
x_max = df['Rush'].max() + 150
y_max = df['Rec'].max() + 150
h = 25 
xx, yy = np.meshgrid(np.arange(0, x_max, h), np.arange(0, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid_points)

# Calculate predictions for background
try:
    if mode == "Exact (SVC)":
        Z_encoded = model.predict(grid_scaled)
    else:
        if hasattr(model, "coef_"):
            grid_features = rbf_feature.transform(grid_scaled)
            Z_encoded = model.predict(grid_features)
        else:
            Z_encoded = np.array([0] * len(grid_points))
    Z = le.inverse_transform(Z_encoded)
except:
    Z = np.array([selected_roles[0]] * len(grid_points))

grid_df = pd.DataFrame({
    'Rush': xx.ravel(), 'Rec': yy.ravel(),
    'Rush2': xx.ravel() + h, 'Rec2': yy.ravel() + h,
    'Predicted_Role': Z
})

# --- Metrics ---
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Current Accuracy</div><div class="metric-value">{accuracy:.1%}</div></div>""", unsafe_allow_html=True)
with m2:
    val = len(model.support_) if mode == "Exact (SVC)" else st.session_state.current_iteration
    lbl = "Support Vectors" if mode == "Exact (SVC)" else "Iterations"
    st.markdown(f"""<div class="metric-card"><div class="metric-label">{lbl}</div><div class="metric-value">{val}</div></div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Training Size</div><div class="metric-value">{len(df)}</div></div>""", unsafe_allow_html=True)

# --- Plotting ---
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
    x='Rush:Q', y='Rec:Q',
    color=alt.Color('True Role:N', scale=color_scale, title="Real Role"),
    tooltip=['True Role', 'Rush', 'Rec']
)

layers = [background, points]

# Add halos for Support Vectors only in Exact mode
if support_vectors is not None:
    sv_layer = alt.Chart(support_vectors).mark_point(
        size=160, shape='circle', stroke='white', strokeWidth=2, opacity=1.0
    ).encode(x='Rush:Q', y='Rec:Q')
    layers.insert(1, sv_layer)

st.altair_chart(alt.layer(*layers).properties(width='container', height=600).interactive(), width='stretch')

# --- Training Loop Logic ---
if mode == "Iterative (SGD)" and st.session_state.is_training and not st.session_state.converged:
    X_features = rbf_feature.transform(X_scaled)
    prev_coef = model.coef_.copy() if hasattr(model, "coef_") else None
    
    model.partial_fit(X_features, y_encoded, classes=classes)
    st.session_state.current_iteration += 1
    st.session_state.history.append(model.score(X_features, y_encoded))
    
    if prev_coef is not None:
        delta = np.linalg.norm(model.coef_ - prev_coef)
        if delta < tol:
            st.session_state.converged = True
            st.session_state.is_training = False
    
    time.sleep(0.01)
    st.rerun()

# --- Explanation Section ---
st.markdown("---")
col_exp1, col_exp2 = st.columns(2)
with col_exp1:
    st.subheader("Exact vs. Iterative")
    st.write("""
    - **Exact (SVC)**: Uses a quadratic programming solver. It finds the absolute maximum margin possible for the kernel. 
    It is the 'correct' way to perform SVM and provides the highest accuracy.
    - **Iterative (SGD)**: Approximates the SVM using gradient descent. It's useful for huge datasets where the exact solver is too slow, 
    but for this size, it is less precise and doesn't explicitly identify support vectors.
    """)
with col_exp2:
    if mode == "Iterative (SGD)" and st.session_state.history:
        hist_df = pd.DataFrame({'Iteration': range(len(st.session_state.history)), 'Accuracy': st.session_state.history})
        st.altair_chart(alt.Chart(hist_df).mark_line(color='#60a5fa').encode(x='Iteration', y='Accuracy').properties(height=150), width='stretch')
    else:
        st.write("**Support Vectors**")
        st.write("In Exact mode, the players with white halos are the 'boundary' players. Only these players matter for the model's logic.")