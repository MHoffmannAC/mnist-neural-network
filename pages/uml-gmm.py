import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from pages.utils.ml import generate_ml_data, select_roles

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="GMM NFL Playground")

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
        border-top: 4px solid #8b5cf6;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #a78bfa;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.title("🟣 GMM Hyperparameters")

    selected_roles = select_roles()

    st.markdown("---")
    st.subheader("2. GMM Configuration")

    n_components = st.slider("Number of Gaussians (K)", 2, 8, 4)

    covariance_type = st.selectbox(
        "Covariance Type",
        ["full", "tied", "diag", "spherical"],
        index=0,
        help="Full allows ellipses to be oriented in any direction. Diag constrains them to be axis-aligned.",
    )

    st.markdown("---")
    st.info("""
    **Gaussian Mixture Models** (GMM) assume the data is composed of $K$ overlapping probability distributions.
    
    Unlike K-Means, GMM allows for **soft assignment**, meaning a player can belong 70% to Cluster A and 30% to Cluster B.
    """)

# --- Main Layout ---
st.title("🧠 Gaussian Mixture Model (GMM) Playground")
st.markdown("### Soft Clustering & Probabilistic Player Archetypes")

if not selected_roles:
    st.warning("Please select at least one player group in the sidebar.")
    st.stop()

df = generate_ml_data(selected_roles)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[["Rush", "Rec"]])

# --- GMM Logic ---
gmm = GaussianMixture(
    n_components=n_components,
    covariance_type=covariance_type,
    random_state=42,
    max_iter=100,
)
gmm.fit(X_scaled)
df["Cluster"] = gmm.predict(X_scaled)

# Probabilities
probs = gmm.predict_proba(X_scaled)
df["Confidence"] = np.max(probs, axis=1)
# Create a string for probabilities to show in tooltip
df["Prob_String"] = [
    ", ".join([f"G{i}: {p:.1%}" for i, p in enumerate(row)]) for row in probs
]

# --- Metrics ---
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-label">Likelihood (Log)</div><div class="metric-value">{gmm.lower_bound_:.2f}</div></div>""",
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-label">Avg Confidence</div><div class="metric-value">{df["Confidence"].mean():.1%}</div></div>""",
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-label">Total Players</div><div class="metric-value">{len(df)}</div></div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

col_plot, col_analysis = st.columns([2.5, 1])

with col_plot:
    # --- Altair Visualization ---
    df["Gaussian_Label"] = df["Cluster"].apply(lambda x: f"Gaussian {x}")

    color_scale = alt.Scale(scheme="tableau10")

    # Scatter plot where size depends on confidence
    scatter = (
        alt.Chart(df)
        .mark_circle(size=100, stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X("Rush:Q", title="Rushing Yards"),
            y=alt.Y("Rec:Q", title="Receiving Yards"),
            color=alt.Color(
                "Gaussian_Label:N",
                scale=color_scale,
                title="Most Likely Gaussian",
            ),
            size=alt.Size(
                "Confidence:Q",
                scale=alt.Scale(range=[40, 300]),
                title="Confidence Score",
            ),
            opacity=alt.condition(
                alt.datum.Confidence < 0.6,
                alt.value(0.5),
                alt.value(0.9),
            ),
            tooltip=["True Role", "Rush", "Rec", "Gaussian_Label", "Prob_String"],
        )
        .properties(width="container", height=600)
        .interactive()
    )

    st.altair_chart(scatter, width="stretch")

with col_analysis:
    st.subheader("Probabilistic Analysis")
    st.write(
        "GMM is perfect for **Hybrid Roles**. Look for players with smaller dots—these are players the model is unsure about.",
    )

    # Show lowest confidence players (the hybrids)
    low_conf = df.sort_values("Confidence").head(5)[
        ["True Role", "Confidence", "Gaussian_Label"]
    ]
    st.markdown("**Top 5 'Identity Crisis' Players:**")
    st.dataframe(low_conf, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("Covariance Influence")
    if covariance_type == "full":
        st.write(
            "Using **Full** covariance allows the 'clouds' to rotate. This captures the correlation between rushing and receiving.",
        )
    elif covariance_type == "diag":
        st.write(
            "Using **Diag** covariance forces the 'clouds' to align with the axes. This ignores correlation.",
        )

# --- Detailed Breakdown Expander ---
with st.expander("📊 View Composition Matrix"):
    st.write("Which Gaussian 'captured' which real-world role?")
    crosstab = pd.crosstab(df["True Role"], df["Gaussian_Label"])
    st.table(crosstab)

st.markdown("---")
st.caption(
    "Gaussian Mixture Models use the Expectation-Maximization (EM) algorithm to find the parameters of the distributions that best represent the data.",
)
