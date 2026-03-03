import altair as alt
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from pages.utils.ml import generate_ml_data, select_roles

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
        padding: 1rem;
        text-align: center;
    }
    .info-box {
        background-color: #1e293b;
        border-left: 5px solid #2563eb;
        padding: 1rem;
        border-radius: 0.3rem;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ DBSCAN Settings")

    selected_roles = select_roles()

    st.subheader("2. Hyperparameters")
    st.markdown("Unlike K-Means, DBSCAN finds clusters based on density.")

    eps = st.slider(
        "Epsilon (Radius)",
        0.1,
        2.0,
        0.5,
        help="Maximum distance between two samples for one to be considered as in the neighborhood of the other.",
    )
    min_samples = st.slider(
        "Min Samples",
        2,
        15,
        5,
        help="Number of samples in a neighborhood for a point to be considered a core point.",
    )

    st.info(
        "💡 **Epsilon** controls how far the algorithm looks for neighbors. **Min Samples** controls how 'crowded' an area must be to form a cluster.",
    )

# --- Main Layout ---
st.title("🛰️ DBSCAN NFL Playground")
st.markdown("### Density-Based Clustering & Outlier Detection")

if not selected_roles:
    st.warning("Select roles in the sidebar to begin.")
    st.stop()

df = generate_ml_data(selected_roles)

# --- Clustering Logic ---
# DBSCAN is sensitive to scale, so we normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[["Rush", "Rec"]])

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
df["Cluster"] = dbscan.fit_predict(X_scaled)

# Analysis of results
n_clusters = len(set(df["Cluster"])) - (1 if -1 in df["Cluster"] else 0)
n_noise = list(df["Cluster"]).count(-1)

# --- Metrics Display ---
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Clusters Found", n_clusters)
with m2:
    st.metric("Outliers (Noise)", n_noise)
with m3:
    st.metric("Total Players", len(df))

col_plot, col_info = st.columns([3, 1])

with col_plot:
    # Prepare Plot Labels
    df["Cluster_Label"] = df["Cluster"].apply(
        lambda x: f"Cluster {x}" if x >= 0 else "Outlier (Noise)",
    )

    # Custom color palette: Outliers are usually dark gray/black in DBSCAN visualizations
    color_scale = alt.Scale(
        domain=[f"Cluster {i}" for i in range(max(0, n_clusters))]
        + ["Outlier (Noise)"],
        range=[
            "#10b981",
            "#3b82f6",
            "#f59e0b",
            "#a78bfa",
            "#ef4444",
            "#ec4899",
            "#06b6d4",
            "#475569",
        ],
    )

    chart = (
        alt.Chart(df)
        .mark_circle(size=100, opacity=0.8, stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X("Rush:Q", title="Rushing Yards"),
            y=alt.Y("Rec:Q", title="Receiving Yards"),
            color=alt.Color(
                "Cluster_Label:N",
                scale=color_scale,
                title="DBSCAN Result",
            ),
            tooltip=["True Role", "Rush", "Rec", "Cluster_Label"],
        )
        .properties(width="container", height=550)
        .interactive()
    )

    st.altair_chart(chart, width="stretch")

with col_info:
    st.markdown("### How to read this:")
    st.markdown("""
    - **Colored Points**: These are part of a 'dense' cluster.
    - **Dark Gray Points (-1)**: These are **Noise**. The algorithm couldn't find enough neighbors within the radius to consider them part of a group.
    
    **Try this:**
    Set **Epsilon** to `0.4` and add **Dual-Threat QBs**. Notice how they often become Outliers because they are statistically unique!
    """)

    if n_noise > 0:
        st.markdown("---")
        st.markdown("### Outlier Breakdown")
        noise_df = df[df["Cluster"] == -1]["True Role"].value_counts().reset_index()
        noise_df.columns = ["Role", "Count"]
        st.dataframe(noise_df, width="stretch", hide_index=True)

# --- Cluster Composition Table ---
if n_clusters > 0:
    with st.expander("📊 Detailed Cluster Breakdown"):
        # Filter out noise for the crosstab if you only want to see clusters
        crosstab = pd.crosstab(df["True Role"], df["Cluster"])
        crosstab.columns = [
            f"Cluster {c}" if c != -1 else "Noise" for c in crosstab.columns
        ]
        st.table(crosstab)
