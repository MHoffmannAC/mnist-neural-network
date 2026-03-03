import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from pages.utils.ml import generate_ml_data, select_roles

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="K-Means NFL Playground")

# Custom CSS for a "Playground" aesthetic
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
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Data Generation (Expanded NFL Archetypes) ---


# --- K-Means Logic Helpers ---
def get_initial_centroids(df, k):
    if df.empty:
        return pd.DataFrame(columns=["Rush", "Rec"])
    return df[["Rush", "Rec"]].sample(k).reset_index(drop=True)


def assign_clusters(df, centroids):
    if df.empty or centroids.empty:
        return np.array([])
    # Vectorized distance calculation
    dists = np.sqrt(
        ((df[["Rush", "Rec"]].values[:, np.newaxis] - centroids.values) ** 2).sum(
            axis=2,
        ),
    )
    return np.argmin(dists, axis=1)


def compute_new_centroids(df, k):
    # Group by cluster and find mean. Reindex ensures we handle clusters with no points
    new_c = df.groupby("Cluster")[["Rush", "Rec"]].mean().reindex(range(k))
    return new_c


# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")

    selected_roles = select_roles()

    st.subheader("2. Hyperparameters")
    k = st.slider(
        "Number of Clusters (K)",
        2,
        8,
        min(len(selected_roles), 8) if selected_roles else 4,
    )

    # Check for config changes to trigger reset
    current_config = {"k": k, "roles": sorted(selected_roles)}
    if (
        "last_config" not in st.session_state
        or st.session_state.last_config != current_config
    ):
        st.session_state.last_config = current_config
        st.session_state.df = generate_ml_data(selected_roles)
        if not st.session_state.df.empty:
            st.session_state.centroids = get_initial_centroids(st.session_state.df, k)
            st.session_state.df["Cluster"] = -1
        st.session_state.iteration = 0
        st.session_state.phase = "Assign"
        st.session_state.converged = False

    if st.button("Reset Centroids"):
        if not st.session_state.df.empty:
            st.session_state.centroids = get_initial_centroids(st.session_state.df, k)
            st.session_state.df["Cluster"] = -1
            st.session_state.iteration = 0
            st.session_state.phase = "Assign"
            st.session_state.converged = False
            st.rerun()

# --- Main Layout ---
st.title("🏈 K-Means NFL Playground")
st.markdown("### Identifying Player Roles via Rushing vs Receiving Stats")

if not selected_roles:
    st.warning(
        "Please select at least one player group in the sidebar to generate data.",
    )
    st.stop()

col_plot, col_info = st.columns([3, 1])

# --- Control Logic ---
with col_info:
    st.markdown("### Controls")

    if st.session_state.converged:
        st.success("✅ Algorithm Converged!")
    else:
        st.info(f"Next Phase: **{st.session_state.phase}**")

    # The "Next Step" Button
    if st.button("Next Step ➡️") and not st.session_state.converged:
        if st.session_state.phase == "Assign":
            st.session_state.df["Cluster"] = assign_clusters(
                st.session_state.df,
                st.session_state.centroids,
            )
            st.session_state.phase = "Update"
        else:
            old_centroids = st.session_state.centroids.copy()
            st.session_state.centroids = compute_new_centroids(
                st.session_state.df,
                k,
            ).combine_first(old_centroids)
            st.session_state.iteration += 1
            st.session_state.phase = "Assign"

            # Check for convergence
            if np.allclose(
                old_centroids.values,
                st.session_state.centroids.values,
                atol=1e-2,
            ):
                st.session_state.converged = True
        st.rerun()

    st.markdown("---")
    st.metric("Iteration", st.session_state.iteration)

    if st.session_state.iteration > 0:
        sse = 0
        for i, center in st.session_state.centroids.iterrows():
            pts = st.session_state.df[st.session_state.df["Cluster"] == i][
                ["Rush", "Rec"]
            ]
            if not pts.empty:
                sse += np.sum((pts.values - center.values) ** 2)
        st.metric("Inertia (SSE)", f"{sse:,.0f}")

# --- Plotting ---
with col_plot:
    plot_df = st.session_state.df.copy()
    plot_df["Cluster_Label"] = plot_df["Cluster"].apply(
        lambda x: f"Cluster {x}" if x >= 0 else "Unassigned",
    )

    # 1. Player Points
    points = (
        alt.Chart(plot_df)
        .mark_circle(size=80, opacity=0.7)
        .encode(
            x=alt.X("Rush:Q", title="Rushing Yards"),
            y=alt.Y("Rec:Q", title="Receiving Yards"),
            color=alt.Color(
                "Cluster_Label:N",
                scale=alt.Scale(
                    domain=[f"Cluster {i}" for i in range(k)] + ["Unassigned"],
                    range=[
                        "#10b981",
                        "#3b82f6",
                        "#f59e0b",
                        "#ef4444",
                        "#a78bfa",
                        "#ec4899",
                        "#64748b",
                    ],
                ),
                title="Assigned Group",
            ),
            tooltip=["True Role", "Rush", "Rec", "Cluster_Label"],
        )
    )

    # 2. Centroids
    centroid_plot_df = st.session_state.centroids.copy()
    centroid_plot_df["Cluster_ID"] = centroid_plot_df.index
    centroid_plot_df["Cluster_Label"] = centroid_plot_df["Cluster_ID"].apply(
        lambda x: f"Cluster {int(x)}",
    )

    centers = (
        alt.Chart(centroid_plot_df)
        .mark_point(
            size=500,
            shape="cross",
            strokeWidth=5,
            color="white",
            fill="black",
        )
        .encode(
            x="Rush:Q",
            y="Rec:Q",
            tooltip=["Cluster_Label", "Rush", "Rec"],
        )
    )

    st.altair_chart(
        (points + centers).properties(width="container", height=550).interactive(),
        width="stretch",
    )

# --- Analysis Table ---
if st.session_state.iteration > 0:
    with st.expander("📊 View Cluster Composition vs. Real Roles", expanded=True):
        table_df = st.session_state.df[st.session_state.df["Cluster"] != -1]
        crosstab = pd.crosstab(table_df["True Role"], table_df["Cluster"])
        crosstab.columns = [f"Cluster {c}" for c in crosstab.columns]
        st.table(crosstab)

st.markdown("---")
st.caption(
    "K-Means identifies spatial patterns. Try adding 'Dual-Threat QB' vs 'Pocket QB' to see if the algorithm can separate them based purely on rushing volume.",
)
