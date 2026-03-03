import base64
import io
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from matplotlib import cm
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --- Page Configuration & CSS ---
st.set_page_config(layout="wide", page_title="Neural Net Playground")

st.markdown(
    """
<style>
    .reportview-container {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    .sidebar .sidebar-content {
        background-color: #1e293b;
    }
    h1, h2, h3 {
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .metric-container {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-family: monospace;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 9999px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    .prediction-bar-container {
        display: flex;
        align-items: center;
        margin-bottom: 0.25rem;
        width: 100%;
    }
    .prediction-label {
        width: 20px;
        font-family: monospace;
        font-weight: bold;
        color: #94a3b8;
    }
    .prediction-bar-bg {
        flex-grow: 1;
        background-color: #0f172a;
        height: 12px;
        border-radius: 9999px;
        margin: 0 10px;
        overflow: hidden;
    }
    .prediction-bar-fill {
        height: 100%;
        background-color: #10b981;
        transition: width 0.3s ease;
    }
    .prediction-value {
        width: 40px;
        font-family: monospace;
        font-size: 0.8rem;
        color: #94a3b8;
        text-align: right;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Constants ---
VAL_EXAMPLES = 10000


# --- Data Loading (Cached) ---
@st.cache_data
def load_mnist_data(train_examples):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    indices = np.random.permutation(len(x_train))
    x_train_sub = x_train[indices[:train_examples]]
    y_train_sub = y_train[indices[:train_examples]]

    x_val_sub = x_test[:VAL_EXAMPLES]
    y_val_sub = y_test[:VAL_EXAMPLES]

    return (x_train_sub, y_train_sub), (x_val_sub, y_val_sub), x_test


# --- Model Building ---
def build_compiled_model(layers, activation, lr):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    regularizer = tf.keras.regularizers.l2(0.0001)

    for units in layers:
        model.add(
            tf.keras.layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizer,
            ),
        )
    model.add(
        tf.keras.layers.Dense(
            10,
            activation="softmax",
            kernel_regularizer=regularizer,
        ),
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --- Utility Functions ---
def get_effective_receptive_field(model, target_layer_idx, node_idx):
    """Back-projects weights to visualize what a neuron is looking for in the 28x28 input."""
    w_matrix = model.layers[target_layer_idx].get_weights()[0]
    eff_weights = w_matrix[:, node_idx]

    for l in range(target_layer_idx - 1, 0, -1):
        prev_w = model.layers[l].get_weights()[0]
        eff_weights = np.dot(prev_w, eff_weights)

    return eff_weights.reshape((28, 28))


def get_rf_base64(model, target_layer_idx, node_idx):
    eff_weights = get_effective_receptive_field(model, target_layer_idx, node_idx)
    max_abs = np.percentile(np.abs(eff_weights), 98) + 1e-12
    norm_w = np.clip(eff_weights / max_abs, -1.0, 1.0)

    cmap = cm.get_cmap("RdBu")
    mapped_img = cmap((norm_w + 1) / 2)
    img = Image.fromarray((mapped_img * 255).astype(np.uint8))
    img = img.resize((84, 84), Image.Resampling.NEAREST)

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def array_to_base64(arr):
    """Helper to convert a gradient/saliency 2D array to a base64 image with robust normalization."""
    abs_arr = np.abs(arr)
    max_raw = np.max(abs_arr)

    if max_raw < 1e-25:
        norm_arr = np.zeros_like(arr)
    else:
        p_val = np.percentile(abs_arr, 99.5)
        divisor = p_val if p_val > (max_raw * 0.001) else max_raw
        norm_arr = np.clip(arr / (divisor + 1e-30), -1.0, 1.0)

    cmap = cm.get_cmap("RdBu")
    mapped_img = cmap((norm_arr + 1) / 2)
    img = Image.fromarray((mapped_img * 255).astype(np.uint8))
    img = img.resize((84, 84), Image.Resampling.NEAREST)

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def compute_all_saliencies(model, test_input):
    """Calculates saliency maps by taking gradients of PRE-ACTIVATION values."""
    img_tensor = tf.convert_to_tensor(test_input, dtype=tf.float32)
    saliencies = [None]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(img_tensor)
        curr = img_tensor
        layer_preactivations = []
        for i, layer in enumerate(model.layers):
            if i == 0:
                curr = layer(curr)
                continue
            w, b = layer.get_weights()
            pre_act = tf.matmul(curr, w) + b
            layer_preactivations.append(pre_act)
            curr = layer.activation(pre_act)

    for pre_act in layer_preactivations:
        layer_sals = []
        for n_idx in range(pre_act.shape[1]):
            grad = tape.gradient(pre_act[0, n_idx], img_tensor)
            if grad is not None:
                layer_sals.append(grad.numpy()[0, :, :, 0])
            else:
                layer_sals.append(np.zeros((28, 28)))
        saliencies.append(layer_sals)

    del tape
    return saliencies


def draw_network_svg(layers_config, activations=None, model=None, saliencies=None):
    """Generates an SVG string representing the neural network architecture with tooltips."""
    width = 700
    height = 500
    rendered_layers = []

    # Input nodes visual (5 top, ellipsis, 5 bottom)
    in_indices = list(range(5)) + ["dots"] + list(range(784 - 5, 784))
    in_nodes = []
    for idx in in_indices:
        if idx == "dots":
            in_nodes.append(
                {"id": "dots", "val": 0, "rf": "", "sal": "", "real_idx": None},
            )
        else:
            in_nodes.append(
                {
                    "id": f"in_{idx}",
                    "val": (activations[0][idx] if activations else 0),
                    "rf": "",
                    "sal": "",
                    "real_idx": idx,
                },
            )
    rendered_layers.append(in_nodes)

    # Hidden layers
    for l_idx, units in enumerate(layers_config):
        layer_acts = (
            activations[l_idx + 2]
            if activations and len(activations) > l_idx + 2
            else np.zeros(units)
        )
        nodes = []
        for i in range(units):
            rf_b64 = get_rf_base64(model, l_idx + 1, i) if model else ""
            sal_b64 = (
                array_to_base64(saliencies[l_idx + 1][i])
                if saliencies and len(saliencies) > l_idx + 1
                else ""
            )
            nodes.append(
                {
                    "id": f"h_{l_idx}_{i}",
                    "val": layer_acts[i],
                    "rf": rf_b64,
                    "sal": sal_b64,
                    "real_idx": i,
                },
            )
        rendered_layers.append(nodes)

    # Output layer
    out_acts = activations[-1] if activations else np.zeros(10)
    out_nodes = []
    for i in range(10):
        rf_b64 = get_rf_base64(model, len(layers_config) + 1, i) if model else ""
        sal_b64 = array_to_base64(saliencies[-1][i]) if saliencies else ""
        out_nodes.append(
            {
                "id": f"out_{i}",
                "val": out_acts[i],
                "rf": rf_b64,
                "sal": sal_b64,
                "real_idx": i,
            },
        )
    rendered_layers.append(out_nodes)

    col_width = width / len(rendered_layers)
    coords = []
    for c_idx, layer in enumerate(rendered_layers):
        x = col_width * c_idx + (col_width / 2)
        row_height = height / max(1, len(layer))
        layer_coords = []
        for r_idx, node in enumerate(layer):
            y = (
                (height / 2)
                - ((len(layer) * min(row_height, 40)) / 2)
                + (r_idx * min(row_height, 40))
                + 20
            )
            layer_coords.append(
                (
                    x,
                    y,
                    node["val"],
                    node["id"],
                    node["rf"],
                    node["sal"],
                    row_height,
                    node["real_idx"],
                ),
            )
        coords.append(layer_coords)

    svg_lines = []
    svg_nodes = []
    svg_tooltips = []

    # Connection logic
    for i in range(len(coords) - 1):
        weights = None
        max_abs_w = 1.0
        if model:
            weights = model.layers[i + 1].get_weights()[0]
            max_abs_w = np.max(np.abs(weights)) + 1e-10

        layer_src_vals = [node[2] for node in coords[i] if node[3] != "dots"]
        layer_dst_vals = [node[2] for node in coords[i + 1] if node[3] != "dots"]

        max_src = max(layer_src_vals) if layer_src_vals and i > 0 else 1.0
        max_dst = max(layer_dst_vals) if layer_dst_vals else 1.0

        if max_src < 1e-6:
            max_src = 1.0
        if max_dst < 1e-6:
            max_dst = 1.0

        for node_a in coords[i]:
            if node_a[3] == "dots":
                continue
            norm_src = float(np.clip(node_a[2] / max_src, 0, 1))

            for node_b in coords[i + 1]:
                if node_b[3] == "dots":
                    continue
                norm_dst = float(np.clip(node_b[2] / max_dst, 0, 1))
                min_opacity = 0.15
                opacity = (
                    min(
                        0.9,
                        min_opacity + norm_dst * 0.4 + (norm_src * norm_dst) * 0.35,
                    )
                    if activations
                    else 0.1
                )
                min_width = 0.8
                stroke_w = min_width
                color = "#475569"

                if weights is not None:
                    idx_src = node_a[7]
                    idx_dst = node_b[7]
                    w_val = weights[idx_src, idx_dst]
                    stroke_w = min_width + 4.0 * (abs(w_val) / max_abs_w)
                    color = "#3b82f6" if activations else "#64748b"

                svg_lines.append(
                    f'<line x1="{node_a[0]}" y1="{node_a[1]}" x2="{node_b[0]}" y2="{node_b[1]}" stroke="{color}" stroke-width="{stroke_w}" opacity="{opacity}" />',
                )

    for c_idx, layer in enumerate(coords):
        is_output = c_idx == len(coords) - 1
        layer_vals = [node[2] for node in layer if node[3] != "dots"]
        layer_max = max(layer_vals) if layer_vals and max(layer_vals) > 0 else 1.0

        for node in layer:
            x, y, val, nid, rf_b64, sal_b64, row_height, _ = node
            if nid == "dots":
                svg_nodes.append(
                    f'<text x="{x}" y="{y}" fill="#64748b" font-size="20" text-anchor="middle">⋮</text>',
                )
                continue

            intensity = (
                float(np.clip(val, 0, 1))
                if is_output
                else float(np.clip(val / layer_max, 0, 1))
            )
            if not activations:
                intensity = 0.0

            base_color = "16, 185, 129" if is_output else "59, 130, 246"
            bg_color = f"rgba({base_color}, {0.2 + 0.8 * intensity})"
            r = min(12 if c_idx > 0 else 6, max(3, (row_height / 2) - 1.5))

            svg_nodes.append(
                f'<circle cx="{x}" cy="{y}" r="{r}" fill="{bg_color}" stroke="#1e293b" stroke-width="2" />',
            )

            if is_output:
                label = nid.split("_")[1]
                svg_nodes.append(
                    f'<text x="{x + 20}" y="{y + 4}" fill="#e2e8f0" font-family="monospace" font-size="12">{label}</text>',
                )

            if rf_b64:
                # Dynamic positioning to avoid clipping
                width_bg = 188 if saliencies else 94

                # Horizontal placement
                dx = 15
                if x > width - width_bg - 30:
                    dx = -width_bg - 15

                # Vertical placement: dy is the image top-left offset
                dy = -42
                if y < 110:  # Near top edge, shift tooltip down
                    dy = 40
                elif y > height - 110:  # Near bottom edge, shift tooltip higher
                    dy = -75

                act_text = f"Act: {val:.3f}" if activations is not None else "Act: --"

                # Internal layout relative to (dx, dy)
                rect_x = dx - 5
                rect_y = dy - 32
                text_header_x = dx + 42
                text_act_x = dx + (width_bg / 2) - 5
                text_y_header = dy - 18
                text_y_act = dy - 5

                sal_svg = ""
                if saliencies:
                    img_data = (
                        sal_b64
                        or "iVBORw0KGgoAAAANSUhEUgAAAFQAAABUCAYAAAAcaxDBAAAAAXNSR0IArs4c6QAAAExJREFUeF7twTEBAAAAwqD1T20ND6AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOBvAHAQAAH8S88fAAAAAElFTkSuQmCC"
                    )
                    sal_svg = f'<text x="{dx + 42 + 94}" y="{text_y_header}" class="tooltip-text" text-anchor="middle">SALIENCY</text><image href="data:image/png;base64,{img_data}" x="{dx + 94}" y="{dy}" width="84" height="84" />'

                svg_tooltips.append(
                    f'<g class="tooltip-container" transform="translate({x}, {y})">'
                    f'<circle cx="0" cy="0" r="{max(r + 5, 15)}" fill="transparent" />'
                    f'<g class="tooltip">'
                    f'<rect x="{rect_x}" y="{rect_y}" width="{width_bg}" height="122" class="tooltip-bg"/>'
                    f'<text x="{text_act_x}" y="{text_y_act}" class="tooltip-text" text-anchor="middle" style="fill:#38bdf8;">{act_text}</text>'
                    f'<text x="{text_header_x}" y="{text_y_header}" class="tooltip-text" text-anchor="middle">PATTERN</text>'
                    f'<image href="data:image/png;base64,{rf_b64}" x="{dx}" y="{dy}" width="84" height="84" />'
                    f"{sal_svg}"
                    f"</g></g>",
                )

    svg = (
        f'<svg width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f"<style>.tooltip-container {{ cursor: pointer; }} .tooltip {{ visibility: hidden; opacity: 0; transition: opacity 0.2s; pointer-events: none; }} .tooltip-container:hover .tooltip {{ visibility: visible; opacity: 1; }} .tooltip-bg {{ fill: #0f172a; stroke: #334155; stroke-width: 1; rx: 4; }} .tooltip-text {{ fill: #94a3b8; font-family: monospace; font-size: 10px; font-weight: bold; }}</style>"
        f"<g>{''.join(svg_lines)}"
        f"{''.join(svg_nodes)}"
        f"{''.join(svg_tooltips)}</g>"
        f"</svg>"
    )
    return svg


# --- Session State Management ---
if "model" not in st.session_state:
    st.session_state.model = None
if "is_training" not in st.session_state:
    st.session_state.is_training = False
if "current_epoch" not in st.session_state:
    st.session_state.current_epoch = 0
if "history" not in st.session_state:
    st.session_state.history = {"loss": [], "acc": [], "val_acc": []}
if "random_test_idx" not in st.session_state:
    st.session_state.random_test_idx = None

# --- Sidebar UI ---
with st.sidebar:
    st.title("⚙️ Hyperparameters")
    st.markdown("### Architecture")
    if "layers_config" not in st.session_state:
        st.session_state.layers_config = [16, 16]

    num_hidden = st.number_input(
        "Hidden Layers",
        0,
        5,
        len(st.session_state.layers_config),
    )
    new_layers_config = []
    for i in range(num_hidden):
        prev_val = (
            st.session_state.layers_config[i]
            if i < len(st.session_state.layers_config)
            else 16
        )
        units = st.slider(f"Neurons in Layer {i + 1}", 1, 32, prev_val)
        new_layers_config.append(units)

    st.markdown("### Training Settings")
    train_size = st.selectbox(
        "Training Data Size",
        [4000, 10000, 30000, 60000],
        index=3,
    )
    lr = st.selectbox("Learning Rate", [0.0001, 0.001, 0.01, 0.1], index=1)
    activation = st.selectbox("Activation", ["relu", "sigmoid", "tanh"], index=0)
    epochs = st.number_input("Max Epochs", min_value=1, value=50)
    update_freq = st.selectbox(
        "Update UI every N Epochs",
        [1, 2, 5, 10, 20, 50, 100],
        index=0,
    )

    current_config = {
        "layers": new_layers_config,
        "activation": activation,
        "lr": lr,
        "train_size": train_size,
    }
    if (
        "last_config" not in st.session_state
        or st.session_state.last_config != current_config
    ):
        st.session_state.last_config = current_config
        st.session_state.layers_config = new_layers_config
        st.session_state.model = build_compiled_model(new_layers_config, activation, lr)
        st.session_state.current_epoch = 0
        st.session_state.history = {"loss": [], "acc": [], "val_acc": []}
        st.session_state.is_training = False

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "Train / Resume" if not st.session_state.is_training else "⏸ Pause",
        ):
            if st.session_state.current_epoch < epochs:
                st.session_state.is_training = not st.session_state.is_training
                st.rerun()
    with col2:
        if st.button("Reset Model"):
            st.session_state.model = build_compiled_model(
                st.session_state.layers_config,
                activation,
                lr,
            )
            st.session_state.current_epoch = 0
            st.session_state.history = {"loss": [], "acc": [], "val_acc": []}
            st.session_state.is_training = False
            st.rerun()

# --- Main App Layout ---
st.title("🧠 Neural Network Playground")
st.markdown(
    f"**Epoch:** `{st.session_state.current_epoch} / {epochs}` | Status: {'🟢 Training...' if st.session_state.is_training else '🟡 Paused / Idle'}",
)

(x_train, y_train), (x_val, y_val), x_test_full = load_mnist_data(train_size)
col_network, col_testing = st.columns([2, 1])

test_input, activations, saliencies = None, None, None

with col_testing:
    st.markdown("### ✍️ Live Testing")
    input_method = st.radio(
        "Input Method:",
        ["Draw Manually", "Random Example"],
        horizontal=True,
    )

    if input_method == "Draw Manually":
        st.caption("Draw a digit (0-9). Entire box is treated as the 20x20 core area.")

        draw_col, input_col = st.columns([1.1, 0.9])

        with draw_col:
            canvas_result = st_canvas(
                fill_color="black",
                stroke_width=15,
                stroke_color="white",
                background_color="black",
                update_streamlit=True,
                height=140,
                width=140,
                drawing_mode="freedraw",
                key="canvas",
            )

        if canvas_result.image_data is not None:
            img_gray = np.mean(canvas_result.image_data[:, :, :3], axis=2).astype(
                np.uint8,
            )
            if np.sum(img_gray) > 0:
                resized = np.clip(
                    np.array(
                        Image.fromarray(img_gray).resize(
                            (20, 20),
                            Image.Resampling.LANCZOS,
                        ),
                    ).astype("float32")
                    / 255.0,
                    0,
                    1,
                )
                canvas_28 = np.zeros((28, 28), dtype=np.float32)
                canvas_28[4:24, 4:24] = resized
                test_input = canvas_28.reshape(1, 28, 28, 1)

                with input_col:
                    st.markdown(
                        "<div style='font-size: 0.8rem; color: #94a3b8;'>Model Input (28x28):</div>",
                        unsafe_allow_html=True,
                    )
                    st.image(canvas_28, width=100, clamp=True)
    else:
        if (
            st.button("🎲 Load Random Image")
            or st.session_state.random_test_idx is None
        ):
            st.session_state.random_test_idx = np.random.randint(0, len(x_test_full))
        if st.session_state.random_test_idx is not None:
            idx = st.session_state.random_test_idx
            st.image(x_test_full[idx].reshape(28, 28), width=140, clamp=True)
            test_input = x_test_full[idx].reshape(1, 28, 28, 1)

    if test_input is not None and st.session_state.model:
        extractor = tf.keras.Model(
            inputs=st.session_state.model.inputs,
            outputs=[l.output for l in st.session_state.model.layers],
        )
        activations_list = extractor.predict(test_input, verbose=0)
        activations = [test_input.flatten()] + [a.flatten() for a in activations_list]

        if not st.session_state.is_training:
            saliencies = compute_all_saliencies(st.session_state.model, test_input)

        st.markdown(
            "<div style='margin-top: 10px; margin-bottom: 5px;'><b>Softmax Confidence</b></div>",
            unsafe_allow_html=True,
        )
        preds = activations[-1]

        pred_df = pd.DataFrame(
            {
                "Digit": [str(i) for i in range(10)],
                "Confidence": preds,
                "Color": [
                    "#10b981" if i == np.argmax(preds) else "#64748b" for i in range(10)
                ],
            },
        )

        y_max = max(preds) if max(preds) > 0.01 else 0.1

        conf_chart = (
            alt.Chart(pred_df)
            .mark_bar()
            .encode(
                x=alt.X("Digit:N", axis=alt.Axis(labelAngle=0, title=None)),
                y=alt.Y(
                    "Confidence:Q",
                    scale=alt.Scale(domain=[0, y_max]),
                    axis=alt.Axis(title=None, format=".0%"),
                ),
                color=alt.Color("Color:N", scale=None),
                tooltip=["Digit", alt.Tooltip("Confidence", format=".2%")],
            )
            .properties(height=120)
        )

        st.altair_chart(conf_chart, width="stretch")

    elif input_method == "Draw Manually":
        st.info("Draw something to test.")

with col_network:
    st.markdown("### ⚡ Network Architecture")
    st.caption("Hover over neurons to inspect Receptive Fields & Saliency Maps!")
    svg_string = draw_network_svg(
        st.session_state.layers_config,
        activations,
        (None if st.session_state.is_training else st.session_state.model),
        saliencies,
    )
    st.markdown(svg_string, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📊 Metrics Tracking")
if len(st.session_state.history["loss"]) > 0:
    c1, c2, c3 = st.columns(3)
    epochs_arr = list(range(1, len(st.session_state.history["loss"]) + 1))

    with c1:
        st.markdown(
            f"<div class='metric-container' style='margin-bottom: 0;'><div>Loss</div><div class='metric-value' style='color:#60a5fa'>{st.session_state.history['loss'][-1]:.4f}</div></div>",
            unsafe_allow_html=True,
        )
        loss_chart = (
            alt.Chart(
                pd.DataFrame(
                    {"Epoch": epochs_arr, "Loss": st.session_state.history["loss"]},
                ),
            )
            .mark_line(color="#60a5fa")
            .encode(
                x="Epoch",
                y=alt.Y("Loss", scale=alt.Scale(type="log")),
            )
            .properties(height=150)
        )
        st.altair_chart(loss_chart, width="stretch")

    with c2:
        tacc = [val * 100 for val in st.session_state.history["acc"]]
        st.markdown(
            f"<div class='metric-container' style='margin-bottom: 0;'><div>Train Acc</div><div class='metric-value' style='color:#34d399'>{tacc[-1]:.1f}%</div></div>",
            unsafe_allow_html=True,
        )
        tacc_chart = (
            alt.Chart(pd.DataFrame({"Epoch": epochs_arr, "Train Acc": tacc}))
            .mark_line(color="#34d399")
            .encode(
                x="Epoch",
                y=alt.Y("Train Acc", scale=alt.Scale(zero=False)),
            )
            .properties(height=150)
        )
        st.altair_chart(tacc_chart, width="stretch")

    with c3:
        vacc = [val * 100 for val in st.session_state.history["val_acc"]]
        st.markdown(
            f"<div class='metric-container' style='margin-bottom: 0;'><div>Val Acc</div><div class='metric-value' style='color:#a78bfa'>{vacc[-1]:.1f}%</div></div>",
            unsafe_allow_html=True,
        )
        vacc_chart = (
            alt.Chart(pd.DataFrame({"Epoch": epochs_arr, "Val Acc": vacc}))
            .mark_line(color="#a78bfa")
            .encode(
                x="Epoch",
                y=alt.Y("Val Acc", scale=alt.Scale(zero=False)),
            )
            .properties(height=150)
        )
        st.altair_chart(vacc_chart, width="stretch")
else:
    st.caption("Hit 'Train' to begin tracking metrics.")

if st.session_state.is_training:
    if st.session_state.current_epoch < epochs:
        run_n = min(update_freq, epochs - st.session_state.current_epoch)
        target = st.session_state.current_epoch + run_n
        hist = st.session_state.model.fit(
            x_train,
            y_train,
            epochs=target,
            initial_epoch=st.session_state.current_epoch,
            validation_data=(x_val, y_val),
            batch_size=128,
            verbose=0,
        )
        st.session_state.current_epoch = target
        st.session_state.history["loss"].extend(hist.history["loss"])
        st.session_state.history["acc"].extend(hist.history["accuracy"])
        st.session_state.history["val_acc"].extend(hist.history["val_accuracy"])
        time.sleep(0.05)
        st.rerun()
    else:
        st.session_state.is_training = False
        st.rerun()
