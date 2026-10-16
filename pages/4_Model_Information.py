import streamlit as st
import pandas as pd
import tensorflow as tf
from utils.model_loader import load_trained_model, load_class_mapping, load_metrics
from utils.class_mapping import get_class_dataframe

st.set_page_config(
    page_title="Specifications • NeuroTraffic",
    page_icon="ℹ️",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    code, pre { font-family: 'JetBrains Mono', monospace !important; }

    .block-container {
        padding-top: 1.75rem !important;
        padding-bottom: 2rem !important;
        max-width: 1100px !important;
    }
    .page-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #FFFFFF 40%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .page-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 1.75rem;
    }
    .metric-chip {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1rem 1.15rem;
        backdrop-filter: blur(10px);
    }
    .metric-chip .chip-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin-bottom: 2px;
    }
    .metric-chip .chip-val {
        font-size: 1.45rem;
        font-weight: 800;
        color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">Model Specifications</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Deep learning architecture, layer hyperparameters, and GTSRB 58-class mapping index.</div>', unsafe_allow_html=True)

model, model_err = load_trained_model()
mapping, map_err = load_class_mapping()
metrics, metrics_err = load_metrics()

num_classes = mapping.get("num_classes", 58) if mapping else 58
img_dim = mapping.get("img_size", 64) if mapping else 64
total_params = metrics.get("total_parameters", 1150330) if metrics else 1150330

# 1. Four Header Metric Chips
s1, s2, s3, s4 = st.columns(4)

with s1:
    st.markdown(f'<div class="metric-chip"><div class="chip-label">Classes</div><div class="chip-val" style="color: #38BDF8;">{num_classes} Signs</div></div>', unsafe_allow_html=True)

with s2:
    st.markdown(f'<div class="metric-chip"><div class="chip-label">Input Shape</div><div class="chip-val" style="color: #60A5FA;">{img_dim}×{img_dim} RGB</div></div>', unsafe_allow_html=True)

with s3:
    st.markdown(f'<div class="metric-chip"><div class="chip-label">Parameters</div><div class="chip-val" style="color: #34D399;">{total_params:,}</div></div>', unsafe_allow_html=True)

with s4:
    st.markdown(f'<div class="metric-chip"><div class="chip-label">Framework</div><div class="chip-val" style="color: #F472B6;">TF {tf.__version__}</div></div>', unsafe_allow_html=True)

st.write("")
st.write("")

# 2. Side-by-Side Summary
col_specs, col_arch = st.columns([1, 1], gap="medium")

with col_specs:
    with st.container(border=True):
        st.markdown("<div style='font-size: 1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;'>Training & Preprocessing Parameters</div>", unsafe_allow_html=True)
        spec_list = [
            {"Parameter": "Dataset Source", "Value": "German / Kaggle GTSRB 58-Class Set"},
            {"Parameter": "Target Size", "Value": f"{img_dim} × {img_dim} pixels, 3 Channels"},
            {"Parameter": "Normalization", "Value": "Pixel / 255.0 to [0.0, 1.0] (float32)"},
            {"Parameter": "Resampling Method", "Value": "Bilinear Resampling (PIL)"},
            {"Parameter": "Batch Size", "Value": "32 mini-batch tensor"},
            {"Parameter": "Loss Function", "Value": "Sparse Categorical Crossentropy"},
            {"Parameter": "Optimizer", "Value": "Adam (Learning rate = 0.001)"},
        ]
        st.dataframe(pd.DataFrame(spec_list), use_container_width=True, hide_index=True)

with col_arch:
    with st.container(border=True):
        st.markdown("<div style='font-size: 1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;'>Layer Block Breakdown</div>", unsafe_allow_html=True)
        summary_layers = [
            {"Block": "Input Tensor", "Shape": f"(None, {img_dim}, {img_dim}, 3)", "Params": "0"},
            {"Block": "Conv Block 1 (32)", "Shape": "(None, 32, 32, 32)", "Params": "1,024"},
            {"Block": "Conv Block 2 (64)", "Shape": "(None, 16, 16, 64)", "Params": "18,752"},
            {"Block": "Conv Block 3 (128)", "Shape": "(None, 8, 8, 128)", "Params": "74,368"},
            {"Block": "Flatten", "Shape": "(None, 8192)", "Params": "0"},
            {"Block": "Dense (128) + Dropout", "Shape": "(None, 128)", "Params": "1,048,704"},
            {"Block": "Dense (Softmax)", "Shape": f"(None, {num_classes})", "Params": "7,482"},
        ]
        st.dataframe(pd.DataFrame(summary_layers), use_container_width=True, hide_index=True)

st.write("")

# 3. Expandable Detailed Panels
with st.expander("🔍 Full Neural Network Layer Hierarchy"):
    arch_full = [
        {"Layer": "InputLayer", "Type": "Input", "Shape": "(None, 64, 64, 3)", "Details": "3-channel RGB image tensor"},
        {"Layer": "conv2d", "Type": "Conv2D", "Shape": "(None, 64, 64, 32)", "Details": "32 filters, 3x3, ReLU, same padding"},
        {"Layer": "batch_normalization", "Type": "BatchNorm", "Shape": "(None, 64, 64, 32)", "Details": "Momentum 0.9"},
        {"Layer": "max_pooling2d", "Type": "MaxPooling2D", "Shape": "(None, 32, 32, 32)", "Details": "2x2 pool window, stride 2"},
        {"Layer": "conv2d_1", "Type": "Conv2D", "Shape": "(None, 32, 32, 64)", "Details": "64 filters, 3x3, ReLU, same padding"},
        {"Layer": "batch_normalization_1", "Type": "BatchNorm", "Shape": "(None, 32, 32, 64)", "Details": "Momentum 0.9"},
        {"Layer": "max_pooling2d_1", "Type": "MaxPooling2D", "Shape": "(None, 16, 16, 64)", "Details": "2x2 pool window, stride 2"},
        {"Layer": "conv2d_2", "Type": "Conv2D", "Shape": "(None, 16, 16, 128)", "Details": "128 filters, 3x3, ReLU, same padding"},
        {"Layer": "batch_normalization_2", "Type": "BatchNorm", "Shape": "(None, 16, 16, 128)", "Details": "Momentum 0.9"},
        {"Layer": "max_pooling2d_2", "Type": "MaxPooling2D", "Shape": "(None, 8, 8, 128)", "Details": "2x2 pool window, stride 2"},
        {"Layer": "flatten", "Type": "Flatten", "Shape": "(None, 8192)", "Details": "Reshapes 8x8x128 feature tensor to 1D vector"},
        {"Layer": "dense", "Type": "Dense", "Shape": "(None, 128)", "Details": "Fully connected with ReLU activation"},
        {"Layer": "dropout", "Type": "Dropout", "Shape": "(None, 128)", "Details": "Dropout rate 0.4"},
        {"Layer": "dense_1", "Type": "Dense (Output)", "Shape": f"(None, {num_classes})", "Details": "Softmax multi-class output distribution"}
    ]
    st.dataframe(pd.DataFrame(arch_full), use_container_width=True, hide_index=True)

with st.expander(f"📋 Complete 58-Class Traffic Sign Mapping Index"):
    df_classes = get_class_dataframe()
    st.dataframe(df_classes, use_container_width=True, height=400, hide_index=True)
