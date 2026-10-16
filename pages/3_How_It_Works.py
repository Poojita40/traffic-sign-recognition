import streamlit as st

st.set_page_config(
    page_title="Architecture • NeuroTraffic",
    page_icon="💡",
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
    .flow-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(30, 58, 138, 0.35);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 6px 14px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 4px;
        margin-bottom: 6px;
    }
    .flow-arrow {
        color: #475569;
        font-weight: bold;
        margin: 0 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">How It Works</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Interactive conceptual tour of the Convolutional Neural Network feature hierarchy and classification pipeline.</div>', unsafe_allow_html=True)

# 1. Glowing Sequence Ribbon
with st.container(border=True):
    st.markdown("<div style='font-size: 0.82rem; text-transform: uppercase; color: #64748B; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 0.6rem;'>Pipeline Data Flow</div>", unsafe_allow_html=True)
    st.markdown("""
    <div>
        <span class="flow-badge">1. Input Image</span>
        <span class="flow-arrow">➔</span>
        <span class="flow-badge">2. Bilinear 64×64</span>
        <span class="flow-arrow">➔</span>
        <span class="flow-badge">3. Conv2D + BatchNorm</span>
        <span class="flow-arrow">➔</span>
        <span class="flow-badge">4. MaxPooling (2×2)</span>
        <span class="flow-arrow">➔</span>
        <span class="flow-badge">5. Flatten & Dense</span>
        <span class="flow-arrow">➔</span>
        <span class="flow-badge">6. Softmax (58 Classes)</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# 2. Balanced 2-Column Explanations
c1, c2 = st.columns(2, gap="medium")

with c1:
    with st.container(border=True):
        st.markdown("<div style='font-size: 1rem; font-weight: 700; color: #60A5FA; margin-bottom: 0.2rem;'>Stage 1: Preprocessing & Normalization</div>", unsafe_allow_html=True)
        st.caption("Input images are converted to 3-channel RGB, resized to standard 64×64 pixels, and pixel values [0, 255] are rescaled by 1/255.0 to float32 values [0.0, 1.0].")

    with st.container(border=True):
        st.markdown("<div style='font-size: 1rem; font-weight: 700; color: #60A5FA; margin-bottom: 0.2rem;'>Stage 2: Multi-Scale Convolutions</div>", unsafe_allow_html=True)
        st.caption("Three Conv2D blocks (32, 64, 128 filters of 3×3 kernels) extract edge gradients, geometric corners, and higher-order road sign symbols.")

    with st.container(border=True):
        st.markdown("<div style='font-size: 1rem; font-weight: 700; color: #60A5FA; margin-bottom: 0.2rem;'>Stage 3: Normalization & Max-Pooling</div>", unsafe_allow_html=True)
        st.caption("Batch Normalization (momentum 0.9) accelerates convergence. 2×2 MaxPooling downsamples spatial dimensions (64×64 → 8×8), granting spatial invariance.")

with c2:
    with st.container(border=True):
        st.markdown("<div style='font-size: 1rem; font-weight: 700; color: #60A5FA; margin-bottom: 0.2rem;'>Stage 4: Feature Flattening & Dense Synthesis</div>", unsafe_allow_html=True)
        st.caption("The 8×8×128 feature map is flattened into an 8,192-dimensional vector and routed into a 128-unit Dense layer with Dropout (0.4) to eliminate co-adaptation.")

    with st.container(border=True):
        st.markdown("<div style='font-size: 1rem; font-weight: 700; color: #60A5FA; margin-bottom: 0.2rem;'>Stage 5: Softmax Probability Distribution</div>", unsafe_allow_html=True)
        st.caption("A 58-unit output Dense layer computes raw logits, converted by the Softmax function into calibrated class probabilities summing to 1.0 (100%):")
        st.latex(r"P(\text{Class } i) = \frac{e^{z_i}}{\sum_{j=1}^{58} e^{z_j}}")

    with st.container(border=True):
        st.markdown("<div style='font-size: 1rem; font-weight: 700; color: #60A5FA; margin-bottom: 0.2rem;'>Stage 6: Class Index Decoupling</div>", unsafe_allow_html=True)
        st.caption("The winning index is translated via `class_mapping.json` into its official GTSRB Class ID and sign name (e.g., 'Speed limit (60km/h)').")
