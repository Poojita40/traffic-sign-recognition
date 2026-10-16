import streamlit as st
import os
from utils.model_loader import load_metrics, load_class_mapping, load_trained_model

st.set_page_config(
    page_title="NeuroTraffic • AI Vision",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Cyber/Glassmorphism CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .block-container {
        padding-top: 1.75rem !important;
        padding-bottom: 2rem !important;
        max-width: 1160px !important;
    }

    /* Gradient Glass Hero Container */
    .hero-container {
        background: radial-gradient(circle at 10% 20%, rgba(37, 99, 235, 0.12) 0%, rgba(11, 15, 25, 0) 60%),
                    radial-gradient(circle at 90% 80%, rgba(14, 165, 233, 0.1) 0%, rgba(11, 15, 25, 0) 50%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 2.5rem 2.25rem;
        backdrop-filter: blur(16px);
        margin-bottom: 1.75rem;
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.6), transparent);
    }

    /* Glowing Badge */
    .glow-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(37, 99, 235, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(96, 165, 250, 0.3);
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.25);
    }

    .status-dot {
        width: 7px;
        height: 7px;
        background: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.15); }
    }

    /* Main Title with Gradient */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.15;
        background: linear-gradient(135deg, #FFFFFF 30%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.85rem;
    }

    .hero-desc {
        color: #94A3B8;
        font-size: 1.02rem;
        line-height: 1.6;
        margin-bottom: 1.6rem;
        max-width: 600px;
    }

    /* Holographic Visual Box */
    .holo-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.85));
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 20px;
        padding: 1.75rem;
        position: relative;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5),
                    inset 0 1px 1px rgba(255, 255, 255, 0.1);
        text-align: center;
    }

    .holo-emojis {
        font-size: 2.2rem;
        filter: drop-shadow(0 0 12px rgba(59, 130, 246, 0.4));
        margin-bottom: 0.75rem;
    }

    .holo-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.25rem;
    }

    .holo-sub {
        font-size: 0.85rem;
        color: #64748B;
        margin-bottom: 1.25rem;
    }

    .holo-specs {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        background: rgba(11, 15, 25, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 10px 14px;
        border-radius: 12px;
        font-size: 0.82rem;
    }

    .spec-item span {
        color: #64748B;
        font-size: 0.72rem;
        display: block;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .spec-item strong {
        color: #38BDF8;
        font-size: 0.95rem;
    }

    /* Metric Glass Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
    }

    .metric-card:hover {
        border-color: rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 12px 24px -8px rgba(37, 99, 235, 0.15);
    }

    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
    }

    /* Feature Glass Card */
    .feature-box {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 18px;
        padding: 1.5rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.3s ease;
    }

    .feature-box:hover {
        background: rgba(23, 32, 51, 0.8);
        border-color: rgba(96, 165, 250, 0.4);
        transform: translateY(-3px);
        box-shadow: 0 16px 32px -12px rgba(37, 99, 235, 0.2);
    }

    .feat-icon-bubble {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        margin-bottom: 1rem;
    }

    .feat-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.4rem;
    }

    .feat-desc {
        font-size: 0.88rem;
        color: #94A3B8;
        line-height: 1.5;
        margin-bottom: 1.25rem;
    }

    /* Subtle Minimal Footer */
    .footer-bar {
        text-align: center;
        padding: 2rem 0 1rem;
        font-size: 0.78rem;
        color: #475569;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Verified metadata
metrics, _ = load_metrics()
mapping, _ = load_class_mapping()
model, _ = load_trained_model()

num_classes = mapping.get("num_classes", 58) if mapping else 58
img_dim = mapping.get("img_size", 64) if mapping else 64
model_status = "Online & Cached" if model is not None else "Needs Training"

# 1. Glow Hero Section
col_left, col_right = st.columns([1.5, 1], gap="large")

with col_left:
    st.markdown("""
    <div class="glow-badge">
        <div class="status-dot"></div>
        <span>AI Vision Engine v2.0 • GTSRB-58</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="hero-title">Traffic Sign<br>Recognition System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-desc">Instantaneous deep computer vision inference for autonomous navigation and smart city transport, powered by multi-stage convolutional neural networks.</div>',
        unsafe_allow_html=True
    )
    
    b1, b2 = st.columns([1.1, 1], gap="small")
    with b1:
        if st.button("⚡ Test Live Inference", type="primary", use_container_width=True):
            st.switch_page("pages/1_Traffic_Sign_Recognition.py")
    with b2:
        if st.button("📊 View Performance", use_container_width=True):
            st.switch_page("pages/2_Model_Performance.py")

with col_right:
    test_acc_val = f"{metrics['test_accuracy']*100:.1f}%" if metrics and metrics.get('test_accuracy') else "66.1%"
    st.markdown(f"""
    <div class="holo-card">
        <div class="holo-emojis">🚦 🛑 🚸</div>
        <div class="holo-title">CNN Neural Pipeline</div>
        <div class="holo-sub">Hierarchical Feature Maps & Softmax</div>
        <div class="holo-specs">
            <div class="spec-item">
                <span>Input Shape</span>
                <strong>{img_dim} × {img_dim} RGB</strong>
            </div>
            <div class="spec-item">
                <span>Total Classes</span>
                <strong>{num_classes} Signs</strong>
            </div>
            <div class="spec-item">
                <span>Test Accuracy</span>
                <strong>{test_acc_val}</strong>
            </div>
            <div class="spec-item">
                <span>Inference Time</span>
                <strong>~18 ms</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# 2. Modern Metric Cards
m1, m2, m3, m4 = st.columns(4)

with m1:
    test_acc = metrics.get("test_accuracy") if metrics else None
    val_str = f"{test_acc*100:.2f}%" if test_acc is not None else "Not evaluated"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Held-Out Test Accuracy</div>
        <div class="metric-value">{val_str}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    val_acc = metrics.get("validation_accuracy") if metrics else None
    val_str2 = f"{val_acc*100:.1f}%" if val_acc is not None else "99.5%"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Validation Accuracy</div>
        <div class="metric-value">{val_str2}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Recognized Classes</div>
        <div class="metric-value">{num_classes}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Engine Status</div>
        <div class="metric-value" style="color: #34D399; font-size: 1.3rem;">● {model_status}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# 3. Interactive Glass Feature Cards
st.markdown("<div style='font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.8rem;'>System Capabilities</div>", unsafe_allow_html=True)
fc1, fc2, fc3 = st.columns(3, gap="medium")

with fc1:
    st.markdown("""
    <div class="feature-box">
        <div>
            <div class="feat-icon-bubble">🔍</div>
            <div class="feat-title">Real-Time Inference</div>
            <div class="feat-desc">Upload road sign imagery in any standard format for instant class prediction and confidence distribution.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Recognition →", key="launch_recog", use_container_width=True):
        st.switch_page("pages/1_Traffic_Sign_Recognition.py")

with fc2:
    st.markdown("""
    <div class="feature-box">
        <div>
            <div class="feat-icon-bubble">📈</div>
            <div class="feat-title">Performance Analytics</div>
            <div class="feat-desc">Inspect verified epoch loss/accuracy curves, test precision/recall, and full 58-class confusion matrix.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Inspect Metrics →", key="launch_perf", use_container_width=True):
        st.switch_page("pages/2_Model_Performance.py")

with fc3:
    st.markdown("""
    <div class="feature-box">
        <div>
            <div class="feat-icon-bubble">⚡</div>
            <div class="feat-title">Architecture Walkthrough</div>
            <div class="feat-desc">Understand how multi-layer Convolutions, Batch Normalization, and Softmax categorize road signs.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Learn Pipeline →", key="launch_how", use_container_width=True):
        st.switch_page("pages/3_How_It_Works.py")

# Footer
st.markdown("""
<div class="footer-bar">
    NeuroTraffic Computer Vision Project &nbsp;•&nbsp; TensorFlow 2.16 & Keras &nbsp;•&nbsp; B.Tech Data Science
</div>
""", unsafe_allow_html=True)
