import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from utils.predictor import predict_traffic_sign
from utils.model_loader import load_trained_model, load_class_mapping

st.set_page_config(
    page_title="Inference • Traffic Sign Recognition",
    page_icon="🔍",
    layout="wide"
)

# Dark Glassmorphism Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    code, pre { font-family: 'JetBrains Mono', monospace !important; }

    .block-container {
        padding-top: 1.75rem !important;
        padding-bottom: 2rem !important;
        max-width: 1140px !important;
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
        margin-bottom: 1.5rem;
    }
    .pred-hero {
        background: linear-gradient(145deg, rgba(30, 58, 138, 0.45), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(59, 130, 246, 0.45);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 28px -6px rgba(37, 99, 235, 0.25);
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-top: 8px;
    }
    .high-conf {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .med-conf {
        background: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .low-conf {
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">Traffic Sign Recognition & Dynamic Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Upload any road sign image to trigger real-time neural inference. The confidence rating, probability breakdown, and dynamic chart update instantly.</div>', unsafe_allow_html=True)

# Session state
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

model, model_err = load_trained_model()
mapping, map_err = load_class_mapping()

if model is None:
    st.error(f"Model Not Ready: {model_err}")
    st.stop()

col_up, col_res = st.columns([1, 1.2], gap="large")

with col_up:
    with st.container(border=True):
        st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;'>1. Upload Sign Image</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Select traffic sign image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed"
        )
        
        # Auto-predict if a new image is selected
        if uploaded_file is not None:
            if st.session_state.last_uploaded_name != uploaded_file.name:
                with st.spinner(f"Extracting features from {uploaded_file.name}..."):
                    ok, res, err = predict_traffic_sign(uploaded_file, top_k=5)
                    if ok:
                        st.session_state.prediction_result = res
                        st.session_state.last_uploaded_name = uploaded_file.name
                    else:
                        st.error(f"Inference error: {err}")
                        st.session_state.prediction_result = None

            try:
                img = Image.open(uploaded_file)
                st.image(img, caption=f"Active File: {uploaded_file.name}", use_column_width=True)
                
                b1, b2 = st.columns([1.5, 1], gap="small")
                with b1:
                    if st.button("⚡ Re-Analyze Sign", type="primary", use_container_width=True):
                        with st.spinner("Recomputing CNN activations..."):
                            ok, res, err = predict_traffic_sign(uploaded_file, top_k=5)
                            if ok:
                                st.session_state.prediction_result = res
                with b2:
                    if st.button("🗑️ Clear Image", use_container_width=True):
                        st.session_state.prediction_result = None
                        st.session_state.last_uploaded_name = None
                        st.rerun()

            except Exception as e:
                st.error(f"Error loading image: {e}")
                st.session_state.prediction_result = None
        else:
            st.info("Upload any traffic sign photo above to instantly update the graph and confidence scores.")

with col_res:
    with st.container(border=True):
        st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;'>2. Real-Time Confidence & Probability Graph</div>", unsafe_allow_html=True)
        
        if st.session_state.prediction_result is not None:
            res = st.session_state.prediction_result
            best = res["best_prediction"]
            conf = best["confidence_pct"]
            
            badge_cls = "high-conf" if conf >= 70 else ("med-conf" if conf >= 40 else "low-conf")
            
            # Dynamic Result Hero
            st.markdown(f"""
            <div class="pred-hero">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #93C5FD; font-weight: 700; letter-spacing: 0.08em;">Primary Classification</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #FFFFFF; margin: 4px 0;">{best['class_name']}</div>
                <div style="font-size: 0.85rem; color: #CBD5E1;">Class ID: <strong>{best['class_id']}</strong> &nbsp;•&nbsp; Output Unit: <strong>{best['class_index']}</strong></div>
                <div class="badge-pill {badge_cls}">Prediction Confidence: {conf:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Dynamic Probability Breakdown
            top_preds = res["top_predictions"]
            
            st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin: 12px 0 8px 0;'>Softmax Probability Breakdown</div>", unsafe_allow_html=True)
            for p in top_preds:
                c_name = p['class_name']
                c_pct = p['confidence_pct']
                c_frac = max(0.0, min(1.0, float(p['confidence'])))
                
                # Highlight top-1 with blue accent, others in clean slate
                label_color = "#60A5FA" if p['rank'] == 1 else "#F8FAFC"
                st.markdown(f"<div style='display: flex; justify-content: space-between; font-size: 0.88rem; font-weight: 600; margin-bottom: 3px;'><span style='color: {label_color};'>#{p['rank']} {c_name} (ID {p['class_id']})</span><span style='color: #38BDF8;'>{c_pct:.1f}%</span></div>", unsafe_allow_html=True)
                st.progress(c_frac)
            
            # Table Breakdown
            df_display = pd.DataFrame([
                {
                    "Rank": f"#{p['rank']}",
                    "Traffic Sign Label": p["class_name"],
                    "Class ID": p["class_id"],
                    "Model Confidence": f"{p['confidence_pct']:.2f}%"
                }
                for p in top_preds[:3]
            ])
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
        else:
            st.markdown("""
            <div style="padding: 4rem 1rem; text-align: center; color: #64748B;">
                <div style="font-size: 2.8rem; margin-bottom: 0.75rem;">📊</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #94A3B8;">Awaiting Image Upload</div>
                <div style="font-size: 0.85rem; margin-top: 4px; max-width: 380px; margin-left: auto; margin-right: auto;">
                    Select or drag any traffic sign image on the left. The neural network will instantly compute activations and render the live probability graph.
                </div>
            </div>
            """, unsafe_allow_html=True)
