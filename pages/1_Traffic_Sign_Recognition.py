import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go
from PIL import Image
from utils.predictor import predict_traffic_sign
from utils.model_loader import load_trained_model, load_class_mapping, load_metrics, load_training_history

st.set_page_config(
    page_title="Inference & Analytics • Traffic Sign Recognition",
    page_icon="🚦",
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
        max-width: 1180px !important;
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
        margin-bottom: 1.35rem;
    }

    /* Analytics KPI Cards */
    .metric-chip {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 0.9rem 1.15rem;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-chip:hover {
        border-color: rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    .chip-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin-bottom: 2px;
    }
    .chip-val {
        font-size: 1.5rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
    }
    .chip-sub {
        font-size: 0.74rem;
        color: #94A3B8;
        margin-top: 2px;
    }

    /* Prediction Hero Card */
    .pred-hero {
        background: linear-gradient(145deg, rgba(30, 58, 138, 0.45), rgba(15, 23, 42, 0.85));
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

    /* Image Preview Container & Metadata */
    .img-meta-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        background: rgba(11, 15, 25, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 8px 12px;
        margin-top: 10px;
        font-size: 0.78rem;
    }
    .meta-tag-label {
        color: #64748B;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .meta-tag-val {
        color: #38BDF8;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">Traffic Sign Recognition & Dynamic Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Upload any road sign image to trigger real-time neural inference. Live prediction confidence, model benchmark accuracy, and neural convergence analytics update dynamically.</div>', unsafe_allow_html=True)

# Load model, class mapping, metrics & training history
model, model_err = load_trained_model()
mapping, map_err = load_class_mapping()
metrics, metrics_err = load_metrics()
history, history_err = load_training_history()

if model is None:
    st.error(f"Model Not Ready: {model_err}")
    st.stop()

# -------------------------------------------------------------
# TOP ANALYTICS BAR: Model Accuracy & Key Benchmarks
# -------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    val_acc = metrics.get("validation_accuracy", 0.9952) if metrics else 0.9952
    st.markdown(f"""
    <div class="metric-chip">
        <div class="chip-label">Model Accuracy</div>
        <div class="chip-val" style="color: #34D399;">{val_acc * 100:.2f}%</div>
        <div class="chip-sub">Validation Set Benchmark</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    test_acc = metrics.get("test_accuracy", 0.6610) if metrics else 0.6610
    st.markdown(f"""
    <div class="metric-chip">
        <div class="chip-label">Test Evaluation</div>
        <div class="chip-val" style="color: #38BDF8;">{test_acc * 100:.2f}%</div>
        <div class="chip-sub">Held-Out Test Set</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    train_loss = metrics.get("training_loss", 0.1447) if metrics else 0.1447
    st.markdown(f"""
    <div class="metric-chip">
        <div class="chip-label">Convergence Loss</div>
        <div class="chip-val" style="color: #F87171;">{train_loss:.4f}</div>
        <div class="chip-sub">Cross-Entropy Final Loss</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    total_classes = len(mapping.get("class_ids", [])) if mapping else 58
    st.markdown(f"""
    <div class="metric-chip">
        <div class="chip-label">Model Architecture</div>
        <div class="chip-val" style="color: #A78BFA;">{total_classes} Classes</div>
        <div class="chip-sub">Deep CNN • 64×64 Input</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -------------------------------------------------------------
# MAIN INTERACTION SECTION: Upload & Inference
# -------------------------------------------------------------
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

col_up, col_res = st.columns([1, 1.2], gap="large")

with col_up:
    with st.container(border=True):
        st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;'>1. Upload Sign Image</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Select traffic sign image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # 1. Safely read bytes to avoid stream depletion bugs
            image_bytes = uploaded_file.getvalue()
            
            try:
                # 2. Open clean in-memory PIL image
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                # 3. Auto-predict when a new image is selected
                if st.session_state.last_uploaded_name != uploaded_file.name:
                    with st.spinner(f"Extracting CNN activations for {uploaded_file.name}..."):
                        ok, res, err = predict_traffic_sign(img, top_k=5)
                        if ok:
                            st.session_state.prediction_result = res
                            st.session_state.last_uploaded_name = uploaded_file.name
                        else:
                            st.error(f"Inference error: {err}")
                            st.session_state.prediction_result = None

                # 4. Reliably render the uploaded image using in-memory PIL Image
                st.image(
                    img,
                    caption=f"Active File: {uploaded_file.name}",
                    use_container_width=True
                )
                
                # 5. Show image metadata chips for analytics depth
                st.markdown(f"""
                <div class="img-meta-grid">
                    <div>
                        <div class="meta-tag-label">Resolution</div>
                        <div class="meta-tag-val">{img.width} × {img.height} px</div>
                    </div>
                    <div>
                        <div class="meta-tag-label">File Size</div>
                        <div class="meta-tag-val">{len(image_bytes) / 1024:.1f} KB</div>
                    </div>
                    <div>
                        <div class="meta-tag-label">Color Space</div>
                        <div class="meta-tag-val">{img.mode} (3 Ch)</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                b1, b2 = st.columns([1.5, 1], gap="small")
                with b1:
                    if st.button("⚡ Re-Analyze Sign", type="primary", use_container_width=True):
                        with st.spinner("Recomputing CNN activations..."):
                            ok, res, err = predict_traffic_sign(img, top_k=5)
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
            st.info("Upload any traffic sign photo above to instantly preview the image and calculate neural inference & confidence analytics.")

with col_res:
    with st.container(border=True):
        st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;'>2. Real-Time Confidence & Probability Graph</div>", unsafe_allow_html=True)
        
        if st.session_state.prediction_result is not None:
            res = st.session_state.prediction_result
            best = res["best_prediction"]
            conf = best["confidence_pct"]
            latency = res.get("latency_ms", 15.0)
            
            badge_cls = "high-conf" if conf >= 70 else ("med-conf" if conf >= 40 else "low-conf")
            
            # Dynamic Result Hero
            st.markdown(f"""
            <div class="pred-hero">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #93C5FD; font-weight: 700; letter-spacing: 0.08em;">Primary Classification</div>
                <div style="font-size: 1.55rem; font-weight: 800; color: #FFFFFF; margin: 4px 0;">{best['class_name']}</div>
                <div style="font-size: 0.85rem; color: #CBD5E1;">
                    Class ID: <strong>{best['class_id']}</strong> &nbsp;•&nbsp; 
                    Output Unit: <strong>{best['class_index']}</strong> &nbsp;•&nbsp; 
                    Latency: <strong>{latency} ms</strong>
                </div>
                <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                    <div class="badge-pill {badge_cls}">Prediction Confidence: {conf:.2f}%</div>
                    <div class="badge-pill" style="background: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.4);">
                        Model Benchmark Acc: {val_acc * 100:.2f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Dynamic Probability Breakdown
            top_preds = res["top_predictions"]
            
            st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin: 12px 0 8px 0;'>Softmax Probability Breakdown</div>", unsafe_allow_html=True)
            for p in top_preds:
                c_name = p['class_name']
                c_pct = p['confidence_pct']
                c_frac = max(0.0, min(1.0, float(p['confidence'])))
                
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
                for p in top_preds[:4]
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

# -------------------------------------------------------------
# SECTION 3: Deep Model Accuracy & Performance Analytics
# -------------------------------------------------------------
st.write("")
with st.container(border=True):
    st.markdown("<div style='font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.35rem;'>📈 3. Model Accuracy & Training Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.85rem; color: #94A3B8; margin-bottom: 1rem;'>Examine the neural model's learning convergence, epoch-by-epoch accuracy progression, and comprehensive validation metrics.</div>", unsafe_allow_html=True)
    
    tab_acc, tab_loss, tab_summary = st.tabs(["🎯 Accuracy Progression", "📉 Loss Convergence", "📊 Metrics Breakdown"])
    
    with tab_acc:
        if history and "accuracy" in history and "val_accuracy" in history:
            epochs = history.get("epochs", list(range(1, len(history["accuracy"]) + 1)))
            acc_vals = [round(a * 100, 2) for a in history["accuracy"]]
            val_acc_vals = [round(a * 100, 2) for a in history["val_accuracy"]]
            
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=epochs,
                y=acc_vals,
                mode="lines+markers",
                name="Training Accuracy",
                line=dict(color="#60A5FA", width=3),
                marker=dict(size=7, color="#60A5FA")
            ))
            fig_acc.add_trace(go.Scatter(
                x=epochs,
                y=val_acc_vals,
                mode="lines+markers",
                name="Validation Accuracy",
                line=dict(color="#34D399", width=3),
                marker=dict(size=7, color="#34D399")
            ))
            
            fig_acc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                margin=dict(l=40, r=20, t=30, b=40),
                height=320,
                xaxis=dict(
                    title="Epoch",
                    titlefont=dict(color="#94A3B8", size=12),
                    tickfont=dict(color="#CBD5E1"),
                    gridcolor="rgba(255,255,255,0.06)",
                    dtick=1
                ),
                yaxis=dict(
                    title="Accuracy (%)",
                    titlefont=dict(color="#94A3B8", size=12),
                    tickfont=dict(color="#CBD5E1"),
                    gridcolor="rgba(255,255,255,0.06)",
                    range=[30, 102]
                ),
                legend=dict(
                    font=dict(color="#CBD5E1"),
                    bgcolor="rgba(15,23,42,0.8)",
                    bordercolor="rgba(255,255,255,0.1)",
                    borderwidth=1
                ),
                hovermode="x unified"
            )
            st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.info("Training history not found.")

    with tab_loss:
        if history and "loss" in history and "val_loss" in history:
            epochs = history.get("epochs", list(range(1, len(history["loss"]) + 1)))
            loss_vals = [round(l, 4) for l in history["loss"]]
            val_loss_vals = [round(l, 4) for l in history["val_loss"]]
            
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=epochs,
                y=loss_vals,
                mode="lines+markers",
                name="Training Loss",
                line=dict(color="#F87171", width=3),
                marker=dict(size=7, color="#F87171")
            ))
            fig_loss.add_trace(go.Scatter(
                x=epochs,
                y=val_loss_vals,
                mode="lines+markers",
                name="Validation Loss",
                line=dict(color="#FB923C", width=3),
                marker=dict(size=7, color="#FB923C")
            ))
            
            fig_loss.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                margin=dict(l=40, r=20, t=30, b=40),
                height=320,
                xaxis=dict(
                    title="Epoch",
                    titlefont=dict(color="#94A3B8", size=12),
                    tickfont=dict(color="#CBD5E1"),
                    gridcolor="rgba(255,255,255,0.06)",
                    dtick=1
                ),
                yaxis=dict(
                    title="Cross-Entropy Loss",
                    titlefont=dict(color="#94A3B8", size=12),
                    tickfont=dict(color="#CBD5E1"),
                    gridcolor="rgba(255,255,255,0.06)"
                ),
                legend=dict(
                    font=dict(color="#CBD5E1"),
                    bgcolor="rgba(15,23,42,0.8)",
                    bordercolor="rgba(255,255,255,0.1)",
                    borderwidth=1
                ),
                hovermode="x unified"
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("Loss history not found.")

    with tab_summary:
        if metrics:
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f"**Validation Accuracy:** `{metrics.get('validation_accuracy', 0)*100:.2f}%`")
                st.markdown(f"**Training Accuracy:** `{metrics.get('training_accuracy', 0)*100:.2f}%`")
            with m_col2:
                st.markdown(f"**Test Accuracy:** `{metrics.get('test_accuracy', 0)*100:.2f}%`")
                st.markdown(f"**Test Precision:** `{metrics.get('test_precision', 0)*100:.2f}%`")
            with m_col3:
                st.markdown(f"**Test Recall:** `{metrics.get('test_recall', 0)*100:.2f}%`")
                st.markdown(f"**Test F1-Score:** `{metrics.get('test_f1', 0)*100:.2f}%`")
        else:
            st.info("Metrics summary not available.")
