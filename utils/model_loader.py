import os
import json
import streamlit as st
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "traffic_sign_cnn.keras")
MAPPING_PATH = os.path.join(MODEL_DIR, "class_mapping.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")

@st.cache_resource(show_spinner="Loading trained CNN model...")
def load_trained_model():
    """
    Loads and caches the trained Keras CNN model.
    """
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found at: {MODEL_PATH}. Run training first."
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

@st.cache_data(show_spinner=False)
def load_class_mapping():
    """
    Loads and caches class mapping dictionary.
    """
    if not os.path.exists(MAPPING_PATH):
        return None, f"Class mapping file not found at: {MAPPING_PATH}"
    try:
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        return mapping, None
    except Exception as e:
        return None, f"Error reading mapping: {str(e)}"

@st.cache_data(show_spinner=False)
def load_metrics():
    """
    Loads saved evaluation metrics. Returns (data, error).
    """
    if not os.path.exists(METRICS_PATH):
        return None, "Metrics file not found. Evaluation not yet recorded."
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        return metrics, None
    except Exception as e:
        return None, f"Error reading metrics: {str(e)}"

@st.cache_data(show_spinner=False)
def load_training_history():
    """
    Loads training epoch history. Returns (data, error).
    """
    if not os.path.exists(HISTORY_PATH):
        return None, "Training history not found."
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
        return history, None
    except Exception as e:
        return None, f"Error reading history: {str(e)}"
