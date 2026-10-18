import os
import io
import json
import time
import numpy as np
from PIL import Image
from utils.model_loader import load_trained_model, load_class_mapping

def preprocess_image(image_input, target_size=(64, 64)):
    """
    Safely preprocesses an input image matching the notebook pipeline:
    1. Read PIL Image, raw bytes, or stream (with seek safety)
    2. Convert to RGB
    3. Resize to target_size (64, 64) with Bilinear resampling
    4. Normalize pixel values from [0, 255] to [0.0, 1.0] (dtype float32)
    5. Add batch dimension -> (1, 64, 64, 3)
    """
    if isinstance(image_input, bytes):
        image = Image.open(io.BytesIO(image_input))
    elif not isinstance(image_input, Image.Image):
        if hasattr(image_input, "seek"):
            try:
                image_input.seek(0)
            except Exception:
                pass
        image = Image.open(image_input)
    else:
        image = image_input

    # Convert to RGB to handle grayscale or RGBA properly
    image_rgb = image.convert("RGB")
    
    # Resize to exact dimensions
    resized = image_rgb.resize(target_size, Image.Resampling.BILINEAR)
    
    # Convert to float32 and normalize
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    
    # Add batch dimension
    batch = np.expand_dims(arr, axis=0)
    return image_rgb, batch

def predict_traffic_sign(image_input, top_k=3):
    """
    Runs inference using the loaded CNN model.
    Returns:
    - success (bool)
    - dict with best_prediction and top_k list
    - error_message (if any)
    """
    model, model_err = load_trained_model()
    if model is None:
        return False, None, f"Model could not be loaded: {model_err}"
    
    mapping, map_err = load_class_mapping()
    if mapping is None:
        return False, None, f"Class mapping could not be loaded: {map_err}"
    
    try:
        t_start = time.perf_counter()
        img_size = mapping.get("img_size", 64)
        rgb_image, preprocessed_batch = preprocess_image(image_input, target_size=(img_size, img_size))
        
        # Inference
        probs = model.predict(preprocessed_batch, verbose=0)[0]
        latency_ms = round((time.perf_counter() - t_start) * 1000, 1)
        
        # Top-k indices
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        class_ids = mapping.get("class_ids", [])
        class_names = mapping.get("class_names", {})
        
        top_predictions = []
        for rank, idx in enumerate(top_indices):
            # Model output index to ClassId
            if idx < len(class_ids):
                cid = class_ids[idx]
            else:
                cid = idx
            
            cname = class_names.get(str(cid), f"Sign Class {cid}")
            conf = float(probs[idx])
            
            top_predictions.append({
                "rank": rank + 1,
                "class_index": int(idx),
                "class_id": int(cid),
                "class_name": cname,
                "confidence": conf,
                "confidence_pct": round(conf * 100, 2)
            })
            
        best = top_predictions[0]
        
        result = {
            "best_prediction": best,
            "top_predictions": top_predictions,
            "all_probabilities": probs.tolist(),
            "image_size": f"{img_size}x{img_size} RGB",
            "latency_ms": latency_ms
        }
        return True, result, None
    except Exception as e:
        return False, None, f"Prediction error: {str(e)}"
