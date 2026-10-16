import os
import json
import pandas as pd
from utils.model_loader import load_class_mapping

def get_class_dataframe():
    """
    Returns pandas DataFrame of all classes for easy reference.
    """
    mapping, err = load_class_mapping()
    if mapping is None:
        return pd.DataFrame(columns=["Output Index", "Class ID", "Traffic Sign Name"])
    
    class_ids = mapping.get("class_ids", [])
    class_names = mapping.get("class_names", {})
    
    rows = []
    for idx, cid in enumerate(class_ids):
        rows.append({
            "Output Index": idx,
            "Class ID": cid,
            "Traffic Sign Name": class_names.get(str(cid), f"Class {cid}")
        })
    return pd.DataFrame(rows)
