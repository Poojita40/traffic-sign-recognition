import os
import json
import math
import random
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Settings matching notebook ---
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10
VAL_SPLIT = 0.2
SEED = 42
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".webp")

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "archive", "traffic_Data")
if not os.path.exists(DATASET_DIR):
    DATASET_DIR = os.path.join(BASE_DIR, "archive")

DATA_DIR = os.path.join(DATASET_DIR, "DATA")
TEST_DIR = os.path.join(DATASET_DIR, "TEST")
LABELS_CSV = os.path.join(BASE_DIR, "archive", "labels.csv")
if not os.path.exists(LABELS_CSV):
    LABELS_CSV = os.path.join(DATASET_DIR, "labels.csv")

MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "traffic_sign_cnn.keras")
MAPPING_PATH = os.path.join(MODEL_DIR, "class_mapping.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")

print("Checking dataset paths:")
print("DATA_DIR:", DATA_DIR)
print("TEST_DIR:", TEST_DIR)
print("LABELS_CSV:", LABELS_CSV)

labels_df = pd.read_csv(LABELS_CSV)
labels_df.columns = [c.strip() for c in labels_df.columns]
labels_df = labels_df[["ClassId", "Name"]].dropna()
labels_df["ClassId"] = pd.to_numeric(labels_df["ClassId"]).astype(int)
labels_df["Name"] = labels_df["Name"].astype(str).str.strip()

id_to_name = dict(zip(labels_df["ClassId"], labels_df["Name"]))

def sort_key(name):
    return (0, int(name), name) if name.strip().isdigit() else (1, 0, name)

data_subdirs = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))], key=sort_key)

def list_images(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS) and os.path.isfile(os.path.join(folder, f))]
    return sorted([os.path.join(folder, f) for f in files])

def load_and_resize(src):
    with Image.open(src) as im:
        im = im.convert("RGB")
        im = im.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.uint8)

print("Loading training images from class folders...")
images, label_ids = [], []
for folder in data_subdirs:
    cid = int(folder)
    folder_path = os.path.join(DATA_DIR, folder)
    img_paths = list_images(folder_path)
    for p in img_paths:
        try:
            images.append(load_and_resize(p))
            label_ids.append(cid)
        except Exception:
            pass

X_all = np.stack(images)
y_ids = np.array(label_ids)
del images

class_ids = sorted(set(y_ids.tolist()))
NUM_CLASSES = len(class_ids)
id_to_index = {cid: idx for idx, cid in enumerate(class_ids)}
index_to_id = {idx: cid for cid, idx in id_to_index.items()}
class_names = [id_to_name.get(cid, f"Class {cid}") for cid in class_ids]
y_all = np.array([id_to_index[c] for c in y_ids])

print(f"Total training images loaded: {len(X_all)}, Total classes: {NUM_CLASSES}")

train_idx, val_idx = train_test_split(
    np.arange(len(y_all)),
    test_size=VAL_SPLIT,
    random_state=SEED,
    stratify=y_all
)

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_val, y_val = X_all[val_idx], y_all[val_idx]
print(f"Train set: {len(X_train)}, Val set: {len(X_val)}")

def to_float(img, lbl):
    return tf.cast(img, tf.float32) / 255.0, lbl

def make_dataset(X, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=SEED)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(to_float, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(X_train, y_train, training=True)
val_ds = make_dataset(X_val, y_val, training=False)

# Build CNN matching notebook architecture
model = keras.Sequential([
    keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(momentum=0.9),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(momentum=0.9),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(momentum=0.9),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(NUM_CLASSES, activation="softmax"),
], name="traffic_sign_cnn")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

print("Training CNN model...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=2,
    shuffle=False
)

# Evaluate Validation
val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc*100:.2f}%")

# Evaluate Test set if available
test_loss, test_acc = None, None
test_f1, test_precision, test_recall = None, None, None
test_cm = None
n_test_samples = 0

test_files = list_images(TEST_DIR) if os.path.exists(TEST_DIR) else []
if test_files:
    test_imgs, test_labels = [], []
    for f in test_files:
        base = os.path.basename(f)
        prefix = base.split("_")[0]
        if prefix.isdigit():
            cid = int(prefix)
            if cid in id_to_index:
                try:
                    test_imgs.append(load_and_resize(f))
                    test_labels.append(id_to_index[cid])
                except Exception:
                    pass
    if test_imgs:
        X_test = np.stack(test_imgs)
        y_test = np.array(test_labels)
        n_test_samples = len(X_test)
        test_ds = make_dataset(X_test, y_test, training=False)
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        
        y_pred_probs = model.predict(test_ds, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
        test_precision = float(prec)
        test_recall = float(rec)
        test_f1 = float(f1)
        test_cm = confusion_matrix(y_test, y_pred).tolist()
        print(f"Test accuracy: {test_acc*100:.2f}%, Test F1: {test_f1*100:.2f}% ({n_test_samples} samples)")

# Save Model
model.save(MODEL_PATH)
print("Saved model to:", MODEL_PATH)

# Save Class Mapping
mapping_data = {
    "img_size": IMG_SIZE,
    "normalization": "pixel / 255.0",
    "num_classes": NUM_CLASSES,
    "class_ids": class_ids,
    "class_names": {str(c): id_to_name.get(c, f"Class {c}") for c in class_ids},
    "index_to_id": {str(idx): cid for idx, cid in index_to_id.items()}
}
with open(MAPPING_PATH, "w") as f:
    json.dump(mapping_data, f, indent=2)
print("Saved mapping to:", MAPPING_PATH)

# Save Training History
hist_data = {
    "epochs": list(range(1, len(history.history["accuracy"]) + 1)),
    "accuracy": [float(x) for x in history.history["accuracy"]],
    "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
    "loss": [float(x) for x in history.history["loss"]],
    "val_loss": [float(x) for x in history.history["val_loss"]],
}
with open(HISTORY_PATH, "w") as f:
    json.dump(hist_data, f, indent=2)
print("Saved history to:", HISTORY_PATH)

# Save Metrics
metrics_data = {
    "training_accuracy": float(history.history["accuracy"][-1]),
    "training_loss": float(history.history["loss"][-1]),
    "validation_accuracy": float(val_acc),
    "validation_loss": float(val_loss),
    "test_accuracy": float(test_acc) if test_acc is not None else None,
    "test_loss": float(test_loss) if test_loss is not None else None,
    "test_precision": test_precision,
    "test_recall": test_recall,
    "test_f1": test_f1,
    "confusion_matrix": test_cm,
    "total_classes": NUM_CLASSES,
    "test_samples": n_test_samples,
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "image_dimensions": f"{IMG_SIZE}x{IMG_SIZE}x3",
    "optimizer": "Adam (lr=0.001)",
    "loss_function": "Sparse Categorical Crossentropy",
    "total_parameters": int(model.count_params())
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics_data, f, indent=2)
print("Saved metrics to:", METRICS_PATH)
print("Training and artifact export complete successfully!")
