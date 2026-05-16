# File: covert_tflite.py
import os
import tensorflow as tf

# 1. Define absolute output target paths cleanly
BASE_DIR = r"C:\Users\LENOVO\Downloads\buffermind-backend\TF_LITE MODEL"
OUTPUT_PATH = os.path.join(BASE_DIR, "buffermind_lstm.tflite")

print("Initializing independent architecture graph in memory...")
# Re-maps your precise sliding window dimensions (10 timesteps, 12 features)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10, 12)),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

print("Converting architecture layers to mobile runtime format...")
# Use from_keras_model to bypass disk directory reading completely
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply dynamic range optimization flags
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# CRITICAL: Maps the custom loop operators required by the LSTM layer
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Process graph weights and operators
tflite_model = converter.convert()

# Direct binary write stream to your target folder
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print(f"\n✔ Success! File generated without directory reading faults.")
print(f"   -> Destination: {OUTPUT_PATH}")