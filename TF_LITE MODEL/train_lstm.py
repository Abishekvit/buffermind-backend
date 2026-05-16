# File: train_lstm.py
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Direct Keras ecosystem imports to bypass TensorFlow wrapper version conflicts
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# 1. Reproducible seed
tf.random.set_seed(42)
np.random.seed(42)

# 2. Load the pre-processed 3D arrays
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Extract shapes directly from your preprocessed 3D numpy arrays
num_timesteps = X_train.shape[1]  # Extracted sequence window depth (10)
num_features = X_train.shape[2]   # Extracted feature spacing length (12)

print(f" Detected Sequence Engine Configuration:")
print(f"   -> Timesteps per Window: {num_timesteps}")
print(f"   -> Features per Timestep: {num_features}\n")

# ==========================================
# 3. MODEL ARCHITECTURE (Mobile-Friendly & Optimized)
# ==========================================
model = Sequential([
    # Input tensor definition matching our sliding background buffer window
    Input(shape=(num_timesteps, num_features)),
    
    # Recurrent memory layer tracking temporal degradation sequences
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    
    # Hidden dense reasoning array
    Dense(16, activation="relu"),
    Dropout(0.2),
    
    # Classification head calculating network drop risk probability
    Dense(1, activation="sigmoid")
])

# Adding target tracking parameters to evaluate predictive performance
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy", 
        tf.keras.metrics.Precision(name="precision"), 
        tf.keras.metrics.Recall(name="recall")
    ]
)

print(model.summary())

# ==========================================
# 4. TRAINING LOOP WITH EARLY STOPPING
# ==========================================
early_stopping = EarlyStopping(
    monitor="val_loss", 
    patience=5, 
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# ==========================================
# 5. COMPREHENSIVE PERFORMANCE EVALUATION
# ==========================================
print("\n=== Model Evaluation ===")
loss, acc, prec, rec = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f} (When it predicts a disconnect, how often is it right?)")
print(f"Recall:    {rec:.4f} (Out of all real disconnects, how many did we catch?)")

# 6. Metrics & Confusion Matrix
y_pred = model.predict(X_test)
y_pred_bin = (y_pred > 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_bin))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_bin, target_names=["Stable Connection", "Will Disconnect"]))

# ==========================================
# 7. EXPORT TO TFLITE (Forced Absolute Path Fix)
# ==========================================
print("\nCompiling graph structure to .tflite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Required settings for LSTM dynamic sequence operations on mobile runtimes
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS     
]

tflite_model = converter.convert()

# Explicit target folder definition using raw string prefix to handle backslashes
target_directory = r"C:\Users\LENOVO\Downloads\buffermind-backend\TF_LITE MODEL"
output_path = os.path.join(target_directory, "buffermind_agent.tflite")

# Confirm directory schema integrity before writing binary data
os.makedirs(target_directory, exist_ok=True)

# Write binary model file directly to your target directory
with open(output_path, "wb") as f:
    f.write(tflite_model)
    
print(f"\n✔ Successfully compiled and saved locally!")
print(f"   -> Location: {output_path}")
print("   -> Copy this file into your Android app's 'app/src/main/assets/' path.")