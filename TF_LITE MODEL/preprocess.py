# File: preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. LOAD DATA WITH ALIGNED SCHEMA
# ==========================================
# Make sure this file matches the output name of your synthetic data script!
df = pd.read_csv("buffermind_synthetic_telemetry.csv")

# Group features dynamically using the exact keys present in your generated CSV
binary_features = [
    "is_walking", 
    "is_weak_signal", 
    "is_repeated_playback", 
    "is_gps_moving"
]

continuous_features = [
    "time_of_day", 
    "movement_speed_kmh", 
    "current_signal",
    "sig_hist_t_1", 
    "sig_hist_t_2", 
    "sig_hist_t_3", 
    "sig_hist_t_4", 
    "sig_hist_t_5"
]

all_features = binary_features + continuous_features
target_variable = "label_will_disconnect"

print(f"Loading dataset. Slicing {len(all_features)} features and 1 target variable...")

# ==========================================
# 2. CHRONOLOGICAL SPLIT (Preserves Temporal Order)
# ==========================================
split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()

# ==========================================
# 3. TARGETED FEATURE NORMALIZATION
# ==========================================
# Fit ONLY on training data continuous metrics, then transform test to avoid data leakage
scaler = StandardScaler()
df_train[continuous_features] = scaler.fit_transform(df_train[continuous_features])
df_test[continuous_features] = scaler.transform(df_test[continuous_features])

# Convert extracted segments to numeric matrix arrays
X_train_flat = df_train[all_features].values.astype(np.float32)
y_train_flat = df_train[target_variable].values.astype(np.float32)

X_test_flat = df_test[all_features].values.astype(np.float32)
y_test_flat = df_test[target_variable].values.astype(np.float32)

# ==========================================
# 4. SLIDING WINDOW ENGINE FOR LSTM (2D -> 3D)
# ==========================================
def create_sliding_windows(X_data, y_data, time_steps=10):
    X_3d, y_3d = [], []
    for i in range(len(X_data) - time_steps):
        # Grab a historical block window of 'time_steps' consecutive rows
        X_3d.append(X_data[i : i + time_steps])
        # Target label maps to whether a disconnect happens at the very end of this sequence window
        y_3d.append(y_data[i + time_steps - 1])
    return np.array(X_3d), np.array(y_3d)

# Set temporal depth to match the size of your background Android ring-buffer framework
TIMESTEPS = 10  

X_train, y_train = create_sliding_windows(X_train_flat, y_train_flat, time_steps=TIMESTEPS)
X_test, y_test = create_sliding_windows(X_test_flat, y_test_flat, time_steps=TIMESTEPS)

# ==========================================
# 5. MATRIX STRUCTURE EXPORT FOR TRAINING
# ==========================================
print(f"\n✔ Preprocessing Completed Successfully:")
print(f"   -> X_train shape (3D Tensor): {X_train.shape} (Samples, Timesteps, Features)")
print(f"   -> X_test shape  (3D Tensor): {X_test.shape}  (Samples, Timesteps, Features)")
print(f"   -> Total Features Tracked:   {X_train.shape[2]}")

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("\n✔ Sequential numpy arrays saved safely to local disk. Ready for 'python .\\train_lstm.py'")