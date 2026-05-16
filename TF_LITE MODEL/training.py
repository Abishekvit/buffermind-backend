import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Set reproducible random seed
np.random.seed(42)

# ==========================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ==========================================
NUM_SAMPLES = 12000  # Exceeds the 10k requirement for deep learning training
SEQUENCE_LENGTH = 10  # 10 timesteps per window (e.g., last 50 seconds of telemetry)

print(f"Generating {NUM_SAMPLES} timesteps of sequential mobile telemetry data...")

# ==========================================
# 2. BASELINE FEATURE GENERATION
# ==========================================
# Time of Day (0-23) - cyclical baseline
time_of_day = np.random.randint(0, 24, size=NUM_SAMPLES)

# Contextual probabilities influenced by Time of Day (less movement at night)
is_night = (time_of_day < 6) | (time_of_day > 22)
walk_prob = np.where(is_night, 0.05, 0.35)
is_walking = np.random.binomial(1, walk_prob).astype(int)

# Movement speed (km/h) based on walking state
# 0 if stationary, 3-6 km/h if walking, with occasional vehicle spikes if GPS is moving
movement_speed_kmh = np.where(
    is_walking == 1, 
    np.random.normal(4.5, 1.0, size=NUM_SAMPLES), 
    np.random.choice([0.0, 25.0], size=NUM_SAMPLES, p=[0.95, 0.05]) # 5% chance of vehicular movement
)
movement_speed_kmh = np.clip(movement_speed_kmh, 0, 80)

# GPS Moving state (highly correlated with speed)
is_gps_moving = np.where(movement_speed_kmh > 0.5, 1, 0)

# Playback duration (minutes) - how long the user has been listening
playback_duration_minutes = np.random.exponential(scale=15.0, size=NUM_SAMPLES)
playback_duration_minutes = np.clip(playback_duration_minutes, 0.5, 120.0)

# Repeated playback (0 or 1) - user looping a track/podcast episode
is_repeated_playback = np.random.binomial(1, 0.15, size=NUM_SAMPLES)

# ==========================================
# 3. TEMPORAL SIGNAL LEVEL SIMULATION (Markov-like chain)
# ==========================================
# Signal strength from 0 (No Signal) to 4 (Excellent)
signal_levels = np.zeros(NUM_SAMPLES, dtype=int)
current_signal = 3 # Start with good signal

for i in range(NUM_SAMPLES):
    # Transition probability influenced by movement and speed
    # High speed or walking increases the probability of signal degradation
    degrade_chance = 0.15
    if is_walking[i] == 1: degrade_chance += 0.15
    if movement_speed_kmh[i] > 15: degrade_chance += 0.30
    
    rand = np.random.rand()
    if rand < degrade_chance:
        current_signal = max(0, current_signal - 1)
    elif rand > (1.0 - 0.20): # 20% chance to recover signal if stable
        current_signal = min(4, current_signal + 1)
        
    signal_levels[i] = current_signal

is_weak_signal = np.where(signal_levels <= 1, 1, 0)

# ==========================================
# 4. SIGNAL HISTORY ENGINE (Windowed Lookback)
# ==========================================
signal_history = np.zeros((NUM_SAMPLES, 5), dtype=int)
for i in range(NUM_SAMPLES):
    for h in range(5):
        if i - (h + 1) >= 0:
            signal_history[i, h] = signal_levels[i - (h + 1)]
        else:
            signal_history[i, h] = signal_levels[i] # Pad with current if edge

# ==========================================
# 5. COGNITIVE DISCONNECT RISK ARCHITECTURE (Deterministic Rules + Noise)
# ==========================================
# Base risk calculation mapping real-world mobile edge conditions
disconnect_probability = (
    0.10 * (4 - signal_levels) +               # Low current signal directly scales risk
    0.15 * is_walking +                        # Walking introduces handoff risk
    0.20 * (movement_speed_kmh / 40.0) +       # High speed introduces cell tower switching pressure
    0.15 * (is_gps_moving * is_weak_signal) +  # Lethal combo: moving through a dead zone
    0.05 * (5 - np.mean(signal_history, axis=1)) # Dropping historical signal trend
)

# Inject Gaussian noise to simulate chaotic real-world RF environments
disconnect_probability += np.random.normal(0, 0.05, size=NUM_SAMPLES)
disconnect_probability = np.clip(disconnect_probability, 0.0, 1.0)

# Concrete Binary Label generation based on a dynamic risk threshold (Risk > 65%)
label_will_disconnect = np.where(disconnect_probability > 0.65, 1, 0)

# ==========================================
# 6. PANDAS DATAFRAME COMPOSITION
# ==========================================
df = pd.DataFrame({
    'time_of_day': time_of_day,
    'is_walking': is_walking,
    'is_gps_moving': is_gps_moving,
    'movement_speed_kmh': movement_speed_kmh,
    'playback_duration_minutes': playback_duration_minutes,
    'is_repeated_playback': is_repeated_playback,
    'current_signal': signal_levels,
    'is_weak_signal': is_weak_signal,
    'sig_hist_t_1': signal_history[:, 0],
    'sig_hist_t_2': signal_history[:, 1],
    'sig_hist_t_3': signal_history[:, 2],
    'sig_hist_t_4': signal_history[:, 3],
    'sig_hist_t_5': signal_history[:, 4],
    'disconnect_probability': disconnect_probability,
    'label_will_disconnect': label_will_disconnect
})

# Export to CSV for local project structures
df.to_csv('buffermind_synthetic_telemetry.csv', index=False)
print("✔ Dataset successfully saved to 'buffermind_synthetic_telemetry.csv'")

# ==========================================
# 7. LSTM SEQUENCE WINDOW PREPARATION ENGINE
# ==========================================
def create_lstm_sequences(dataframe, seq_length):
    """
    Transforms a continuous DataFrame into [Samples, Timesteps, Features] 3D arrays
    suitable for Keras/TF Lite LSTM consumption.
    """
    # Drop target metrics to prevent data leakage during training
    feature_cols = [col for col in dataframe.columns if col not in ['disconnect_probability', 'label_will_disconnect']]
    feature_data = dataframe[feature_cols].values
    target_data = dataframe['label_will_disconnect'].values
    
    X, y = [], []
    for i in range(len(dataframe) - seq_length):
        X.append(feature_data[i : i + seq_length])
        # Predict if a disconnect happens at the END of this window sequence
        y.append(target_data[i + seq_length - 1])
        
    return np.array(X), np.array(y), feature_cols

X, y, feature_names = create_lstm_sequences(df, SEQUENCE_LENGTH)
print(f"✔ LSTM Sequence Generation Complete.")
print(f"   -> Input Shape (X): {X.shape} (Samples, Timesteps, Features)")
print(f"   -> Output Shape (y): {y.shape} (Samples,)")

# Train/Validation Split for Hackathon Validation Pipeline
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   -> Training Set: {X_train.shape}, Validation Set: {X_val.shape}")

# ==========================================
# 8. VISUALIZATION PIPELINE
# ==========================================
plt.figure(figsize=(14, 5))

# Plot 1: Target Label Distribution
plt.subplot(1, 2, 1)
sns.countplot(x='label_will_disconnect', data=df, palette='Set2')
plt.title('Distribution of Target Disconnect Labels')
plt.xlabel('Will Disconnect (0=No, 1=Yes)')
plt.ylabel('Count')

# Plot 2: Correlation Heatmap focusing on target relationships
plt.subplot(1, 2, 2)
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Matrix')

plt.tight_layout()
plt.savefig('buffermind_data_analysis.png', dpi=300)
print("✔ Visualization generated and saved as 'buffermind_data_analysis.png'")
plt.show()