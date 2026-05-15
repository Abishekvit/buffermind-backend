# File: fake_lstm.py

def calculate_prediction(
    walking: bool,
    weak_signal: bool,
    repeated_playback: bool,
    gps_moving: bool,
    playback_duration_minutes: float,
) -> dict:
    """
    Simulate an LSTM prediction:
    - weighted scoring of context features
    - target 78% disconnect probability when walking + weak + repeat
    """
    score = 0.0
    components = []

    if walking:
        score += 0.30
        components.append("Walking detected")
    if weak_signal:
        score += 0.30
        components.append("Weak signal")
    if repeated_playback:
        score += 0.20
        components.append("Repeat playback")
    if gps_moving:
        score += 0.10
        components.append("GPS movement")
    if playback_duration_minutes >= 5.0:
        score += 0.10
        components.append("Long playback")

    # Clamp to 0.0–1.0, center around 0.78 for demo
    score = max(0.0, min(1.0, score))
    probability = 0.78 if (walking and weak_signal and repeated_playback) else score

    return {
        "disconnect_probability": probability,
        "confidence": 0.90,
        "should_buffer": probability >= 0.7,
        "buffer_minutes": 30,
        "reason": " + ".join(components)
    }