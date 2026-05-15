## Day 7: Docker‑based AI Backend

BufferMind now runs the AI prediction backend as a **Docker‑containerized microservice**:

- `Dockerfile` builds a lightweight image from `python:3.10-slim`.
- `docker-compose.yml` orchestrates the FastAPI service with:
  - health checks,
  - logging,
  - automatic restart.
- A `/health` endpoint allows tools to verify backend liveness.
- The `/predict` API continues to accept the same context JSON and returns:
  - `disconnect_probability`,
  - `confidence`,
  - `should_buffer`,
  - `buffer_minutes`,
  - `reason`.

The Android app communicates with this Dockerized backend using Retrofit, making the architecture look like a real **Samsung AI platform** instead of just a local fake model.

Future:
- Host on cloud (AWS ECS, Google Cloud Run, Kubernetes).
- Add real LSTM model behind the same `/predict` contract.

# 1. Ensure you have Docker + Docker Compose installed.

# 2. Navigate to backend folder
cd buffermind-backend

# 3. Build and run
docker compose up --build

# 4. Check health
curl http://localhost:8000/health

# 5. Test prediction
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '
{
    "walking": true,
    "weak_signal": true,
    "repeated_playback": true,
    "gps_moving": true,
    "playback_duration_minutes": 15.0
}'

# 6. Android: set Retrofit base URL to your laptop IP (Wi‑Fi interface)
