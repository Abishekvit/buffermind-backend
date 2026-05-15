# File: main.py

from fastapi import FastAPI, HTTPException, logger
from fastapi.responses import JSONResponse
from models import PredictionRequest, PredictionResponse
from fake_lstm import calculate_prediction
import logging

app = FastAPI(
    title="BufferMind AI Backend",
    description="Context‑aware adaptive memory prediction service",
    version="0.1.0",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/health")
def health():
    return JSONResponse(content={"status": "healthy"})

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        result = calculate_prediction(
            walking=request.walking,
            weak_signal=request.weak_signal,
            repeated_playback=request.repeated_playback,
            gps_moving=request.gps_moving,
            playback_duration_minutes=request.playback_duration_minutes,
        )
        logger.info(
            f"Prediction: p={result['disconnect_probability']}, "
            f"conf={result['confidence']}, "
            f"should_buffer={result['should_buffer']}"
        )
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")