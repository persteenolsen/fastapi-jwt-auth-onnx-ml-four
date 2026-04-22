import os
import jwt
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Literal

# -----------------------------
# INIT APP
# -----------------------------
app = FastAPI(
    title="FastAPI + JWT + ONNX ML (v4)",
    description="22-04-2026 - House Price Prediction API with ONNX ML pipeline + JWT auth",
    version="4.0.0",
    contact={
        "name": "Per Olsen",
        "url": "https://persteenolsen.netlify.app",
    },
)

# -----------------------------
# ENV
# -----------------------------
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
FAKE_USERNAME = os.getenv("FAKE_USERNAME")
FAKE_PASSWORD = os.getenv("FAKE_PASSWORD")

if not SECRET_KEY:
    raise ValueError("SECRET_KEY missing")

# -----------------------------
# AUTH
# -----------------------------
bearer = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    try:
        decoded = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=["HS256"]
        )
        return decoded["username"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

# -----------------------------
# MODEL (LAZY LOADING)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.onnx")

session = None  # 👈 IMPORTANT: not loaded at startup

def get_session():
    global session
    if session is None:
        print("🚀 Loading ONNX model from:", MODEL_PATH)
        print("📦 Model exists:", os.path.exists(MODEL_PATH))
        session = ort.InferenceSession(MODEL_PATH)
    return session

# -----------------------------
# REQUEST MODELS
# -----------------------------
class LoginRequest(BaseModel):
    username: str
    password: str

class PredictionRequest(BaseModel):
    size: float
    rooms: int
    year_built: int
    location: Literal["city", "suburb", "rural"]
    condition: Literal["poor", "fair", "good", "excellent"]

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def root():
    return {"message": "ONNX ML API running"}

# -----------------------------
# LOGIN
# -----------------------------
@app.post("/login")
def login(req: LoginRequest):
    if req.username == FAKE_USERNAME and req.password == FAKE_PASSWORD:
        token = jwt.encode(
            {"username": req.username},
            SECRET_KEY,
            algorithm="HS256"
        )
        return {"token": token}

    raise HTTPException(status_code=401, detail="Bad credentials")

# -----------------------------
# PREDICT
# -----------------------------
@app.post("/predict")
def predict(data: PredictionRequest, username: str = Depends(verify_token)):
    try:
        session = get_session()  # 👈 lazy load happens here

        input_dict = {
            "size": np.array([[data.size]], dtype=np.float32),
            "rooms": np.array([[data.rooms]], dtype=np.int64),
            "year_built": np.array([[data.year_built]], dtype=np.int64),
            "location": np.array([[data.location]], dtype=np.str_),
            "condition": np.array([[data.condition]], dtype=np.str_),
        }

        result = session.run(None, input_dict)

        prediction = float(result[0][0])

        return {
            "user": username,
            "predicted_price": round(prediction, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))