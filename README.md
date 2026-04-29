# 🏠 v4 - FastAPI + JWT + ONNX House Price Prediction API

Last updated: 

- 29-04-2026

A production-ready machine learning API built with **FastAPI**, featuring **JWT authentication** and an **ONNX-optimized model** for fast, portable inference.

This project demonstrates a complete ML pipeline:

* Synthetic data generation
* Model training with preprocessing
* Export to ONNX
* Secure API deployment with FastAPI

---

# 🚀 Features

* 🔐 JWT Authentication
* ⚡ ONNX Runtime for fast inference
* 🧠 Scikit-learn pipeline (preprocessing + model)
* 📊 Synthetic dataset generation
* 🧪 Swagger UI for testing
* 🐍 Clean, minimal Python structure

---

# 📁 Project Structure

```
.
├── app.py              # FastAPI app (API + inference)
├── train.py            # Model training + ONNX export
├── generate_data.py    # Synthetic dataset generator
├── dataset.csv         # Generated dataset
├── model.onnx          # Exported ONNX model
├── .env                # Environment variables
└── README.md
```

---

# ⚙️ Setup

## 1. Clone repository

```
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

---

## 2. Create virtual environment

```
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

---

## 3. Install dependencies

```
pip install fastapi uvicorn numpy pandas scikit-learn onnxruntime skl2onnx python-dotenv pyjwt
```
or 

```
pip install -r requirements/train.txt
```
---

## 4. Configure environment variables

Create a `.env` file:

```
SECRET_KEY=your_secret_key
FAKE_USERNAME=testuser
FAKE_PASSWORD=testpass
```

---

# 📊 Generate Dataset

```
python generate_data.py
```

This creates a synthetic dataset (`dataset.csv`) with features:

* size
* rooms
* location
* year_built
* condition
* price (target)

---

# 🧠 Train Model + Export ONNX

```
python train.py
```

This will:

* Train a Scikit-learn pipeline
* Apply preprocessing (StandardScaler + OneHotEncoder)
* Export `model.onnx`

---

# ▶️ Run API

```
uvicorn app:app --reload
```

API will be available at:

```
http://127.0.0.1:8000
```

---

# 📖 API Usage

## 🔐 Login

**POST** `/login`

Request:

```
{
  "username": "testuser",
  "password": "testpass"
}
```

Response:

```
{
  "token": "your_jwt_token"
}
```

---

## 📈 Predict House Price

**POST** `/predict`

Headers:

```
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json
```

Request:

```
{
  "size": 120,
  "rooms": 3,
  "year_built": 2000,
  "location": "suburb",
  "condition": "good"
}
```

Response:

```
{
  "user": "testuser",
  "predicted_price": 500000.0
}
```

---

# 🧪 Swagger UI

Interactive API docs available at:

```
http://127.0.0.1:8000/docs
```

---

# 🧠 Model Details

* Model: Linear Regression
* Preprocessing:

  * StandardScaler (numeric features)
  * OneHotEncoder (categorical features)
* Export: ONNX via `skl2onnx`
* Inference: `onnxruntime`

---

# ⚠️ Important Implementation Detail

The ONNX model expects **column-based inputs**, not a single feature array.

Each feature must be passed separately:

* size → float32
* rooms → int64
* year_built → int64
* location → string
* condition → string

Shape must be:

```
[batch_size, 1]
```

Example (internal representation):

```
{
  "size": [[120.0]],
  "rooms": [[3]],
  "year_built": [[2000]],
  "location": [["suburb"]],
  "condition": [["good"]]
}
```

---

# 🧩 Common Issues

### ❌ Invalid input shape

Make sure values are 2D:

```
[[value]]  ✔
[value]    ❌
```

### ❌ Wrong data types

* Use `float32` for size
* Use `int64` for integers
* Use string tensors for categories

### ❌ Mismatched preprocessing

Do NOT manually encode categorical values — the model already includes preprocessing.

---

# 🚀 Future Improvements

* Batch predictions
* Model versioning (`/v1/predict`)
* Dockerization
* Logging & monitoring
* Switch to more advanced models (e.g. RandomForest)

---

## 2. Add serverless note (important context)

```md
## ⚠️ Serverless Note

The ONNX model is loaded using lazy loading to support serverless environments (e.g., Vercel).

This prevents cold start failures and reduces initialization errors.

## 🧪 Local vs Production

Local:
- Runs with Uvicorn
- Model loads on startup

Production (Vercel):
- Runs as serverless function
- Model loads on first request (lazy loading)

# 📜 License

MIT License

---

# 👨‍💻 Author

Built with FastAPI, ONNX, and Scikit-learn for demonstration and learning purposes.
