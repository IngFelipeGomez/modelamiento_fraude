# src/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# --- Inicializar la app ---
app = FastAPI(
    title="API Predicción de Fraude - MODELAMIENTO_FRAUDE",
    description="Servicio REST que predice la probabilidad de fraude utilizando un modelo entrenado.",
    version="1.0.0"
)

# --- Rutas de carga ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
ENCODER_PATH = BASE_DIR / "model" / "encoder.pkl"

# --- Cargar modelo y encoder ---
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

# --- Definir clase de entrada ---
class DatosEntrada(BaseModel):
    Edad: float
    Nivel_Educacional: int
    Años_Trabajando: float
    Ingresos: float
    Deuda_Comercial: float
    Deuda_Credito: float
    Otras_Deudas: float
    Ratio_Ingresos_Deudas: float

# --- Endpoint principal ---
@app.post("/predict")
def predecir_fraude(data: DatosEntrada):
    df_input = pd.DataFrame([data.dict()])
    X_encoded = encoder.transform(df_input)
    prob = model.predict_proba(X_encoded)[:, 1][0]
    pred = int(prob >= 0.5)
    return {
        "probabilidad_fraude": round(float(prob), 4),
        "prediccion": "Sí" if pred == 1 else "No"
    }

# --- Endpoint raíz ---
@app.get("/")
def home():
    return {"mensaje": "API de MODELAMIENTO_FRAUDE. Usa /docs para probar la predicción."}