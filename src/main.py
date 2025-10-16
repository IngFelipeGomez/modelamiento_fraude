# main.py

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from enum import Enum
import pandas as pd
import pickle
import os
from typing import List
from pathlib import Path

# --- CONFIGURACIÓN DE ARTEFACTOS Y CONSTANTES ---

# Rutas y nombres de archivos de artefactos.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.pkl"
ENCODER_PATH = Path(__file__).resolve().parent.parent / "model" / "encoder.pkl"

print(f"🟢 Cargando modelo desde: {MODEL_PATH}")
print(f"🟢 Cargando encoder desde: {ENCODER_PATH}")

# Variables de entrada esperadas por el modelo (orden y tipo).
COLUMNAS_INPUT = [
    'Edad', 'Nivel_Educacional', 'Años_Trabajando', 'Ingresos', 
    'Deuda_Comercial', 'Deuda_Credito', 'Otras_Deudas', 'Ratio_Ingresos_Deudas'
]

# --- ESTRUCTURA DE DATOS (Pydantic) ---

# Define el Enum para generar el desplegable en Swagger y la validación estricta.
class NivelEducacionalEnum(str, Enum):
    """Opciones válidas para el Nivel Educacional."""
    med = "Med"
    supinc = "SupInc"
    supcom = "SupCom"
    bas = "Bas"
    posg = "Posg"

# Define la estructura de entrada con valores de ejemplo y tipos correctos.
class ClienteData(BaseModel):
    Edad: int = Field(default=56, description="Edad del cliente en años.")
    # El Enum garantiza el desplegable y la validación.
    Nivel_Educacional: NivelEducacionalEnum = Field(
        default=NivelEducacionalEnum.posg, 
        description="Nivel educacional. Debe ser uno de: Bas, Med, SupInc, SupCom, Posg.",
        examples=["Bas", "Med", "SupInc", "SupCom", "Posg"]
    )
    Años_Trabajando: int = Field(default=16, description="Años de experiencia laboral.")
    Ingresos: float = Field(default=232.0, description="Ingresos anuales en miles de USD.")
    Deuda_Comercial: float = Field(default=2.8, description="Deuda comercial en miles de USD.")
    Deuda_Credito: float = Field(default=2.1, description="Deuda de tarjeta de crédito en miles de USD.")
    Otras_Deudas: float = Field(default=4.39, description="Otras deudas en miles de USD.")
    Ratio_Ingresos_Deudas: float = Field(default=0.04, description="Ratio de ingresos respecto a la deuda total.")
    
# --- INICIALIZACIÓN DE LA API Y CARGA DE MODELO ---

app = FastAPI(
    title="API de Predicción de Default (Riesgo Crediticio)",
    description="Modelo de Machine Learning desplegado para evaluar el riesgo de incumplimiento de pago. **Por favor, vea las opciones válidas para 'Nivel_Educacional' antes de enviar la solicitud.**"
)

MODELO_ML = None
ENCODER_TARGET = None

def cargar_artefactos(ruta_modelo, ruta_encoder):
    """Carga los archivos .pkl del modelo y el encoder."""
    try:
        ruta_base = os.path.dirname(os.path.abspath(__file__))
        
        with open(os.path.join(ruta_base, ruta_modelo), 'rb') as file:
            modelo = pickle.load(file)
        
        with open(os.path.join(ruta_base, ruta_encoder), 'rb') as file:
            encoder = pickle.load(file)
        
        return modelo, encoder
    except FileNotFoundError as e:
        print(f"Error FATAL: No se pudo cargar un artefacto: {e}. Asegúrese de que los archivos .pkl están en la misma carpeta.")
        return None, None
    except Exception as e:
        print(f"Error FATAL al cargar los archivos .pkl: {e}")
        return None, None

# Carga de artefactos
MODELO_ML, ENCODER_TARGET = cargar_artefactos(MODEL_PATH, ENCODER_PATH)

# --- FUNCIÓN DE PREDICCIÓN CENTRAL ---

def predecir(df_input: pd.DataFrame):
    """Función central que maneja el preprocesamiento y la predicción."""
    
    if MODELO_ML is None or ENCODER_TARGET is None:
         raise HTTPException(
            status_code=500,
            detail="Error de inicialización: Los archivos model.pkl o encoder.pkl no se pudieron cargar al iniciar el servidor."
        )
    
    # Preprocesamiento (Codificación con TargetEncoder)
    # Convertir el Enum de vuelta a string para que el encoder lo procese
    df_input['Nivel_Educacional'] = df_input['Nivel_Educacional'].astype(str)
    
    # Aplicar la transformación de las columnas existentes
    df_encoded = ENCODER_TARGET.transform(df_input)

    # Predicción y probabilidades
    probabilidades = MODELO_ML.predict_proba(df_encoded)
    prob_default = probabilidades[0][1] 
    pred_class = MODELO_ML.predict(df_encoded)[0] 
    
    resultado_texto = "ALTO RIESGO de Default (1)" if pred_class == 1 else "BAJO RIESGO / PAGADOR (0)"

    return {
        "prediction_status": resultado_texto,
        "prediction_class": int(pred_class), 
        "probability_default": round(prob_default, 4)
    }

# --- ENDPOINTS ---

@app.get("/")
def home():
    """Endpoint de bienvenida."""
    return {"message": "API de Predicción de Default Activa. Vaya a /docs para la documentación interactiva."}


@app.post(
    "/predict", 
    summary="Predicción de Riesgo Crediticio (JSON)",
    description="""
    Realiza la predicción de riesgo de Default (0: Pagador, 1: Riesgo/Default) usando 8 variables.
    
    ### ⚠️ Nivel Educacional
    El campo **'Nivel_Educacional'** solo acepta los siguientes valores: **Bas, Med, SupInc, SupCom, Posg**.
        """
)
def predecir_default(cliente: ClienteData):
    """
    Realiza la predicción de riesgo de Default.
    """
    try:
        input_dict = cliente.dict()
        df_input = pd.DataFrame([input_dict], columns=COLUMNAS_INPUT)
        
        return predecir(df_input)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor al procesar la predicción: {e}. Por favor, verifique el formato de entrada."
        )
