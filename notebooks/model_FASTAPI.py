from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pandas as pd
import pickle
import os
import sys

# --- CONFIGURACIÓN DE ARTEFACTOS Y CONSTANTES ---

# Rutas de los artefactos que generaste
MODEL_PATH = 'model.pkl'
ENCODER_PATH = 'encoder.pkl' # Usamos el TargetEncoder que ya tienes

# Variables Categóricas y sus categorías válidas
VARIABLE_CATEGORICA = 'Nivel_Educacional'
CATEGORIAS_VALIDAS = ["Med", "SupInc", "SupCom", "Bas", "Posg", "Name"]

# Variables de entrada esperadas (8 características de tu proyecto)
COLUMNAS_INPUT = [
    'Edad', 'Nivel_Educacional', 'Años_Trabajando', 'Ingresos', 
    'Deuda_Comercial', 'Deuda_Credito', 'Otras_Deudas', 'Ratio_Ingresos_Deudas'
]

# --- ESTRUCTURA DE DATOS (Pydantic) ---

# Define la estructura de datos para la solicitud POST
class ClienteData(BaseModel):
    Edad: int
    Nivel_Educacional: str
    Años_Trabajando: int
    Ingresos: float
    Deuda_Comercial: float
    Deuda_Credito: float
    Otras_Deudas: float
    Ratio_Ingresos_Deudas: float

    # Validador para la variable categórica (maneja el error 400 antes de la predicción)
    @validator('Nivel_Educacional')
    def validar_nivel_educacional(cls, v):
        if v not in CATEGORIAS_VALIDAS:
            raise ValueError(
                f"Nivel Educacional inválido: '{v}'. Debe ser uno de: {CATEGORIAS_VALIDAS}"
            )
        return v

# --- INICIALIZACIÓN DE LA API Y CARGA DE MODELO ---

app = FastAPI(
    title="API de Predicción de Default (Riesgo Crediticio)",
    description="Modelo de Regresión Logística para evaluar el riesgo de incumplimiento de pago."
)

MODELO_ML = None
ENCODER_TARGET = None

# Función para cargar artefactos al iniciar la API
def cargar_artefactos(ruta_modelo, ruta_encoder):
    try:
        ruta_base = os.path.dirname(os.path.abspath(__file__))
        
        with open(os.path.join(ruta_base, ruta_modelo), 'rb') as file:
            modelo = pickle.load(file)
        
        with open(os.path.join(ruta_base, ruta_encoder), 'rb') as file:
            encoder = pickle.load(file)
        
        return modelo, encoder
    except FileNotFoundError as e:
        # Error 500 si no encuentra los archivos al inicio
        print(f"ERROR FATAL: No se pudo cargar un artefacto: {e}")
        return None, None
    except Exception as e:
        print(f"ERROR FATAL al cargar los archivos .pkl: {e}")
        return None, None

# Cargar los artefactos una sola vez al inicio
MODELO_ML, ENCODER_TARGET = cargar_artefactos(MODEL_PATH, ENCODER_PATH)

# --- ENDPOINTS ---

@app.get("/")
def home():
    """Endpoint de bienvenida."""
    return {"message": "API de Predicción de Default Activa. Vaya a /docs para interactuar."}


@app.post("/predict")
async def predecir_default(cliente: ClienteData):
    """
    Realiza la predicción de riesgo de Default (1) o Pagador (0).
    """
    
    # 1. Chequeo de Artefactos (Manejo de Error 500)
    if MODELO_ML is None or ENCODER_TARGET is None:
         raise HTTPException(
            status_code=500,
            detail="Error de inicialización: El modelo o el codificador no se cargaron correctamente en el servidor."
        )

    try:
        # 2. Preparar los datos
        input_dict = cliente.dict()
        
        # Convertir a DataFrame con las columnas en el orden correcto
        df_input = pd.DataFrame([input_dict], columns=COLUMNAS_INPUT)

        # 3. Preprocesamiento: Codificación
        # Se aplica el TargetEncoder guardado (encoder.pkl)
        df_encoded = ENCODER_TARGET.transform(df_input)

        # 4. Predicción
        # El modelo predice la probabilidad de [Clase 0, Clase 1]
        probabilidades = MODELO_ML.predict_proba(df_encoded)
        
        # Probabilidad de Default (Clase 1)
        prob_default = float(probabilidades[0][1]) 
        
        # La clase predicha es 0 o 1
        pred_class = int(MODELO_ML.predict(df_encoded)[0])
        
        # 5. Construir la Respuesta
        if pred_class == 1:
            resultado_texto = "ALTO RIESGO de Default (1)"
        else:
            resultado_texto = "BAJO RIESGO / PAGADOR (0)"

        return {
            "estado_prediccion": resultado_texto,
            "prediccion_default": pred_class,
            "probabilidad_default": round(prob_default, 4)
        }

    except Exception as e:
        # Captura errores que podrían surgir durante la predicción/transformación
        raise HTTPException(
            status_code=400,
            detail=f"Error en el procesamiento de la solicitud: {e}. Verifique la integridad de los datos de entrada."
        )
