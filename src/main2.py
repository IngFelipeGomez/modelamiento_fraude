# main2.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from enum import Enum
import pandas as pd
import pickle
import os
import json
from pathlib import Path
# --- CONFIGURACIÓN DE ARTEFACTOS Y CONSTANTES ---

# Rutas y nombres de archivos de artefactos.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.pkl"
ENCODER_PATH = Path(__file__).resolve().parent.parent / "model" / "encoder.pkl"

print(f" Cargando modelo desde: {MODEL_PATH}")
print(f" Cargando encoder desde: {ENCODER_PATH}")


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
    Nivel_Educacional: NivelEducacionalEnum = Field(
        default=NivelEducacionalEnum.posg, 
        description="Nivel educacional. Debe ser uno de: Bas, Med, SupInc, SupCom, Posg."
    )
    Años_Trabajando: int = Field(default=16, description="Años de experiencia laboral.")
    Ingresos: float = Field(default=232.0, description="Ingresos anuales en miles de USD.")
    Deuda_Comercial: float = Field(default=2.8, description="Deuda comercial en miles de USD.")
    Deuda_Credito: float = Field(default=2.1, description="Deuda de tarjeta de crédito en miles de USD.")
    Otras_Deudas: float = Field(default=4.39, description="Otras deudas en miles de USD.")
    Ratio_Ingresos_Deudas: float = Field(default=0.04, description="Ratio de ingresos respecto a la deuda total.")
    
# --- INICIALIZACIÓN DE LA API Y CARGA DE MODELO ---

app = FastAPI(
    title="API de Predicción de Riesgo Crediticio",
    description="Modelo de Machine Learning desplegado para evaluar el riesgo de incumplimiento de pago."
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
    return {"message": "API de Predicción de Default Activa. Vaya a /docs para la documentación interactiva, o a /form para la interfaz amigable."}


@app.post(
    "/predict", 
    summary="Predicción de Riesgo Crediticio (JSON)",
    description="""
    Realiza la predicción de riesgo de Default (0: Pagador, 1: Riesgo/Default) usando 8 variables.
    
    ### ⚠️ Nivel Educacional
    El campo **'Nivel_Educacional'** solo acepta los siguientes valores: **Bas, Med, SupInc, SupCom, Posg**.
    Utilice el menú desplegable al hacer clic en el valor predeterminado del JSON para seleccionarlo.
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

# --- INTERFAZ DE FORMULARIO AMIGABLE ---

# Diccionario para mapear los campos a etiquetas y tipos amigables en el formulario
FORM_FIELDS = [
    {"id": "Edad", "label": "Edad", "type": "number", "min": 18, "default": 56, "hint": "Tipo: Número Entero (Ej: 56)"},
    {"id": "Nivel_Educacional", "label": "Nivel Educacional", "type": "select", "options": NivelEducacionalEnum, "default": "Posg", "hint": "Selecciona una opción."},
    {"id": "Años_Trabajando", "label": "Años Trabajando", "type": "number", "min": 0, "default": 16, "hint": "Tipo: Número Entero (Ej: 16)"},
    {"id": "Ingresos", "label": "Ingresos (miles USD)", "type": "number", "step": 0.01, "default": 232.0, "hint": "Tipo: Número Decimal (Ej: 232.0)"},
    {"id": "Deuda_Comercial", "label": "Deuda Comercial (miles USD)", "type": "number", "step": 0.01, "default": 2.8, "hint": "Tipo: Número Decimal (Ej: 2.8)"},
    {"id": "Deuda_Credito", "label": "Deuda de Crédito (miles USD)", "type": "number", "step": 0.01, "default": 2.1, "hint": "Tipo: Número Decimal (Ej: 2.1)"},
    {"id": "Otras_Deudas", "label": "Otras Deudas (miles USD)", "type": "number", "step": 0.01, "default": 4.39, "hint": "Tipo: Número Decimal (Ej: 4.39)"},
    {"id": "Ratio_Ingresos_Deudas", "label": "Ratio Ingresos/Deudas", "type": "number", "step": 0.0001, "default": 0.04, "hint": "Tipo: Número Decimal (Ej: 0.04)"},
]

def generate_form_html(request: Request) -> str:
    """Genera la estructura HTML del formulario."""
    
    # 1. Generar los campos de entrada
    form_fields_html = ""
    for field in FORM_FIELDS:
        field_id = field['id']
        label = field['label']
        default_value = str(field['default'])
        hint = field['hint']

        if field['type'] == 'select':
            options_html = ""
            for name, value in field['options'].__members__.items():
                selected = "selected" if value.value == default_value else ""
                options_html += f'<option value="{value.value}" {selected}>{label} ({value.value})</option>'
            
            input_field = f"""
            <select id="{field_id}" name="{field_id}" class="w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                {options_html}
            </select>
            """
        else:
            input_field = f"""
            <input type="{field['type']}" id="{field_id}" name="{field_id}" 
                   value="{default_value}" 
                   min="{field.get('min', '')}" step="{field.get('step', '1')}"
                   class="w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500" required>
            """

        form_fields_html += f"""
        <div class="mb-4">
            <label for="{field_id}" class="block text-sm font-medium text-gray-700">{label}</label>
            <div class="flex items-center space-x-2">
                <p class="text-xs text-indigo-600 font-mono p-1 bg-indigo-50 rounded-md w-1/3 text-center truncate" title="{hint}">{hint}</p>
                {input_field}
            </div>
        </div>
        """

    # 2. Obtener la URL base para el JavaScript
    base_url = str(request.base_url).rstrip('/')

    # 3. Estructura HTML completa con Tailwind y JavaScript
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Evaluación de Riesgo Crediticio</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {{ font-family: 'Inter', sans-serif; background-color: #f4f7f9; }}
            .loader {{ border-top-color: #3498db; -webkit-animation: spin 1s ease-in-out infinite; animation: spin 1s ease-in-out infinite; }}
            @-webkit-keyframes spin {{ 0% {{ -webkit-transform: rotate(0deg); }} 100% {{ -webkit-transform: rotate(360deg); }} }}
            @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        </style>
    </head>
    <body class="p-4 sm:p-8 flex justify-center items-start min-h-screen">

        <div class="w-full max-w-lg bg-white p-6 sm:p-8 rounded-2xl shadow-xl transition duration-300 hover:shadow-2xl">
            <h1 class="text-3xl font-extrabold text-gray-900 mb-2 text-center">
                Calculadora de Riesgo Crediticio
            </h1>
            <p class="text-center text-sm text-gray-500 mb-6">
                Complete el formulario para enviar la solicitud al modelo `/predict`.
            </p>
            
            <form id="predictionForm" class="space-y-4">
                {form_fields_html}
                <button type="submit" id="submitButton" class="w-full py-3 bg-indigo-600 text-white font-semibold rounded-xl shadow-md hover:bg-indigo-700 transition duration-150 flex items-center justify-center">
                    <span id="buttonText">Evaluar Riesgo</span>
                    <div id="loader" class="loader ease-linear rounded-full border-4 border-t-4 border-gray-200 h-6 w-6 ml-3 hidden"></div>
                </button>
            </form>

            <div id="resultBox" class="mt-8 p-5 rounded-xl border-l-4 hidden transition duration-300" role="alert">
                <h3 id="resultTitle" class="text-xl font-bold mb-2">Resultado:</h3>
                <p id="resultText" class="text-lg"></p>
                <div id="probabilityText" class="text-sm mt-1"></div>
            </div>

            <div id="errorBox" class="mt-8 p-4 bg-red-100 border-l-4 border-red-500 text-red-700 hidden rounded-lg" role="alert">
                <p class="font-bold">Error en la Solicitud</p>
                <p id="errorText"></p>
            </div>
        </div>

        <script>
            const form = document.getElementById('predictionForm');
            const resultBox = document.getElementById('resultBox');
            const resultTitle = document.getElementById('resultTitle');
            const resultText = document.getElementById('resultText');
            const probabilityText = document.getElementById('probabilityText');
            const errorBox = document.getElementById('errorBox');
            const errorText = document.getElementById('errorText');
            const submitButton = document.getElementById('submitButton');
            const buttonText = document.getElementById('buttonText');
            const loader = document.getElementById('loader');
            
            const API_URL = "{base_url}/predict";

            form.addEventListener('submit', async (e) => {{
                e.preventDefault();
                
                // Limpiar resultados anteriores
                resultBox.classList.add('hidden');
                errorBox.classList.add('hidden');
                
                // Mostrar Loader
                submitButton.disabled = true;
                buttonText.textContent = "Evaluando...";
                loader.classList.remove('hidden');

                const formData = new FormData(form);
                const payload = {{}};

                // Convertir FormData a JSON
                for (let [key, value] of formData.entries()) {{
                    // Pydantic espera los tipos correctos
                    if (['Edad', 'Años_Trabajando'].includes(key)) {{
                        payload[key] = parseInt(value);
                    }} else if (['Ingresos', 'Deuda_Comercial', 'Deuda_Credito', 'Otras_Deudas', 'Ratio_Ingresos_Deudas'].includes(key)) {{
                        payload[key] = parseFloat(value);
                    }} else {{
                        payload[key] = value;
                    }}
                }}

                try {{
                    const response = await fetch(API_URL, {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(payload)
                    }});

                    const data = await response.json();

                    if (!response.ok) {{
                        // Error del servidor (400, 500, etc.)
                        errorText.textContent = data.detail || "Error desconocido del servidor.";
                        errorBox.classList.remove('hidden');
                        return;
                    }}

                    // Predicción exitosa (200)
                    const isDefault = data.prediction_class === 1;
                    const prob = (data.probability_default * 100).toFixed(2);
                    
                    resultTitle.textContent = "Resultado de Evaluación";
                    resultText.textContent = data.prediction_status;
                    
                    // FIX: Uso de concatenación JS (+) en lugar de template literals (`) para evitar error de Python.
                    probabilityText.textContent = 'Probabilidad de Default: ' + prob + '%';

                    // Estilos dinámicos
                    if (isDefault) {{
                        resultBox.className = 'mt-8 p-5 rounded-xl border-l-4 border-red-500 bg-red-50 text-red-800';
                    }} else {{
                        resultBox.className = 'mt-8 p-5 rounded-xl border-l-4 border-green-500 bg-green-50 text-green-800';
                    }}
                    resultBox.classList.remove('hidden');

                }} catch (error) {{
                    // FIX: Uso de concatenación JS (+) en lugar de template literals (`) para evitar error de Python.
                    errorText.textContent = 'No se pudo conectar con la API: ' + error.message;
                    errorBox.classList.remove('hidden');
                }} finally {{
                    // Ocultar Loader
                    submitButton.disabled = false;
                    buttonText.textContent = "Evaluar Riesgo";
                    loader.classList.add('hidden');
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/form", response_class=HTMLResponse, summary="Formulario Web Amigable")
async def get_prediction_form(request: Request):
    """
    Sirve una página web con un formulario para ingresar datos y probar la API de forma visual.
    """
    return generate_form_html(request)
