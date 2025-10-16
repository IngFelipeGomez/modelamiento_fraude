# 🧠 API de Predicción de Default – Proyecto de Machine Learning

### Universidad Adolfo Ibáñez  
**Curso:** Cloud Computing  
**Profesor:** Ahmad Armoush  
**Fecha:** 15-10-2025  

---

## 👥 Integrantes del Grupo
- Desiréé Vera  
- Felipe Gómez  
- Harmynn Garrido  
- Diego Granados  

---

## 🎯 Objetivo del Proyecto
Desarrollar un proyecto completo de *Machine Learning* que prediga la probabilidad de **default** (no pago de deudas) por parte de un cliente.  
El proyecto integra las etapas de análisis de datos, entrenamiento de modelos, creación de una API con FastAPI y documentación para su despliegue.

---

## 📊 Descripción del Problema y Datos
El problema consiste en identificar qué clientes tienen mayor probabilidad de no cumplir con sus pagos.

**Dataset:** `Tabla Trabajo Grupal N°2.xlsx`  
**Filas:** 12.356 | **Columnas:** 10  

**Variables principales**

| Variable | Tipo | Descripción |
|-----------|------|-------------|
| Edad | Numérica | Edad del cliente |
| Nivel_Educacional | Categórica | Nivel educacional |
| Años_Trabajando | Numérica | Años de experiencia laboral |
| Ingresos | Numérica | Monto encriptado del ingreso |
| Deuda_Comercial | Numérica | Monto de deuda comercial |
| Deuda_Credito | Numérica | Monto de deuda de consumo |
| Otras_Deudas | Numérica | Otras deudas |
| Ratio_Ingresos_Deudas | Numérica | Proporción entre ingresos y deudas |
| Default | Binaria | 1 = incurre en default / 0 = paga correctamente |

---

## ⚙️ Modelamiento

Se entrenaron dos modelos supervisados de clasificación:

| Modelo | AUC | KS | Accuracy | Precision | Recall | F1 |
|---------|-----|----|-----------|------------|---------|----|
| Regresión Logística (Logit) | 0.8386 | 0.5166 | 0.7448 | 0.7305 | 0.9449 | **0.8240** |
| Árbol de Decisión (max_depth=7) | 0.8103 | 0.4789 | 0.7337 | 0.7178 | 0.9539 | 0.8191 |

**Modelo seleccionado:** *Regresión Logística (Logit)*  
Se eligió por su mejor equilibrio entre precisión y recall.

---

## 🌿 Estructura del Proyecto

```bash
modelamiento_fraude/
│
├── data/                         # Datos originales
│   └── Tabla Trabajo Grupal N°2.xlsx
│
├── model/                        # Modelos entrenados y codificadores
│   ├── encoder.pkl
│   └── model.pkl
│
├── notebooks/                   # Exploración y modelamiento
│   ├── AED_fraude.py
│   ├── Tarea_Grupal_Tech.ipynb
│   ├── modelamiento_fraude.py
│   └── test_model.py
│
├── src/                          # Código fuente de la API
│   ├── __init__.py
│   ├── main.py
│   ├── .gitattributes
│   ├── .gitignore
│   ├── python-version
│   ├── runtime.txt
│   ├── README.md
│   └── requirements.txt
│
├── documentos/                   # Documentación técnica y ejecutiva
│   ├── Analisis y decisiones metodologicas.pdf
│   └── Resumen de los Resultados.pdf
│
├── demo/                         # Evidencia de despliegue
│   └── Despliegue_local.mp4
│
└── requirements.txt              # Dependencias del proyecto

```


## 🚀 Ejecución Local

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/IngFelipeGomez/modelamiento_fraude.git
   cd modelamiento_fraude

2. **Crear entorno virtual** (importante instalar Python 3.12)

   ```bash
   python3.12 -m venv venv
   venv\Scripts\activate        # En Windows  
   source venv/bin/activate     # En Linux/Mac

   (en caso de error al activar intentar correr el siguiente codigo:
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   )
   
   
4. **Instalar dependencias**
   ```bash
    pip install -r requirements.txt

7. **Ejecutar la API**
   ```bash
   cd src
   uvicorn main:app --reload

8. **Abrir en el navegador**
   http://127.0.0.1:8000/docs



**Uso de la API**
   En la interfaz interactiva (/docs) puedes probar el endpoint /predict.

**Ejemplo de entrada:**


| Variable | Tipo | Descripción |
|-----------|------|-------------|
| Edad | Numérica | Edad del cliente |
| Nivel_Educacional | Categórica | Nivel educacional |
| Años_Trabajando | Numérica | Años de experiencia laboral |
| Ingresos | Numérica | Monto encriptado del ingreso |
| Deuda_Comercial | Numérica | Monto de deuda comercial |
| Deuda_Credito | Numérica | Monto de deuda de consumo |
| Otras_Deudas | Numérica | Otras deudas |
| Ratio_Ingresos_Deudas | Numérica | Proporción entre ingresos y deudas |
| Default | Binaria | 1 = incurre en default / 0 = paga correctamente |

*Para el campo  "Nivel_Educacional debe ingresar uno de los siguientes valores (entre comillas): "Bas": Educación Básica, "Med": Educación Media, "SupInc": Superior Incompleta, "SupCom": Superior Completa, "Posg": Post Grado*

Para el campo "Ratio_Ingresos_Deudas": Debe ingresar un valor entre 0 y 1.
```bash
{
  "Edad": 35,
  "Nivel_Educacional": "SupInc",
  "Años_Trabajando": 10,
  "Ingresos": 45.0,
  "Deuda_Comercial": 10.5,
  "Deuda_Credito": 3.5,
  "Otras_Deudas": 2.0,
  "Ratio_Ingresos_Deudas": 0.35
}
```
**Ejemplo de salida:**

{
  "prediction_status": "ALTO RIESGO de Default (1)",
  
  "prediction_class": 1,
  
  "probability_default": 0.6055
}

| Variable | Tipo | Descripción |
|-----------|------|-------------|
| Default | Binaria | 1 = incurre en default / 0 = paga correctamente |


**Dependencias principales**
```bash
catboost==1.2.8
fastapi==0.110.0
uvicorn==0.29.0
pydantic>=2.7.0
pytest==7.1.2
pylint ==2.15.0
black == 22.6.0
pandas == 2.2.0
numpy==1.26.4
scikit-learn==1.6.1
category_encoders==2.0.0
matplotlib==3.8.0
seaborn==0.12.2
openpyxl==3.1.2
text
Copiar código
fastapi
uvicorn
pandas
scikit-learn
joblib
numpy
```



































