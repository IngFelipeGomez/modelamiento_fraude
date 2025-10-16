# 🧠 API de Predicción de Default – Proyecto de Machine Learning

### Universidad Adolfo Ibáñez  
**Curso:** Cloud Computing  
**Profesor:** Ahmad Armoush  
**Fecha:** 04-10-2025  

---

## 👥 Integrantes del Grupo
- Desirée Vera  
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

## 🧩 Estructura del Proyecto

modelamiento_fraude/
│
├── data/                  # Datos originales
│   └── Tabla Trabajo Grupal N°2.xlsx
│
├── model/                 # Modelos entrenados y encoder
│   ├── encoder.pkl
│   └── model.pkl
│
├── notebooks/             # Exploración y modelamiento
│   └── Tarea_Grupal_Tech.ipynb
│
├── src/                   # Código fuente de la API
│   └── main.py
│
├── requirements.txt       # Dependencias del proyecto
└── evidencia_api.png      # Captura de la API funcionando

## 🚀 Ejecución Local

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/tu_usuario/modelamiento_fraude.git
   cd modelamiento_fraude

2. **(Opcional) Crear entorno virtual**
  python -m venv venv
venv\Scripts\activate        # En Windows  
source venv/bin/activate     # En Linux/Mac
   
3. **Instalar dependencias**
 pip install -r requirements.txt

4. **Ejecutar la API**
cd src
uvicorn main:app --reload

5. **Abrir en el navegador**
   http://127.0.0.1:8000/docs



**Uso de la API**
   En la interfaz interactiva (/docs) puedes probar el endpoint /predict.

**Ejemplo de entrada:**

{
  "Edad": 35,
  "Nivel_Educacional": 2,
  "Años_Trabajando": 10,
  "Ingresos": 45.0,
  "Deuda_Comercial": 10.5,
  "Deuda_Credito": 3.5,
  "Otras_Deudas": 2.0,
  "Ratio_Ingresos_Deudas": 0.35
}

**Ejemplo de salida:**
{
  "probabilidad_fraude": 0.6032,
  "prediccion": "Sí"
}


**Dependencias principales**
   text
   Copiar código
   fastapi
   uvicorn
   pandas
   scikit-learn
   joblib
   numpy






