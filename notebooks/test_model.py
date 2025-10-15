import pickle
import pandas as pd
import os
import sys

# --- Constantes y Artefactos ---
# Las categorías para el Nivel Educacional
CATEGORIAS_EDUCACION = ["Med", "SupInc", "SupCom", "Bas", "Posg"]

# Definir los nombres de las columnas que espera el modelo
COLUMNAS_INPUT = [
    'Edad', 'Nivel_Educacional', 'Años_Trabajando', 'Ingresos', 
    'Deuda_Comercial', 'Deuda_Credito', 'Otras_Deudas', 'Ratio_Ingresos_Deudas'
]

# --- Funciones de Utilidad ---

def cargar_artefactos(ruta_modelo='model.pkl', ruta_encoder='encoder.pkl'):
    """Carga el modelo de ML y el encoder desde archivos .pkl."""
    
    # Ruta base es donde se está ejecutando el script
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    
    archivos = {'modelo': ruta_modelo, 'encoder': ruta_encoder}
    artefactos = {}
    
    for nombre, archivo in archivos.items():
        ruta_completa = os.path.join(ruta_base, archivo)
        if not os.path.exists(ruta_completa):
            print(f"\n❌ ERROR: Archivo '{archivo}' no encontrado.")
            print(f"Asegúrese de que '{archivo}' se encuentra en: {ruta_base}")
            sys.exit(1) # Detiene la ejecución si falta un archivo
            
        try:
            with open(ruta_completa, 'rb') as file:
                artefactos[nombre] = pickle.load(file)
            print(f"✅ Artefacto '{archivo}' cargado correctamente.")
        except Exception as e:
            print(f"\n❌ ERROR al cargar {archivo}: {e}")
            sys.exit(1)
            
    return artefactos['modelo'], artefactos['encoder']


def obtener_input_usuario():
    """Solicita al usuario que ingrese los valores de las 8 variables."""
    print("\n--- INGRESO DE DATOS DEL CLIENTE ---")
    datos_cliente = {}

    # 1. Edad (int)
    while True:
        try:
            edad = int(input("1. Edad (años, ej: 45): "))
            datos_cliente['Edad'] = edad
            break
        except ValueError:
            print("Entrada inválida. Ingrese un número entero para la edad.")

    # 2. Nivel Educacional (Categórico)
    print("\n2. Nivel Educacional:")
    for i, cat in enumerate(CATEGORIAS_EDUCACION):
        print(f"   [{i + 1}] {cat}")
        
    while True:
        try:
            opcion = int(input(f"Seleccione la opción [1-{len(CATEGORIAS_EDUCACION)}]: "))
            if 1 <= opcion <= len(CATEGORIAS_EDUCACION):
                educacion = CATEGORIAS_EDUCACION[opcion - 1]
                datos_cliente['Nivel_Educacional'] = educacion
                break
            else:
                print("Opción fuera de rango.")
        except ValueError:
            print("Entrada inválida. Ingrese el número de la opción.")

    # 3. Años Trabajando (int)
    while True:
        try:
            trabajando = int(input("3. Años Trabajando (ej: 10): "))
            datos_cliente['Años_Trabajando'] = trabajando
            break
        except ValueError:
            print("Entrada inválida. Ingrese un número entero.")
            
    # 4. Ingresos (float)
    while True:
        try:
            ingresos = float(input("4. Ingresos (monto encriptado, ej: 45000.50): "))
            datos_cliente['Ingresos'] = ingresos
            break
        except ValueError:
            print("Entrada inválida. Ingrese un número (puede ser decimal).")
            
    # 5. Deuda Comercial (float)
    while True:
        try:
            d_comercial = float(input("5. Deuda Comercial (ej: 5000.00): "))
            datos_cliente['Deuda_Comercial'] = d_comercial
            break
        except ValueError:
            print("Entrada inválida. Ingrese un número.")
            
    # 6. Deuda Crédito (float)
    while True:
        try:
            d_credito = float(input("6. Deuda Crédito (ej: 2500.00): "))
            datos_cliente['Deuda_Credito'] = d_credito
            break
        except ValueError:
            print("Entrada inválida. Ingrese un número.")
            
    # 7. Otras Deudas (float)
    while True:
        try:
            o_deudas = float(input("7. Otras Deudas (ej: 1000.00): "))
            datos_cliente['Otras_Deudas'] = o_deudas
            break
        except ValueError:
            print("Entrada inválida. Ingrese un número.")
            
    # 8. Ratio Ingresos Deudas (float)
    while True:
        try:
            ratio = float(input("8. Ratio Ingresos Deudas (ej: 0.15): "))
            datos_cliente['Ratio_Ingresos_Deudas'] = ratio
            break
        except ValueError:
            print("Entrada inválida. Ingrese un número.")

    return datos_cliente


def hacer_prediccion(datos_cliente, modelo, encoder):
    """Procesa los datos de entrada y realiza la predicción."""
    
    # 1. Convertir a DataFrame de Pandas (crucial para el encoder)
    # Se añade como una lista para garantizar que Pandas lo trate como una fila.
    df_input = pd.DataFrame([datos_cliente], columns=COLUMNAS_INPUT)
    
    # 2. Preprocesamiento (Codificación con TargetEncoder)
    # El encoder.transform() codifica la columna 'Nivel_Educacional'
    df_encoded = encoder.transform(df_input)
    
    # 3. Predicción
    # El modelo de Logit de Scikit-learn (model.pkl) espera datos codificados
    probabilidades = modelo.predict_proba(df_encoded)
    pred_class = modelo.predict(df_encoded)[0]

    # Probabilidad de Default (Clase 1)
    prob_default = probabilidades[0][1] 
    
    # 4. Resultado
    print("\n--- RESULTADOS DE LA PREDICCIÓN ---")
    
    if pred_class == 1:
        resultado_texto = "ALTO RIESGO de Default (1)"
    else:
        resultado_texto = "BAJO RIESGO / PAGADOR (0)"

    print(f"CLASIFICACIÓN FINAL: {resultado_texto}")
    print(f"Probabilidad de Default (Clase 1): {prob_default:.4f} ({prob_default*100:.2f}%)")


def main():
    """Función principal para la ejecución del script de prueba."""
    
    # 1. Cargar el modelo y el encoder (solo una vez)
    modelo, encoder = cargar_artefactos()
    
    # 2. Obtener la entrada del usuario
    datos_cliente = obtener_input_usuario()
    
    # 3. Realizar la predicción y mostrar resultados
    hacer_prediccion(datos_cliente, modelo, encoder)
    
    print("\n--- PRUEBA FINALIZADA ---")

if __name__ == "__main__":
    main()
