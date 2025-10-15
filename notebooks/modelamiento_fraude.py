# modelamiento_fraude.py
# Código corregido para compatibilidad con Python 3.12 y serialización con Scikit-learn.

import pandas as pd
import numpy as np
import os
import pickle  # Necesario para guardar los modelos (.pkl)

# Reemplazamos statsmodels con la versión de Scikit-learn (fácil de serializar)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, roc_auc_score, f1_score, confusion_matrix,
    accuracy_score, precision_score, recall_score
)
from category_encoders import TargetEncoder

# Para visualización (opcional, puede causar problemas si no hay entorno gráfico)
import matplotlib.pyplot as plt
import seaborn as sns


# --- Carga y limpieza de datos ---
def cargar_datos(nombre_archivo):
    # Detecta la ruta del script actual
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_excel = os.path.join(ruta_base, nombre_archivo)

    if not os.path.exists(ruta_excel):
        raise FileNotFoundError(f"⚠️ Archivo no encontrado: {ruta_excel}")

    # Carga la hoja correcta
    df = pd.read_excel(ruta_excel, sheet_name='Desarrollo')
    df.columns = df.columns.str.strip()
    df.drop_duplicates(inplace=True)
    df.drop(columns=['Id_Cliente'], inplace=True)
    return df


# --- Codificación (Devuelve el encoder entrenado) ---
def codificar_target(df_train, df_test, columna='Nivel_Educacional'):
    # Creamos el encoder de Target
    encoder = TargetEncoder(cols=[columna])
    # fit solo con los datos de entrenamiento
    encoder.fit(df_train.drop('Default', axis=1), df_train['Default']) 
    
    X_train_encoded = encoder.transform(df_train.drop('Default', axis=1))
    X_test_encoded = encoder.transform(df_test.drop('Default', axis=1))
    
    # Devolvemos el encoder para guardarlo en .pkl
    return X_train_encoded, X_test_encoded, encoder 


# --- Entrenamiento de modelos (Ambos Scikit-learn) ---
def entrenar_modelos(X_train_encoded, y_train):
    # Modelo 1: Árbol de Decisión
    tree_model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=75, random_state=21)
    tree_model.fit(X_train_encoded, y_train)

    # Modelo 2: Regresión Logística de Scikit-learn
    # Usamos 'class_weight=balanced' ya que el dataset está desbalanceado.
    modelo_logit_sklearn = LogisticRegression(
        random_state=21, 
        max_iter=1000, 
        class_weight='balanced' 
    ) 
    modelo_logit_sklearn.fit(X_train_encoded, y_train)

    # Devolvemos ambos modelos de Scikit-learn
    return modelo_logit_sklearn, tree_model

# --- Evaluación por F1 (Unificada para Scikit-learn) ---
def evaluar_modelo_por_f1(modelo, X, y, muestra, tipo):
    # Ambos modelos de sklearn usan predict_proba
    probs = modelo.predict_proba(X)[:, 1]

    fpr, tpr, thresholds = roc_curve(y, probs)
    auc = roc_auc_score(y, probs)
    ks = max(tpr - fpr)
    
    # Encontramos el umbral óptimo basado en F1-Score
    f1_scores = [f1_score(y, (probs >= thr).astype(int)) for thr in thresholds]
    idx_max = np.argmax(f1_scores)
    threshold_optimo = thresholds[idx_max]
    pred_class = (probs >= threshold_optimo).astype(int)

    acc = accuracy_score(y, pred_class)
    prec = precision_score(y, pred_class)
    rec = recall_score(y, pred_class)
    f1 = f1_score(y, pred_class)
    cm = confusion_matrix(y, pred_class)

    print(f"\n📊 {tipo.upper()} - {muestra.upper()}")
    print(f"AUC: {auc:.4f} | KS: {ks:.4f} | Threshold óptimo (F1): {threshold_optimo:.4f}")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print("Matriz de Confusión:")
    print(cm)

    return {
        'Modelo': tipo,
        'Muestra': muestra,
        'AUC': auc,
        'KS': ks,
        'Threshold': threshold_optimo,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1
    }

# --- Visualización ROC ---
def graficar_roc(modelo_logit, tree_model, X_train_encoded, X_test_encoded, y_train, y_test, metricas):
    plt.figure(figsize=(10, 6))
    for resultado in metricas:
        modelo = resultado['Modelo']
        muestra = resultado['Muestra']
        
        # Seleccionamos el modelo y los datos correspondientes
        model_obj = modelo_logit if 'logit' in modelo else tree_model
        X_data = X_train_encoded if muestra == 'Train' else X_test_encoded
        y_real = y_train if muestra == 'Train' else y_test
        
        probs = model_obj.predict_proba(X_data)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_real, probs)
        plt.plot(fpr, tpr, label=f'{modelo.upper()} {muestra} (AUC={resultado["AUC"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('FPR (Tasa de Falsos Positivos)')
    plt.ylabel('TPR (Tasa de Verdaderos Positivos)')
    plt.title('Curvas ROC - Comparación de Modelos')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Guardar Artefactos ---
def guardar_artefactos(modelo_a_desplegar, encoder, nombre_modelo='model.pkl', nombre_encoder='encoder.pkl'):
    """Serializa el modelo (de scikit-learn) y el codificador para el despliegue."""
    
    # Guardar el Modelo
    with open(nombre_modelo, 'wb') as file:
        pickle.dump(modelo_a_desplegar, file)
    print(f"✅ Modelo para despliegue guardado como: {nombre_modelo}")
    
    # Guardar el Codificador (TargetEncoder)
    with open(nombre_encoder, 'wb') as file:
        pickle.dump(encoder, file)
    print(f"✅ Codificador (TargetEncoder) guardado como: {nombre_encoder}")


# --- Main (COMPLETO) ---
def main():
    # 1. Carga, división y Codificación
    # Asegúrate de que "Tabla Trabajo Grupal N°2.xlsx" está en el mismo directorio
    df = cargar_datos("./Tabla Trabajo Grupal N°2.xlsx")
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=21)
    
    # Recibimos también el encoder
    X_train_encoded, X_test_encoded, encoder = codificar_target(df_train, df_test)
    y_train = df_train['Default']
    y_test = df_test['Default']

    # 2. Entrenamiento
    modelo_logit, tree_model = entrenar_modelos(X_train_encoded, y_train)

    # 3. Evaluación (Muestra las métricas para la decisión)
    metricas = []
    # Usamos 'logit_sk' 
    for modelo, tipo in [(modelo_logit, 'logit_sk'), (tree_model, 'tree')]:
        for X, y, muestra in [(X_train_encoded, y_train, 'Train'), (X_test_encoded, y_test, 'Test')]:
            resultado = evaluar_modelo_por_f1(modelo, X, y, muestra, tipo)
            metricas.append(resultado)

    graficar_roc(modelo_logit, tree_model, X_train_encoded, X_test_encoded, y_train, y_test, metricas)

    metricas_df = pd.DataFrame(metricas)
    print("\n📋 Comparación de modelos:")
    print(metricas_df)
    
    # 4. Serialización (Elegimos el Logit ya que tuvo mejor AUC/F1 en la evaluación anterior)
    print("\n📦 Serializando el modelo de Regresión Logística (Logit_sk) y el Codificador...")
    guardar_artefactos(modelo_logit, encoder, nombre_modelo='model.pkl', nombre_encoder='encoder.pkl')


if __name__ == "__main__":
    main()