# --- Importar librerías necesarias ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Cargar datos ---
archivo = "Tabla Trabajo Grupal N°2.xlsx"
ruta_completa = os.path.join(os.path.dirname(__file__), archivo)

if not os.path.exists(ruta_completa):
    raise FileNotFoundError(f"Archivo no encontrado en: {ruta_completa}")
else:
    df = pd.read_excel(ruta_completa, sheet_name='Desarrollo', engine='openpyxl')
df.columns = df.columns.str.strip()

print(f'La base de datos cuenta con {df.shape[0]} registros y {df.shape[1]} columnas')

# --- Limpieza básica ---
df.drop_duplicates(inplace=True)
df.drop(columns=['Id_Cliente'], inplace=True)

print(f'Dataset limpio: {df.shape[0]} registros y {df.shape[1]} columnas')
print(f"Datos faltantes por columna:\n{df.isnull().sum()}")

# --- Análisis exploratorio ---
print(df.info())
print(df.describe())
print(df.describe(include='object'))
print(df['Nivel_Educacional'].value_counts())

# --- Histograma de variables numéricas ---
variables_numericas = [
    'Edad', 'Años_Trabajando', 'Ingresos', 'Deuda_Comercial',
    'Deuda_Credito', 'Otras_Deudas', 'Ratio_Ingresos_Deudas', 'Default'
]

df_histograma = df[variables_numericas].melt(var_name='variable', value_name='valor')

g = sns.FacetGrid(df_histograma, col='variable', col_wrap=2, sharex=False, sharey=False, height=4, aspect=1.2)
g.map(sns.histplot, 'valor', bins=20, kde=True)
g.set_axis_labels("Valor", "Frecuencia")
g.set_titles(col_template="{col_name}")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Distribución de Variables Numéricas', fontsize=16)
plt.tight_layout()
plt.show()

# --- Gráfico de Nivel Educacional ---
conteo = df['Nivel_Educacional'].value_counts().reset_index()
conteo.columns = ['Nivel_Educacional', 'Cantidad']

plt.figure(figsize=(12,6))
sns.barplot(data=conteo, x='Nivel_Educacional', y='Cantidad', color='skyblue')
plt.title('Distribución de los Clientes por Nivel de Educación', y=1.02)
plt.xlabel("Nivel")
plt.ylabel("N° Clientes")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Configuración global ---
sns.set_style("whitegrid")

# --- Variables ---
variables_numericas = [
    'Edad', 'Años_Trabajando', 'Ingresos', 'Deuda_Comercial',
    'Deuda_Credito', 'Otras_Deudas', 'Ratio_Ingresos_Deudas'
]
variables_categoricas = ['Nivel_Educacional']
variable_objetivo = 'Default'

# --- Función: Gráficos numéricos vs Default ---
def graficos_numericos_vs_objetivo(df, variables, objetivo):
    fig, axes = plt.subplots(nrows=len(variables), ncols=2, figsize=(14, 6 * len(variables)))
    for i, col in enumerate(variables):
        sns.violinplot(x=objetivo, y=col, data=df, ax=axes[i, 0])
        axes[i, 0].set_title(f'Distribución de {col} por {objetivo}')
        sns.boxplot(x=objetivo, y=col, data=df, ax=axes[i, 1])
        axes[i, 1].set_title(f'Distribución de {col} por {objetivo} (Boxplot)')
    plt.tight_layout()
    plt.show()

# --- Función: Gráficos categóricos vs Default ---
def graficos_categoricos_vs_objetivo(df, variables, objetivo):
    for col in variables:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=col, hue=objetivo, data=df)
        plt.title(f'Conteo de {col} por {objetivo}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# --- Función: Mapa de calor de correlaciones ---
def mapa_correlacion(df, variables):
    corr_matrix = df[variables].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='viridis',
        fmt=".2f",
        linewidths=.5
    )
    plt.title('Mapa de Calor de Correlaciones de Variables Numéricas', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# --- Ejecución de funciones ---
graficos_numericos_vs_objetivo(df, variables_numericas, variable_objetivo)
graficos_categoricos_vs_objetivo(df, variables_categoricas, variable_objetivo)
mapa_correlacion(df, variables_numericas + [variable_objetivo])
