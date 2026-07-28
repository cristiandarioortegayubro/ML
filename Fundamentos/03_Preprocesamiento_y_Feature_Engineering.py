# Databricks notebook source
# DBTITLE 1,# Preprocesamiento de Datos y Feature Engineering
# MAGIC %md
# MAGIC # Preprocesamiento de Datos y Feature Engineering
# MAGIC
# MAGIC ## 🎯 Objetivo
# MAGIC
# MAGIC En este notebook aprenderemos las técnicas **más importantes** para preparar datos antes de entrenar modelos de Machine Learning.
# MAGIC
# MAGIC ### 📌 Tema Central
# MAGIC
# MAGIC > **"Los modelos son tan buenos como los datos que reciben."**
# MAGIC
# MAGIC El 80% del trabajo en un proyecto de ML es **preparación de datos**. Un modelo mediocre con datos bien preparados supera a un modelo sofisticado con datos sucios.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Contenido
# MAGIC
# MAGIC ### Parte 1: Preprocesamiento de Datos
# MAGIC 1. **Limpieza de datos**
# MAGIC    - Valores faltantes
# MAGIC    - Duplicados
# MAGIC    - Outliers
# MAGIC
# MAGIC 2. **Transformaciones**
# MAGIC    - Encoding de variables categóricas
# MAGIC    - Scaling y normalización
# MAGIC    - Transformaciones matemáticas
# MAGIC
# MAGIC ### Parte 2: Feature Engineering
# MAGIC 3. **Creación de features**
# MAGIC    - Features de fecha/tiempo
# MAGIC    - Interacciones
# MAGIC    - Agregaciones
# MAGIC
# MAGIC 4. **Selección de features**
# MAGIC    - Correlación
# MAGIC    - Importancia de features
# MAGIC    - Reducción de dimensionalidad
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🛠️ Herramientas
# MAGIC
# MAGIC * **Pandas**: Manipulación de datos
# MAGIC * **NumPy**: Operaciones numéricas
# MAGIC * **Scikit-learn**: Transformers (StandardScaler, OneHotEncoder, etc.)
# MAGIC * **PySpark**: Para big data (opcional en este notebook)

# COMMAND ----------

# DBTITLE 1,## Parte 1: Limpieza de Datos
# MAGIC %md
# MAGIC ## Parte 1: Limpieza de Datos
# MAGIC
# MAGIC ### 🧹 1.1 Valores Faltantes (Missing Values)
# MAGIC
# MAGIC Los valores faltantes son uno de los problemas más comunes en datos reales.
# MAGIC
# MAGIC #### Estrategias:
# MAGIC
# MAGIC | Método | Cuándo usarlo | Ventajas | Desventajas |
# MAGIC |--------|----------------|----------|-------------|
# MAGIC | **Eliminar filas** | <5% missing | Simple | Pierde información |
# MAGIC | **Imputar con media/mediana** | Numéricos | Preserva tamaño | Introduce bias |
# MAGIC | **Imputar con moda** | Categóricos | Simple | Pierde variabilidad |
# MAGIC | **Imputar con modelo** | >10% missing | Preciso | Complejo |
# MAGIC | **Flag de missing** | Patrón importante | Info adicional | Más features |
# MAGIC
# MAGIC #### Fórmulas:
# MAGIC
# MAGIC **Media (Mean):**
# MAGIC $$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$
# MAGIC
# MAGIC **Mediana (Median):**
# MAGIC $$\text{mediana} = \begin{cases} 
# MAGIC x_{(n+1)/2} & \text{si } n \text{ es impar} \\
# MAGIC \frac{x_{n/2} + x_{(n/2)+1}}{2} & \text{si } n \text{ es par}
# MAGIC \end{cases}$$
# MAGIC
# MAGIC **Moda (Mode):**
# MAGIC $$\text{moda} = \text{valor más frecuente}$$

# COMMAND ----------

# DBTITLE 1,Ejemplo: Manejo de valores faltantes
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Crear dataset con valores faltantes
np.random.seed(42)
df = pd.DataFrame({
    'edad': [25, 30, np.nan, 45, np.nan, 35, 50, np.nan],
    'salario': [50000, 60000, np.nan, 80000, 70000, np.nan, 90000, 75000],
    'ciudad': ['NY', 'LA', np.nan, 'CHI', 'NY', 'LA', np.nan, 'CHI'],
    'compro': [1, 0, 1, 1, 0, np.nan, 1, 0]
})

print("┌──────────────────────────────────────────────────┐")
print("│        DATASET ORIGINAL CON VALORES FALTANTES       │")
print("└──────────────────────────────────────────────────┘")
print(df)
print("\n📉 Valores faltantes por columna:")
print(df.isnull().sum())
print(f"\n📊 Porcentaje de valores faltantes:")
print((df.isnull().sum() / len(df) * 100).round(2))

# Estrategia 1: Imputar numéricas con mediana
print("\n" + "="*50)
print("🔧 ESTRATEGIA 1: Imputar numéricas con mediana")
print("="*50)

df_imputed = df.copy()
imputer_median = SimpleImputer(strategy='median')
df_imputed[['edad', 'salario']] = imputer_median.fit_transform(df[['edad', 'salario']])

print("\n✅ Resultado:")
print(df_imputed[['edad', 'salario']])
print(f"\nMediana edad: {df['edad'].median():.1f}")
print(f"Mediana salario: ${df['salario'].median():,.0f}")

# Estrategia 2: Imputar categóricas con moda
print("\n" + "="*50)
print("🔧 ESTRATEGIA 2: Imputar categóricas con moda")
print("="*50)

df_imputed['ciudad'].fillna(df['ciudad'].mode()[0], inplace=True)
df_imputed['compro'].fillna(df['compro'].mode()[0], inplace=True)

print("\n✅ Dataset completo (sin valores faltantes):")
print(df_imputed)
print(f"\n✅ Total valores faltantes: {df_imputed.isnull().sum().sum()}")

# COMMAND ----------

# DBTITLE 1,## 1.2 Outliers (Valores Atípicos)
# MAGIC %md
# MAGIC ## 1.2 Outliers (Valores Atípicos)
# MAGIC
# MAGIC ### 🔍 Definición
# MAGIC
# MAGIC Outliers son observaciones que se desvían significativamente del resto de los datos.
# MAGIC
# MAGIC ### Métodos de Detección:
# MAGIC
# MAGIC #### 1️⃣ **Método IQR (Interquartile Range)**
# MAGIC
# MAGIC $$\text{IQR} = Q_3 - Q_1$$
# MAGIC
# MAGIC $$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$$
# MAGIC $$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$
# MAGIC
# MAGIC Donde:
# MAGIC * $Q_1$ = Primer cuartil (25%)
# MAGIC * $Q_3$ = Tercer cuartil (75%)
# MAGIC
# MAGIC #### 2️⃣ **Método Z-Score**
# MAGIC
# MAGIC $$z = \frac{x - \mu}{\sigma}$$
# MAGIC
# MAGIC Outlier si $|z| > 3$ (o 2.5 según el umbral)
# MAGIC
# MAGIC ### Estrategias de Manejo:
# MAGIC
# MAGIC * **Eliminar**: Si son errores de medición
# MAGIC * **Transformar**: Log, sqrt para reducir impacto
# MAGIC * **Cap (Winsorization)**: Reemplazar con percentiles extremos
# MAGIC * **Dejar**: Si son valores legítimos importantes

# COMMAND ----------

# DBTITLE 1,Ejemplo: Detección y manejo de outliers
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Crear datos con outliers
np.random.seed(42)
data_normal = np.random.normal(100, 15, 95)
outliers = np.array([200, 210, 220, 5, 10])  # Outliers
data = np.concatenate([data_normal, outliers])

df_outliers = pd.DataFrame({'valor': data})

print("┌──────────────────────────────────────────────────┐")
print("│         DETECCIÓN DE OUTLIERS CON IQR              │")
print("└──────────────────────────────────────────────────┘")

# Método 1: IQR
Q1 = df_outliers['valor'].quantile(0.25)
Q3 = df_outliers['valor'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\n📊 Estadísticas:")
print(f"  Q1 (25%): {Q1:.2f}")
print(f"  Q3 (75%): {Q3:.2f}")
print(f"  IQR: {IQR:.2f}")
print(f"\n🚨 Límites para outliers:")
print(f"  Lower Bound: {lower_bound:.2f}")
print(f"  Upper Bound: {upper_bound:.2f}")

# Detectar outliers
outliers_mask = (df_outliers['valor'] < lower_bound) | (df_outliers['valor'] > upper_bound)
outliers_detected = df_outliers[outliers_mask]

print(f"\n⚠️  Outliers detectados: {len(outliers_detected)}")
print(f"  Valores: {outliers_detected['valor'].values}")

# Método 2: Z-Score
print("\n" + "="*50)
print("🔧 MÉTODO 2: Z-SCORE")
print("="*50)

z_scores = np.abs(stats.zscore(df_outliers['valor']))
outliers_zscore = df_outliers[z_scores > 3]

print(f"\n⚠️  Outliers con |z| > 3: {len(outliers_zscore)}")
print(f"  Valores: {outliers_zscore['valor'].values}")

# Estrategia: Winsorization (cap outliers)
print("\n" + "="*50)
print("🔧 ESTRATEGIA: WINSORIZATION (CAP)")
print("="*50)

df_capped = df_outliers.copy()
df_capped['valor'] = np.clip(df_capped['valor'], lower_bound, upper_bound)

print(f"\n✅ Outliers reemplazados con límites:")
print(f"  Antes: min={df_outliers['valor'].min():.2f}, max={df_outliers['valor'].max():.2f}")
print(f"  Después: min={df_capped['valor'].min():.2f}, max={df_capped['valor'].max():.2f}")

print(f"\n📊 Comparación de medias:")
print(f"  Con outliers: {df_outliers['valor'].mean():.2f}")
print(f"  Sin outliers: {df_capped['valor'].mean():.2f}")
print(f"  Diferencia: {abs(df_outliers['valor'].mean() - df_capped['valor'].mean()):.2f}")

# COMMAND ----------

# DBTITLE 1,## Parte 2: Transformaciones
# MAGIC %md
# MAGIC ## Parte 2: Transformaciones de Datos
# MAGIC
# MAGIC ### 2.1 Encoding de Variables Categóricas
# MAGIC
# MAGIC Los modelos de ML requieren inputs numéricos. Debemos convertir variables categóricas.
# MAGIC
# MAGIC | Método | Uso | Ejemplo | Ventajas | Desventajas |
# MAGIC |--------|-----|---------|----------|-------------|
# MAGIC | **Label Encoding** | Ordinales | Educación: Primaria=1, Secundaria=2, Universidad=3 | Simple, compacto | Implica orden |
# MAGIC | **One-Hot Encoding** | Nominales de baja cardinalidad | Ciudad: NY, LA, CHI → 3 columnas binarias | Sin orden implícito | Muchas columnas |
# MAGIC | **Target Encoding** | Alta cardinalidad | Reemplazar por media del target | Compacto | Riesgo de overfitting |
# MAGIC | **Frequency Encoding** | Alta cardinalidad | Reemplazar por frecuencia | Simple | Pierde identidad |
# MAGIC
# MAGIC ### 2.2 Scaling y Normalización
# MAGIC
# MAGIC Algunos algoritmos (KNN, SVM, redes neuronales) son sensibles a la escala de features.
# MAGIC
# MAGIC #### Standardization (Z-Score Normalization)
# MAGIC
# MAGIC $$x' = \frac{x - \mu}{\sigma}$$
# MAGIC
# MAGIC * Media = 0, Desviación estándar = 1
# MAGIC * **Usar cuando**: Features siguen distribución normal
# MAGIC * **Algoritmos**: SVM, Redes Neuronales, PCA
# MAGIC
# MAGIC #### Min-Max Normalization
# MAGIC
# MAGIC $$x' = \frac{x - \min(x)}{\max(x) - \min(x)}$$
# MAGIC
# MAGIC * Rango [0, 1]
# MAGIC * **Usar cuando**: Quieres un rango específico
# MAGIC * **Sensible a outliers**
# MAGIC
# MAGIC #### Robust Scaling
# MAGIC
# MAGIC $$x' = \frac{x - \text{median}(x)}{\text{IQR}}$$
# MAGIC
# MAGIC * Robusto a outliers
# MAGIC * Usa mediana y IQR
# MAGIC * **Usar cuando**: Hay outliers

# COMMAND ----------

# DBTITLE 1,Ejemplo: Encoding y Scaling
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler

# Dataset de ejemplo
df_transform = pd.DataFrame({
    'ciudad': ['NY', 'LA', 'CHI', 'NY', 'LA', 'CHI'],
    'educacion': ['Secundaria', 'Universidad', 'Primaria', 'Universidad', 'Secundaria', 'Universidad'],
    'edad': [25, 35, 22, 40, 28, 38],
    'salario': [50000, 75000, 35000, 85000, 55000, 80000]
})

print("┌──────────────────────────────────────────────────┐")
print("│          ENCODING Y SCALING - EJEMPLOS            │")
print("└──────────────────────────────────────────────────┘")

print("\n📊 Dataset original:")
print(df_transform)

# 1. Label Encoding (para ordinales)
print("\n" + "="*50)
print("🔧 1. LABEL ENCODING (Educación - Ordinal)")
print("="*50)

educacion_mapping = {'Primaria': 1, 'Secundaria': 2, 'Universidad': 3}
df_transform['educacion_encoded'] = df_transform['educacion'].map(educacion_mapping)

print("\n✅ Resultado:")
print(df_transform[['educacion', 'educacion_encoded']])

# 2. One-Hot Encoding (para nominales)
print("\n" + "="*50)
print("🔧 2. ONE-HOT ENCODING (Ciudad - Nominal)")
print("="*50)

df_onehot = pd.get_dummies(df_transform['ciudad'], prefix='ciudad')
print("\n✅ Resultado:")
print(df_onehot)

# 3. Standardization
print("\n" + "="*50)
print("🔧 3. STANDARDIZATION (Z-Score)")
print("="*50)

scaler_std = StandardScaler()
df_transform['salario_std'] = scaler_std.fit_transform(df_transform[['salario']])

print("\n✅ Resultado (Salario):")
print(df_transform[['salario', 'salario_std']])
print(f"\nMedia: {df_transform['salario_std'].mean():.10f}")
print(f"Std: {df_transform['salario_std'].std():.2f}")

# 4. Min-Max Normalization
print("\n" + "="*50)
print("🔧 4. MIN-MAX NORMALIZATION")
print("="*50)

scaler_minmax = MinMaxScaler()
df_transform['edad_minmax'] = scaler_minmax.fit_transform(df_transform[['edad']])

print("\n✅ Resultado (Edad):")
print(df_transform[['edad', 'edad_minmax']])
print(f"\nMin: {df_transform['edad_minmax'].min()}")
print(f"Max: {df_transform['edad_minmax'].max()}")

# Comparación final
print("\n" + "="*50)
print("📊 DATASET TRANSFORMADO COMPLETO")
print("="*50)
print("\n")
print(df_transform)

# COMMAND ----------

# DBTITLE 1,## Parte 3: Feature Engineering
# MAGIC %md
# MAGIC ## Parte 3: Feature Engineering
# MAGIC
# MAGIC ### 🎯 Definición
# MAGIC
# MAGIC > **Feature Engineering** es el arte de crear nuevas features a partir de datos existentes para mejorar el performance del modelo.
# MAGIC
# MAGIC ### 💡 Regla de Oro
# MAGIC
# MAGIC **"Features beat algorithms"** - Mejores features tienen más impacto que mejores algoritmos.
# MAGIC
# MAGIC ### Tipos de Features
# MAGIC
# MAGIC #### 1️⃣ **Features de Fecha/Tiempo**
# MAGIC
# MAGIC De una columna `fecha`:
# MAGIC * Año, mes, día
# MAGIC * Día de la semana, fin de semana
# MAGIC * Trimestre, semestre
# MAGIC * Hora, minuto (para timestamps)
# MAGIC * Días desde evento
# MAGIC
# MAGIC #### 2️⃣ **Interacciones**
# MAGIC
# MAGIC Combinar dos o más features:
# MAGIC * Producto: `precio * cantidad = gasto_total`
# MAGIC * Ratio: `ingresos / gastos = ratio_ahorro`
# MAGIC * Diferencia: `fecha_entrega - fecha_pedido = dias_envio`
# MAGIC
# MAGIC #### 3️⃣ **Agregaciones**
# MAGIC
# MAGIC Por grupos:
# MAGIC * `promedio_compra_por_cliente`
# MAGIC * `total_ventas_por_ciudad`
# MAGIC * `max_precio_por_categoria`
# MAGIC
# MAGIC #### 4️⃣ **Binning**
# MAGIC
# MAGIC Convertir continuas en categóricas:
# MAGIC * Edad → Grupos etáreos: 0-18, 19-35, 36-65, 65+
# MAGIC * Salario → Rangos: Bajo, Medio, Alto
# MAGIC
# MAGIC #### 5️⃣ **Transformaciones Matemáticas**
# MAGIC
# MAGIC * **Log**: $\log(x)$ - Para distribución sesgada
# MAGIC * **Sqrt**: $\sqrt{x}$ - Reduce impacto de valores grandes
# MAGIC * **Polynomial**: $x^2, x^3$ - Captura relaciones no-lineales

# COMMAND ----------

# DBTITLE 1,Ejemplo: Feature Engineering completo
# Dataset de ejemplo: Transacciones de e-commerce
df_fe = pd.DataFrame({
    'fecha_compra': pd.to_datetime(['2024-01-15', '2024-02-20', '2024-03-10', '2024-01-22', '2024-02-14']),
    'precio': [50, 150, 30, 200, 75],
    'cantidad': [2, 1, 5, 1, 3],
    'cliente_id': ['C1', 'C2', 'C1', 'C3', 'C1'],
    'categoria': ['Electrónica', 'Ropa', 'Libros', 'Electrónica', 'Libros'],
    'edad_cliente': [25, 45, 25, 60, 25]
})

print("┌──────────────────────────────────────────────────┐")
print("│            FEATURE ENGINEERING - EJEMPLOS          │")
print("└──────────────────────────────────────────────────┘")

print("\n📊 Dataset original:")
print(df_fe)

# 1. Features de fecha/tiempo
print("\n" + "="*50)
print("🔧 1. FEATURES DE FECHA/TIEMPO")
print("="*50)

df_fe['mes'] = df_fe['fecha_compra'].dt.month
df_fe['dia_semana'] = df_fe['fecha_compra'].dt.dayofweek
df_fe['es_fin_semana'] = (df_fe['dia_semana'] >= 5).astype(int)
df_fe['trimestre'] = df_fe['fecha_compra'].dt.quarter

print("\n✅ Features creadas:")
print(df_fe[['fecha_compra', 'mes', 'dia_semana', 'es_fin_semana', 'trimestre']])

# 2. Interacciones (Features derivadas)
print("\n" + "="*50)
print("🔧 2. INTERACCIONES")
print("="*50)

df_fe['gasto_total'] = df_fe['precio'] * df_fe['cantidad']
df_fe['precio_por_unidad'] = df_fe['precio']  # Ya está, pero podría ser gasto_total / cantidad

print("\n✅ Features de interacción:")
print(df_fe[['precio', 'cantidad', 'gasto_total']])

# 3. Agregaciones por cliente
print("\n" + "="*50)
print("🔧 3. AGREGACIONES POR CLIENTE")
print("="*50)

agg_cliente = df_fe.groupby('cliente_id').agg({
    'gasto_total': ['sum', 'mean', 'count'],
    'cantidad': 'sum'
}).reset_index()
agg_cliente.columns = ['cliente_id', 'total_gastado', 'gasto_promedio', 'num_compras', 'total_items']

df_fe = df_fe.merge(agg_cliente, on='cliente_id', how='left')

print("\n✅ Features de agregación:")
print(df_fe[['cliente_id', 'total_gastado', 'gasto_promedio', 'num_compras']])

# 4. Binning (edad en grupos)
print("\n" + "="*50)
print("🔧 4. BINNING (Edad en grupos)")
print("="*50)

df_fe['grupo_edad'] = pd.cut(df_fe['edad_cliente'], 
                              bins=[0, 18, 35, 65, 100], 
                              labels=['Joven', 'Adulto', 'Senior', 'Anciano'])

print("\n✅ Binning de edad:")
print(df_fe[['edad_cliente', 'grupo_edad']])

# 5. Transformaciones matemáticas
print("\n" + "="*50)
print("🔧 5. TRANSFORMACIONES MATEMÁTICAS")
print("="*50)

df_fe['log_precio'] = np.log1p(df_fe['precio'])  # log(1+x) para evitar log(0)
df_fe['sqrt_cantidad'] = np.sqrt(df_fe['cantidad'])

print("\n✅ Transformaciones:")
print(df_fe[['precio', 'log_precio', 'cantidad', 'sqrt_cantidad']])

# Resumen final
print("\n" + "="*50)
print("🏆 DATASET CON TODAS LAS FEATURES")
print("="*50)
print(f"\n📊 Features originales: 6")
print(f"🎉 Features nuevas: {len(df_fe.columns) - 6}")
print(f"🔢 Total features: {len(df_fe.columns)}")
print("\nColumnas:")
for i, col in enumerate(df_fe.columns, 1):
    print(f"  {i}. {col}")

# COMMAND ----------

# DBTITLE 1,## Parte 4: Selección de Features
# MAGIC %md
# MAGIC ## Parte 4: Selección de Features
# MAGIC
# MAGIC ### ⚠️ El Problema de Muchas Features
# MAGIC
# MAGIC **"Curse of Dimensionality"** - Demasiadas features:
# MAGIC * ❌ Overfitting
# MAGIC * ❌ Mayor tiempo de entrenamiento
# MAGIC * ❌ Modelos difíciles de interpretar
# MAGIC * ❌ Ruido en los datos
# MAGIC
# MAGIC ### Métodos de Selección
# MAGIC
# MAGIC #### 1️⃣ **Correlación**
# MAGIC
# MAGIC **Eliminar features altamente correlacionadas:**
# MAGIC * Si $\text{corr}(X_1, X_2) > 0.9$ → Eliminar una
# MAGIC * Evita redundancia
# MAGIC
# MAGIC **Correlación de Pearson:**
# MAGIC $$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$
# MAGIC
# MAGIC $r \in [-1, 1]$:
# MAGIC * $r = 1$: Correlación positiva perfecta
# MAGIC * $r = 0$: Sin correlación
# MAGIC * $r = -1$: Correlación negativa perfecta
# MAGIC
# MAGIC #### 2️⃣ **Feature Importance**
# MAGIC
# MAGIC Modelos basados en árboles (Random Forest, XGBoost) proveen importancia de features:
# MAGIC * **Gini Importance**: Cuánto reduce la impureza
# MAGIC * **Permutation Importance**: Caída en accuracy al permutar feature
# MAGIC
# MAGIC #### 3️⃣ **Recursive Feature Elimination (RFE)**
# MAGIC
# MAGIC 1. Entrenar modelo con todas las features
# MAGIC 2. Rankear features por importancia
# MAGIC 3. Eliminar la menos importante
# MAGIC 4. Repetir hasta N features
# MAGIC
# MAGIC #### 4️⃣ **Variance Threshold**
# MAGIC
# MAGIC Eliminar features con baja varianza:
# MAGIC * Si $\text{Var}(X) < \text{threshold}$ → Eliminar
# MAGIC * Features constantes o casi-constantes no aportan

# COMMAND ----------

# DBTITLE 1,Ejemplo: Selección de features
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Crear dataset para clasificación
np.random.seed(42)
n = 100

df_selection = pd.DataFrame({
    'feature1': np.random.randn(n),
    'feature2': np.random.randn(n) * 2,
    'feature3': np.random.randn(n) * 0.1,  # Baja varianza
    'feature4': np.ones(n),  # Constante
    'feature5': np.random.randn(n),
})

# Feature 2 es altamente correlacionada con feature 1
df_selection['feature2'] = df_selection['feature1'] * 0.9 + np.random.randn(n) * 0.1

# Target
df_selection['target'] = (df_selection['feature1'] + df_selection['feature5'] > 0).astype(int)

print("┌──────────────────────────────────────────────────┐")
print("│         SELECCIÓN DE FEATURES - EJEMPLOS           │")
print("└──────────────────────────────────────────────────┘")

# 1. Variance Threshold
print("\n" + "="*50)
print("🔧 1. VARIANCE THRESHOLD")
print("="*50)

features = df_selection.drop('target', axis=1)
print("\n📊 Varianza de cada feature:")
for col in features.columns:
    print(f"  {col}: {features[col].var():.4f}")

var_threshold = VarianceThreshold(threshold=0.05)
features_high_var = var_threshold.fit_transform(features)
selected_features = features.columns[var_threshold.get_support()]

print(f"\n✅ Features seleccionadas (varianza > 0.05): {list(selected_features)}")
print(f"❌ Features eliminadas: {list(set(features.columns) - set(selected_features))}")

# 2. Correlación
print("\n" + "="*50)
print("🔧 2. MATRIZ DE CORRELACIÓN")
print("="*50)

corr_matrix = features.corr()
print("\n📊 Matriz de correlación:")
print(corr_matrix.round(2))

print("\n⚠️  Pares altamente correlacionados (|r| > 0.8):")
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            print(f"  {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")

# 3. Feature Importance con Random Forest
print("\n" + "="*50)
print("🔧 3. FEATURE IMPORTANCE (Random Forest)")
print("="*50)

X = df_selection.drop('target', axis=1)
y = df_selection['target']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🏆 Feature Importance:")
print(importances.to_string(index=False))

print(f"\n✅ Top 3 features más importantes:")
for i, row in importances.head(3).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print("\n\n📝 CONCLUSIONES:")
print("="*50)
print("✅ Eliminar features con baja varianza (feature3, feature4)")
print("✅ Eliminar una de las features correlacionadas (feature1 o feature2)")
print("✅ Priorizar features con alta importancia (feature1, feature5)")
print("\n🎯 Features finales recomendadas: feature1, feature5")

# COMMAND ----------

# DBTITLE 1,## 📝 Conclusiones y Mejores Prácticas
# MAGIC %md
# MAGIC ## 📝 Conclusiones y Mejores Prácticas
# MAGIC
# MAGIC ### 🏆 Key Takeaways
# MAGIC
# MAGIC 1. **El 80% del trabajo en ML es preparación de datos**
# MAGIC    - Limpieza, transformaciones, feature engineering
# MAGIC    - Modelos mediocres con buenos datos > Modelos sofisticados con datos sucios
# MAGIC
# MAGIC 2. **No existe una "receta única"**
# MAGIC    - Depende del problema, datos y algoritmo
# MAGIC    - Experimentar con diferentes técnicas
# MAGIC
# MAGIC 3. **Feature Engineering es un arte**
# MAGIC    - Requiere domain knowledge
# MAGIC    - Creatividad y experimentación
# MAGIC    - "Features beat algorithms"
# MAGIC
# MAGIC 4. **Menos es más**
# MAGIC    - Eliminar features redundantes o irrelevantes
# MAGIC    - Evitar overfitting
# MAGIC    - Modelos más interpretables
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ✅ Checklist de Preprocesamiento
# MAGIC
# MAGIC **Antes de entrenar cualquier modelo:**
# MAGIC
# MAGIC ☐ **Explorar datos**
# MAGIC   - Estadísticas descriptivas
# MAGIC   - Visualizaciones
# MAGIC   - Identificar problemas
# MAGIC
# MAGIC ☐ **Limpieza**
# MAGIC   - Valores faltantes → Imputar o eliminar
# MAGIC   - Duplicados → Eliminar
# MAGIC   - Outliers → Manejar apropiadamente
# MAGIC
# MAGIC ☐ **Transformaciones**
# MAGIC   - Encoding categóricas → Label, One-Hot, Target
# MAGIC   - Scaling numéricas → StandardScaler, MinMaxScaler
# MAGIC   - Transformaciones matemáticas si es necesario
# MAGIC
# MAGIC ☐ **Feature Engineering**
# MAGIC   - Crear features de dominio
# MAGIC   - Interacciones
# MAGIC   - Agregaciones
# MAGIC
# MAGIC ☐ **Selección de Features**
# MAGIC   - Eliminar baja varianza
# MAGIC   - Eliminar alta correlación
# MAGIC   - Feature importance
# MAGIC
# MAGIC ☐ **Validar**
# MAGIC   - Train/test split ANTES de preprocesar
# MAGIC   - Fit en train, transform en test
# MAGIC   - Evitar data leakage
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚠️ Errores Comunes a Evitar
# MAGIC
# MAGIC 1. **Data Leakage**
# MAGIC    ❌ Fit/transform en todo el dataset
# MAGIC    ✅ Fit en train, transform en train y test por separado
# MAGIC
# MAGIC 2. **Imputar antes de split**
# MAGIC    ❌ Imputar con media de todo el dataset
# MAGIC    ✅ Split primero, luego imputar con media del train
# MAGIC
# MAGIC 3. **Eliminar outliers sin justificación**
# MAGIC    ❌ Eliminar automáticamente todos los outliers
# MAGIC    ✅ Analizar si son errores o valores legítimos
# MAGIC
# MAGIC 4. **One-Hot Encoding de alta cardinalidad**
# MAGIC    ❌ 100+ categorías → 100+ columnas
# MAGIC    ✅ Usar Target Encoding o Frequency Encoding
# MAGIC
# MAGIC 5. **Scaling innecesario**
# MAGIC    ❌ Scaling en árboles de decisión (no necesario)
# MAGIC    ✅ Scaling solo para algoritmos sensibles (SVM, KNN, NN)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📚 Recursos Adicionales
# MAGIC
# MAGIC **Libros:**
# MAGIC * **"Feature Engineering for Machine Learning"** - Alice Zheng
# MAGIC * **"Python for Data Analysis"** - Wes McKinney
# MAGIC
# MAGIC **Cursos:**
# MAGIC * Kaggle Learn - Feature Engineering
# MAGIC * Fast.ai - Practical Deep Learning
# MAGIC
# MAGIC **Herramientas:**
# MAGIC * **Pandas Profiling**: EDA automático
# MAGIC * **featuretools**: Feature engineering automatizado
# MAGIC * **category_encoders**: Múltiples métodos de encoding
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚀 Próximos Pasos
# MAGIC
# MAGIC **En el siguiente notebook** (`04_Evaluacion_y_Validacion.ipynb`):
# MAGIC * Métricas de evaluación
# MAGIC * Validación cruzada
# MAGIC * Overfitting y Underfitting
# MAGIC * Bias-Variance Tradeoff
# MAGIC
# MAGIC **Luego:**
# MAGIC * Aprendizaje Supervisado (algoritmos específicos)
# MAGIC * Aprendizaje No Supervisado
# MAGIC * Aprendizaje por Refuerzo
# MAGIC * AutoML y MLOps
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎉 ¡Felicidades!
# MAGIC
# MAGIC Ahora dominas las técnicas fundamentales de **Preprocesamiento y Feature Engineering**.
# MAGIC
# MAGIC Estos conocimientos son la base para **cualquier proyecto de Machine Learning exitoso**. 🚀

# COMMAND ----------

