# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Título y Caso de Negocio
# MAGIC %md
# MAGIC # Predicción de Precios de Propiedades Inmobiliarias usando Árbol de Decisión
# MAGIC
# MAGIC ## 🏠 Objetivo de Negocio
# MAGIC
# MAGIC En este notebook desarrollaremos un modelo de **Machine Learning para predecir el precio de venta de propiedades inmobiliarias** basándonos en sus características físicas y de ubicación.
# MAGIC
# MAGIC ### ¿Por qué es importante?
# MAGIC
# MAGIC * **Para agentes inmobiliarios**: Valorar propiedades correctamente para cerrar ventas más rápido
# MAGIC * **Para compradores**: Identificar propiedades sobrevaloradas o con buen precio
# MAGIC * **Para inversores**: Detectar oportunidades de inversión con alto potencial de revalorización
# MAGIC * **Para bancos**: Evaluar garantías hipotecarias con mayor precisión
# MAGIC
# MAGIC ### Caso de uso
# MAGIC
# MAGIC Una agencia inmobiliaria quiere:
# MAGIC 1. Estimar automáticamente el precio justo de mercado de una propiedad
# MAGIC 2. Entender qué características tienen mayor impacto en el precio
# MAGIC 3. Asesorar a vendedores sobre mejoras que incrementen el valor de la propiedad
# MAGIC 4. Detectar propiedades con precio muy por debajo o por encima del mercado
# MAGIC
# MAGIC **Resultado esperado**: Un modelo que prediga el precio de venta (variable continua) con un error aceptable para el negocio

# COMMAND ----------

# DBTITLE 1,Qué es un Árbol de Decisión para Regresión
# MAGIC %md
# MAGIC ## 🌳 ¿Qué es un Árbol de Decisión para Regresión?
# MAGIC
# MAGIC ### Concepto
# MAGIC
# MAGIC Un **Árbol de Decisión para Regresión** funciona similar a uno de clasificación, pero en lugar de predecir una clase (categoría), predice un **valor numérico continuo** (precio, temperatura, ventas, etc.).
# MAGIC
# MAGIC ### Diferencias clave con Clasificación:
# MAGIC
# MAGIC | Aspecto | Clasificación | Regresión |
# MAGIC |---------|---------------|------------|
# MAGIC | **Variable objetivo** | Categórica (Churn/No Churn) | Numérica continua (Precio $) |
# MAGIC | **Predicción en hojas** | Clase mayoritaria | Promedio de los valores |
# MAGIC | **Métricas** | Accuracy, Precision, Recall | RMSE, MAE, R² |
# MAGIC | **Criterio de split** | Gini, Entropy | Variance, MSE |
# MAGIC | **Ejemplo** | ¿Abandona el cliente? Sí/No | ¿Cuánto vale la casa? $350,000 |
# MAGIC
# MAGIC ### ¿Cómo funciona?
# MAGIC
# MAGIC 1. **Divide** el espacio de features recursivamente
# MAGIC 2. En cada nodo, pregunta: "¿Qué variable y valor de corte minimizan la varianza en los grupos resultantes?"
# MAGIC 3. En las **hojas**, la predicción es el **promedio** de los valores de entrenamiento que caen en esa hoja
# MAGIC 4. Ejemplo: "Si área > 150m² Y habitaciones > 3 → Precio promedio = $420,000"
# MAGIC
# MAGIC ### ¿Cuándo usar Árboles de Decisión para Regresión?
# MAGIC
# MAGIC ✅ **Ventajas:**
# MAGIC * **Interpretables**: Puedes explicar por qué el modelo predijo ese precio
# MAGIC * **No requieren normalización**: Variables en diferentes escalas (m², años, km) sin problema
# MAGIC * **Capturan no-linealidad**: Detectan que "área > 200m² tiene impacto exponencial en precio"
# MAGIC * **Manejo de interacciones**: "Piscina solo aumenta precio si área > 150m²"
# MAGIC
# MAGIC ⚠️ **Limitaciones:**
# MAGIC * **Propensos a overfitting**: Pueden memorizar precios de entrenamiento
# MAGIC * **No extrapolan bien**: Si entrenaste con casas de $100k-$500k, no predecirá bien $1M
# MAGIC * **Sensibles a outliers**: Un precio atípico puede sesgar una rama completa
# MAGIC
# MAGIC ### Ideal para:
# MAGIC * Predicción de precios (inmuebles, autos, productos)
# MAGIC * Estimación de demanda o ventas
# MAGIC * Valoración de activos
# MAGIC * Como modelo baseline o parte de ensembles (Random Forest, GBT)

# COMMAND ----------

# DBTITLE 1,Importar Librerías
# =============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# =============================================================================

# PySpark ML - Framework de Machine Learning distribuido de Databricks
from pyspark.ml import Pipeline  # Para encadenar transformaciones
from pyspark.ml.regression import DecisionTreeRegressor  # Algoritmo de árbol de decisión para REGRESIÓN
from pyspark.ml.feature import StringIndexer, VectorAssembler  # Para preparar datos
from pyspark.ml.evaluation import RegressionEvaluator  # Para evaluar modelos de regresión

# PySpark SQL - Para manipulación de datos
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Visualización - Para gráficos y análisis exploratorio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configuración de visualización
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("✅ Librerías importadas correctamente")

# COMMAND ----------

# DBTITLE 1,Descripción del Dataset
# MAGIC %md
# MAGIC ## 📊 Dataset: Propiedades Inmobiliarias
# MAGIC
# MAGIC ### Descripción
# MAGIC
# MAGIC Crearemos un dataset sintético que simula datos reales de propiedades inmobiliarias en venta. Este dataset contiene información sobre:
# MAGIC
# MAGIC ### Variables del Dataset
# MAGIC
# MAGIC **Características Físicas:**
# MAGIC * `propiedad_id`: Identificador único de la propiedad
# MAGIC * `area_m2`: Área construida en metros cuadrados (50-300 m²)
# MAGIC * `habitaciones`: Número de dormitorios (1-5)
# MAGIC * `banos`: Número de baños (1-4)
# MAGIC * `antiguedad_anos`: Años desde la construcción (0-50 años)
# MAGIC * `piso`: Número de piso si es departamento (0-10, 0 = casa/planta baja)
# MAGIC * `garaje`: Si tiene garaje (1 = Sí, 0 = No)
# MAGIC * `piscina`: Si tiene piscina (1 = Sí, 0 = No)
# MAGIC
# MAGIC **Ubicación:**
# MAGIC * `barrio`: Zona de la ciudad ("Centro", "Norte", "Sur", "Este", "Oeste")
# MAGIC * `distancia_centro_km`: Distancia al centro de la ciudad en kilómetros (0-20 km)
# MAGIC
# MAGIC **Variable Objetivo (a predecir):**
# MAGIC * `precio_venta`: Precio de venta en USD (100,000 - 800,000 USD)
# MAGIC
# MAGIC ### Relación con el precio:
# MAGIC
# MAGIC **Aumentan el precio:**
# MAGIC * Mayor área (m²) → Más espacio = mayor valor
# MAGIC * Más habitaciones y baños → Más funcionalidad
# MAGIC * Menor antigüedad → Propiedad más nueva
# MAGIC * Amenidades (garaje, piscina) → Mayor confort
# MAGIC * Ubicación céntrica → Menos distancia al centro
# MAGIC
# MAGIC **Disminuyen el precio:**
# MAGIC * Mayor antigüedad → Requiere mantenimiento
# MAGIC * Ubicación periférica → Menos servicios
# MAGIC * Menos área → Espacio limitado
# MAGIC
# MAGIC ### Tasa de cambio del dataset
# MAGIC * **10,000 registros** de propiedades
# MAGIC * **Distribución realista** de precios con mayoría en rango medio

# COMMAND ----------

# DBTITLE 1,Cargar y Explorar Datos
# =============================================================================
# CREACIÓN Y EXPLORACIÓN DEL DATASET
# =============================================================================

# Creamos un dataset sintético que simula propiedades inmobiliarias reales
np.random.seed(42)  # Semilla para reproducibilidad

# Número de propiedades
n_propiedades = 10000

# Función para generar precios realistas basados en características
def generar_precio(area, habitaciones, banos, antiguedad, distancia, piso, garaje, piscina, barrio):
    # Precio base proporcional al área
    precio_base = area * 2000  # $2000 por m²
    
    # Ajustes por características
    precio_base += habitaciones * 15000  # Cada habitación agrega valor
    precio_base += banos * 10000  # Cada baño agrega valor
    precio_base -= antiguedad * 1500  # La antigüedad reduce el valor
    precio_base -= distancia * 3000  # Mayor distancia al centro reduce valor
    precio_base += piso * 2000 if piso > 0 else 5000  # Casas (piso=0) valen más que deptos bajos
    precio_base += 25000 if garaje == 1 else 0
    precio_base += 40000 if piscina == 1 else 0
    
    # Ajuste por barrio
    multiplicador_barrio = {"Centro": 1.3, "Norte": 1.15, "Este": 1.0, "Sur": 0.9, "Oeste": 0.95}
    precio_base *= multiplicador_barrio.get(barrio, 1.0)
    
    # Añadir ruido aleatorio (±10%)
    ruido = np.random.uniform(0.9, 1.1)
    precio_final = precio_base * ruido
    
    # Limitar al rango realista
    return max(100000, min(800000, precio_final))

# Generar datos
data = []
for i in range(n_propiedades):
    area = int(np.random.gamma(9, 15))  # Distribución gamma (realista para áreas)
    area = max(50, min(300, area))  # Limitar rango
    habitaciones = int(np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.25, 0.35, 0.2, 0.1]))
    banos = int(np.random.choice([1, 2, 3, 4], p=[0.2, 0.45, 0.25, 0.1]))
    antiguedad = int(np.random.exponential(12))  # Mayoría propiedades relativamente nuevas
    antiguedad = min(50, antiguedad)
    distancia = round(np.random.uniform(0, 20), 2)
    piso = int(np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], p=[0.3, 0.15, 0.12, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.02, 0.01]))
    garaje = int(np.random.choice([0, 1], p=[0.3, 0.7]))
    piscina = int(np.random.choice([0, 1], p=[0.85, 0.15]))  # Piscinas son menos comunes
    barrio = np.random.choice(["Centro", "Norte", "Sur", "Este", "Oeste"])
    
    precio = generar_precio(area, habitaciones, banos, antiguedad, distancia, piso, garaje, piscina, barrio)
    
    data.append((i, area, habitaciones, banos, antiguedad, distancia, piso, garaje, piscina, barrio, round(precio, 2)))

# Definir schema
schema = StructType([
    StructField("propiedad_id", IntegerType(), False),
    StructField("area_m2", IntegerType(), True),
    StructField("habitaciones", IntegerType(), True),
    StructField("banos", IntegerType(), True),
    StructField("antiguedad_anos", IntegerType(), True),
    StructField("distancia_centro_km", DoubleType(), True),
    StructField("piso", IntegerType(), True),
    StructField("garaje", IntegerType(), True),
    StructField("piscina", IntegerType(), True),
    StructField("barrio", StringType(), True),
    StructField("precio_venta", DoubleType(), True)
])

# Crear DataFrame
df = spark.createDataFrame(data, schema)

print("=" * 80)
print("📋 PRIMERAS 10 PROPIEDADES DEL DATASET")
print("=" * 80)
display(df.limit(10))

print("\n" + "=" * 80)
print("🔍 SCHEMA DEL DATASET")
print("=" * 80)
df.printSchema()

print("\n" + "=" * 80)
print("📊 ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 80)
display(df.describe())

print("\n" + "=" * 80)
print("📈 ESTADÍSTICAS DEL PRECIO DE VENTA")
print("=" * 80)
precio_stats = df.select(
    F.min("precio_venta").alias("Precio Mínimo"),
    F.percentile_approx("precio_venta", 0.25).alias("Q1 (25%)"),
    F.percentile_approx("precio_venta", 0.5).alias("Mediana"),
    F.mean("precio_venta").alias("Precio Promedio"),
    F.percentile_approx("precio_venta", 0.75).alias("Q3 (75%)"),
    F.max("precio_venta").alias("Precio Máximo")
)
display(precio_stats)

print("\n✅ Datos cargados y explorados correctamente")

# COMMAND ----------

# DBTITLE 1,Análisis Exploratorio
# MAGIC %md
# MAGIC ## 🔎 Análisis Exploratorio de Datos (EDA) para Regresión
# MAGIC
# MAGIC ### ¿Qué buscar en Regresión?
# MAGIC
# MAGIC A diferencia de clasificación, en regresión queremos entender:
# MAGIC
# MAGIC 1. **Distribución de la variable objetivo**: ¿Es normal? ¿Tiene outliers? ¿Está sesgada?
# MAGIC 2. **Correlaciones lineales**: ¿Qué features se correlacionan fuertemente con el precio?
# MAGIC 3. **Relaciones no lineales**: ¿El impacto de una variable cambia en diferentes rangos?
# MAGIC 4. **Outliers**: Propiedades con precios extremos que pueden sesgar el modelo
# MAGIC
# MAGIC ### Preguntas clave a responder:
# MAGIC
# MAGIC * ¿Cuál es la distribución de precios? ¿Simétrica o sesgada?
# MAGIC * ¿Qué features tienen mayor correlación con el precio?
# MAGIC * ¿Existen outliers (propiedades muy caras o baratas)?
# MAGIC * ¿La relación área-precio es lineal o tiene umbrales?
# MAGIC * ¿Cómo varía el precio por barrio?
# MAGIC
# MAGIC ### Visualizaciones a crear:
# MAGIC
# MAGIC 1. **Distribución del precio**: Histograma y boxplot
# MAGIC 2. **Correlaciones**: Heatmap de correlación entre variables numéricas
# MAGIC 3. **Scatter plots**: Precio vs Área, Precio vs Habitaciones, etc.
# MAGIC 4. **Precio por categoría**: Boxplot de precio por barrio

# COMMAND ----------

# DBTITLE 1,Visualizaciones Exploratorias
# =============================================================================
# ANÁLISIS EXPLORATORIO - VISUALIZACIONES
# =============================================================================

# Convertir a Pandas para visualización (dataset pequeño, es seguro)
df_pandas = df.toPandas()

# VISUALIZACIÓN 1: Distribución del Precio de Venta
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma
axes[0].hist(df_pandas['precio_venta'], bins=50, color='skyblue', edgecolor='black')
axes[0].set_title('Distribución de Precios de Venta', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Precio de Venta (USD)')
axes[0].set_ylabel('Frecuencia')
axes[0].axvline(df_pandas['precio_venta'].mean(), color='red', linestyle='--', label=f'Promedio: ${df_pandas["precio_venta"].mean():,.0f}')
axes[0].axvline(df_pandas['precio_venta'].median(), color='green', linestyle='--', label=f'Mediana: ${df_pandas["precio_venta"].median():,.0f}')
axes[0].legend()

# Boxplot
axes[1].boxplot(df_pandas['precio_venta'], vert=True)
axes[1].set_title('Boxplot de Precios', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Precio de Venta (USD)')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("📊 Interpretación:")
print("   - Distribución relativamente simétrica indica que el precio está bien distribuido")
print("   - Outliers en boxplot son propiedades atípicamente caras o baratas\n")

# VISUALIZACIÓN 2: Correlación entre Variables Numéricas
fig, ax = plt.subplots(figsize=(10, 8))

# Seleccionar solo variables numéricas
numeric_cols = ['area_m2', 'habitaciones', 'banos', 'antiguedad_anos', 'distancia_centro_km', 'piso', 'garaje', 'piscina', 'precio_venta']
corr_matrix = df_pandas[numeric_cols].corr()

# Crear heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Matriz de Correlación', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

print("📊 Interpretación de Correlaciones:")
print("   - Valores cercanos a +1: Fuerte correlación positiva (a mayor X, mayor precio)")
print("   - Valores cercanos a -1: Fuerte correlación negativa (a mayor X, menor precio)")
print("   - Valores cercanos a 0: Sin correlación lineal\n")

# VISUALIZACIÓN 3: Scatter Plots - Precio vs Features Clave
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Precio vs Área
axes[0, 0].scatter(df_pandas['area_m2'], df_pandas['precio_venta'], alpha=0.5, s=10)
axes[0, 0].set_title('Precio vs Área')
axes[0, 0].set_xlabel('Área (m²)')
axes[0, 0].set_ylabel('Precio (USD)')
axes[0, 0].grid(alpha=0.3)

# Precio vs Habitaciones
axes[0, 1].scatter(df_pandas['habitaciones'], df_pandas['precio_venta'], alpha=0.5, s=10)
axes[0, 1].set_title('Precio vs Habitaciones')
axes[0, 1].set_xlabel('Número de Habitaciones')
axes[0, 1].set_ylabel('Precio (USD)')
axes[0, 1].grid(alpha=0.3)

# Precio vs Antigüedad
axes[1, 0].scatter(df_pandas['antiguedad_anos'], df_pandas['precio_venta'], alpha=0.5, s=10)
axes[1, 0].set_title('Precio vs Antigüedad')
axes[1, 0].set_xlabel('Antigüedad (años)')
axes[1, 0].set_ylabel('Precio (USD)')
axes[1, 0].grid(alpha=0.3)

# Precio vs Distancia al Centro
axes[1, 1].scatter(df_pandas['distancia_centro_km'], df_pandas['precio_venta'], alpha=0.5, s=10)
axes[1, 1].set_title('Precio vs Distancia al Centro')
axes[1, 1].set_xlabel('Distancia (km)')
axes[1, 1].set_ylabel('Precio (USD)')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("📊 Interpretación de Scatter Plots:")
print("   - Tendencia ascendente: A mayor X, mayor precio")
print("   - Tendencia descendente: A mayor X, menor precio")
print("   - Dispersión: Variabilidad en la relación\n")

# VISUALIZACIÓN 4: Precio por Barrio
fig, ax = plt.subplots(figsize=(10, 6))
df_pandas.boxplot(column='precio_venta', by='barrio', ax=ax)
ax.set_title('Precio de Venta por Barrio', fontsize=14, fontweight='bold')
ax.set_xlabel('Barrio')
ax.set_ylabel('Precio de Venta (USD)')
plt.suptitle('')  # Remover título por defecto de pandas
plt.tight_layout()
plt.show()

print("📊 Interpretación:")
print("   - Barrios céntricos suelen tener precios más altos")
print("   - Diferencias entre medianas indican que 'barrio' es predictivo\n")

print("✅ Análisis exploratorio completado")

# COMMAND ----------

# DBTITLE 1,Preparación de Datos
# MAGIC %md
# MAGIC ## 🔧 Preparación de Datos para Regresión
# MAGIC
# MAGIC ### Pasos de preparación:
# MAGIC
# MAGIC 1. **Codificar variables categóricas**: Convertir 'barrio' a índices numéricos
# MAGIC 2. **Vector Assembly**: Combinar todas las features en un vector
# MAGIC 3. **Sin normalización**: Los árboles de decisión NO requieren normalizar variables
# MAGIC
# MAGIC ### Features que usaremos:
# MAGIC
# MAGIC **Numéricas directas:**
# MAGIC * area_m2
# MAGIC * habitaciones
# MAGIC * banos
# MAGIC * antiguedad_anos
# MAGIC * distancia_centro_km
# MAGIC * piso
# MAGIC * garaje (0/1)
# MAGIC * piscina (0/1)
# MAGIC
# MAGIC **Categórica a codificar:**
# MAGIC * barrio (Centro, Norte, Sur, Este, Oeste)
# MAGIC
# MAGIC **Variable objetivo:**
# MAGIC * precio_venta (la que queremos predecir)

# COMMAND ----------

# DBTITLE 1,Preparar Datos
# =============================================================================
# PREPARACIÓN DE DATOS
# =============================================================================

print("🔧 CODIFICACIÓN DE VARIABLE CATEGÓRICA")
print("=" * 50)

# StringIndexer para 'barrio'
barrio_indexer = StringIndexer(
    inputCol="barrio",
    outputCol="barrio_idx"
)

df = barrio_indexer.fit(df).transform(df)

print("Ejemplo de codificación de barrio:")
df.select("barrio", "barrio_idx").distinct().orderBy("barrio_idx").show()

print("\n✅ Variable categórica codificada\n")

print("🔧 CREACIÓN DEL VECTOR DE FEATURES")
print("=" * 50)

# Lista de features
feature_columns = [
    "area_m2",
    "habitaciones",
    "banos",
    "antiguedad_anos",
    "distancia_centro_km",
    "piso",
    "garaje",
    "piscina",
    "barrio_idx"
]

# VectorAssembler
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

df_final = assembler.transform(df)

# Seleccionar columnas necesarias
# En regresión, la columna objetivo se llama "label" (convención de PySpark ML)
df_final = df_final.select("propiedad_id", "features", F.col("precio_venta").alias("label"))

print("Dataset preparado:")
df_final.printSchema()

print("\nEjemplo de vectores de features:")
df_final.select("propiedad_id", "features", "label").show(5, truncate=False)

print("\n✅ Datos preparados correctamente")
print(f"   - Total de features: {len(feature_columns)}")

# COMMAND ----------

# DBTITLE 1,División Train Test
# MAGIC %md
# MAGIC ## 📊 División de Datos: Entrenamiento y Prueba
# MAGIC
# MAGIC ### ¿Por qué dividir?
# MAGIC
# MAGIC Igual que en clasificación, necesitamos:
# MAGIC * **Train set (80%)**: Para que el modelo aprenda la relación entre features y precio
# MAGIC * **Test set (20%)**: Para evaluar cuán bien predice en propiedades nuevas
# MAGIC
# MAGIC ### Importancia en Regresión:
# MAGIC
# MAGIC * **Evitar overfitting**: El modelo podría memorizar precios exactos del entrenamiento
# MAGIC * **Medir error real**: RMSE en test indica el error promedio en USD que cometeremos
# MAGIC * **Validar generalización**: ¿El modelo funciona con propiedades que nunca vio?
# MAGIC
# MAGIC ### División:
# MAGIC * **80% train / 20% test**
# MAGIC * **Seed = 42** para reproducibilidad

# COMMAND ----------

# DBTITLE 1,Split Train Test
# =============================================================================
# DIVISIÓN DE DATOS: TRAIN / TEST
# =============================================================================

print("📊 DIVISIÓN DEL DATASET")
print("=" * 50)

# Dividir 80/20
train_data, test_data = df_final.randomSplit([0.8, 0.2], seed=42)

train_count = train_data.count()
test_count = test_data.count()
total_count = train_count + test_count

print(f"📈 Total de propiedades: {total_count}")
print(f"📈 Datos de entrenamiento: {train_count} ({train_count/total_count*100:.1f}%)")
print(f"📈 Datos de prueba: {test_count} ({test_count/total_count*100:.1f}%)")

# Verificar distribución de precios en ambos conjuntos
print("\n📊 ESTADÍSTICAS DE PRECIO EN TRAIN")
train_stats = train_data.select(
    F.min("label").alias("Min"),
    F.mean("label").alias("Promedio"),
    F.max("label").alias("Max")
).collect()[0]
print(f"   Mínimo: ${train_stats['Min']:,.2f} | Promedio: ${train_stats['Promedio']:,.2f} | Máximo: ${train_stats['Max']:,.2f}")

print("\n📊 ESTADÍSTICAS DE PRECIO EN TEST")
test_stats = test_data.select(
    F.min("label").alias("Min"),
    F.mean("label").alias("Promedio"),
    F.max("label").alias("Max")
).collect()[0]
print(f"   Mínimo: ${test_stats['Min']:,.2f} | Promedio: ${test_stats['Promedio']:,.2f} | Máximo: ${test_stats['Max']:,.2f}")

print("\n✅ División completada correctamente")

# COMMAND ----------

# DBTITLE 1,Entrenamiento del Modelo
# MAGIC %md
# MAGIC ## 🌳 Entrenamiento del Modelo: Decision Tree Regressor
# MAGIC
# MAGIC ### Hiperparámetros para Regresión:
# MAGIC
# MAGIC #### 1. **maxDepth** (Profundidad máxima)
# MAGIC * Similar a clasificación: controla complejidad del árbol
# MAGIC * Valores bajos (3-5): Modelo simple, puede sub-ajustar
# MAGIC * Valores altos (15-20): Modelo complejo, riesgo de overfitting
# MAGIC
# MAGIC #### 2. **minInstancesPerNode**
# MAGIC * Mínimo de propiedades en un nodo para dividirlo
# MAGIC * Regularización: valores altos (50-100) evitan overfitting
# MAGIC
# MAGIC #### 3. **maxBins**
# MAGIC * Puntos de corte para variables continuas
# MAGIC * Default: 32 (suficiente para la mayoría de casos)
# MAGIC
# MAGIC #### 4. **impurity** (para regresión)
# MAGIC * **"variance"**: Minimiza la varianza dentro de cada grupo (default y recomendado)
# MAGIC * Busca splits que hagan grupos más homogéneos en precio
# MAGIC
# MAGIC ### Configuración:
# MAGIC * **maxDepth = 10**: Profundo para capturar relaciones complejas precio-features
# MAGIC * **minInstancesPerNode = 50**: Evitar nodos con pocas propiedades (regularización)
# MAGIC * **impurity = "variance"**: Criterio estándar para regresión

# COMMAND ----------

# DBTITLE 1,Crear y Entrenar Modelo
# =============================================================================
# ENTRENAMIENTO DEL MODELO: DECISION TREE REGRESSOR
# =============================================================================

print("🌳 CONFIGURACIÓN Y ENTRENAMIENTO DEL MODELO")
print("=" * 50)

# Crear modelo Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(
    featuresCol="features",
    labelCol="label",  # En regresión, "label" es el precio
    maxDepth=10,  # Profundidad máxima del árbol
    minInstancesPerNode=50,  # Mínimo de ejemplos por nodo (regularización)
    seed=42
)

print("Hiperparámetros configurados:")
print(f"  - Profundidad máxima: {dt_regressor.getMaxDepth()}")
print(f"  - Mínimo de instancias por nodo: {dt_regressor.getMinInstancesPerNode()}")
print(f"  - Semilla: {dt_regressor.getSeed()}")

# Entrenar el modelo
print("\n⏳ Entrenando modelo de regresión...")
model = dt_regressor.fit(train_data)
print("✅ Modelo entrenado correctamente\n")

# Información del modelo
print("📊 INFORMACIÓN DEL MODELO ENTRENADO")
print("=" * 50)
print(f"Número de nodos: {model.numNodes}")
print(f"Profundidad real: {model.depth}")
print(f"Número de features: {model.numFeatures}")

# Feature Importance
print("\n📊 IMPORTANCIA DE FEATURES")
print("=" * 50)

feature_importances = model.featureImportances.toArray()
importances_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print(importances_df.to_string(index=False))

print("\n💡 Interpretación:")
print("   - Features con alta importancia tienen mayor impacto en la predicción del precio")
print("   - Suma de todas las importancias = 1.0")

print("\n✅ Modelo listo para predecir precios")

# COMMAND ----------

# DBTITLE 1,Evaluación del Modelo de Regresión
# MAGIC %md
# MAGIC ## 📈 Evaluación de Modelos de Regresión
# MAGIC
# MAGIC ### Métricas para Regresión (diferentes a clasificación):
# MAGIC
# MAGIC #### 1. **RMSE (Root Mean Squared Error)**
# MAGIC * **Fórmula**: Raíz cuadrada del promedio de (error al cuadrado)
# MAGIC * **Unidades**: Mismas que la variable objetivo (USD en nuestro caso)
# MAGIC * **Interpretación**: Error promedio en USD que comete el modelo
# MAGIC * **Ejemplo**: RMSE = $30,000 significa que en promedio nos equivocamos ±$30,000
# MAGIC * **Sensible a outliers**: Errores grandes se penalizan más (al cuadrado)
# MAGIC
# MAGIC #### 2. **MAE (Mean Absolute Error)**
# MAGIC * **Fórmula**: Promedio del valor absoluto de los errores
# MAGIC * **Unidades**: Mismas que la variable objetivo (USD)
# MAGIC * **Interpretación**: Error promedio absoluto en USD
# MAGIC * **Menos sensible a outliers** que RMSE
# MAGIC * **Ejemplo**: MAE = $25,000 significa error promedio de $25,000
# MAGIC
# MAGIC #### 3. **R² (Coeficiente de Determinación)**
# MAGIC * **Rango**: 0 a 1 (puede ser negativo si el modelo es muy malo)
# MAGIC * **Interpretación**: % de varianza del precio explicada por el modelo
# MAGIC * **Ejemplo**: R² = 0.85 significa que el modelo explica el 85% de la variabilidad en precios
# MAGIC * **0 = modelo aleatorio, 1 = modelo perfecto**
# MAGIC * **No tiene unidades**: Útil para comparar modelos
# MAGIC
# MAGIC ### ¿Qué métrica usar?
# MAGIC
# MAGIC * **RMSE**: Si outliers son importantes y queremos penalizarlos
# MAGIC * **MAE**: Si queremos una medida más robusta a outliers
# MAGIC * **R²**: Para entender qué tan bien el modelo "explica" la variabilidad
# MAGIC * **Idealmente, reportar las 3** para visión completa
# MAGIC
# MAGIC ### Análisis de Residuos:
# MAGIC
# MAGIC **Residuos** = Precio Real - Precio Predicho
# MAGIC
# MAGIC * **Residuos cercanos a 0**: Buenas predicciones
# MAGIC * **Distribución normal de residuos**: Modelo bien calibrado
# MAGIC * **Patrones en residuos**: Indican que el modelo no captura algo

# COMMAND ----------

# DBTITLE 1,Evaluar Modelo
# =============================================================================
# EVALUACIÓN DEL MODELO EN DATOS DE PRUEBA
# =============================================================================

print("📊 EVALUACIÓN DEL MODELO")
print("=" * 70)

# Hacer predicciones en test set
print("⏳ Generando predicciones...")
predictions = model.transform(test_data)
print("✅ Predicciones generadas\n")

# Ver ejemplos de predicciones
print("Ejemplos de predicciones (primeras 10 propiedades):")
predictions.select("propiedad_id", "label", "prediction").show(10)

print("\nColumnas:")
print("  - label: Precio real de venta (USD)")
print("  - prediction: Precio predicho por el modelo (USD)\n")

# Crear evaluador de regresión
evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction"
)

# Calcular métricas
print("=" * 70)
print("📈 MÉTRICAS DE RENDIMIENTO")
print("=" * 70)

# RMSE (Root Mean Squared Error)
evaluator.setMetricName("rmse")
rmse = evaluator.evaluate(predictions)
print(f"🎯 RMSE (Root Mean Squared Error): ${rmse:,.2f}")
print(f"   Interpretación: En promedio, el modelo se equivoca ±${rmse:,.0f} en el precio")
print(f"   Mientras más bajo, mejor\n")

# MAE (Mean Absolute Error)
evaluator.setMetricName("mae")
mae = evaluator.evaluate(predictions)
print(f"🎯 MAE (Mean Absolute Error): ${mae:,.2f}")
print(f"   Interpretación: Error promedio absoluto de ${mae:,.0f}")
print(f"   Menos sensible a outliers que RMSE\n")

# R² (R-squared)
evaluator.setMetricName("r2")
r2 = evaluator.evaluate(predictions)
print(f"🎯 R² (Coeficiente de Determinación): {r2:.4f}")
print(f"   Interpretación: El modelo explica {r2*100:.2f}% de la variabilidad en precios")
print(f"   - R² > 0.90: Excelente")
print(f"   - R² 0.70-0.90: Muy bueno")
print(f"   - R² 0.50-0.70: Bueno")
print(f"   - R² < 0.50: Mejorable\n")

# Calcular error porcentual promedio
predictions_pd = predictions.select("label", "prediction").toPandas()
error_porcentual = abs((predictions_pd['label'] - predictions_pd['prediction']) / predictions_pd['label']) * 100
print(f"🎯 Error Porcentual Promedio: {error_porcentual.mean():.2f}%")
print(f"   Interpretación: En promedio, el error es {error_porcentual.mean():.1f}% del precio real\n")

print("=" * 70)
print("✅ EVALUACIÓN COMPLETADA")
print("=" * 70)

# COMMAND ----------

# DBTITLE 1,Interpretación para el Negocio
# MAGIC %md
# MAGIC ## 💼 Interpretación para el Negocio Inmobiliario
# MAGIC
# MAGIC ### ¿Qué significan las métricas para la agencia?
# MAGIC
# MAGIC #### Escenario Real: Valoración de Propiedades
# MAGIC
# MAGIC **Contexto:**
# MAGIC * Precio promedio de propiedades: ~$350,000
# MAGIC * Comisión de venta: 3% del precio
# MAGIC * Tiempo promedio de venta: 60 días
# MAGIC
# MAGIC #### Análisis de Error:
# MAGIC
# MAGIC **Si RMSE = $30,000:**
# MAGIC * **Para el vendedor**: Podríamos sobrevalorar o subvalorar su propiedad en ~$30k
# MAGIC * **Impacto**: Propiedad sobrevalorada tarda más en vender; subvalorada pierde dinero
# MAGIC * **Error relativo**: $30k / $350k = 8.6% de error (aceptable para tasaciones iniciales)
# MAGIC
# MAGIC **Si R² = 0.85:**
# MAGIC * El modelo explica 85% de la variación en precios
# MAGIC * El 15% restante depende de factores no capturados (vistas, renovaciones, negociación)
# MAGIC * **Conclusión**: Buen punto de partida, pero requiere ajuste manual
# MAGIC
# MAGIC ### Casos de Uso:
# MAGIC
# MAGIC **1. Tasación Rápida:**
# MAGIC * Cliente llama: "Tengo una casa de 150m², 3 habitaciones, en zona Norte"
# MAGIC * Modelo predice: $385,000 ± $30,000
# MAGIC * Agente: "Precio estimado entre $355k-$415k, visitemos para afinar"
# MAGIC
# MAGIC **2. Detección de Oportunidades:**
# MAGIC * Propiedad listada a $250,000
# MAGIC * Modelo predice: $320,000
# MAGIC * ¡Oportunidad! Precio 22% bajo mercado (potencial para inversionistas)
# MAGIC
# MAGIC **3. Asesoramiento a Vendedores:**
# MAGIC * Vendedor quiere $450,000
# MAGIC * Modelo predice: $380,000
# MAGIC * Agente: "El mercado valora su propiedad ~$380k. Consideremos renovaciones o ajustar precio"
# MAGIC
# MAGIC **4. Identificar Drivers de Valor:**
# MAGIC * Feature Importance muestra que 'área' y 'barrio' son más importantes
# MAGIC * Recomendación: "Invertir en extensión de área tiene mayor ROI que una piscina"
# MAGIC
# MAGIC ### Limitaciones a considerar:
# MAGIC
# MAGIC ⚠️ **El modelo NO captura:**
# MAGIC * Estado de conservación (remodelada vs deteriorada)
# MAGIC * Vistas (montaña, mar, parque)
# MAGIC * Urgencia del vendedor (divorcio, mudanza, herencia)
# MAGIC * Tendencias de mercado recientes (burbuja, crisis)
# MAGIC * Factores emocionales (casa de famoso, historia)
# MAGIC
# MAGIC ### Recomendaciones:
# MAGIC
# MAGIC 1. **Usar como screening inicial**, no como tasación final
# MAGIC 2. **Combinar con visita presencial** de agente experimentado
# MAGIC 3. **Actualizar el modelo trimestralmente** con ventas recientes
# MAGIC 4. **Crear alertas** cuando predicción ≠ precio listado (>20% diferencia)

# COMMAND ----------

# DBTITLE 1,Visualizar Resultados
# =============================================================================
# VISUALIZACIÓN DE RESULTADOS
# =============================================================================

print("📊 VISUALIZACIÓN DE PREDICCIONES Y RESIDUOS")
print("=" * 70)

# Convertir predicciones a pandas
predictions_pd = predictions.select("label", "prediction").toPandas()

# VISUALIZACIÓN 1: Precio Real vs Precio Predicho
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot: Real vs Predicho
axes[0].scatter(predictions_pd['label'], predictions_pd['prediction'], alpha=0.5, s=10)
axes[0].plot([predictions_pd['label'].min(), predictions_pd['label'].max()],
             [predictions_pd['label'].min(), predictions_pd['label'].max()],
             'r--', lw=2, label='Predicción Perfecta')
axes[0].set_xlabel('Precio Real (USD)', fontsize=12)
axes[0].set_ylabel('Precio Predicho (USD)', fontsize=12)
axes[0].set_title('Precio Real vs Precio Predicho', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Histograma de errores
error = predictions_pd['label'] - predictions_pd['prediction']
axes[1].hist(error, bins=50, color='coral', edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
axes[1].set_xlabel('Error (Real - Predicho) en USD', fontsize=12)
axes[1].set_ylabel('Frecuencia', fontsize=12)
axes[1].set_title('Distribución de Errores de Predicción', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n💡 Interpretación del Scatter Plot:")
print("   - Puntos cercanos a la línea roja: Buenas predicciones")
print("   - Puntos alejados: Propiedades con precio difícil de predecir")
print("   - Dispersión uniforme: Modelo bien calibrado\n")

print("💡 Interpretación del Histograma de Errores:")
print("   - Centrado en 0: Modelo no sesgado (no sobrestima ni subestima sistemáticamente)")
print("   - Forma de campana: Distribución normal de errores (ideal)")
print("   - Colas largas: Existen propiedades atípicas\n")

# VISUALIZACIÓN 2: Feature Importance (gráfico horizontal)
fig, ax = plt.subplots(figsize=(10, 6))

importances_sorted = importances_df.sort_values('Importance', ascending=True)
colors = plt.cm.RdYlGn(importances_sorted['Importance'] / importances_sorted['Importance'].max())

ax.barh(importances_sorted['Feature'], importances_sorted['Importance'], color=colors)
ax.set_xlabel('Importancia', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('Importancia de Features en Predicción de Precio', fontsize=14, fontweight='bold', pad=20)

for i, (feature, importance) in enumerate(zip(importances_sorted['Feature'], importances_sorted['Importance'])):
    ax.text(importance + 0.005, i, f'{importance:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.show()

print("\n🏆 TOP 3 FEATURES MÁS IMPORTANTES:")
for idx, row in importances_df.head(3).iterrows():
    print(f"   {row['Feature']}: {row['Importance']:.4f}")

print("\n📌 Insights de Negocio:")
print("   - Features más importantes son los mayores drivers del precio")
print("   - Recomendar a vendedores invertir en mejorar estos aspectos")
print("   - Para tasaciones, enfocarse en estas características\n")

# VISUALIZACIÓN 3: Residuos vs Predicciones (detectar patrones)
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(predictions_pd['prediction'], error, alpha=0.5, s=10)
ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
ax.set_xlabel('Precio Predicho (USD)', fontsize=12)
ax.set_ylabel('Residuo (Real - Predicho)', fontsize=12)
ax.set_title('Residuos vs Precio Predicho', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("💡 Interpretación de Residuos:")
print("   - Residuos distribuidos aleatoriamente: Modelo capta bien la relación")
print("   - Patrón (curva, embudo): Modelo no captura algo (considerar transformaciones)")
print("   - Outliers: Propiedades con características únicas no modeladas\n")

print("✅ Visualizaciones completadas")

# COMMAND ----------

# DBTITLE 1,Conclusiones y Próximos Pasos
# MAGIC %md
# MAGIC ## 🎓 Conclusiones y Próximos Pasos
# MAGIC
# MAGIC ### ✅ Lo que logramos en este notebook:
# MAGIC
# MAGIC 1. **Creamos un dataset sintético** de 10,000 propiedades inmobiliarias con generación realista de precios
# MAGIC 2. **Realizamos análisis exploratorio** identificando correlaciones entre features y precio
# MAGIC 3. **Preparamos los datos** para regresión con PySpark ML
# MAGIC 4. **Entrenamos un Árbol de Decisión para Regresión** que predice precios
# MAGIC 5. **Evaluamos el modelo** con métricas de regresión (RMSE, MAE, R²)
# MAGIC 6. **Interpretamos resultados** desde una perspectiva de negocio inmobiliario
# MAGIC
# MAGIC ### 📊 Resultados Clave:
# MAGIC
# MAGIC * El modelo predice precios con un error promedio medido en RMSE y MAE
# MAGIC * R² indica qué proporción de la variabilidad en precios el modelo explica
# MAGIC * Las features más importantes revelan qué características impactan más el precio
# MAGIC * El modelo es **interpretable** y útil para tasaciones rápidas y detección de oportunidades
# MAGIC
# MAGIC ### 🚀 Próximos Pasos para Mejorar:
# MAGIC
# MAGIC #### 1. **Optimización de Hiperparámetros**
# MAGIC ```python
# MAGIC from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# MAGIC
# MAGIC paramGrid = ParamGridBuilder() \
# MAGIC     .addGrid(dt_regressor.maxDepth, [5, 10, 15, 20]) \
# MAGIC     .addGrid(dt_regressor.minInstancesPerNode, [20, 50, 100]) \
# MAGIC     .build()
# MAGIC
# MAGIC cv = CrossValidator(
# MAGIC     estimator=dt_regressor,
# MAGIC     evaluator=RegressionEvaluator(metricName="rmse"),
# MAGIC     estimatorParamMaps=paramGrid,
# MAGIC     numFolds=5
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC #### 2. **Probar Modelos más Avanzados**
# MAGIC * **Random Forest Regressor**: Ensemble de árboles (mayor accuracy, reduce overfitting)
# MAGIC * **Gradient Boosted Trees (GBT)**: Árboles secuenciales optimizando errores
# MAGIC * **Linear Regression**: Modelo baseline simple para comparar
# MAGIC
# MAGIC #### 3. **Feature Engineering Avanzado**
# MAGIC * **Interacciones**: `area_m2 × barrio_idx` (el impacto del área varía por barrio)
# MAGIC * **Transformaciones**: Log(precio) si la distribución es sesgada
# MAGIC * **Binning**: Agrupar antigüedad en categorías (nueva: 0-5años, media: 5-20, vieja: >20)
# MAGIC * **Ratios**: `banos / habitaciones`, `precio_por_m2`
# MAGIC * **Distancias a puntos de interés**: Colegios, hospitales, transporte público
# MAGIC
# MAGIC #### 4. **Manejo de Outliers**
# MAGIC ```python
# MAGIC # Detectar outliers en precio usando IQR
# MAGIC Q1 = df.approxQuantile("precio_venta", [0.25], 0.01)[0]
# MAGIC Q3 = df.approxQuantile("precio_venta", [0.75], 0.01)[0]
# MAGIC IQR = Q3 - Q1
# MAGIC
# MAGIC # Filtrar outliers extremos
# MAGIC df_clean = df.filter(
# MAGIC     (F.col("precio_venta") >= Q1 - 3*IQR) &
# MAGIC     (F.col("precio_venta") <= Q3 + 3*IQR)
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC #### 5. **Validación Cruzada Temporal**
# MAGIC * Dividir datos por fecha de venta
# MAGIC * Entrenar con meses 1-10, evaluar en meses 11-12
# MAGIC * Simula cómo el modelo funcionaría en predicciones futuras
# MAGIC
# MAGIC #### 6. **Análisis de Subgrupos**
# MAGIC ```python
# MAGIC # Evaluar RMSE por rango de precio
# MAGIC predictions.withColumn(
# MAGIC     "rango_precio",
# MAGIC     F.when(F.col("label") < 200000, "Bajo")
# MAGIC      .when(F.col("label") < 400000, "Medio")
# MAGIC      .otherwise("Alto")
# MAGIC ).groupBy("rango_precio").agg(
# MAGIC     F.sqrt(F.avg((F.col("label") - F.col("prediction"))**2)).alias("RMSE")
# MAGIC ).show()
# MAGIC ```
# MAGIC
# MAGIC #### 7. **Deployment en Producción**
# MAGIC * **API de Tasación**: Endpoint REST que recibe features y devuelve precio estimado
# MAGIC * **Dashboard de Agentes**: Interfaz donde ingresan datos de propiedad y obtienen valoración
# MAGIC * **Alertas Automáticas**: Notificar cuando una propiedad listada está >15% fuera del rango predicho
# MAGIC * **Reentrenamiento Automático**: Actualizar modelo mensualmente con ventas recientes
# MAGIC
# MAGIC ### 📚 Recursos para Aprender Más:
# MAGIC
# MAGIC * [PySpark ML Regression](https://spark.apache.org/docs/latest/ml-classification-regression.html#regression)
# MAGIC * [Interpretación de métricas de regresión](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
# MAGIC * [Feature Engineering para Regresión](https://www.databricks.com/blog/2021/05/27/feature-engineering-on-databricks.html)
# MAGIC
# MAGIC ### 🎯 Aplica este Notebook:
# MAGIC
# MAGIC * Reemplaza el dataset sintético con datos reales de tu mercado inmobiliario
# MAGIC * Añade features específicas de tu región (ej: tipo de calefacción, certificación energética)
# MAGIC * Ajusta hiperparámetros según tus datos
# MAGIC * Compara con otros algoritmos (Random Forest, GBT, Linear Regression)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **¡Felicitaciones!** Has completado un pipeline end-to-end de Machine Learning para Regresión. 🎉

# COMMAND ----------

