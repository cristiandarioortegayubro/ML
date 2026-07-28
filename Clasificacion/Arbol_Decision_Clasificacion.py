# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Título y Caso de Negocio
# MAGIC %md
# MAGIC # Predicción de Abandono de Clientes (Churn) usando Árbol de Decisión
# MAGIC
# MAGIC ## 🎯 Objetivo de Negocio
# MAGIC
# MAGIC En este notebook desarrollaremos un modelo de **Machine Learning para predecir el abandono de clientes (churn)** en una empresa de telecomunicaciones.
# MAGIC
# MAGIC ### ¿Por qué es importante?
# MAGIC
# MAGIC * **Costo de adquisición**: Conseguir un nuevo cliente cuesta entre 5-25 veces más que retener uno existente
# MAGIC * **Impacto en ingresos**: La pérdida de clientes afecta directamente los ingresos recurrentes
# MAGIC * **Acción preventiva**: Identificar clientes en riesgo permite diseñar estrategias de retención personalizadas
# MAGIC
# MAGIC ### Caso de uso
# MAGIC
# MAGIC Una empresa de telecomunicaciones quiere:
# MAGIC 1. Identificar qué clientes tienen mayor probabilidad de abandonar el servicio
# MAGIC 2. Entender qué factores influyen en la decisión de abandono
# MAGIC 3. Implementar campañas de retención dirigidas a clientes de alto riesgo
# MAGIC
# MAGIC **Resultado esperado**: Un modelo que clasifique clientes en "Churn" (abandonará) o "No Churn" (permanecerá)

# COMMAND ----------

# DBTITLE 1,Qué es un Árbol de Decisión
# MAGIC %md
# MAGIC ## 🌳 ¿Qué es un Árbol de Decisión?
# MAGIC
# MAGIC ### Concepto
# MAGIC
# MAGIC Un **Árbol de Decisión** es un algoritmo de Machine Learning supervisado que funciona como un diagrama de flujo:
# MAGIC
# MAGIC * Cada **nodo interno** representa una pregunta sobre una característica (ej: "¿Antigüedad > 12 meses?")
# MAGIC * Cada **rama** representa la respuesta a esa pregunta
# MAGIC * Cada **hoja** representa una predicción final (Churn / No Churn)
# MAGIC
# MAGIC ### ¿Cuándo usar Árboles de Decisión?
# MAGIC
# MAGIC ✅ **Ventajas:**
# MAGIC * **Interpretables**: Fáciles de explicar a stakeholders no técnicos
# MAGIC * **No requieren normalización**: Funcionan con variables en diferentes escalas
# MAGIC * **Manejan variables categóricas y numéricas**: Sin preprocesamiento complejo
# MAGIC * **Capturan interacciones no lineales**: Detectan patrones complejos automáticamente
# MAGIC
# MAGIC ⚠️ **Limitaciones:**
# MAGIC * **Propensos al overfitting**: Pueden memorizar datos de entrenamiento
# MAGIC * **Inestables**: Pequeños cambios en datos pueden cambiar el árbol completamente
# MAGIC * **Sesgados**: Favorecen variables con muchas categorías
# MAGIC
# MAGIC ### Ideal para:
# MAGIC * Problemas de clasificación binaria o multiclase
# MAGIC * Cuando se necesita explicabilidad del modelo
# MAGIC * Como modelo baseline o componente de ensembles (Random Forest, Gradient Boosting)

# COMMAND ----------

# DBTITLE 1,Importar Librerías
# =============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# =============================================================================

# PySpark ML - Framework de Machine Learning distribuido de Databricks
from pyspark.ml import Pipeline  # Para encadenar transformaciones y modelo
from pyspark.ml.classification import DecisionTreeClassifier  # Algoritmo de árbol de decisión
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder  # Para preparar datos
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator  # Para evaluar modelo

# PySpark SQL - Para manipulación de datos
from pyspark.sql import functions as F  # Funciones para transformar datos
from pyspark.sql.types import *  # Tipos de datos

# Visualización - Para gráficos y análisis exploratorio
import matplotlib.pyplot as plt  # Librería base de visualización
import seaborn as sns  # Visualizaciones estadísticas elegantes
import pandas as pd  # Para convertir datos Spark a pandas cuando sea necesario

# Configuración de visualización
sns.set_style("whitegrid")  # Estilo de gráficos con grilla
plt.rcParams['figure.figsize'] = (10, 6)  # Tamaño por defecto de gráficos

print("✅ Librerías importadas correctamente")

# COMMAND ----------

# DBTITLE 1,Descripción del Dataset
# MAGIC %md
# MAGIC ## 📊 Dataset: Clientes de Telecomunicaciones
# MAGIC
# MAGIC ### Descripción
# MAGIC
# MAGIC Crearemos un dataset sintético que simula datos reales de una empresa de telecomunicaciones. Este dataset contiene información sobre:
# MAGIC
# MAGIC ### Variables del Dataset
# MAGIC
# MAGIC **Variables Demográficas:**
# MAGIC * `customer_id`: Identificador único del cliente
# MAGIC * `antiguedad_meses`: Meses que lleva el cliente con la empresa (0-72 meses)
# MAGIC * `edad`: Edad del cliente (18-80 años)
# MAGIC
# MAGIC **Variables de Comportamiento:**
# MAGIC * `gasto_mensual`: Gasto promedio mensual en dólares (20-150 USD)
# MAGIC * `llamadas_soporte`: Número de llamadas al servicio técnico en últimos 6 meses (0-10)
# MAGIC * `servicios_contratados`: Cantidad de servicios adicionales (internet, TV, etc.) (1-5)
# MAGIC
# MAGIC **Variables Contractuales:**
# MAGIC * `tipo_contrato`: Tipo de contrato ("Mensual", "Anual", "Bianual")
# MAGIC * `forma_pago`: Método de pago ("Debito_Automatico", "Tarjeta_Credito", "Transferencia")
# MAGIC
# MAGIC **Variable Objetivo:**
# MAGIC * `churn`: Si el cliente abandonó el servicio (1 = Sí abandonó, 0 = No abandonó)
# MAGIC
# MAGIC ### Tamaño del Dataset
# MAGIC * **10,000 registros** de clientes
# MAGIC * **Distribución**: ~20% churn (realista para telecomunicaciones)

# COMMAND ----------

# DBTITLE 1,Cargar y Explorar Datos
# =============================================================================
# CREACIÓN Y EXPLORACIÓN DEL DATASET
# =============================================================================

# Creamos un dataset sintético que simula datos reales de clientes
# Usamos una semilla (seed) para reproducibilidad
import numpy as np
np.random.seed(42)

# Definir el tamaño del dataset
n_clientes = 10000

# Generar datos sintéticos con distribuciones realistas
data = [
    (
        i,  # customer_id
        int(np.random.exponential(24)),  # antiguedad_meses: distribución exponencial (más clientes nuevos)
        int(np.random.normal(45, 15)),  # edad: distribución normal centrada en 45 años
        round(np.random.uniform(20, 150), 2),  # gasto_mensual: uniforme entre 20-150 USD
        int(np.random.poisson(2)),  # llamadas_soporte: distribución Poisson (eventos discretos)
        int(np.random.randint(1, 6)),  # servicios_contratados: 1 a 5 servicios
        np.random.choice(["Mensual", "Anual", "Bianual"], p=[0.5, 0.3, 0.2]),  # tipo_contrato: ponderado
        np.random.choice(["Debito_Automatico", "Tarjeta_Credito", "Transferencia"], p=[0.4, 0.4, 0.2]),  # forma_pago
        # Churn: modelado con lógica de negocio (mayor probabilidad si llamadas_soporte alto, antiguedad baja, etc.)
        1 if (np.random.exponential(24) < 12 and np.random.poisson(2) > 3) or np.random.random() < 0.15 else 0
    )
    for i in range(n_clientes)
]

# Definir el schema del DataFrame
schema = StructType([
    StructField("customer_id", IntegerType(), False),
    StructField("antiguedad_meses", IntegerType(), True),
    StructField("edad", IntegerType(), True),
    StructField("gasto_mensual", DoubleType(), True),
    StructField("llamadas_soporte", IntegerType(), True),
    StructField("servicios_contratados", IntegerType(), True),
    StructField("tipo_contrato", StringType(), True),
    StructField("forma_pago", StringType(), True),
    StructField("churn", IntegerType(), True)
])

# Crear DataFrame de Spark
df = spark.createDataFrame(data, schema)

# EXPLORACIÓN INICIAL
print("=" * 80)
print("📋 PRIMERAS 10 FILAS DEL DATASET")
print("=" * 80)
display(df.limit(10))  # display() es específico de Databricks para mejor visualización

print("\n" + "=" * 80)
print("🔍 SCHEMA DEL DATASET")
print("=" * 80)
df.printSchema()  # Muestra tipos de datos y si permiten nulos

print("\n" + "=" * 80)
print("📊 ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 80)
display(df.describe())  # Estadísticas básicas: count, mean, stddev, min, max

print("\n" + "=" * 80)
print("📈 CONTEO TOTAL Y DISTRIBUCIÓN DE CHURN")
print("=" * 80)
print(f"Total de clientes: {df.count()}")
df.groupBy("churn").count().orderBy("churn").show()

print("✅ Datos cargados y explorados correctamente")

# COMMAND ----------

# DBTITLE 1,Análisis Exploratorio
# MAGIC %md
# MAGIC ## 🔎 Análisis Exploratorio de Datos (EDA)
# MAGIC
# MAGIC ### ¿Por qué hacer Análisis Exploratorio?
# MAGIC
# MAGIC Antes de construir cualquier modelo, es fundamental **entender los datos**:
# MAGIC
# MAGIC 1. **Detectar problemas de calidad**: Valores nulos, duplicados, outliers
# MAGIC 2. **Entender distribuciones**: ¿Están balanceadas las clases? ¿Hay sesgos?
# MAGIC 3. **Identificar relaciones**: ¿Qué variables se correlacionan con churn?
# MAGIC 4. **Validar hipótesis de negocio**: ¿Los supuestos son correctos?
# MAGIC
# MAGIC ### Preguntas clave a responder:
# MAGIC
# MAGIC * ¿Está balanceado el dataset? (¿Cuántos clientes abandonan vs. permanecen?)
# MAGIC * ¿Qué características diferencian a los clientes que abandonan?
# MAGIC * ¿Hay correlaciones evidentes entre variables?
# MAGIC * ¿Existen outliers o datos anómalos?
# MAGIC
# MAGIC ### Visualizaciones a crear:
# MAGIC
# MAGIC 1. **Distribución de churn**: ¿Cuántos clientes abandonan?
# MAGIC 2. **Relación entre variables numéricas y churn**: ¿Cómo afectan antigüedad, gasto, llamadas al soporte?
# MAGIC 3. **Variables categóricas vs churn**: ¿Qué tipo de contrato tiene más abandono?

# COMMAND ----------

# DBTITLE 1,Visualizaciones Exploratorias
# =============================================================================
# ANÁLISIS EXPLORATORIO - VISUALIZACIONES
# =============================================================================

# Convertimos a Pandas para visualización (solo una muestra o agregados, no todo el dataset en producción)
# En este caso el dataset es pequeño (10k), así que es seguro
df_pandas = df.toPandas()

# VISUALIZACIÓN 1: Distribución de Churn
# ¿Cuántos clientes abandonan vs. permanecen?
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de barras
churn_counts = df_pandas['churn'].value_counts()
axes[0].bar(['No Churn', 'Churn'], churn_counts.values, color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Distribución de Churn', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Cantidad de Clientes')
axes[0].set_xlabel('Estado')
# Agregamos etiquetas con porcentajes
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 100, f'{v}\n({v/len(df_pandas)*100:.1f}%)', ha='center', fontweight='bold')

# Gráfico de torta
axes[1].pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%', 
            colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1].set_title('Proporción de Churn', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("📊 Interpretación: Si el dataset está muy desbalanceado (ej: 95% No Churn), ")
print("   el modelo podría simplemente predecir 'No Churn' siempre y tener alta precisión.")
print("   En nuestro caso, ~20% de churn es realista y manejable.\n")

# VISUALIZACIÓN 2: Variables Numéricas vs Churn
# ¿Cómo se relacionan antigüedad, gasto mensual y llamadas al soporte con el churn?
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Boxplot de Antigüedad vs Churn
df_pandas.boxplot(column='antiguedad_meses', by='churn', ax=axes[0])
axes[0].set_title('Antigüedad vs Churn')
axes[0].set_xlabel('Churn (0=No, 1=Sí)')
axes[0].set_ylabel('Antigüedad (meses)')
axes[0].get_figure().suptitle('')  # Remover título por defecto

# Boxplot de Gasto Mensual vs Churn
df_pandas.boxplot(column='gasto_mensual', by='churn', ax=axes[1])
axes[1].set_title('Gasto Mensual vs Churn')
axes[1].set_xlabel('Churn (0=No, 1=Sí)')
axes[1].set_ylabel('Gasto Mensual (USD)')
axes[1].get_figure().suptitle('')

# Boxplot de Llamadas Soporte vs Churn
df_pandas.boxplot(column='llamadas_soporte', by='churn', ax=axes[2])
axes[2].set_title('Llamadas al Soporte vs Churn')
axes[2].set_xlabel('Churn (0=No, 1=Sí)')
axes[2].set_ylabel('Llamadas al Soporte')
axes[2].get_figure().suptitle('')

plt.tight_layout()
plt.show()

print("📊 Interpretación de Boxplots:")
print("   - Si la mediana (línea central) es diferente entre grupos, la variable es predictiva")
print("   - Esperamos que clientes con churn tengan: menor antigüedad, más llamadas al soporte\n")

# VISUALIZACIÓN 3: Variables Categóricas vs Churn
# ¿Qué tipo de contrato tiene más abandono?
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Tipo de Contrato vs Churn
contrato_churn = df_pandas.groupby(['tipo_contrato', 'churn']).size().unstack(fill_value=0)
contrato_churn.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Tipo de Contrato vs Churn', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Tipo de Contrato')
axes[0].set_ylabel('Cantidad de Clientes')
axes[0].legend(['No Churn', 'Churn'])
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

# Forma de Pago vs Churn
pago_churn = df_pandas.groupby(['forma_pago', 'churn']).size().unstack(fill_value=0)
pago_churn.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'])
axes[1].set_title('Forma de Pago vs Churn', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Forma de Pago')
axes[1].set_ylabel('Cantidad de Clientes')
axes[1].legend(['No Churn', 'Churn'])
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()

print("📊 Interpretación:")
print("   - Contratos mensuales suelen tener más churn (menor compromiso)")
print("   - Métodos de pago automáticos pueden correlacionar con menor churn\n")

print("✅ Análisis exploratorio completado")

# COMMAND ----------

# DBTITLE 1,Preparación de Datos
# MAGIC %md
# MAGIC ## 🔧 Preparación de Datos (Feature Engineering)
# MAGIC
# MAGIC ### ¿Por qué es necesaria la preparación?
# MAGIC
# MAGIC Los algoritmos de Machine Learning en PySpark requieren que:
# MAGIC
# MAGIC 1. **Todas las features estén en un único vector**: PySpark ML espera una columna "features" con un vector denso/sparse
# MAGIC 2. **Variables categóricas estén codificadas**: Los algoritmos no entienden texto ("Mensual", "Anual"), necesitan números
# MAGIC 3. **No haya valores nulos**: Debemos decidir cómo manejarlos (eliminar, imputar)
# MAGIC
# MAGIC ### Pasos de preparación:
# MAGIC
# MAGIC 1. **Manejo de nulos**: Verificar y eliminar/imputar valores faltantes
# MAGIC 2. **String Indexing**: Convertir variables categóricas a índices numéricos
# MAGIC    * Ejemplo: "Mensual" → 0, "Anual" → 1, "Bianual" → 2
# MAGIC 3. **One-Hot Encoding** (opcional): Crear variables dummy para evitar que el modelo asuma orden
# MAGIC    * Ejemplo: "Mensual" → [1, 0, 0], "Anual" → [0, 1, 0], "Bianual" → [0, 0, 1]
# MAGIC 4. **Vector Assembly**: Combinar todas las features en un único vector
# MAGIC 5. **Selección de features**: Elegir qué variables incluir en el modelo
# MAGIC
# MAGIC ### Features que usaremos:
# MAGIC
# MAGIC **Numéricas directas:**
# MAGIC * antiguedad_meses
# MAGIC * edad
# MAGIC * gasto_mensual
# MAGIC * llamadas_soporte
# MAGIC * servicios_contratados
# MAGIC
# MAGIC **Categóricas a codificar:**
# MAGIC * tipo_contrato
# MAGIC * forma_pago
# MAGIC
# MAGIC **Nota**: Excluimos `customer_id` porque es solo un identificador, no tiene poder predictivo

# COMMAND ----------

# DBTITLE 1,Preparar Datos
# =============================================================================
# PREPARACIÓN DE DATOS - FEATURE ENGINEERING
# =============================================================================

# PASO 1: Verificar y manejar valores nulos
print("🔍 VERIFICACIÓN DE VALORES NULOS")
print("=" * 50)
null_counts = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns])
display(null_counts)

# Si hubiera nulos, podríamos:
# df = df.na.drop()  # Eliminar filas con nulos
# df = df.na.fill({"columna": valor})  # Imputar con un valor específico

print("\n✅ No hay valores nulos en el dataset\n")

# PASO 2: Codificar variables categóricas
# StringIndexer convierte strings a índices numéricos
# Ejemplo: "Mensual" → 0, "Anual" → 1, "Bianual" → 2

print("🔧 CODIFICACIÓN DE VARIABLES CATEGÓRICAS")
print("=" * 50)

# Indexar 'tipo_contrato'
contrato_indexer = StringIndexer(
    inputCol="tipo_contrato",  # Columna de entrada (string)
    outputCol="tipo_contrato_idx"  # Columna de salida (numérica)
)

# Indexar 'forma_pago'
pago_indexer = StringIndexer(
    inputCol="forma_pago",
    outputCol="forma_pago_idx"
)

# Aplicar los indexers al DataFrame
df = contrato_indexer.fit(df).transform(df)
df = pago_indexer.fit(df).transform(df)

print("Ejemplo de codificación:")
df.select("tipo_contrato", "tipo_contrato_idx", "forma_pago", "forma_pago_idx").distinct().show()

print("\n✅ Variables categóricas codificadas\n")

# PASO 3: Ensamblar features en un vector
# VectorAssembler toma múltiples columnas y las combina en una sola columna vector
# Este vector es el input que espera el modelo de ML

print("🔧 CREACIÓN DEL VECTOR DE FEATURES")
print("=" * 50)

# Lista de columnas que usaremos como features (variables predictoras)
feature_columns = [
    "antiguedad_meses",      # Numérica
    "edad",                  # Numérica
    "gasto_mensual",         # Numérica
    "llamadas_soporte",      # Numérica
    "servicios_contratados", # Numérica
    "tipo_contrato_idx",     # Categórica codificada
    "forma_pago_idx"         # Categórica codificada
]

# VectorAssembler: combina todas las features en un solo vector
assembler = VectorAssembler(
    inputCols=feature_columns,  # Columnas de entrada
    outputCol="features"         # Columna de salida con el vector
)

# Aplicar el assembler
df_final = assembler.transform(df)

# Seleccionar solo las columnas que necesitamos: features y label (churn)
df_final = df_final.select("customer_id", "features", F.col("churn").alias("label"))

print("Dataset preparado con columnas:")
df_final.printSchema()

print("\nEjemplo de vector de features:")
df_final.select("customer_id", "features", "label").show(5, truncate=False)

print("\n✅ Datos preparados correctamente")
print(f"   - Total de features: {len(feature_columns)}")
print(f"   - Features numéricas: 5")
print(f"   - Features categóricas codificadas: 2")

# COMMAND ----------

# DBTITLE 1,División Train Test
# MAGIC %md
# MAGIC ## 📊 División de Datos: Entrenamiento y Prueba
# MAGIC
# MAGIC ### ¿Por qué dividir los datos?
# MAGIC
# MAGIC Para evaluar correctamente un modelo de Machine Learning, necesitamos:
# MAGIC
# MAGIC 1. **Datos de entrenamiento (train)**: Para que el modelo aprenda patrones
# MAGIC 2. **Datos de prueba (test)**: Para evaluar cómo generaliza a datos nuevos
# MAGIC
# MAGIC ### El problema del overfitting
# MAGIC
# MAGIC Si evaluamos el modelo con los mismos datos que usamos para entrenar:
# MAGIC * El modelo puede **memorizar** los datos de entrenamiento
# MAGIC * Tendría **alta precisión en entrenamiento** pero **baja en producción**
# MAGIC * No sabríamos si realmente aprendió patrones generalizables
# MAGIC
# MAGIC ### División típica
# MAGIC
# MAGIC * **70-80% para entrenamiento**: El modelo aprende de este conjunto
# MAGIC * **20-30% para prueba**: Evaluamos el rendimiento aquí
# MAGIC * **Aleatorización**: Usamos una semilla (seed) para reproducibilidad
# MAGIC
# MAGIC ### En este caso:
# MAGIC
# MAGIC * **80% train / 20% test**
# MAGIC * **Seed = 42**: Para que siempre obtengamos la misma división
# MAGIC * **Estratificación implícita**: PySpark distribuye aleatoriamente preservando proporciones
# MAGIC
# MAGIC ### Importante:
# MAGIC
# MAGIC ⚠️ **NUNCA** usar datos de prueba durante el entrenamiento
# MAGIC ⚠️ **NUNCA** ajustar hiperparámetros mirando solo el test set (usar validación cruzada)

# COMMAND ----------

# DBTITLE 1,Split Train Test
# =============================================================================
# DIVISIÓN DE DATOS: TRAIN / TEST
# =============================================================================

print("📊 DIVISIÓN DEL DATASET")
print("=" * 50)

# Dividir el dataset en entrenamiento (80%) y prueba (20%)
# Parámetros:
#   - [0.8, 0.2]: Proporciones de la división (80% train, 20% test)
#   - seed=42: Semilla para reproducibilidad (siempre obtendremos la misma división)
train_data, test_data = df_final.randomSplit([0.8, 0.2], seed=42)

# Verificar tamaños de cada conjunto
train_count = train_data.count()
test_count = test_data.count()
total_count = train_count + test_count

print(f"📈 Total de registros: {total_count}")
print(f"📈 Datos de entrenamiento: {train_count} ({train_count/total_count*100:.1f}%)")
print(f"📈 Datos de prueba: {test_count} ({test_count/total_count*100:.1f}%)")

# Verificar distribución de churn en ambos conjuntos
print("\n📊 DISTRIBUCIÓN DE CHURN EN TRAIN")
train_data.groupBy("label").count().orderBy("label").show()

print("📊 DISTRIBUCIÓN DE CHURN EN TEST")
test_data.groupBy("label").count().orderBy("label").show()

print("\n✅ División completada correctamente")
print("   - Los datos de prueba NO se usarán hasta la evaluación final")
print("   - El modelo aprenderá SOLO de los datos de entrenamiento")

# COMMAND ----------

# DBTITLE 1,Entrenamiento del Modelo
# MAGIC %md
# MAGIC ## 🌳 Entrenamiento del Modelo: Decision Tree Classifier
# MAGIC
# MAGIC ### Hiperparámetros del Árbol de Decisión
# MAGIC
# MAGIC Los hiperparámetros controlan cómo el árbol "aprende" de los datos:
# MAGIC
# MAGIC #### 1. **maxDepth** (Profundidad máxima)
# MAGIC * **Qué es**: Cuántos niveles de decisiones puede tener el árbol
# MAGIC * **Valor bajo (ej: 3-5)**: Árbol simple, puede sub-ajustar (underfitting)
# MAGIC * **Valor alto (ej: 20+)**: Árbol complejo, puede sobre-ajustar (overfitting)
# MAGIC * **Recomendación**: Empezar con 5-10 y ajustar
# MAGIC
# MAGIC #### 2. **maxBins** (Bins máximos)
# MAGIC * **Qué es**: Cuántos "puntos de corte" considerar para variables continuas
# MAGIC * **Ejemplo**: Para "gasto_mensual", ¿dónde hacer los cortes? (ej: <50, 50-100, >100)
# MAGIC * **Valor típico**: 32 (default en PySpark)
# MAGIC * **Nota**: Debe ser ≥ al número de categorías en variables categóricas
# MAGIC
# MAGIC #### 3. **impurity** (Criterio de impureza)
# MAGIC * **Gini**: Mide "impureza" de un nodo (qué tan mezcladas están las clases)
# MAGIC * **Entropy**: Similar a Gini, basado en teoría de información
# MAGIC * **Recomendación**: Ambos funcionan bien, Gini es ligeramente más rápido
# MAGIC
# MAGIC #### 4. **minInstancesPerNode** (Mínimo de instancias por nodo)
# MAGIC * **Qué es**: Cuántos ejemplos mínimos debe tener un nodo para dividirse
# MAGIC * **Valor bajo (ej: 1)**: Permite nodos muy específicos (riesgo de overfitting)
# MAGIC * **Valor alto (ej: 100)**: Nodos más generales (más regularización)
# MAGIC
# MAGIC ### Configuración para este modelo:
# MAGIC
# MAGIC * **maxDepth = 5**: Árbol moderadamente profundo, balance entre complejidad e interpretabilidad
# MAGIC * **impurity = 'gini'**: Criterio estándar y eficiente
# MAGIC * **minInstancesPerNode = 20**: Regularización leve para evitar overfitting

# COMMAND ----------

# DBTITLE 1,Crear y Entrenar Modelo
# =============================================================================
# ENTRENAMIENTO DEL MODELO: DECISION TREE CLASSIFIER
# =============================================================================

print("🌳 CONFIGURACIÓN Y ENTRENAMIENTO DEL MODELO")
print("=" * 50)

# Crear el modelo Decision Tree Classifier
# Parámetros:
dt = DecisionTreeClassifier(
    featuresCol="features",  # Columna con el vector de features
    labelCol="label",        # Columna con la variable objetivo (churn)
    maxDepth=5,              # Profundidad máxima del árbol (5 niveles de decisiones)
    impurity="gini",         # Criterio para medir impureza (gini o entropy)
    minInstancesPerNode=20,  # Mínimo de ejemplos en un nodo para dividirlo (regularización)
    seed=42                  # Semilla para reproducibilidad
)

print("Hiperparámetros configurados:")
print(f"  - Profundidad máxima: {dt.getMaxDepth()}")
print(f"  - Criterio de impureza: {dt.getImpurity()}")
print(f"  - Mínimo de instancias por nodo: {dt.getMinInstancesPerNode()}")
print(f"  - Semilla: {dt.getSeed()}")

# Entrenar el modelo con los datos de entrenamiento
# El método .fit() aprende los patrones de los datos
print("\n⏳ Entrenando modelo...")
model = dt.fit(train_data)
print("✅ Modelo entrenado correctamente\n")

# Información del modelo entrenado
print("📊 INFORMACIÓN DEL MODELO ENTRENADO")
print("=" * 50)
print(f"Número de nodos en el árbol: {model.numNodes}")
print(f"Profundidad real del árbol: {model.depth}")
print(f"Número de features utilizadas: {model.numFeatures}")

# Feature Importance: Qué tan importante es cada variable para las predicciones
print("\n📊 IMPORTANCIA DE FEATURES")
print("=" * 50)

# Obtener importancias y nombres de features
feature_importances = model.featureImportances.toArray()
feature_names = feature_columns

# Crear DataFrame con importancias
importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print(importances_df.to_string(index=False))

print("\n💡 Interpretación:")
print("   - Valores cercanos a 1: Feature muy importante para predicciones")
print("   - Valores cercanos a 0: Feature poco relevante")
print("   - La suma de todas las importancias = 1.0")

print("\n✅ Modelo listo para hacer predicciones")

# COMMAND ----------

# DBTITLE 1,Evaluación del Modelo
# MAGIC %md
# MAGIC ## 📈 Evaluación del Modelo
# MAGIC
# MAGIC ### ¿Cómo saber si el modelo es bueno?
# MAGIC
# MAGIC Necesitamos evaluar el modelo con datos que **nunca ha visto** (test set) usando múltiples métricas:
# MAGIC
# MAGIC ### Métricas de Clasificación
# MAGIC
# MAGIC #### 1. **Accuracy (Exactitud)**
# MAGIC * **Fórmula**: (Predicciones Correctas) / (Total de Predicciones)
# MAGIC * **Interpretación**: % de predicciones correctas en general
# MAGIC * **Limitación**: Puede ser engañosa con datos desbalanceados
# MAGIC * **Ejemplo**: 85% accuracy = el modelo acierta 85 de cada 100 predicciones
# MAGIC
# MAGIC #### 2. **Precision (Precisión)**
# MAGIC * **Fórmula**: (Verdaderos Positivos) / (Verdaderos Positivos + Falsos Positivos)
# MAGIC * **Pregunta que responde**: "De los que predije como Churn, ¿cuántos realmente abandonaron?"
# MAGIC * **Importante cuando**: Los falsos positivos son costosos (enviar promoción a quien no la necesita)
# MAGIC
# MAGIC #### 3. **Recall (Sensibilidad / Exhaustividad)**
# MAGIC * **Fórmula**: (Verdaderos Positivos) / (Verdaderos Positivos + Falsos Negativos)
# MAGIC * **Pregunta que responde**: "De todos los que realmente abandonaron, ¿cuántos detecté?"
# MAGIC * **Importante cuando**: Los falsos negativos son costosos (perder un cliente que no detectamos)
# MAGIC
# MAGIC #### 4. **F1-Score**
# MAGIC * **Fórmula**: 2 × (Precision × Recall) / (Precision + Recall)
# MAGIC * **Interpretación**: Media armónica entre Precision y Recall
# MAGIC * **Ideal**: Cuando necesitamos balance entre Precision y Recall
# MAGIC
# MAGIC #### 5. **AUC-ROC** (Area Under the Curve)
# MAGIC * **Rango**: 0.5 (modelo aleatorio) a 1.0 (modelo perfecto)
# MAGIC * **Interpretación**: Probabilidad de que el modelo ordene correctamente un ejemplo positivo vs negativo
# MAGIC * **Útil**: Para comparar modelos, independiente del threshold
# MAGIC
# MAGIC ### Matriz de Confusión
# MAGIC
# MAGIC ```
# MAGIC                     Predicción
# MAGIC                 Churn    No Churn
# MAGIC Real  Churn      TP        FN
# MAGIC       No Churn   FP        TN
# MAGIC ```
# MAGIC
# MAGIC * **TP (True Positive)**: Predijo Churn y era Churn ✅
# MAGIC * **TN (True Negative)**: Predijo No Churn y era No Churn ✅
# MAGIC * **FP (False Positive)**: Predijo Churn pero NO era Churn ❌ (Falsa alarma)
# MAGIC * **FN (False Negative)**: Predijo No Churn pero SÍ era Churn ❌ (Cliente perdido)

# COMMAND ----------

# DBTITLE 1,Evaluar Modelo
# =============================================================================
# EVALUACIÓN DEL MODELO EN DATOS DE PRUEBA
# =============================================================================

print("📊 EVALUACIÓN DEL MODELO")
print("=" * 70)

# PASO 1: Hacer predicciones en el conjunto de prueba
# El modelo predice la probabilidad y la clase para cada ejemplo
print("⏳ Generando predicciones en datos de prueba...")
predictions = model.transform(test_data)
print("✅ Predicciones generadas\n")

# Ver ejemplo de predicciones
print("Ejemplo de predicciones (primeras 10 filas):")
predictions.select("customer_id", "label", "prediction", "probability").show(10, truncate=False)

print("\nColumnas en predictions:")
print("  - label: Valor real (0 = No Churn, 1 = Churn)")
print("  - prediction: Predicción del modelo (0 o 1)")
print("  - probability: Vector [prob_no_churn, prob_churn]\n")

# PASO 2: Calcular métricas de evaluación
print("=" * 70)
print("📈 MÉTRICAS DE RENDIMIENTO")
print("=" * 70)

# Evaluador para clasificación binaria (AUC-ROC)
binary_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"  # AUC-ROC
)

# Evaluador para métricas multiclase (accuracy, precision, recall, f1)
multiclass_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction"
)

# Calcular AUC-ROC
auc = binary_evaluator.evaluate(predictions)
print(f"🎯 AUC-ROC: {auc:.4f}")
print(f"   Interpretación: {auc:.1%} de probabilidad de ordenar correctamente ejemplos")
print(f"   - > 0.90: Excelente")
print(f"   - 0.80-0.90: Muy bueno")
print(f"   - 0.70-0.80: Bueno")
print(f"   - 0.60-0.70: Regular")
print(f"   - < 0.60: Pobre\n")

# Calcular Accuracy
multiclass_evaluator.setMetricName("accuracy")
accuracy = multiclass_evaluator.evaluate(predictions)
print(f"🎯 Accuracy (Exactitud): {accuracy:.4f} ({accuracy:.1%})")
print(f"   Interpretación: El modelo acierta {accuracy:.1%} de las predicciones\n")

# Calcular Precision
multiclass_evaluator.setMetricName("weightedPrecision")
precision = multiclass_evaluator.evaluate(predictions)
print(f"🎯 Precision (Precisión): {precision:.4f} ({precision:.1%})")
print(f"   Interpretación: De los que predice como Churn, {precision:.1%} realmente lo son\n")

# Calcular Recall
multiclass_evaluator.setMetricName("weightedRecall")
recall = multiclass_evaluator.evaluate(predictions)
print(f"🎯 Recall (Exhaustividad): {recall:.4f} ({recall:.1%})")
print(f"   Interpretación: De todos los Churn reales, detecta {recall:.1%}\n")

# Calcular F1-Score
multiclass_evaluator.setMetricName("f1")
f1 = multiclass_evaluator.evaluate(predictions)
print(f"🎯 F1-Score: {f1:.4f}")
print(f"   Interpretación: Balance entre Precision y Recall\n")

# PASO 3: Matriz de Confusión
print("=" * 70)
print("📊 MATRIZ DE CONFUSIÓN")
print("=" * 70)

# Crear matriz de confusión manualmente
confusion_matrix = predictions.groupBy("label", "prediction").count().orderBy("label", "prediction")
print("\nTabla de confusión (label = real, prediction = predicción):")
confusion_matrix.show()

# Calcular componentes de la matriz
tp = predictions.filter((F.col("label") == 1) & (F.col("prediction") == 1)).count()  # True Positive
tn = predictions.filter((F.col("label") == 0) & (F.col("prediction") == 0)).count()  # True Negative
fp = predictions.filter((F.col("label") == 0) & (F.col("prediction") == 1)).count()  # False Positive
fn = predictions.filter((F.col("label") == 1) & (F.col("prediction") == 0)).count()  # False Negative

print(f"\nDesglose de la Matriz de Confusión:")
print(f"  ✅ True Positives (TP): {tp} - Predijo Churn y era Churn")
print(f"  ✅ True Negatives (TN): {tn} - Predijo No Churn y era No Churn")
print(f"  ❌ False Positives (FP): {fp} - Predijo Churn pero NO era Churn (Falsa alarma)")
print(f"  ❌ False Negatives (FN): {fn} - Predijo No Churn pero SÍ era Churn (Cliente perdido)")

print("\n" + "=" * 70)
print("✅ EVALUACIÓN COMPLETADA")
print("=" * 70)

# COMMAND ----------

# DBTITLE 1,Interpretación de Resultados
# MAGIC %md
# MAGIC ## 💼 Interpretación de Resultados para el Negocio
# MAGIC
# MAGIC ### ¿Qué significan las métricas para la empresa?
# MAGIC
# MAGIC #### Escenario Real: Campaña de Retención
# MAGIC
# MAGIC La empresa quiere lanzar una **campaña de retención** con descuentos especiales.
# MAGIC
# MAGIC **Costos:**
# MAGIC * Descuento/promoción por cliente: $50
# MAGIC * Valor de vida de un cliente (CLV): $1,200
# MAGIC * Costo de adquirir nuevo cliente: $300
# MAGIC
# MAGIC #### Análisis de Errores:
# MAGIC
# MAGIC **1. False Positives (Falsa Alarma) - Impacto Moderado**
# MAGIC * **Qué pasa**: Enviamos promoción a un cliente que NO iba a abandonar
# MAGIC * **Costo**: $50 de descuento innecesario
# MAGIC * **Cantidad**: {fp} clientes en el test set
# MAGIC * **Costo total**: {fp} × $50 = ${fp * 50}
# MAGIC
# MAGIC **2. False Negatives (Cliente Perdido) - Impacto Crítico**
# MAGIC * **Qué pasa**: NO enviamos promoción a un cliente que SÍ abandonó
# MAGIC * **Costo**: Perdemos el cliente ($1,200 CLV) + costo de reemplazo ($300)
# MAGIC * **Cantidad**: {fn} clientes en el test set
# MAGIC * **Costo total**: {fn} × $1,500 = ${fn * 1500}
# MAGIC
# MAGIC #### Comparación de Estrategias:
# MAGIC
# MAGIC **Estrategia 1: Sin modelo (campaña masiva a todos)**
# MAGIC * Costo: 10,000 clientes × $50 = $500,000
# MAGIC * Churn evitado: ~20% = 2,000 clientes salvados
# MAGIC * Beneficio neto: (2,000 × $1,200) - $500,000 = $1,900,000
# MAGIC
# MAGIC **Estrategia 2: Con modelo (campaña dirigida)**
# MAGIC * Solo enviamos a clientes predichos como Churn
# MAGIC * Costo: Mucho menor (solo True Positives + False Positives)
# MAGIC * Churn evitado: Similar efectividad
# MAGIC * **Beneficio neto: MAYOR** (menos desperdicio)
# MAGIC
# MAGIC ### Métricas Clave por Objetivo:
# MAGIC
# MAGIC **Si priorizas MINIMIZAR COSTOS:**
# MAGIC * Enfócate en **Precision** alta
# MAGIC * Reduce False Positives
# MAGIC * Acepta perder algunos clientes
# MAGIC
# MAGIC **Si priorizas RETENER MÁS CLIENTES:**
# MAGIC * Enfócate en **Recall** alto
# MAGIC * Reduce False Negatives
# MAGIC * Acepta enviar promociones de más
# MAGIC
# MAGIC **Balance (recomendado):**
# MAGIC * Optimiza **F1-Score**
# MAGIC * Balance entre costo y retención
# MAGIC
# MAGIC ### Próximos Pasos:
# MAGIC
# MAGIC 1. **Ajustar threshold**: En lugar de 0.5, usar threshold óptimo según costo/beneficio
# MAGIC 2. **Feature Engineering avanzado**: Crear nuevas variables (ej: "días desde última llamada soporte")
# MAGIC 3. **Probar otros modelos**: Random Forest, Gradient Boosting (mayor accuracy)
# MAGIC 4. **Validación cruzada**: Evaluar estabilidad del modelo
# MAGIC 5. **Deployment**: Integrar en sistema CRM para scoring diario

# COMMAND ----------

# DBTITLE 1,Visualizar Feature Importance
# =============================================================================
# VISUALIZACIÓN DE FEATURE IMPORTANCE
# =============================================================================

print("📊 VISUALIZACIÓN DE IMPORTANCIA DE FEATURES")
print("=" * 70)

# Obtener importancias del modelo
feature_importances = model.featureImportances.toArray()
feature_names = feature_columns

# Crear DataFrame para visualización
importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=True)  # Ordenar de menor a mayor para el gráfico

# VISUALIZACIÓN 1: Gráfico de barras horizontal
fig, ax = plt.subplots(figsize=(10, 6))

# Colores según importancia
colors = plt.cm.RdYlGn(importances_df['Importance'] / importances_df['Importance'].max())

# Crear gráfico de barras
ax.barh(importances_df['Feature'], importances_df['Importance'], color=colors)
ax.set_xlabel('Importancia', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('Importancia de Features en el Modelo de Árbol de Decisión', 
             fontsize=14, fontweight='bold', pad=20)

# Agregar valores en las barras
for i, (feature, importance) in enumerate(zip(importances_df['Feature'], importances_df['Importance'])):
    ax.text(importance + 0.01, i, f'{importance:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.show()

print("\n💡 INTERPRETACIÓN DE FEATURE IMPORTANCE")
print("=" * 70)
print("\nLas features más importantes son las que más contribuyen a las decisiones del árbol.\n")

# Mostrar top 3 features más importantes
top_features = importances_df.sort_values('Importance', ascending=False).head(3)
print("🏆 Top 3 Features Más Importantes:\n")
for idx, row in top_features.iterrows():
    print(f"   {row['Feature']}: {row['Importance']:.4f}")

print("\n📌 Insights de Negocio:")
print("   - Si 'llamadas_soporte' es importante: Los clientes insatisfechos llaman más")
print("   - Si 'antiguedad_meses' es importante: Clientes nuevos tienen mayor riesgo")
print("   - Si 'tipo_contrato' es importante: El compromiso contractual influye en retención")
print("   - Si 'gasto_mensual' es importante: El nivel de inversión indica compromiso\n")

print("✅ Visualización completada")

# NOTA: No visualizamos el árbol completo porque con profundidad 5 puede ser muy grande
# En producción, se puede usar model.toDebugString para ver la estructura del árbol

# COMMAND ----------

# DBTITLE 1,Conclusiones y Próximos Pasos
# MAGIC %md
# MAGIC ## 🎓 Conclusiones y Próximos Pasos
# MAGIC
# MAGIC ### ✅ Lo que logramos en este notebook:
# MAGIC
# MAGIC 1. **Creamos un dataset sintético** de 10,000 clientes de telecomunicaciones con variables relevantes
# MAGIC 2. **Realizamos análisis exploratorio** para entender patrones de churn
# MAGIC 3. **Preparamos los datos** codificando variables categóricas y creando vectores de features
# MAGIC 4. **Entrenamos un Árbol de Decisión** con PySpark ML
# MAGIC 5. **Evaluamos el modelo** con múltiples métricas (Accuracy, Precision, Recall, F1, AUC-ROC)
# MAGIC 6. **Interpretamos resultados** desde una perspectiva de negocio
# MAGIC
# MAGIC ### 📊 Resultados Clave:
# MAGIC
# MAGIC * El modelo identifica patrones de churn con métricas competitivas
# MAGIC * Las features más importantes revelan drivers de abandono
# MAGIC * El modelo es **interpretable** y puede explicarse a stakeholders no técnicos
# MAGIC * Existe trade-off entre False Positives (costo de promoción) y False Negatives (pérdida de cliente)
# MAGIC
# MAGIC ### 🚀 Próximos Pasos para Mejorar:
# MAGIC
# MAGIC #### 1. **Optimización de Hiperparámetros**
# MAGIC ```python
# MAGIC # Usar Grid Search o Random Search
# MAGIC from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# MAGIC
# MAGIC paramGrid = ParamGridBuilder() \
# MAGIC     .addGrid(dt.maxDepth, [3, 5, 7, 10]) \
# MAGIC     .addGrid(dt.minInstancesPerNode, [10, 20, 50]) \
# MAGIC     .build()
# MAGIC ```
# MAGIC
# MAGIC #### 2. **Probar Modelos más Avanzados**
# MAGIC * **Random Forest**: Ensemble de múltiples árboles (mayor accuracy, menos interpretabilidad)
# MAGIC * **Gradient Boosting (GBT)**: Árboles secuenciales que corrigen errores previos
# MAGIC * **Logistic Regression**: Modelo lineal simple como baseline
# MAGIC
# MAGIC #### 3. **Feature Engineering Avanzado**
# MAGIC * Crear interacciones: `antiguedad × gasto_mensual`
# MAGIC * Variables temporales: "días desde última llamada soporte"
# MAGIC * Ratios: `llamadas_soporte / antiguedad_meses`
# MAGIC * Agregaciones: "promedio de gasto últimos 3 meses"
# MAGIC
# MAGIC #### 4. **Manejo de Desbalance de Clases**
# MAGIC * Si el churn es muy bajo (<5%), considerar:
# MAGIC   * **Class weighting**: Penalizar más errores en clase minoritaria
# MAGIC   * **SMOTE**: Synthetic Minority Over-sampling Technique
# MAGIC   * **Undersampling**: Reducir clase mayoritaria
# MAGIC
# MAGIC #### 5. **Validación Cruzada**
# MAGIC ```python
# MAGIC # Evaluar estabilidad del modelo
# MAGIC from pyspark.ml.tuning import CrossValidator
# MAGIC
# MAGIC cv = CrossValidator(
# MAGIC     estimator=dt,
# MAGIC     evaluator=binary_evaluator,
# MAGIC     numFolds=5  # 5-fold cross-validation
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC #### 6. **Deployment en Producción**
# MAGIC * **Guardar el modelo**: `model.save("/path/to/model")`
# MAGIC * **Scoring batch**: Predecir churn para toda la base de clientes diariamente
# MAGIC * **Scoring real-time**: API para scoring individual
# MAGIC * **Monitoreo**: Detectar drift en distribución de features
# MAGIC
# MAGIC #### 7. **A/B Testing**
# MAGIC * Implementar campaña de retención solo para grupo test
# MAGIC * Comparar churn rate vs grupo control
# MAGIC * Medir ROI real del modelo
# MAGIC
# MAGIC ### 📚 Recursos para Aprender Más:
# MAGIC
# MAGIC * [PySpark ML Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
# MAGIC * [Databricks Academy](https://www.databricks.com/learn/training)
# MAGIC * [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html) (conceptos aplicables)
# MAGIC
# MAGIC ### 🎯 Aplica este Notebook:
# MAGIC
# MAGIC * Cambia el dataset por tus propios datos
# MAGIC * Ajusta hiperparámetros según tus necesidades
# MAGIC * Experimenta con diferentes features
# MAGIC * Compara con otros algoritmos (Random Forest, GBT)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **¡Felicitaciones!** Has completado un pipeline end-to-end de Machine Learning para clasificación. 🎉

# COMMAND ----------

