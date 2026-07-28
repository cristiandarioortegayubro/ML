# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Título y Caso de Negocio
# MAGIC %md
# MAGIC # Segmentación de Clientes con K-Means Clustering
# MAGIC
# MAGIC ## 🎯 Objetivo de Negocio
# MAGIC
# MAGIC Desarrollaremos un modelo de **K-Means Clustering** para segmentar clientes de un e-commerce y diseñar estrategias de marketing personalizadas.
# MAGIC
# MAGIC ### ¿Qué es Clustering?
# MAGIC
# MAGIC **Clustering** (agrupamiento) es una técnica de **aprendizaje no supervisado** que agrupa datos similares sin etiquetas previas.
# MAGIC
# MAGIC * **Diferencia con Clasificación**:
# MAGIC   * Clasificación: Predice etiquetas conocidas (Churn Sí/No)
# MAGIC   * Clustering: Descubre grupos naturales en los datos
# MAGIC
# MAGIC ### K-Means: Algoritmo de Clustering
# MAGIC
# MAGIC **K-Means** agrupa datos en `K` clusters minimizando la distancia entre puntos y el centroide de su cluster.
# MAGIC
# MAGIC **Funcionamiento:**
# MAGIC 1. Elegir K (número de clusters)
# MAGIC 2. Inicializar K centroides aleatoriamente
# MAGIC 3. Asignar cada punto al centroide más cercano
# MAGIC 4. Recalcular centroides (media de puntos en cada cluster)
# MAGIC 5. Repetir hasta convergencia
# MAGIC
# MAGIC ### Caso de Uso: Segmentación de Clientes
# MAGIC
# MAGIC **Contexto**: E-commerce con 5,000 clientes
# MAGIC
# MAGIC **Variables**:
# MAGIC * **Recency**: Días desde última compra
# MAGIC * **Frequency**: Número de compras en el último año
# MAGIC * **Monetary**: Gasto total en el último año
# MAGIC * **Avg_Order_Value**: Valor promedio por pedido
# MAGIC * **Tenure_Months**: Meses como cliente
# MAGIC
# MAGIC **Objetivo**:
# MAGIC * Identificar segmentos naturales de clientes
# MAGIC * Diseñar campañas de marketing específicas para cada segmento
# MAGIC * Optimizar retención y cross-selling
# MAGIC
# MAGIC **Segmentos esperados**:
# MAGIC * **VIP**: Alta frecuencia, alto gasto, recientes
# MAGIC * **Regulares**: Frecuencia media, gasto medio
# MAGIC * **En Riesgo**: No compran recientemente
# MAGIC * **Nuevos**: Poca antigüedad, actividad baja

# COMMAND ----------

# DBTITLE 1,Importar Librerías
# =============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# =============================================================================

from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import functions as F
from pyspark.sql.types import *

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("✅ Librerías importadas")
print("💡 K-Means requiere normalización (distancias euclídeas)")

# COMMAND ----------

# DBTITLE 1,Crear Dataset de Clientes
# =============================================================================
# CREACIÓN DEL DATASET: CLIENTES E-COMMERCE
# =============================================================================

np.random.seed(42)
n_customers = 5000

# Generar 4 segmentos con características distintas
segments = []

# Segmento 1: VIP (15%)
for _ in range(int(n_customers * 0.15)):
    segments.append({
        'recency': int(np.random.uniform(1, 30)),
        'frequency': int(np.random.uniform(15, 30)),
        'monetary': round(np.random.uniform(5000, 15000), 2),
        'avg_order_value': round(np.random.uniform(200, 600), 2),
        'tenure_months': int(np.random.uniform(12, 60))
    })

# Segmento 2: Regulares (40%)
for _ in range(int(n_customers * 0.40)):
    segments.append({
        'recency': int(np.random.uniform(20, 90)),
        'frequency': int(np.random.uniform(5, 15)),
        'monetary': round(np.random.uniform(1000, 5000), 2),
        'avg_order_value': round(np.random.uniform(80, 250), 2),
        'tenure_months': int(np.random.uniform(6, 36))
    })

# Segmento 3: En Riesgo (25%)
for _ in range(int(n_customers * 0.25)):
    segments.append({
        'recency': int(np.random.uniform(120, 365)),
        'frequency': int(np.random.uniform(2, 8)),
        'monetary': round(np.random.uniform(200, 2000), 2),
        'avg_order_value': round(np.random.uniform(50, 150), 2),
        'tenure_months': int(np.random.uniform(12, 48))
    })

# Segmento 4: Nuevos/Ocasionales (20%)
for _ in range(int(n_customers * 0.20)):
    segments.append({
        'recency': int(np.random.uniform(30, 180)),
        'frequency': int(np.random.uniform(1, 5)),
        'monetary': round(np.random.uniform(50, 1000), 2),
        'avg_order_value': round(np.random.uniform(30, 120), 2),
        'tenure_months': int(np.random.uniform(1, 12))
    })

# Convertir a DataFrame
data = [(i, s['recency'], s['frequency'], s['monetary'], s['avg_order_value'], s['tenure_months']) 
        for i, s in enumerate(segments)]

schema = StructType([
    StructField("customer_id", IntegerType(), False),
    StructField("recency_days", IntegerType(), True),
    StructField("frequency", IntegerType(), True),
    StructField("monetary", DoubleType(), True),
    StructField("avg_order_value", DoubleType(), True),
    StructField("tenure_months", IntegerType(), True)
])

df = spark.createDataFrame(data, schema)

print("=" * 80)
print("📋 DATASET DE CLIENTES CREADO")
print("=" * 80)
print(f"Total de clientes: {df.count()}")
df.show(10)

print("\n📊 ESTADÍSTICAS DESCRIPTIVAS")
df.describe().show()

print("✅ Dataset listo para clustering")

# COMMAND ----------

# DBTITLE 1,EDA
# MAGIC %md
# MAGIC ## 🔍 Análisis Exploratorio (EDA)
# MAGIC
# MAGIC Antes de aplicar K-Means, exploramos las distribuciones y correlaciones entre variables.

# COMMAND ----------

# DBTITLE 1,EDA Visual
# =============================================================================
# ANÁLISIS EXPLORATORIO
# =============================================================================

print("🔍 EDA: EXPLORACIÓN DE VARIABLES")
print("=" * 70)

# Convertir a pandas para visualizar
df_pd = df.toPandas()

# Distribución de variables
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribución de Variables de Clientes', fontsize=16, fontweight='bold')

variables = ['recency_days', 'frequency', 'monetary', 'avg_order_value', 'tenure_months']
for i, var in enumerate(variables):
    row = i // 3
    col = i % 3
    axes[row, col].hist(df_pd[var], bins=50, color='skyblue', edgecolor='black')
    axes[row, col].set_xlabel(var, fontsize=10)
    axes[row, col].set_ylabel('Frecuencia', fontsize=10)
    axes[row, col].set_title(f'Distribución de {var}', fontsize=11, fontweight='bold')
    axes[row, col].grid(alpha=0.3)

axes[1, 2].axis('off')
plt.tight_layout()
plt.show()

# Matriz de correlación
print("\n📊 MATRIZ DE CORRELACIÓN")
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = df_pd[variables].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, ax=ax)
ax.set_title('Matriz de Correlación entre Variables', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("✅ EDA completado")

# COMMAND ----------

# DBTITLE 1,Preparar y Normalizar Datos
# =============================================================================
# PREPARACIÓN Y NORMALIZACIÓN DE DATOS
# =============================================================================

print("🔧 PREPARACIÓN DE DATOS PARA K-MEANS")
print("=" * 50)

# 1. Features
feature_columns = ['recency_days', 'frequency', 'monetary', 'avg_order_value', 'tenure_months']

# 2. Vector Assembly
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features_raw")
df = assembler.transform(df)

# 3. Normalización (CRUCIAL para K-Means - usa distancias euclídeas)
print("\n⚠️  Normalizando features (K-Means es sensible a escala)...")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
scaler_model = scaler.fit(df)
df_scaled = scaler_model.transform(df)

print("✅ Features normalizadas (media=0, std=1)")
print("   - Esto asegura que todas las variables tengan el mismo peso\n")

# Dataset final
df_final = df_scaled.select("customer_id", "features", *feature_columns)

print("Dataset preparado:")
df_final.show(5)
print("✅ Datos listos para K-Means")

# COMMAND ----------

# DBTITLE 1,Elbow Method
# MAGIC %md
# MAGIC ## 📉 Elbow Method: ¿Cuántos Clusters?
# MAGIC
# MAGIC ### Problema: Elegir K
# MAGIC
# MAGIC K-Means requiere especificar `K` (número de clusters) **a priori**. ¿Cómo elegir el mejor K?
# MAGIC
# MAGIC ### Método del Codo (Elbow Method)
# MAGIC
# MAGIC **Procedimiento:**
# MAGIC 1. Entrenar K-Means con diferentes valores de K (ej: 2 a 10)
# MAGIC 2. Calcular **WSSSE** (Within Set Sum of Squared Errors) para cada K
# MAGIC    * WSSSE = Suma de distancias al cuadrado de cada punto a su centroide
# MAGIC    * Menor WSSSE = clusters más compactos
# MAGIC 3. Graficar K vs WSSSE
# MAGIC 4. Buscar el **"codo"** (punto donde la mejora se estabiliza)
# MAGIC
# MAGIC ### Interpretación:
# MAGIC
# MAGIC * **K pequeño**: Pocos clusters grandes, WSSSE alto
# MAGIC * **K grande**: Muchos clusters pequeños, WSSSE bajo (pero overfitting)
# MAGIC * **K óptimo**: En el "codo" del gráfico (balance entre compactación y simplicit)
# MAGIC
# MAGIC ### Otros métodos:
# MAGIC
# MAGIC * **Silhouette Score**: Mide qué tan bien separados están los clusters
# MAGIC * **Conocimiento del negocio**: A veces sabemos cuántos segmentos queremos

# COMMAND ----------

# DBTITLE 1,Elbow Method - Determinar K
# =============================================================================
# ELBOW METHOD: DETERMINAR NÚMERO ÓPTIMO DE CLUSTERS
# =============================================================================

print("📉 ELBOW METHOD: BUSCANDO K ÓPTIMO")
print("=" * 70)

# Probar diferentes valores de K
k_values = range(2, 11)
wssse_values = []

print("Entrenando K-Means para K = 2 a 10...")
for k in k_values:
    kmeans = KMeans(k=k, seed=42, featuresCol="features")
    model = kmeans.fit(df_final)
    wssse = model.summary.trainingCost
    wssse_values.append(wssse)
    print(f"  K={k}: WSSSE={wssse:.2f}")

print("\n✅ Cálculo completado\n")

# Visualizar Elbow
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_values, wssse_values, marker='o', linewidth=2, markersize=8, color='blue')
ax.set_xlabel('Número de Clusters (K)', fontsize=12, fontweight='bold')
ax.set_ylabel('WSSSE (Within Set Sum of Squared Errors)', fontsize=12, fontweight='bold')
ax.set_title('Elbow Method: Determinación de K Óptimo', fontsize=14, fontweight='bold', pad=20)
ax.grid(alpha=0.3)
ax.set_xticks(k_values)

# Marcar el codo sugerido (ejemplo: K=4)
ax.axvline(4, color='red', linestyle='--', linewidth=2, label='K sugerido = 4')
ax.legend(fontsize=11)

plt.tight_layout()
plt.show()

print("💡 Interpretación:")
print("   - Buscar el 'codo' donde la curva se aplana")
print("   - En este caso, K=4 parece un buen balance")
print("   - K muy grande: Overfitting (muchos clusters pequeños)")
print("   - K muy pequeño: Underfitting (pocos clusters grandes)\n")

print("✅ Seleccionaremos K=4 para el modelo final")

# COMMAND ----------

# DBTITLE 1,Entrenar Modelo Final
# =============================================================================
# ENTRENAMIENTO DEL MODELO FINAL: K=4
# =============================================================================

print("🎯 ENTRENAMIENTO DE K-MEANS CON K=4")
print("=" * 70)

# Entrenar K-Means con K=4
kmeans = KMeans(
    k=4,
    seed=42,
    featuresCol="features",
    predictionCol="cluster",
    maxIter=20
)

print("⏳ Entrenando modelo...")
model = kmeans.fit(df_final)
print("✅ Modelo entrenado\n")

# Información del modelo
print("📊 INFORMACIÓN DEL MODELO")
print("=" * 70)
print(f"Número de clusters: {model.summary.k}")
print(f"WSSSE: {model.summary.trainingCost:.2f}")
print(f"Iteraciones: {model.summary.numIter}")
print(f"Tamaño de clusters: {model.summary.clusterSizes}")

# Hacer predicciones (asignar clusters)
predictions = model.transform(df_final)

print("\n📈 DISTRIBUCIÓN DE CLIENTES POR CLUSTER")
predictions.groupBy("cluster").count().orderBy("cluster").show()

print("✅ Clientes asignados a clusters")

# COMMAND ----------

# DBTITLE 1,Interpretación de Clusters
# MAGIC %md
# MAGIC ## 💡 Interpretación de Clusters
# MAGIC
# MAGIC ### Perfil de cada segmento
# MAGIC
# MAGIC Calcularemos las **características promedio** de cada cluster para entender qué tipo de clientes agrupa cada uno.
# MAGIC
# MAGIC ### Métricas clave:
# MAGIC
# MAGIC * **Recency**: Menor = más reciente (mejor)
# MAGIC * **Frequency**: Mayor = más compras (mejor)
# MAGIC * **Monetary**: Mayor = más gasto (mejor)
# MAGIC * **Avg_Order_Value**: Mayor = ticket promedio alto
# MAGIC * **Tenure**: Mayor = más tiempo como cliente
# MAGIC
# MAGIC ### Segmentos esperados:
# MAGIC
# MAGIC * **Cluster 0**: VIP - Recientes, alta frecuencia, alto gasto
# MAGIC * **Cluster 1**: Regulares - Actividad media
# MAGIC * **Cluster 2**: En Riesgo - No compran hace tiempo
# MAGIC * **Cluster 3**: Nuevos/Ocasionales - Baja antigüedad, poca actividad

# COMMAND ----------

# DBTITLE 1,Perfiles de Clusters
# =============================================================================
# PERFILES DE CLUSTERS
# =============================================================================

print("📊 PERFILES DE CLUSTERS")
print("=" * 70)

# Calcular estadísticas por cluster
cluster_profiles = predictions.groupBy("cluster").agg(
    F.count("customer_id").alias("num_customers"),
    F.avg("recency_days").alias("avg_recency"),
    F.avg("frequency").alias("avg_frequency"),
    F.avg("monetary").alias("avg_monetary"),
    F.avg("avg_order_value").alias("avg_order_value"),
    F.avg("tenure_months").alias("avg_tenure")
).orderBy("cluster")

print("Perfiles promedio por cluster:\n")
cluster_profiles.show(truncate=False)

# Convertir a pandas para visualización
profiles_pd = cluster_profiles.toPandas()

# Visualización: Radar chart de perfiles
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Perfiles de Clusters de Clientes', fontsize=16, fontweight='bold')

metrics = ['avg_recency', 'avg_frequency', 'avg_monetary', 'avg_order_value', 'avg_tenure']
metric_names = ['Recency (días)', 'Frequency', 'Monetary (USD)', 'Avg Order Value', 'Tenure (meses)']

for idx, cluster_id in enumerate(range(4)):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    cluster_data = profiles_pd[profiles_pd['cluster'] == cluster_id]
    values = [cluster_data[m].values[0] for m in metrics]
    
    ax.bar(metric_names, values, color=plt.cm.Set2(cluster_id), alpha=0.7, edgecolor='black')
    ax.set_title(f'Cluster {cluster_id} ({int(cluster_data["num_customers"].values[0])} clientes)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor Promedio')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("✅ Perfiles visualizados")

# COMMAND ----------

# DBTITLE 1,Estrategias de Marketing
# MAGIC %md
# MAGIC ## 💼 Estrategias de Marketing por Segmento
# MAGIC
# MAGIC ### Análisis de Perfiles y Recomendaciones
# MAGIC
# MAGIC Basados en los perfiles de cada cluster, diseñaremos estrategias personalizadas:
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **Cluster 0: VIP / Champions** 🏆
# MAGIC **Perfil**:
# MAGIC * Recency: Baja (compran frecuentemente)
# MAGIC * Frequency: Alta (muchas compras)
# MAGIC * Monetary: Alto (mucho gasto)
# MAGIC * Tenure: Media-Alta (clientes establecidos)
# MAGIC
# MAGIC **Estrategia**:
# MAGIC * ✅ **Programa VIP exclusivo**: Acceso anticipado a productos, envío gratis
# MAGIC * ✅ **Cross-sell premium**: Recomendar productos de alta gama
# MAGIC * ✅ **Eventos exclusivos**: Invitaciones a lanzamientos, webinars
# MAGIC * ✅ **Retención**: Contacto periódico, agradecimiento personalizado
# MAGIC * 🚨 **Prioridad**: NO perderlos (LTV muy alto)
# MAGIC
# MAGIC **KPIs**: Mantener frecuencia, aumentar ticket promedio
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **Cluster 1: Regulares / Potenciales** 📈
# MAGIC **Perfil**:
# MAGIC * Recency: Media (compran periódicamente)
# MAGIC * Frequency: Media (varias compras al año)
# MAGIC * Monetary: Medio (gasto moderado)
# MAGIC * Tenure: Media
# MAGIC
# MAGIC **Estrategia**:
# MAGIC * ✅ **Up-sell**: Promover productos de mayor valor
# MAGIC * ✅ **Programas de lealtad**: Puntos, descuentos por volumen
# MAGIC * ✅ **Email marketing**: Newsletters con ofertas personalizadas
# MAGIC * ✅ **Reactivación**: Recordatorios si pasan >60 días sin comprar
# MAGIC * 🎯 **Objetivo**: Convertir en VIP
# MAGIC
# MAGIC **KPIs**: Aumentar frecuencia y monetary
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **Cluster 2: En Riesgo / Dormant** ⚠️
# MAGIC **Perfil**:
# MAGIC * Recency: Alta (hace mucho no compran)
# MAGIC * Frequency: Baja-Media (compraban antes)
# MAGIC * Monetary: Bajo-Medio (gastaban algo)
# MAGIC * Tenure: Media-Alta (llevan tiempo pero inactivos)
# MAGIC
# MAGIC **Estrategia**:
# MAGIC * ✅ **Campaña de reactivación**: "Te extrañamos" con descuento agresivo (20-30%)
# MAGIC * ✅ **Encuesta de abandono**: ¿Por qué dejaron de comprar?
# MAGIC * ✅ **Win-back offers**: Ofertas limitadas, envío gratis
# MAGIC * ✅ **Contenido relevante**: Newsletter con novedades que puedan interesar
# MAGIC * 🚨 **Prioridad**: Recuperar antes de perderlos definitivamente
# MAGIC
# MAGIC **KPIs**: Reducir recency, aumentar frequency
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **Cluster 3: Nuevos / Ocasionales** 🌱
# MAGIC **Perfil**:
# MAGIC * Recency: Variable
# MAGIC * Frequency: Muy baja (1-2 compras)
# MAGIC * Monetary: Bajo (poco gasto)
# MAGIC * Tenure: Baja (clientes nuevos)
# MAGIC
# MAGIC **Estrategia**:
# MAGIC * ✅ **Onboarding**: Email series educativa sobre productos
# MAGIC * ✅ **Descuento de bienvenida**: Incentivo para segunda compra
# MAGIC * ✅ **Social proof**: Reseñas, testimonios
# MAGIC * ✅ **Facilitar segunda compra**: Recomendaciones personalizadas
# MAGIC * 🎯 **Objetivo**: Convertir en Regulares
# MAGIC
# MAGIC **KPIs**: Aumentar frequency (de 1 a 3+ compras)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Resumen de Presupuesto:
# MAGIC
# MAGIC | Cluster | % Clientes | Prioridad | Inversión Marketing | ROI Esperado |
# MAGIC |---------|------------|-----------|------------------------|---------------|
# MAGIC | VIP (0) | 15% | 🔴 Muy Alta | 40% | Alto (retención) |
# MAGIC | Regulares (1) | 40% | 🟡 Alta | 30% | Muy Alto (up-sell) |
# MAGIC | En Riesgo (2) | 25% | 🟠 Media | 20% | Medio (recuperación) |
# MAGIC | Nuevos (3) | 20% | 🟢 Baja | 10% | Bajo (experimentación) |

# COMMAND ----------

# DBTITLE 1,Visualización con PCA
# MAGIC %md
# MAGIC ## 🎨 Visualización de Clusters con PCA
# MAGIC
# MAGIC ### Problema: Visualizar 5 Dimensiones
# MAGIC
# MAGIC Nuestros datos tienen 5 features (recency, frequency, monetary, avg_order_value, tenure). ¿Cómo visualizarlos en 2D?
# MAGIC
# MAGIC ### Solución: PCA (Principal Component Analysis)
# MAGIC
# MAGIC **PCA** reduce dimensionalidad proyectando datos a 2 componentes principales que capturan la mayor varianza.
# MAGIC
# MAGIC **Limitación**: Perdemos información (solo vemos 2 de 5 dimensiones), pero ganamos **visualización intuitiva**.

# COMMAND ----------

# DBTITLE 1,Visualizar con PCA
# =============================================================================
# VISUALIZACIÓN DE CLUSTERS CON PCA
# =============================================================================

from pyspark.ml.feature import PCA

print("🎨 VISUALIZACIÓN DE CLUSTERS (PCA)")
print("=" * 70)

# Aplicar PCA para reducir a 2 dimensiones
pca = PCA(k=2, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(predictions)
predictions_pca = pca_model.transform(predictions)

print(f"✅ PCA aplicado (5D → 2D)")
print(f"Varianza explicada: {sum(pca_model.explainedVariance.toArray()):.2%}\n")

# Convertir a pandas para visualizar
visualization_data = predictions_pca.select("cluster", "pca_features").toPandas()

# Extraer componentes PCA
visualization_data['PC1'] = visualization_data['pca_features'].apply(lambda x: float(x[0]))
visualization_data['PC2'] = visualization_data['pca_features'].apply(lambda x: float(x[1]))

# Visualizar
fig, ax = plt.subplots(figsize=(10, 8))

colors = ['red', 'blue', 'green', 'orange']
for cluster_id in range(4):
    cluster_data = visualization_data[visualization_data['cluster'] == cluster_id]
    ax.scatter(cluster_data['PC1'], cluster_data['PC2'], 
               c=colors[cluster_id], label=f'Cluster {cluster_id}', 
               alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Componente Principal 1', fontsize=12, fontweight='bold')
ax.set_ylabel('Componente Principal 2', fontsize=12, fontweight='bold')
ax.set_title('Visualización de Clusters con PCA (5D → 2D)', fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='best')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("💡 Interpretación:")
print("   - Clusters bien separados = segmentación exitosa")
print("   - Clusters solapados = posible ajuste de K o features\n")

print("✅ Visualización completada")

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## 🎓 Conclusiones y Próximos Pasos
# MAGIC
# MAGIC ### ✅ Lo que logramos:
# MAGIC
# MAGIC 1. Creamos un dataset de **5,000 clientes** con variables RFM
# MAGIC 2. Aplicamos **normalización** (crítico para K-Means)
# MAGIC 3. Usamos **Elbow Method** para determinar K óptimo = 4
# MAGIC 4. Entrenamos **K-Means** y segmentamos clientes en 4 clusters
# MAGIC 5. **Perfilamos cada segmento** con estadísticas descriptivas
# MAGIC 6. Diseñamos **estrategias de marketing personalizadas**
# MAGIC 7. Visualizamos clusters con **PCA**
# MAGIC
# MAGIC ### 📊 Segmentos identificados:
# MAGIC
# MAGIC * **Cluster 0 (VIP)**: Clientes de alto valor - Retener con programa VIP
# MAGIC * **Cluster 1 (Regulares)**: Potencial de up-sell - Convertir en VIP
# MAGIC * **Cluster 2 (En Riesgo)**: Inactivos - Campaña de reactivación
# MAGIC * **Cluster 3 (Nuevos)**: Baja actividad - Onboarding y segunda compra
# MAGIC
# MAGIC ### 📈 Impacto de Negocio:
# MAGIC
# MAGIC **Antes de segmentación**:
# MAGIC * Marketing genérico para todos
# MAGIC * Desperdicio de presupuesto
# MAGIC * Baja personalización
# MAGIC
# MAGIC **Después de segmentación**:
# MAGIC * ✅ **40% del presupuesto** a VIP (15% de clientes, 60% de revenue)
# MAGIC * ✅ **30% del presupuesto** a Regulares (up-sell)
# MAGIC * ✅ **20% del presupuesto** a En Riesgo (win-back)
# MAGIC * ✅ **10% del presupuesto** a Nuevos (experimentación)
# MAGIC * 📈 **ROI aumentado** por targeting preciso
# MAGIC
# MAGIC ### 🚀 Próximos Pasos:
# MAGIC
# MAGIC #### 1. **Validación de Clusters**
# MAGIC ```python
# MAGIC # Silhouette Score (mide separación)
# MAGIC evaluator = ClusteringEvaluator()
# MAGIC silhouette = evaluator.evaluate(predictions)
# MAGIC print(f"Silhouette Score: {silhouette}")
# MAGIC # > 0.5 = buena separación
# MAGIC ```
# MAGIC
# MAGIC #### 2. **Features Adicionales**
# MAGIC * **Categoría de productos**: ¿Qué compran?
# MAGIC * **Canal preferido**: Web, app, tienda física
# MAGIC * **Estacionalidad**: ¿Cuándo compran?
# MAGIC * **Tasa de devolución**: Satisfacción implícita
# MAGIC * **Engagement**: Aperturas de email, visitas sin compra
# MAGIC
# MAGIC #### 3. **Modelos Híbridos**
# MAGIC * **K-Means + Clasificación**: Entrenar clasificador para asignar nuevos clientes a clusters automáticamente
# MAGIC * **Clustering jerárquico**: Subclusters dentro de VIP (VIP-Premium, VIP-Standard)
# MAGIC
# MAGIC #### 4. **Algoritmos Alternativos**
# MAGIC * **DBSCAN**: Detecta clusters de forma arbitraria (no asume esferas)
# MAGIC * **Gaussian Mixture Models**: Asigna probabilidades de pertenencia
# MAGIC * **Hierarchical Clustering**: Dendrograma de similitud
# MAGIC
# MAGIC #### 5. **Deployment y Monitoreo**
# MAGIC * **ETL automático**: Recalcular segmentos mensualmente
# MAGIC * **Dashboard en tiempo real**: Tamaño y composición de clusters
# MAGIC * **Alertas**: Si un VIP pasa a En Riesgo → contactar inmediatamente
# MAGIC * **A/B Testing**: Medir impacto de estrategias personalizadas
# MAGIC
# MAGIC #### 6. **Integración con CRM**
# MAGIC * Exportar segmentos a plataforma de email marketing
# MAGIC * Personalizar website según cluster del usuario
# MAGIC * Notificaciones push personalizadas
# MAGIC
# MAGIC ### 📚 Recursos:
# MAGIC
# MAGIC * [PySpark ML Clustering](https://spark.apache.org/docs/latest/ml-clustering.html)
# MAGIC * [K-Means en Databricks](https://docs.databricks.com/machine-learning/train-model/clustering.html)
# MAGIC * [RFM Analysis](https://en.wikipedia.org/wiki/RFM_(market_research))
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **¡Felicitaciones!** Has completado un proyecto de **Segmentación de Clientes** con K-Means Clustering. 🎉