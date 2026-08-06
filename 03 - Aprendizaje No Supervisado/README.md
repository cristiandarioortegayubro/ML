# Aprendizaje No Supervisado

## 🎯 Definición

**Aprendizaje No Supervisado** es un paradigma de Machine Learning donde el modelo aprende patrones y estructuras **ocultas** en datos **sin etiquetas**. No hay "respuestas correctas" predefinidas; el objetivo es descubrir la organización intrínseca de los datos.

### Características Clave

* 🔍 **Datos no etiquetados**: Sin respuestas correctas conocidas
* 🧩 **Descubrimiento de patrones**: Encontrar estructura, grupos, relaciones
* 📉 **Reducción de complejidad**: Simplificar datos manteniendo información esencial
* 🎨 **Exploración**: Entender la naturaleza de los datos

## 📁 Contenido

Esta carpeta contiene los principales tipos de aprendizaje no supervisado:

### 1️⃣ Clustering (Agrupamiento)

**Objetivo**: Agrupar datos en **clusters** o grupos naturales basados en similitud.

* **Carpeta**: `Clustering/`
* **Ejemplos**:
  - Segmentación de clientes (perfiles de compra)
  - Agrupación de documentos por tema
  - Detección de comunidades en redes sociales
  - Compresión de imágenes
* **Algoritmos**:
  - **K-Means**: Particional, minimiza varianza intra-cluster
  - **DBSCAN**: Basado en densidad, detecta outliers
  - **Hierarchical Clustering**: Construye dendrogramas
  - **Gaussian Mixture Models (GMM)**: Probabilístico
  - **Bisecting K-Means**: Divisivo jerárquico

### 2️⃣ Reducción de Dimensionalidad (Próximamente)

**Objetivo**: Reducir el número de features preservando información relevante.

* **Ejemplos**:
  - Visualización de datos de alta dimensión
  - Feature engineering automático
  - Compresión de datos
  - Eliminación de ruido
* **Algoritmos** (a desarrollar):
  - **PCA (Principal Component Analysis)**: Componentes principales
  - **t-SNE**: Visualización no lineal
  - **UMAP**: Preservación de estructura local y global
  - **Autoencoders**: Deep learning para compresión

### 3️⃣ Detección de Anomalías (Próximamente)

**Objetivo**: Identificar datos atípicos o raros.

* **Ejemplos**:
  - Detección de fraude
  - Fallas en sistemas industriales
  - Intrusión en redes
* **Algoritmos** (a desarrollar):
  - **Isolation Forest**
  - **One-Class SVM**
  - **Local Outlier Factor (LOF)**

## 🔄 Proceso de Aprendizaje No Supervisado

```
1. Recolección de Datos (sin etiquetas)
   ↓
2. Exploración y Visualización
   ↓
3. Preprocesamiento (normalización crucial)
   ↓
4. Selección de Algoritmo
   ↓
5. Entrenamiento / Ajuste
   ↓
6. Evaluación (métricas intrínsecas)
   ↓
7. Interpretación de Resultados
   ↓
8. Validación con expertos del dominio
```

## 📊 Clustering: Conceptos Clave

### Métricas de Similitud

* **Distancia Euclidiana**: $d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$
* **Distancia Manhattan**: $d(x, y) = \sum_{i=1}^{n}|x_i - y_i|$
* **Distancia Coseno**: Ángulo entre vectores
* **Distancia de Mahalanobis**: Normalizada por covarianza

### Métricas de Evaluación

**Intrínsecas** (sin etiquetas):
* **Silhouette Score**: Cohesión intra-cluster vs separación inter-cluster
* **Davies-Bouldin Index**: Ratio compacidad/separación (menor es mejor)
* **Calinski-Harabasz Index**: Ratio varianza inter/intra (mayor es mejor)
* **Inertia (SSE)**: Suma de distancias al cuadrado a centroides

**Extrínsecas** (con etiquetas de referencia, si existen):
* **Adjusted Rand Index (ARI)**
* **Normalized Mutual Information (NMI)**
* **V-measure**

### Determinación del Número de Clusters (K)

* **Elbow Method**: Buscar "codo" en gráfico de Inertia vs K
* **Silhouette Analysis**: Maximizar silhouette score promedio
* **Gap Statistic**: Comparar con distribución aleatoria
* **Business Knowledge**: Conocimiento del dominio

## 📊 Comparación: Supervisado vs No Supervisado

| Aspecto | Supervisado | No Supervisado |
|---------|-------------|----------------|
| **Datos** | Etiquetados | Sin etiquetas |
| **Objetivo** | Predecir etiqueta | Descubrir estructura |
| **Evaluación** | Métricas claras (accuracy, RMSE) | Métricas intrínsecas (silhouette) |
| **Interpretación** | Directa (correcto/incorrecto) | Requiere análisis experto |
| **Ejemplos** | Clasificación, Regresión | Clustering, PCA, Anomalías |
| **Uso** | Predicción en producción | Exploración, segmentación |

## 🚀 Orden de Estudio Recomendado

### Para Principiantes:

1. **Fundamentos** (carpeta `../Fundamentos/`)
   - Introducción a Machine Learning
   - Conceptos de vectores, distancias

2. **Clustering** (`./Clustering/`)
   - Teoría de clustering (conceptos, métricas)
   - K-Means (algoritmo más usado)
   - Interpretación de resultados
   - Problema: Segmentación de clientes

3. **Aprendizaje Supervisado** (`../Aprendizaje Supervisado/`)
   - Comparar con clustering supervisado

### Para Avanzados:

1. DBSCAN y clustering basado en densidad
2. PCA y reducción de dimensionalidad
3. Autoencoders y embeddings
4. Detección de anomalías

## 💼 Aplicaciones Reales

### Clustering

* 🛒 **Retail**: Segmentación de clientes (RFM: Recency, Frequency, Monetary)
* 🧬 **Bioinformática**: Agrupación de genes, análisis de secuencias
* 📰 **Media**: Agrupación de noticias por tema
* 🌍 **Geografía**: Identificación de regiones similares
* 🏥 **Healthcare**: Identificación de subtipos de enfermedades
* 📱 **Telecom**: Segmentación de patrones de uso

### Reducción de Dimensionalidad

* 🖼️ **Computer Vision**: Compresión de imágenes, extracción de features
* 🧠 **Neuroscience**: Análisis de datos de fMRI
* 💳 **Finance**: Reducción de factores de riesgo en portafolios
* 🎵 **Music**: Recomendación basada en embeddings

### Detección de Anomalías

* 💳 **Banking**: Detección de transacciones fraudulentas
* 🏭 **Manufacturing**: Detección de fallas en equipos
* 🔐 **Cybersecurity**: Intrusión en redes
* 🏥 **Healthcare**: Detección de valores médicos atípicos

## 🛠️ Herramientas en Databricks

* **PySpark ML**: K-Means, Bisecting K-Means, Gaussian Mixture
* **MLflow**: Tracking de experimentos de clustering
* **Distributed Computing**: Clustering a escala masiva
* **Visualization**: Dashboards interactivos para explorar clusters

## 📚 Recursos

* [PySpark Clustering Guide](https://spark.apache.org/docs/latest/ml-clustering.html)
* [Unsupervised Learning - Stanford](https://web.stanford.edu/~jurafsky/slp3/)
* [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
* [Databricks Feature Engineering](https://docs.databricks.com/machine-learning/feature-store/index.html)

---

**Siguiente paso**: Explora `../Aprendizaje por Refuerzo/` para aprender cómo un agente aprende mediante interacción y recompensas.
