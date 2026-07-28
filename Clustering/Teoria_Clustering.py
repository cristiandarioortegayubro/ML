# Databricks notebook source
# DBTITLE 1,Teoría de Clustering
# MAGIC %md
# MAGIC # Teoría de Clustering
# MAGIC
# MAGIC ## 1. Introducción
# MAGIC
# MAGIC El **Clustering (Agrupamiento)** es una técnica de aprendizaje no supervisado que agrupa datos similares sin etiquetas previas.
# MAGIC
# MAGIC ### Definición
# MAGIC
# MAGIC Dado un conjunto de datos $\mathcal{D} = \{\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n\}$ donde $\mathbf{x}_i \in \mathbb{R}^d$:
# MAGIC
# MAGIC **Objetivo:** Particionar $\mathcal{D}$ en $k$ grupos (clusters) $C_1, C_2, ..., C_k$ tales que:
# MAGIC
# MAGIC * **Cohesión intra-cluster**: Alta similitud dentro de cada grupo
# MAGIC * **Separación inter-cluster**: Baja similitud entre grupos
# MAGIC
# MAGIC ### Aplicaciones
# MAGIC
# MAGIC * **Segmentación de clientes**: Agrupar por comportamiento de compra
# MAGIC * **Compresión de imágenes**: Reducir colores similares
# MAGIC * **Detección de anomalías**: Puntos que no pertenecen a ningún cluster
# MAGIC * **Recomendaciones**: Usuarios con gustos similares
# MAGIC * **Biología**: Taxonomía, genes con funciones similares

# COMMAND ----------

# DBTITLE 1,Métricas de Similitud
# MAGIC %md
# MAGIC ## 2. Métricas de Similitud y Distancia
# MAGIC
# MAGIC ### 2.1 Distancia Euclidiana
# MAGIC
# MAGIC $$d(\mathbf{x}, \mathbf{y}) = ||\mathbf{x} - \mathbf{y}||_2 = \sqrt{\sum_{j=1}^{d}(x_j - y_j)^2}$$
# MAGIC
# MAGIC **Más común, sensible a escala.**
# MAGIC
# MAGIC ### 2.2 Distancia de Manhattan
# MAGIC
# MAGIC $$d(\mathbf{x}, \mathbf{y}) = ||\mathbf{x} - \mathbf{y}||_1 = \sum_{j=1}^{d}|x_j - y_j|$$
# MAGIC
# MAGIC ### 2.3 Distancia de Minkowski
# MAGIC
# MAGIC Generalización:
# MAGIC
# MAGIC $$d(\mathbf{x}, \mathbf{y}) = \left(\sum_{j=1}^{d}|x_j - y_j|^p\right)^{1/p}$$
# MAGIC
# MAGIC * $p=1$: Manhattan
# MAGIC * $p=2$: Euclidiana
# MAGIC * $p \rightarrow \infty$: Chebyshev
# MAGIC
# MAGIC ### 2.4 Similitud Coseno
# MAGIC
# MAGIC $$\text{sim}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{||\mathbf{x}|| \cdot ||\mathbf{y}||} = \cos(\theta)$$
# MAGIC
# MAGIC **Útil para vectores de texto (TF-IDF).**

# COMMAND ----------

# DBTITLE 1,K-Means
# MAGIC %md
# MAGIC ## 3. K-Means Clustering
# MAGIC
# MAGIC ### Algoritmo
# MAGIC
# MAGIC **Objetivo:** Minimizar la varianza intra-cluster (WCSS):
# MAGIC
# MAGIC $$J = \sum_{i=1}^{k} \sum_{\mathbf{x} \in C_i} ||\mathbf{x} - \boldsymbol{\mu}_i||^2$$
# MAGIC
# MAGIC Donde $\boldsymbol{\mu}_i$ es el centroide del cluster $i$.
# MAGIC
# MAGIC ### Algoritmo Lloyd (K-Means estándar)
# MAGIC
# MAGIC ```python
# MAGIC 1. Inicializar: Seleccionar k centroides aleatorios μ1, ..., μk
# MAGIC
# MAGIC 2. Repetir hasta convergencia:
# MAGIC    a) Asignación: Para cada punto xi:
# MAGIC       C(i) = argmin_j ||xi - μj||^2
# MAGIC    
# MAGIC    b) Actualización: Para cada cluster j:
# MAGIC       μj = (1/|Cj|) ∑_{xi ∈ Cj} xi
# MAGIC    
# MAGIC 3. Retornar: Clusters y centroides
# MAGIC ```
# MAGIC
# MAGIC ### Propiedades
# MAGIC
# MAGIC * **Convergencia**: Garantizada (puede ser óptimo local)
# MAGIC * **Complejidad**: $O(n \cdot k \cdot d \cdot t)$ donde $t$ = iteraciones
# MAGIC * **Forma de clusters**: Esféricos, similar tamaño
# MAGIC * **Sensibilidad**: A inicialización y outliers
# MAGIC
# MAGIC ### K-Means++ (Inicialización Inteligente)
# MAGIC
# MAGIC 1. Elegir primer centroide al azar
# MAGIC 2. Para cada siguiente centroide:
# MAGIC    - Calcular $D(x)$ = distancia de cada punto al centroide más cercano
# MAGIC    - Elegir nuevo centroide con probabilidad proporcional a $D(x)^2$
# MAGIC 3. Ejecutar K-Means estándar
# MAGIC
# MAGIC **Ventaja:** Mejores inicializaciones, menos iteraciones.

# COMMAND ----------

# DBTITLE 1,Selección de K
# MAGIC %md
# MAGIC ## 4. Selección del Número de Clusters (k)
# MAGIC
# MAGIC ### 4.1 Método del Codo (Elbow Method)
# MAGIC
# MAGIC Graficar WCSS vs $k$:
# MAGIC
# MAGIC $$WCSS(k) = \sum_{i=1}^{k} \sum_{\mathbf{x} \in C_i} ||\mathbf{x} - \boldsymbol{\mu}_i||^2$$
# MAGIC
# MAGIC Buscar "codo" donde la mejora se desacelera.
# MAGIC
# MAGIC ### 4.2 Coeficiente de Silueta
# MAGIC
# MAGIC Para cada punto $\mathbf{x}_i$:
# MAGIC
# MAGIC $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
# MAGIC
# MAGIC Donde:
# MAGIC - $a(i)$: Distancia promedio a puntos en su cluster
# MAGIC - $b(i)$: Distancia promedio al cluster más cercano
# MAGIC
# MAGIC **Interpretación:**
# MAGIC - $s(i) \approx 1$: Bien asignado
# MAGIC - $s(i) \approx 0$: En la frontera
# MAGIC - $s(i) < 0$: Mal asignado
# MAGIC
# MAGIC **Promedio del conjunto:**
# MAGIC
# MAGIC $$\bar{s} = \frac{1}{n}\sum_{i=1}^{n} s(i)$$
# MAGIC
# MAGIC Elegir $k$ que maximice $\bar{s}$.
# MAGIC
# MAGIC ### 4.3 Davies-Bouldin Index
# MAGIC
# MAGIC $$DB = \frac{1}{k}\sum_{i=1}^{k} \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(\boldsymbol{\mu}_i, \boldsymbol{\mu}_j)}$$
# MAGIC
# MAGIC **Más bajo es mejor** (clusters compactos y separados).

# COMMAND ----------

# DBTITLE 1,Otros Algoritmos
# MAGIC %md
# MAGIC ## 5. Otros Algoritmos de Clustering
# MAGIC
# MAGIC ### 5.1 Hierarchical Clustering (Jerárquico)
# MAGIC
# MAGIC **Aglomerativo (bottom-up):**
# MAGIC
# MAGIC 1. Empezar: Cada punto es un cluster
# MAGIC 2. Repetir: Unir los 2 clusters más cercanos
# MAGIC 3. Hasta: Un solo cluster o $k$ clusters
# MAGIC
# MAGIC **Linkage (criterio de unión):**
# MAGIC
# MAGIC * **Single**: $\min_{\mathbf{x} \in C_i, \mathbf{y} \in C_j} d(\mathbf{x}, \mathbf{y})$
# MAGIC * **Complete**: $\max_{\mathbf{x} \in C_i, \mathbf{y} \in C_j} d(\mathbf{x}, \mathbf{y})$
# MAGIC * **Average**: Promedio de distancias
# MAGIC * **Ward**: Minimiza varianza
# MAGIC
# MAGIC **Ventajas:** Dendrograma, no requiere $k$ a priori
# MAGIC
# MAGIC **Desventajas:** $O(n^2 \log n)$ o $O(n^3)$
# MAGIC
# MAGIC ### 5.2 DBSCAN (Density-Based)
# MAGIC
# MAGIC Agrupa puntos densos, marca outliers.
# MAGIC
# MAGIC **Parámetros:**
# MAGIC - $\epsilon$: Radio de vecindad
# MAGIC - MinPts: Mínimo de puntos para ser denso
# MAGIC
# MAGIC **Ventajas:**
# MAGIC * Detecta formas arbitrarias
# MAGIC * Identifica outliers
# MAGIC * No requiere $k$
# MAGIC
# MAGIC **Desventajas:**
# MAGIC * Sensible a $\epsilon$ y MinPts
# MAGIC * Problemas con densidades variables
# MAGIC
# MAGIC ### 5.3 Gaussian Mixture Models (GMM)
# MAGIC
# MAGIC Modela datos como mezcla de distribuciones gaussianas:
# MAGIC
# MAGIC $$P(\mathbf{x}) = \sum_{i=1}^{k} \pi_i \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$$
# MAGIC
# MAGIC **Algoritmo EM (Expectation-Maximization)**
# MAGIC
# MAGIC **Ventajas:**
# MAGIC * Clusters elípticos
# MAGIC * Asignación probabilística (soft clustering)
# MAGIC * Fundamento estadístico
# MAGIC
# MAGIC **Desventajas:**
# MAGIC * Más costoso que K-Means
# MAGIC * Sensible a inicialización

# COMMAND ----------

# DBTITLE 1,Evaluación
# MAGIC %md
# MAGIC ## 6. Evaluación de Clustering
# MAGIC
# MAGIC ### Métricas Internas (sin etiquetas)
# MAGIC
# MAGIC 1. **Coeficiente de Silueta**: $[-1, 1]$, mayor mejor
# MAGIC 2. **Davies-Bouldin Index**: $[0, \infty)$, menor mejor
# MAGIC 3. **Calinski-Harabasz Index**: Mayor mejor
# MAGIC
# MAGIC ### Métricas Externas (con etiquetas ground truth)
# MAGIC
# MAGIC 1. **Adjusted Rand Index (ARI)**: $[-1, 1]$, 1 = perfect match
# MAGIC 2. **Normalized Mutual Information (NMI)**: $[0, 1]$, 1 = perfect
# MAGIC 3. **V-Measure**: $[0, 1]$, combina homogeneity y completeness
# MAGIC
# MAGIC ### Consideraciones
# MAGIC
# MAGIC * **No hay métrica universal** para elegir el mejor clustering
# MAGIC * Combinar múltiples métricas
# MAGIC * **Validación visual** crucial en 2D/3D
# MAGIC * **Interpretación de dominio** es clave

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## 7. Conclusiones
# MAGIC
# MAGIC ### Resumen
# MAGIC
# MAGIC | Algoritmo | Forma Clusters | Requiere k | Outliers | Complejidad |
# MAGIC |-----------|---------------|-----------|----------|-------------|
# MAGIC | K-Means | Esféricos | Sí | No | $O(nkdt)$ |
# MAGIC | Hierarchical | Cualquiera | No | No | $O(n^2 \log n)$ |
# MAGIC | DBSCAN | Arbitraria | No | Sí | $O(n \log n)$ |
# MAGIC | GMM | Elípticos | Sí | No | $O(nkdt)$ |
# MAGIC
# MAGIC ### Cuándo Usar Cada Método
# MAGIC
# MAGIC * **K-Means**: Rápido, clusters esféricos, datos grandes
# MAGIC * **Hierarchical**: Exploración, visualización, datos pequeños
# MAGIC * **DBSCAN**: Formas irregulares, outliers importantes
# MAGIC * **GMM**: Asignación probabilística, modelo generativo
# MAGIC
# MAGIC ### Consejos Prácticos
# MAGIC
# MAGIC 1. **Estandarizar datos** antes de clustering
# MAGIC 2. **Probar múltiples valores de $k$**
# MAGIC 3. **Ejecutar varias veces** (inicializaciones aleatorias)
# MAGIC 4. **Visualizar** resultados cuando sea posible
# MAGIC 5. **Validar** con conocimiento de dominio

# COMMAND ----------

