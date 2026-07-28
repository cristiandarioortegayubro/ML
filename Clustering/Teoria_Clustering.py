# Databricks notebook source
# DBTITLE 1,Título y Objetivos
# MAGIC %md
# MAGIC # Teoría de Clustering: Fundamentos Matemáticos
# MAGIC
# MAGIC ## 🎯 Objetivos
# MAGIC
# MAGIC * **K-Means**: Algoritmo, derivación, convergencia
# MAGIC * **Métricas**: Silhouette, Elbow, Davies-Bouldin
# MAGIC * **Comparación de algoritmos**: K-Means, DBSCAN, Hierárquico
# MAGIC * **Aplicaciones**: Segmentación, reducción de datos
# MAGIC
# MAGIC ### Contenido
# MAGIC
# MAGIC 1. Introducción a clustering
# MAGIC 2. K-Means: Matemáticas
# MAGIC 3. Elección del número de clusters
# MAGIC 4. Otros algoritmos
# MAGIC 5. Comparación y aplicaciones

# COMMAND ----------

# DBTITLE 1,Introducción
# MAGIC %md
# MAGIC ## 1. Introducción a Clustering
# MAGIC
# MAGIC ### 1.1 ¿Qué es Clustering?
# MAGIC
# MAGIC **Definición:** Agrupar datos en clusters (grupos) donde:
# MAGIC * Observaciones en el **mismo cluster** son **similares**
# MAGIC * Observaciones en **diferentes clusters** son **diferentes**
# MAGIC
# MAGIC **Aprendizaje no supervisado:** No hay etiquetas $y$, solo features $\mathbf{x}$
# MAGIC
# MAGIC ### 1.2 Tipos de Clustering
# MAGIC
# MAGIC **1. Particional (K-Means, K-Medoids)**
# MAGIC * Divide datos en $k$ clusters disjuntos
# MAGIC * Cada punto pertenece a exactamente un cluster
# MAGIC
# MAGIC **2. Jerárquico (Hierarchical)**
# MAGIC * Dendrograma (estructura de árbol)
# MAGIC * Aglomerativo (bottom-up) o divisivo (top-down)
# MAGIC
# MAGIC **3. Basado en densidad (DBSCAN)**
# MAGIC * Identifica regiones densas
# MAGIC * Puede encontrar clusters de forma arbitraria
# MAGIC * Maneja ruido (outliers)
# MAGIC
# MAGIC ### 1.3 Medidas de Distancia
# MAGIC
# MAGIC **Euclidiana (L2):**
# MAGIC $$d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{p}(x_i - y_i)^2}$$
# MAGIC
# MAGIC **Manhattan (L1):**
# MAGIC $$d(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{p}|x_i - y_i|$$
# MAGIC
# MAGIC **Coseno (para texto):**
# MAGIC $$d(\mathbf{x}, \mathbf{y}) = 1 - \frac{\mathbf{x}^T\mathbf{y}}{||\mathbf{x}|| \cdot ||\mathbf{y}||}$$

# COMMAND ----------

# DBTITLE 1,K-Means Matemático
# MAGIC %md
# MAGIC ## 2. K-Means: Derivación Matemática
# MAGIC
# MAGIC ### 2.1 Formulación del Problema
# MAGIC
# MAGIC **Objetivo:** Minimizar la inercia (within-cluster sum of squares)
# MAGIC
# MAGIC $$\min_{\{C_k\}_{k=1}^{K}} W = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} ||\mathbf{x}_i - \boldsymbol{\mu}_k||^2$$
# MAGIC
# MAGIC donde:
# MAGIC * $C_k$: Cluster $k$
# MAGIC * $\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i$: Centroide del cluster $k$
# MAGIC
# MAGIC ### 2.2 Algoritmo de Lloyd
# MAGIC
# MAGIC **Iteración:**
# MAGIC
# MAGIC **Paso 1 (Asignación):** Para cada $\mathbf{x}_i$:
# MAGIC $$c_i = \arg\min_k ||\mathbf{x}_i - \boldsymbol{\mu}_k||^2$$
# MAGIC
# MAGIC **Paso 2 (Actualización):** Para cada cluster $k$:
# MAGIC $$\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i$$
# MAGIC
# MAGIC **Inicialización:** K-Means++ (mejora sobre aleatorio)
# MAGIC
# MAGIC **Convergencia:**
# MAGIC * Función objetivo $W$ decrece monótonamente
# MAGIC * Converge a mínimo local (no necesariamente global)
# MAGIC * Típicamente converge en 10-20 iteraciones
# MAGIC
# MAGIC ### 2.3 Demostración de Optimalidad de Centroides
# MAGIC
# MAGIC **Objetivo:** Minimizar
# MAGIC $$J_k = \sum_{\mathbf{x}_i \in C_k} ||\mathbf{x}_i - \boldsymbol{\mu}_k||^2$$
# MAGIC
# MAGIC **Expandir:**
# MAGIC $$J_k = \sum_{\mathbf{x}_i \in C_k} (\mathbf{x}_i - \boldsymbol{\mu}_k)^T(\mathbf{x}_i - \boldsymbol{\mu}_k)$$
# MAGIC
# MAGIC **Derivar con respecto a $\boldsymbol{\mu}_k$:**
# MAGIC $$\frac{\partial J_k}{\partial \boldsymbol{\mu}_k} = \sum_{\mathbf{x}_i \in C_k} -2(\mathbf{x}_i - \boldsymbol{\mu}_k) = 0$$
# MAGIC
# MAGIC **Despejar:**
# MAGIC $$\sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i = |C_k| \boldsymbol{\mu}_k$$
# MAGIC
# MAGIC $$\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i \quad \text{(media)}$$
# MAGIC
# MAGIC ∴ El centroide óptimo es la **media** del cluster.

# COMMAND ----------

# DBTITLE 1,Elección de K
# MAGIC %md
# MAGIC ## 3. Elección del Número de Clusters ($k$)
# MAGIC
# MAGIC ### 3.1 Método del Codo (Elbow Method)
# MAGIC
# MAGIC **Idea:** Graficar inercia vs $k$
# MAGIC
# MAGIC $$\text{Inercia}(k) = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} ||\mathbf{x}_i - \boldsymbol{\mu}_k||^2$$
# MAGIC
# MAGIC **Observación:**
# MAGIC * $k \uparrow$ → Inercia ↓ (siempre)
# MAGIC * $k = n$ → Inercia = 0
# MAGIC
# MAGIC **Criterio:** Elegir $k$ donde la curva hace un "codo" (disminución marginal se reduce)
# MAGIC
# MAGIC **Limitación:** Subjetivo, puede no haber codo claro
# MAGIC
# MAGIC ### 3.2 Coeficiente de Silueta (Silhouette)
# MAGIC
# MAGIC **Para cada punto $\mathbf{x}_i$:**
# MAGIC
# MAGIC 1. **$a_i$:** Distancia promedio a puntos en su cluster
# MAGIC    $$a_i = \frac{1}{|C_{k_i}| - 1} \sum_{\mathbf{x}_j \in C_{k_i}, j \neq i} d(\mathbf{x}_i, \mathbf{x}_j)$$
# MAGIC
# MAGIC 2. **$b_i$:** Distancia promedio al cluster más cercano
# MAGIC    $$b_i = \min_{k \neq k_i} \frac{1}{|C_k|} \sum_{\mathbf{x}_j \in C_k} d(\mathbf{x}_i, \mathbf{x}_j)$$
# MAGIC
# MAGIC 3. **Silueta:**
# MAGIC    $$s_i = \frac{b_i - a_i}{\max(a_i, b_i)} \in [-1, 1]$$
# MAGIC
# MAGIC **Interpretación:**
# MAGIC * $s_i \approx 1$: Bien asignado (lejos de otros clusters)
# MAGIC * $s_i \approx 0$: En el borde entre clusters
# MAGIC * $s_i < 0$: Mal asignado (más cerca de otro cluster)
# MAGIC
# MAGIC **Silueta promedio:**
# MAGIC $$\bar{s} = \frac{1}{n} \sum_{i=1}^{n} s_i$$
# MAGIC
# MAGIC **Criterio:** Elegir $k$ que maximice $\bar{s}$
# MAGIC
# MAGIC ### 3.3 Índice de Davies-Bouldin
# MAGIC
# MAGIC $$DB = \frac{1}{K} \sum_{k=1}^{K} \max_{k' \neq k} \frac{\sigma_k + \sigma_{k'}}{d(\boldsymbol{\mu}_k, \boldsymbol{\mu}_{k'})}$$
# MAGIC
# MAGIC donde:
# MAGIC * $\sigma_k$: Dispersión promedio en cluster $k$
# MAGIC * $d(\boldsymbol{\mu}_k, \boldsymbol{\mu}_{k'})$: Distancia entre centroides
# MAGIC
# MAGIC **Criterio:** Menor $DB$ es mejor
# MAGIC
# MAGIC ### 3.4 Gap Statistic
# MAGIC
# MAGIC **Idea:** Comparar inercia con distribución nula (datos aleatorios)
# MAGIC
# MAGIC $$\text{Gap}(k) = \mathbb{E}[\log W_k^*] - \log W_k$$
# MAGIC
# MAGIC **Criterio:** Elegir $k$ que maximice Gap

# COMMAND ----------

# DBTITLE 1,Otros Algoritmos
# MAGIC %md
# MAGIC ## 4. Otros Algoritmos de Clustering
# MAGIC
# MAGIC ### 4.1 Clustering Jerárquico
# MAGIC
# MAGIC **Aglomerativo (bottom-up):**
# MAGIC
# MAGIC ```
# MAGIC 1. Inicializar: Cada punto es un cluster
# MAGIC 2. Repetir:
# MAGIC    - Fusionar los dos clusters más cercanos
# MAGIC 3. Hasta: Tener un solo cluster (o K clusters)
# MAGIC ```
# MAGIC
# MAGIC **Linkage (criterio de distancia entre clusters):**
# MAGIC
# MAGIC * **Single linkage:** $\min_{\mathbf{x}_i \in C_k, \mathbf{x}_j \in C_{k'}} d(\mathbf{x}_i, \mathbf{x}_j)$
# MAGIC * **Complete linkage:** $\max_{\mathbf{x}_i \in C_k, \mathbf{x}_j \in C_{k'}} d(\mathbf{x}_i, \mathbf{x}_j)$
# MAGIC * **Average linkage:** $\frac{1}{|C_k| |C_{k'}|} \sum_{\mathbf{x}_i \in C_k} \sum_{\mathbf{x}_j \in C_{k'}} d(\mathbf{x}_i, \mathbf{x}_j)$
# MAGIC * **Ward:** Minimiza incremento de varianza intra-cluster
# MAGIC
# MAGIC **Ventaja:** No requiere especificar $k$ a priori (dendrograma)
# MAGIC
# MAGIC ### 4.2 DBSCAN
# MAGIC
# MAGIC **Parámetros:**
# MAGIC * $\epsilon$: Radio de vecindad
# MAGIC * MinPts: Puntos mínimos en vecindad
# MAGIC
# MAGIC **Conceptos:**
# MAGIC * **Core point:** Tiene ≥ MinPts en su vecindad
# MAGIC * **Border point:** En vecindad de core point, pero no es core
# MAGIC * **Noise:** Ni core ni border
# MAGIC
# MAGIC **Ventajas:**
# MAGIC * Encuentra clusters de forma arbitraria
# MAGIC * Identifica outliers
# MAGIC * No requiere especificar $k$
# MAGIC
# MAGIC **Desventajas:**
# MAGIC * Sensible a $\epsilon$ y MinPts
# MAGIC * No funciona bien con densidades variables
# MAGIC
# MAGIC ### 4.3 Gaussian Mixture Models (GMM)
# MAGIC
# MAGIC **Modelo:**
# MAGIC $$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$
# MAGIC
# MAGIC * Asignación "suave" (probabilidades)
# MAGIC * Clusters elípticos (vs esféricos en K-Means)
# MAGIC * Estimación via EM (Expectation-Maximization)
# MAGIC
# MAGIC **Ventaja:** Incertidumbre en asignaciones

# COMMAND ----------

# DBTITLE 1,Comparación
# MAGIC %md
# MAGIC ## 5. Comparación de Algoritmos
# MAGIC
# MAGIC ### 5.1 Tabla Comparativa
# MAGIC
# MAGIC | Algoritmo | Forma Clusters | Requiere $k$ | Outliers | Complejidad |
# MAGIC |---|---|---|---|---|
# MAGIC | **K-Means** | Esféricos | ✅ Sí | ❌ No maneja | $O(nkdi)$ |
# MAGIC | **Jerárquico** | Cualquiera | ❌ No | ❌ No maneja | $O(n^2 \log n)$ |
# MAGIC | **DBSCAN** | Arbitraria | ❌ No | ✅ Identifica | $O(n \log n)$ |
# MAGIC | **GMM** | Elípticos | ✅ Sí | ❌ No maneja | $O(nkd^2i)$ |
# MAGIC
# MAGIC ### 5.2 ¿Cuándo usar qué?
# MAGIC
# MAGIC **K-Means:**
# MAGIC * Clusters esféricos, similar tamaño
# MAGIC * Datos grandes ($n > 10,000$)
# MAGIC * Conoces $k$ aproximadamente
# MAGIC
# MAGIC **Jerárquico:**
# MAGIC * Desconoces $k$, quieres dendrograma
# MAGIC * Datos pequeños ($n < 1,000$)
# MAGIC * Estructura jerárquica importante
# MAGIC
# MAGIC **DBSCAN:**
# MAGIC * Clusters de forma arbitraria
# MAGIC * Muchos outliers
# MAGIC * Densidad variable
# MAGIC
# MAGIC **GMM:**
# MAGIC * Clusters elípticos
# MAGIC * Necesitas probabilidades de asignación
# MAGIC * Datos siguen distribución gaussiana
# MAGIC
# MAGIC ### 5.3 Aplicaciones de Clustering
# MAGIC
# MAGIC **Marketing:**
# MAGIC * Segmentación de clientes
# MAGIC * Personalización de campañas
# MAGIC
# MAGIC **Biología:**
# MAGIC * Clasificación de genes
# MAGIC * Taxonomía
# MAGIC
# MAGIC **Visión por computadora:**
# MAGIC * Segmentación de imágenes
# MAGIC * Compresión de colores
# MAGIC
# MAGIC **Análisis de redes sociales:**
# MAGIC * Detección de comunidades
# MAGIC
# MAGIC **Reducción de datos:**
# MAGIC * Cuantización vectorial
# MAGIC * Feature learning

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## 6. Conclusiones
# MAGIC
# MAGIC ### 📚 Resumen
# MAGIC
# MAGIC 1. **K-Means:** Minimiza inercia, algoritmo de Lloyd, converge a mínimo local
# MAGIC 2. **Elección de $k$:** Método del codo, Silueta, Davies-Bouldin
# MAGIC 3. **Variantes:** Jerárquico (dendrograma), DBSCAN (forma arbitraria), GMM (soft assignments)
# MAGIC 4. **Aplicaciones:** Segmentación, compresión, exploración de datos
# MAGIC
# MAGIC ### 🎯 Siguiente
# MAGIC
# MAGIC **Notebook práctico:**
# MAGIC * `KMeans_Clustering.ipynb` - Segmentación de clientes completa
# MAGIC
# MAGIC ### 📖 Referencias
# MAGIC
# MAGIC * MacQueen (1967) - "Some Methods for Classification and Analysis of Multivariate Observations"
# MAGIC * Ester et al. (1996) - "A Density-Based Algorithm for Discovering Clusters" (DBSCAN)
# MAGIC * Rousseeuw (1987) - "Silhouettes: A Graphical Aid to the Interpretation"
# MAGIC * Hastie et al. (2009) - "The Elements of Statistical Learning", Cap. 14

# COMMAND ----------

