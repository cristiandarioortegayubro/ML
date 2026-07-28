# Databricks notebook source
# DBTITLE 1,Título y Objetivos
# MAGIC %md
# MAGIC # Teoría de Árboles de Decisión
# MAGIC
# MAGIC ## 🎯 Objetivos
# MAGIC
# MAGIC * **Fundamentos matemáticos**: Entropía, ganancia de información, índice Gini
# MAGIC * **Algoritmos**: ID3, C4.5, CART
# MAGIC * **Poda de árboles**: Pre-poda y post-poda
# MAGIC * **Análisis comparativo**: Cuándo usar árboles vs otros clasificadores
# MAGIC
# MAGIC ### Contenido
# MAGIC
# MAGIC 1. Fundamentos de árboles de decisión
# MAGIC 2. Medidas de impureza
# MAGIC 3. Algoritmos de construcción
# MAGIC 4. Poda y regularización
# MAGIC 5. Ventajas y limitaciones

# COMMAND ----------

# DBTITLE 1,Fundamentos
# MAGIC %md
# MAGIC ## 1. Fundamentos de Árboles de Decisión
# MAGIC
# MAGIC ### 1.1 ¿Qué es un Árbol de Decisión?
# MAGIC
# MAGIC **Estructura:**
# MAGIC * **Nodo raíz**: Primera pregunta
# MAGIC * **Nodos internos**: Decisiones basadas en features
# MAGIC * **Hojas**: Predicciones finales
# MAGIC
# MAGIC ### 1.2 Entropía de Shannon
# MAGIC
# MAGIC $$H(S) = -\sum_{i=1}^{k} p_i \log_2(p_i)$$
# MAGIC
# MAGIC **Propiedades:**
# MAGIC * $H(S) = 0$: Conjunto puro (todas las muestras de una clase)
# MAGIC * $H(S)$ máxima: Distribución uniforme
# MAGIC
# MAGIC **Ejemplo:**
# MAGIC Dataset: 10 ejemplos (6 positivos, 4 negativos)
# MAGIC $$H = -[\frac{6}{10}\log_2(\frac{6}{10}) + \frac{4}{10}\log_2(\frac{4}{10})] = 0.97$$
# MAGIC
# MAGIC ### 1.3 Ganancia de Información
# MAGIC
# MAGIC $$IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$
# MAGIC
# MAGIC **Objetivo:** Seleccionar feature que maximice $IG$

# COMMAND ----------

# DBTITLE 1,Índice Gini
# MAGIC %md
# MAGIC ## 2. Índice Gini
# MAGIC
# MAGIC ### 2.1 Definición
# MAGIC
# MAGIC $$\text{Gini}(S) = 1 - \sum_{i=1}^{k} p_i^2$$
# MAGIC
# MAGIC **Interpretación:** Probabilidad de clasificar incorrectamente si elegimos aleatoriamente.
# MAGIC
# MAGIC **Comparación con Entropía:**
# MAGIC
# MAGIC | Medida | Rango | Computación | Uso |
# MAGIC |---|---|---|---|
# MAGIC | Entropía | [0, log₂k] | Más costosa (logaritmo) | ID3, C4.5 |
# MAGIC | Gini | [0, 1-1/k] | Más rápida | CART |
# MAGIC
# MAGIC **En la práctica:** Ambas dan resultados similares.
# MAGIC
# MAGIC ### 2.2 Algoritmo CART
# MAGIC
# MAGIC **Classification And Regression Trees**
# MAGIC
# MAGIC **Para clasificación:**
# MAGIC $$\min \sum_{m=1}^{M} \sum_{i \in R_m} \text{Gini}(R_m)$$
# MAGIC
# MAGIC **Para regresión:**
# MAGIC $$\min \sum_{m=1}^{M} \sum_{i \in R_m} (y_i - \hat{y}_{R_m})^2$$
# MAGIC
# MAGIC donde $\hat{y}_{R_m}$ es la media de $y$ en región $R_m$.

# COMMAND ----------

# DBTITLE 1,Algoritmos de Construcción
# MAGIC %md
# MAGIC ## 3. Algoritmos de Construcción
# MAGIC
# MAGIC ### 3.1 ID3 (Iterative Dichotomiser 3)
# MAGIC
# MAGIC ```python
# MAGIC def ID3(S, Features):
# MAGIC     if todas_misma_clase(S):
# MAGIC         return Hoja(clase)
# MAGIC     if Features == vacío:
# MAGIC         return Hoja(clase_mayoritaria)
# MAGIC     
# MAGIC     mejor_feature = max(Features, key=lambda f: IG(S, f))
# MAGIC     arbol = Nodo(mejor_feature)
# MAGIC     
# MAGIC     for valor in valores(mejor_feature):
# MAGIC         S_v = subconjunto(S, mejor_feature == valor)
# MAGIC         arbol.agregar_rama(valor, ID3(S_v, Features - {mejor_feature}))
# MAGIC     
# MAGIC     return arbol
# MAGIC ```
# MAGIC
# MAGIC **Limitaciones:**
# MAGIC * Solo features categóricas
# MAGIC * No maneja valores faltantes
# MAGIC * Propenso a overfitting
# MAGIC
# MAGIC ### 3.2 C4.5 (Mejoras sobre ID3)
# MAGIC
# MAGIC **Innovaciones:**
# MAGIC
# MAGIC 1. **Gain Ratio** (en lugar de IG):
# MAGIC    $$\text{GainRatio}(S, A) = \frac{IG(S, A)}{\text{SplitInfo}(S, A)}$$
# MAGIC    
# MAGIC    donde:
# MAGIC    $$\text{SplitInfo}(S, A) = -\sum_{v} \frac{|S_v|}{|S|} \log_2(\frac{|S_v|}{|S|})$$
# MAGIC
# MAGIC 2. **Features numéricas**: Prueba umbrales $A \leq \theta$
# MAGIC
# MAGIC 3. **Valores faltantes**: Distribución proporcional
# MAGIC
# MAGIC 4. **Poda**: Post-poda usando error estimado
# MAGIC
# MAGIC ### 3.3 Comparación de Algoritmos
# MAGIC
# MAGIC | Característica | ID3 | C4.5 | CART |
# MAGIC |---|---|---|---|
# MAGIC | **Criterio división** | IG (Entropía) | Gain Ratio | Gini |
# MAGIC | **Features categóricas** | ✅ | ✅ | ✅ |
# MAGIC | **Features numéricas** | ❌ | ✅ | ✅ |
# MAGIC | **Valores faltantes** | ❌ | ✅ | ✅ |
# MAGIC | **Multi-way splits** | ✅ | ✅ | ❌ (binario) |
# MAGIC | **Poda** | ❌ | ✅ Post-poda | ✅ Pre/Post-poda |
# MAGIC | **Regresión** | ❌ | ❌ | ✅ |

# COMMAND ----------

# DBTITLE 1,Poda y Regularización
# MAGIC %md
# MAGIC ## 4. Poda y Regularización
# MAGIC
# MAGIC ### 4.1 El Problema del Overfitting
# MAGIC
# MAGIC **Árbol sin poda:**
# MAGIC * Profundidad infinita → memoriza entrenamiento
# MAGIC * Error entrenamiento = 0%, Error test = 40%
# MAGIC
# MAGIC **Soluciones:**
# MAGIC
# MAGIC ### 4.2 Pre-poda (Early Stopping)
# MAGIC
# MAGIC **Detener crecimiento si:**
# MAGIC
# MAGIC 1. **max_depth**: Profundidad máxima alcanzada
# MAGIC 2. **min_samples_split**: Muestras mínimas para dividir
# MAGIC 3. **min_samples_leaf**: Muestras mínimas en hoja
# MAGIC 4. **min_impurity_decrease**: Reducción mínima de impureza
# MAGIC
# MAGIC **Ejemplo en PySpark:**
# MAGIC ```python
# MAGIC dt = DecisionTreeClassifier(
# MAGIC     maxDepth=5,           # Pre-poda
# MAGIC     minInstancesPerNode=20,
# MAGIC     minInfoGain=0.01
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### 4.3 Post-poda (Pruning)
# MAGIC
# MAGIC **Idea:** Construir árbol completo, luego podar.
# MAGIC
# MAGIC **Reduced Error Pruning:**
# MAGIC 1. Dividir datos: train + validation
# MAGIC 2. Construir árbol completo en train
# MAGIC 3. Para cada nodo interno:
# MAGIC    * Reemplazar subárbol por hoja
# MAGIC    * Si accuracy en validation no empeora → podar
# MAGIC
# MAGIC **Cost Complexity Pruning (α-pruning):**
# MAGIC
# MAGIC **Función objetivo:**
# MAGIC $$C_\alpha(T) = \sum_{m=1}^{|T|} \sum_{i \in R_m} L(y_i, \hat{y}_{R_m}) + \alpha |T|$$
# MAGIC
# MAGIC donde:
# MAGIC * $|T|$: Número de nodos hoja
# MAGIC * $\alpha \geq 0$: Parámetro de complejidad
# MAGIC
# MAGIC **Interpretación:**
# MAGIC * $\alpha = 0$: Árbol completo (sin penalización)
# MAGIC * $\alpha \to \infty$: Solo raíz (máxima penalización)
# MAGIC
# MAGIC **Proceso:**
# MAGIC 1. Para cada $\alpha$, encontrar árbol óptimo $T_\alpha$
# MAGIC 2. Usar cross-validation para elegir mejor $\alpha$
# MAGIC
# MAGIC ### 4.4 Hiperparámetros en PySpark
# MAGIC
# MAGIC ```python
# MAGIC from pyspark.ml.classification import DecisionTreeClassifier
# MAGIC
# MAGIC dt = DecisionTreeClassifier(
# MAGIC     # Pre-poda
# MAGIC     maxDepth=10,              # Profundidad máxima
# MAGIC     maxBins=32,               # Bins para features numéricas
# MAGIC     minInstancesPerNode=10,   # Mínimo por nodo
# MAGIC     minInfoGain=0.0,          # Ganancia mínima
# MAGIC     
# MAGIC     # Criterio
# MAGIC     impurity="gini",          # "gini" o "entropy"
# MAGIC     
# MAGIC     # Otros
# MAGIC     seed=42
# MAGIC )
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Ventajas y Limitaciones
# MAGIC %md
# MAGIC ## 5. Ventajas y Limitaciones
# MAGIC
# MAGIC ### 5.1 Ventajas ✅
# MAGIC
# MAGIC **1. Interpretabilidad**
# MAGIC * Reglas IF-THEN fáciles de explicar
# MAGIC * Visualización intuitiva
# MAGIC
# MAGIC **2. No requiere preparación de datos**
# MAGIC * No necesita normalización
# MAGIC * Maneja features categóricas y numéricas
# MAGIC * Robusto a outliers
# MAGIC
# MAGIC **3. Maneja relaciones no lineales**
# MAGIC * Captura interacciones automáticamente
# MAGIC
# MAGIC **4. Feature selection implícita**
# MAGIC * Features importantes cerca de la raíz
# MAGIC
# MAGIC **5. Rápido en predicción**
# MAGIC * $O(\log n)$ en árbol balanceado
# MAGIC
# MAGIC ### 5.2 Limitaciones ❌
# MAGIC
# MAGIC **1. Overfitting**
# MAGIC * Árboles profundos memorizan
# MAGIC * **Solución**: Poda, max_depth
# MAGIC
# MAGIC **2. Inestabilidad**
# MAGIC * Pequeños cambios en datos → árbol muy diferente
# MAGIC * **Solución**: Ensembles (Random Forest, XGBoost)
# MAGIC
# MAGIC **3. Fronteras de decisión rectas**
# MAGIC * Solo divisiones ortogonales (paralelas a ejes)
# MAGIC * Ineficiente para relaciones oblicuas
# MAGIC
# MAGIC **4. Sesgo hacia features con muchos valores**
# MAGIC * Gain Ratio mitiga esto (C4.5)
# MAGIC
# MAGIC **5. No captura relaciones lineales simples eficientemente**
# MAGIC * Para $y = 2x + 3$, árbol necesita muchas divisiones
# MAGIC * Regresión lineal lo hace mejor
# MAGIC
# MAGIC ### 5.3 Comparación con Otros Clasificadores
# MAGIC
# MAGIC | Aspecto | Árboles | Regresión Logística | SVM | Random Forest |
# MAGIC |---|---|---|---|---|
# MAGIC | **Interpretabilidad** | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐ |
# MAGIC | **Accuracy** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
# MAGIC | **Velocidad train** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐ |
# MAGIC | **Velocidad predict** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
# MAGIC | **Overfitting** | Alto | Bajo | Bajo | Bajo |
# MAGIC | **Features no normalizadas** | ✅ | ❌ | ❌ | ✅ |
# MAGIC | **Relaciones no lineales** | ✅ | ❌ | ✅ | ✅ |
# MAGIC
# MAGIC ### 5.4 ¿Cuándo usar Árboles de Decisión?
# MAGIC
# MAGIC **✅ Usa árboles cuando:**
# MAGIC * Interpretabilidad es crítica
# MAGIC * Features son mixtas (categóricas + numéricas)
# MAGIC * Relaciones no lineales complejas
# MAGIC * No puedes normalizar datos
# MAGIC * Necesitas prototipar rápido
# MAGIC
# MAGIC **❌ No uses árboles cuando:**
# MAGIC * Tienes pocos datos (< 100 muestras)
# MAGIC * Relaciones son claramente lineales
# MAGIC * Accuracy es más importante que interpretabilidad → usa ensembles
# MAGIC * Datos tienen mucho ruido

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## 6. Conclusiones
# MAGIC
# MAGIC ### 📚 Resumen
# MAGIC
# MAGIC 1. **Entropía y Gini**: Medidas de impureza para seleccionar divisiones
# MAGIC 2. **Algoritmos**: ID3 → C4.5 → CART (mejoras progresivas)
# MAGIC 3. **Poda**: Pre-poda (detener) y Post-poda (construir luego podar)
# MAGIC 4. **Trade-off**: Interpretabilidad vs Accuracy
# MAGIC 5. **Ensembles**: Random Forest y XGBoost superan árboles individuales
# MAGIC
# MAGIC ### 🎯 Siguiente
# MAGIC
# MAGIC **Notebook práctico:**
# MAGIC * `Arbol_Decision_Clasificacion.ipynb` - Implementación completa con caso churn
# MAGIC
# MAGIC ### 📖 Referencias
# MAGIC
# MAGIC * Quinlan (1986) - "Induction of Decision Trees"
# MAGIC * Breiman et al. (1984) - "Classification and Regression Trees"
# MAGIC * Hastie et al. (2009) - "The Elements of Statistical Learning", Cap. 9

# COMMAND ----------

