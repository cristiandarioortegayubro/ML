# Databricks notebook source
# DBTITLE 1,# Teoría de Random Forest
# MAGIC %md
# MAGIC # Teoría de Random Forest
# MAGIC
# MAGIC ## 1. Introducción
# MAGIC
# MAGIC **Random Forest** (Bosque Aleatorio) es un algoritmo de **aprendizaje ensamblado (ensemble learning)** que construye múltiples árboles de decisión durante el entrenamiento y combina sus predicciones para obtener un resultado más preciso y robusto.
# MAGIC
# MAGIC ### Concepto Clave: Ensemble Learning
# MAGIC
# MAGIC **Ensemble Learning** combina múltiples modelos débiles (weak learners) para crear un modelo fuerte (strong learner):
# MAGIC
# MAGIC $$
# MAGIC \text{Predicción Final} = \text{Agregación}(\text{Modelo}_1, \text{Modelo}_2, ..., \text{Modelo}_n)
# MAGIC $$
# MAGIC
# MAGIC **Ventajas del ensemble:**
# MAGIC * Reduce **overfitting** (sobreajuste)
# MAGIC * Mejora **accuracy** (precisión)
# MAGIC * Mayor **robustez** ante datos ruidosos
# MAGIC * Reduce **varianza** del modelo
# MAGIC
# MAGIC ### Historia
# MAGIC
# MAGIC * **1995**: Leo Breiman introduce **Bagging** (Bootstrap Aggregating)
# MAGIC * **2001**: Leo Breiman publica el paper "Random Forests"
# MAGIC * **Hoy**: Uno de los algoritmos más usados en competencias de ML (Kaggle)
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,## 2. Algoritmo Random Forest
# MAGIC %md
# MAGIC ## 2. Algoritmo Random Forest
# MAGIC
# MAGIC ### Pseudocódigo
# MAGIC
# MAGIC ```
# MAGIC Entrada: Dataset D = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
# MAGIC          Número de árboles: T
# MAGIC          Número de features por árbol: m
# MAGIC
# MAGIC Para t = 1 hasta T:
# MAGIC     1. Bootstrap: Crear muestra Dₜ de D con reemplazo (mismo tamaño)
# MAGIC     2. Feature Sampling: Seleccionar aleatoriamente m features
# MAGIC     3. Entrenar árbol de decisión Tₜ con Dₜ y m features
# MAGIC     4. Sin poda (árboles crecen completamente)
# MAGIC
# MAGIC Predicción:
# MAGIC     - Clasificación: Voto mayoritario de los T árboles
# MAGIC     - Regresión: Promedio de las predicciones de los T árboles
# MAGIC ```
# MAGIC
# MAGIC ### Proceso Visual
# MAGIC
# MAGIC ```
# MAGIC Dataset Original
# MAGIC       ↓
# MAGIC    Bootstrap
# MAGIC    /    |    \
# MAGIC   D₁   D₂   D₃  ...  Dₜ  (T muestras con reemplazo)
# MAGIC   ↓    ↓    ↓         ↓
# MAGIC Tree₁ Tree₂ Tree₃ ... Treeₜ (T árboles)
# MAGIC   ↓    ↓    ↓         ↓
# MAGIC   P₁   P₂   P₃  ...  Pₜ  (T predicciones)
# MAGIC    \    |    /         /
# MAGIC     \   |   /        /
# MAGIC      Agregación (Voto/Promedio)
# MAGIC            ↓
# MAGIC    Predicción Final
# MAGIC ```
# MAGIC
# MAGIC ### Componentes Clave
# MAGIC
# MAGIC #### 1. **Bagging (Bootstrap Aggregating)**
# MAGIC
# MAGIC Cada árbol se entrena con una muestra **bootstrap** (con reemplazo):
# MAGIC
# MAGIC $$
# MAGIC D_t = \{\text{sample}(D, n, \text{replace=True})\}
# MAGIC $$
# MAGIC
# MAGIC * Cada $D_t$ tiene el mismo tamaño que $D$
# MAGIC * Aproximadamente **63%** de las muestras aparecen en cada bootstrap
# MAGIC * **37%** quedan fuera (**Out-Of-Bag samples** o OOB)
# MAGIC
# MAGIC #### 2. **Feature Randomness**
# MAGIC
# MAGIC En cada división de nodo, solo se consideran **m features aleatorias**:
# MAGIC
# MAGIC * **Clasificación**: $m = \sqrt{p}$ (raíz cuadrada del total de features)
# MAGIC * **Regresión**: $m = p/3$ (un tercio del total de features)
# MAGIC * Donde $p$ es el número total de features
# MAGIC
# MAGIC Esto **decorrelaciona** los árboles, reduciendo la varianza.
# MAGIC
# MAGIC #### 3. **Agregación de Predicciones**
# MAGIC
# MAGIC **Para Clasificación (voto mayoritario):**
# MAGIC
# MAGIC $$
# MAGIC \hat{y} = \text{mode}(T_1(x), T_2(x), ..., T_n(x))
# MAGIC $$
# MAGIC
# MAGIC **Para Regresión (promedio):**
# MAGIC
# MAGIC $$
# MAGIC \hat{y} = \frac{1}{T} \sum_{t=1}^{T} T_t(x)
# MAGIC $$
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,## 3. Ventajas y Desventajas
# MAGIC %md
# MAGIC ## 3. Ventajas y Desventajas
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC 1. **Alta Precisión**: Supera a un solo árbol de decisión
# MAGIC 2. **Robustez**:
# MAGIC    * Maneja bien **outliers** y datos ruidosos
# MAGIC    * No requiere normalización de features
# MAGIC    * Funciona con features categóricas y numéricas
# MAGIC 3. **Prevención de Overfitting**: La aleatoriedad reduce sobreajuste
# MAGIC 4. **Feature Importance**: Mide la importancia de cada variable
# MAGIC 5. **OOB Error**: Estimación de error sin necesidad de validación cruzada
# MAGIC 6. **Paralelizable**: Los árboles se entrenan independientemente
# MAGIC 7. **Versatilidad**: Funciona para clasificación y regresión
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC 1. **Menos Interpretable**: Caja negra (vs un solo árbol)
# MAGIC 2. **Computacionalmente Costoso**: Entrena T árboles
# MAGIC 3. **Predicción Lenta**: Debe evaluar T árboles (vs 1 árbol)
# MAGIC 4. **Tamaño del Modelo**: Ocupa más memoria (T árboles en RAM)
# MAGIC 5. **Extrapolación Pobre**: No predice bien fuera del rango de entrenamiento
# MAGIC 6. **Sesgado hacia Features con Muchas Categorías**: En clasificación
# MAGIC
# MAGIC ### Comparación: Random Forest vs Decision Tree
# MAGIC
# MAGIC | Aspecto | Decision Tree | Random Forest |
# MAGIC |---------|---------------|---------------|
# MAGIC | **Accuracy** | Menor | Mayor |
# MAGIC | **Overfitting** | Alto riesgo | Bajo riesgo |
# MAGIC | **Interpretabilidad** | Alta | Baja |
# MAGIC | **Velocidad entrenamiento** | Rápido | Lento |
# MAGIC | **Velocidad predicción** | Muy rápido | Moderado |
# MAGIC | **Robustez** | Baja | Alta |
# MAGIC | **Feature Importance** | Sí | Sí (más confiable) |
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,## 4. Hiperparámetros Clave
# MAGIC %md
# MAGIC ## 4. Hiperparámetros Clave
# MAGIC
# MAGIC ### Parámetros de Ensemble
# MAGIC
# MAGIC 1. **`numTrees` (número de árboles)**
# MAGIC    * Más árboles → Mayor accuracy, pero más lento
# MAGIC    * **Típico**: 100-500 árboles
# MAGIC    * **Regla**: Aumentar hasta que el error se estabilice
# MAGIC
# MAGIC 2. **`featureSubsetStrategy` (m features por split)**
# MAGIC    * **Clasificación**: `"sqrt"` ($m = \sqrt{p}$)
# MAGIC    * **Regresión**: `"onethird"` ($m = p/3$)
# MAGIC    * **Otros**: `"log2"`, `"all"`, o número fijo
# MAGIC
# MAGIC ### Parámetros de Árboles Individuales
# MAGIC
# MAGIC 3. **`maxDepth` (profundidad máxima)**
# MAGIC    * Random Forest suele usar árboles profundos (sin poda)
# MAGIC    * **Típico**: 10-30 (o sin límite)
# MAGIC
# MAGIC 4. **`minInstancesPerNode` (mínimo de ejemplos por hoja)**
# MAGIC    * Controla el tamaño mínimo de las hojas
# MAGIC    * **Típico**: 1-5
# MAGIC
# MAGIC 5. **`maxBins` (número de bins para discretización)**
# MAGIC    * Mayor → Más splits posibles, pero más lento
# MAGIC    * **Típico**: 32
# MAGIC
# MAGIC ### Parámetros de Bootstrap
# MAGIC
# MAGIC 6. **`subsamplingRate` (fracción de datos por árbol)**
# MAGIC    * **Por defecto**: 1.0 (bootstrap con reemplazo)
# MAGIC    * **Alternativa**: 0.8 (80% sin reemplazo, más rápido)
# MAGIC
# MAGIC ### Regla de Oro para Hiperparámetros
# MAGIC
# MAGIC ```python
# MAGIC # Configuración recomendada inicial
# MAGIC rf = RandomForestClassifier(
# MAGIC     numTrees=100,                    # Aumentar si tienes recursos
# MAGIC     featureSubsetStrategy="sqrt",    # sqrt para clasificación
# MAGIC     maxDepth=None,                   # Sin límite (árboles profundos)
# MAGIC     minInstancesPerNode=1,           # Hojas pequeñas OK
# MAGIC     subsamplingRate=1.0              # Bootstrap completo
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,## 5. Feature Importance
# MAGIC %md
# MAGIC ## 5. Feature Importance
# MAGIC
# MAGIC Random Forest calcula la **importancia de cada feature** basándose en cuánto reducen la impureza (Gini o entropía) en promedio.
# MAGIC
# MAGIC ### Cálculo
# MAGIC
# MAGIC Para cada feature $f$:
# MAGIC
# MAGIC 1. En cada árbol $t$, sumar la reducción de impureza de todos los splits que usan $f$:
# MAGIC    $$
# MAGIC    \text{Importance}_t(f) = \sum_{\text{nodos que usan } f} \Delta \text{Impureza}
# MAGIC    $$
# MAGIC
# MAGIC 2. Promediar sobre todos los árboles:
# MAGIC    $$
# MAGIC    \text{Importance}(f) = \frac{1}{T} \sum_{t=1}^{T} \text{Importance}_t(f)
# MAGIC    $$
# MAGIC
# MAGIC 3. Normalizar para que sumen 1:
# MAGIC    $$
# MAGIC    \text{Importance}(f) = \frac{\text{Importance}(f)}{\sum_{f'} \text{Importance}(f')}
# MAGIC    $$
# MAGIC
# MAGIC ### Interpretación
# MAGIC
# MAGIC * **Valores altos**: Feature muy importante para las predicciones
# MAGIC * **Valores bajos**: Feature poco relevante (candidata a eliminar)
# MAGIC * **Suma = 1.0**: Las importancias son proporciones
# MAGIC
# MAGIC ### Ejemplo
# MAGIC
# MAGIC ```
# MAGIC Feature Importance:
# MAGIC 1. Monthly_Charges    : 0.35  (35% de importancia)
# MAGIC 2. Tenure              : 0.28
# MAGIC 3. Contract_Type       : 0.15
# MAGIC 4. Total_Charges       : 0.12
# MAGIC 5. Internet_Service    : 0.07
# MAGIC 6. Payment_Method      : 0.03
# MAGIC ```
# MAGIC
# MAGIC **Interpretación**: `Monthly_Charges` es el factor más importante para predecir churn.
# MAGIC
# MAGIC ### Uso Práctico
# MAGIC
# MAGIC * **Feature Selection**: Eliminar features con importancia < 0.01
# MAGIC * **Interpretabilidad**: Explicar qué variables importan más
# MAGIC * **Domain Insights**: Validar hipótesis de negocio
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,## 6. Out-Of-Bag (OOB) Error
# MAGIC %md
# MAGIC ## 6. Out-Of-Bag (OOB) Error
# MAGIC
# MAGIC ### Concepto
# MAGIC
# MAGIC En cada bootstrap, aproximadamente **37%** de las muestras quedan **fuera** (Out-Of-Bag).
# MAGIC
# MAGIC Estas muestras OOB se pueden usar como **conjunto de validación implícito**:
# MAGIC
# MAGIC 1. Para cada muestra $x_i$, usar solo los árboles que **NO** la incluyeron en su bootstrap
# MAGIC 2. Predecir $x_i$ con esos árboles
# MAGIC 3. Comparar con $y_i$ para calcular el error
# MAGIC
# MAGIC ### Ventajas
# MAGIC
# MAGIC * **No requiere train/test split**: OOB simula un conjunto de validación
# MAGIC * **Uso eficiente de datos**: Todo el dataset se usa para entrenar
# MAGIC * **Estimación no sesgada**: Similar a cross-validation
# MAGIC
# MAGIC ### Fórmula
# MAGIC
# MAGIC **OOB Error (clasificación):**
# MAGIC
# MAGIC $$
# MAGIC \text{OOB Error} = \frac{1}{n} \sum_{i=1}^{n} I(y_i \neq \hat{y}_i^{\text{OOB}})
# MAGIC $$
# MAGIC
# MAGIC **OOB Error (regresión):**
# MAGIC
# MAGIC $$
# MAGIC \text{OOB Error} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i^{\text{OOB}})^2
# MAGIC $$
# MAGIC
# MAGIC Donde $\hat{y}_i^{\text{OOB}}$ es la predicción usando solo árboles que no vieron $x_i$.
# MAGIC
# MAGIC ### Uso Práctico
# MAGIC
# MAGIC ```python
# MAGIC # En PySpark ML, obtener OOB error:
# MAGIC rf_model = rf.fit(train_data)
# MAGIC oob_error = rf_model.oobError  # Si está habilitado
# MAGIC print(f"OOB Error: {oob_error:.4f}")
# MAGIC ```
# MAGIC
# MAGIC **Interpretación**: Si OOB Error ≈ Test Error, el modelo generaliza bien.
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,## 7. Conclusiones
# MAGIC %md
# MAGIC ## 7. Conclusiones
# MAGIC
# MAGIC ### Cuándo Usar Random Forest
# MAGIC
# MAGIC ✅ **Úsalo cuando:**
# MAGIC * Necesitas alta precisión
# MAGIC * Tienes suficiente tiempo de entrenamiento
# MAGIC * No requieres interpretabilidad extrema
# MAGIC * Tienes features mixtas (numéricas y categóricas)
# MAGIC * Datos ruidosos o con outliers
# MAGIC
# MAGIC ❌ **Evítalo cuando:**
# MAGIC * Necesitas interpretabilidad perfecta (usa Decision Tree)
# MAGIC * Tiempo de predicción es crítico (usa modelos lineales)
# MAGIC * Dataset muy pequeño (< 1000 muestras)
# MAGIC * Recursos computacionales limitados
# MAGIC * Necesitas extrapolación fuera del rango de entrenamiento
# MAGIC
# MAGIC ### Comparación con Otros Algoritmos
# MAGIC
# MAGIC | Algoritmo | Accuracy | Velocidad | Interpretabilidad | Robustez |
# MAGIC |-----------|----------|-----------|-------------------|----------|
# MAGIC | **Decision Tree** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
# MAGIC | **Random Forest** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
# MAGIC | **Gradient Boosting** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
# MAGIC | **Logistic Regression** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
# MAGIC
# MAGIC ### Fórmulas Clave para Recordar
# MAGIC
# MAGIC 1. **Features por split (clasificación)**: $m = \sqrt{p}$
# MAGIC 2. **Features por split (regresión)**: $m = p/3$
# MAGIC 3. **Predicción (clasificación)**: $\hat{y} = \text{mode}(T_1(x), ..., T_n(x))$
# MAGIC 4. **Predicción (regresión)**: $\hat{y} = \frac{1}{T} \sum_{t=1}^{T} T_t(x)$
# MAGIC 5. **OOB muestras por árbol**: ~37% del dataset
# MAGIC
# MAGIC ### Próximos Pasos
# MAGIC
# MAGIC 1. **Práctica**: Implementar Random Forest en PySpark ML
# MAGIC 2. **Comparar**: Random Forest vs Decision Tree en el mismo problema
# MAGIC 3. **Optimizar**: Tunear hiperparámetros con Grid Search
# MAGIC 4. **Avanzado**: Explorar Gradient Boosted Trees (GBT)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Referencias
# MAGIC
# MAGIC * **Paper Original**: Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.
# MAGIC * **PySpark ML**: [Random Forest Documentation](https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier)
# MAGIC * **Scikit-Learn**: [Random Forest Guide](https://scikit-learn.org/stable/modules/ensemble.html#forest)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **¡Ahora estás listo para aplicar Random Forest en problemas reales!** 🌲🌲🌲

# COMMAND ----------

