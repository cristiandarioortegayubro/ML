# Databricks notebook source
# DBTITLE 1,Título y Objetivos
# MAGIC %md
# MAGIC # Fundamentos de Machine Learning: Teoría y Matemáticas
# MAGIC
# MAGIC ## 🎯 Objetivos de Aprendizaje
# MAGIC
# MAGIC Este notebook proporciona una base teórica rigurosa de Machine Learning, combinando:
# MAGIC
# MAGIC * **Fundamentos matemáticos**: Teoría del aprendizaje estadístico
# MAGIC * **Conceptos fundamentales**: Tipos de aprendizaje, paradigmas, procesos
# MAGIC * **Análisis comparativo**: Cuándo aplicar cada enfoque
# MAGIC
# MAGIC ### Prerrequisitos
# MAGIC
# MAGIC * Álgebra lineal básica (vectores, matrices)
# MAGIC * Cálculo diferencial (derivadas parciales)
# MAGIC * Probabilidad y estadística (distribuciones, esperanza, varianza)
# MAGIC
# MAGIC ### Contenido
# MAGIC
# MAGIC 1. ¿Qué es Machine Learning? - Definición formal
# MAGIC 2. Tipos de aprendizaje
# MAGIC 3. Teoría del aprendizaje estadístico
# MAGIC 4. El problema del sobreajuste
# MAGIC 5. Proceso de desarrollo de modelos ML
# MAGIC 6. Comparación de enfoques

# COMMAND ----------

# DBTITLE 1,Definición Formal de ML
# MAGIC %md
# MAGIC ## 1. ¿Qué es Machine Learning? - Definición Formal
# MAGIC
# MAGIC ### Definición de Tom Mitchell (1997)
# MAGIC
# MAGIC > *"Se dice que un programa de computadora **aprende** de la experiencia E con respecto a alguna clase de tareas T y medida de rendimiento P, si su rendimiento en las tareas en T, medido por P, mejora con la experiencia E."*
# MAGIC
# MAGIC **Formalmente:**
# MAGIC
# MAGIC $$\text{Aprendizaje} \Leftrightarrow P(T) \text{ mejora con } E$$
# MAGIC
# MAGIC ### Componentes clave:
# MAGIC
# MAGIC * **Tarea (T)**: El problema a resolver (clasificación, regresión, clustering)
# MAGIC * **Experiencia (E)**: Los datos de entrenamiento
# MAGIC * **Medida de rendimiento (P)**: Métrica para evaluar el éxito (accuracy, RMSE, etc.)
# MAGIC
# MAGIC ### Ejemplo: Predicción de abandono de clientes
# MAGIC
# MAGIC * **T**: Clasificar clientes como "abandonarán" o "no abandonarán"
# MAGIC * **E**: Dataset histórico de 10,000 clientes con variables demográficas y comportamiento
# MAGIC * **P**: Accuracy, Precision, Recall en datos de prueba
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Aprendizaje vs Programación Tradicional
# MAGIC
# MAGIC | Programación Tradicional | Machine Learning |
# MAGIC |---|---|
# MAGIC | Reglas explícitas codificadas | Reglas aprendidas de datos |
# MAGIC | `if-else` statements | Modelos estadísticos/matemáticos |
# MAGIC | Funciona en problemas bien definidos | Funciona en problemas complejos/ambiguos |
# MAGIC | Ejemplo: Calculadora | Ejemplo: Reconocimiento facial |

# COMMAND ----------

# DBTITLE 1,Tipos de Aprendizaje
# MAGIC %md
# MAGIC ## 2. Tipos de Aprendizaje
# MAGIC
# MAGIC ### 2.1 Aprendizaje Supervisado
# MAGIC
# MAGIC **Definición:** El algoritmo aprende de datos etiquetados $(x_i, y_i)$ donde:
# MAGIC * $x_i$: Vector de características (input)
# MAGIC * $y_i$: Etiqueta/valor objetivo (output)
# MAGIC
# MAGIC **Objetivo:** Aprender una función $f: X \rightarrow Y$ tal que $f(x_i) \approx y_i$
# MAGIC
# MAGIC **Tipos:**
# MAGIC
# MAGIC **a) Clasificación** ($y$ categórica):
# MAGIC * **Binaria**: $y \in \{0, 1\}$ (churn: sí/no)
# MAGIC * **Multiclase**: $y \in \{1, 2, ..., k\}$ (tipo de producto: A, B, C)
# MAGIC
# MAGIC **Ejemplos:**
# MAGIC * Detección de spam
# MAGIC * Diagnóstico médico
# MAGIC * Reconocimiento de imágenes
# MAGIC
# MAGIC **b) Regresión** ($y$ numérica):
# MAGIC * $y \in \mathbb{R}$ (valores continuos)
# MAGIC
# MAGIC **Ejemplos:**
# MAGIC * Predicción de precios inmobiliarios
# MAGIC * Pronóstico de ventas
# MAGIC * Estimación de temperatura
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2.2 Aprendizaje No Supervisado
# MAGIC
# MAGIC **Definición:** El algoritmo aprende de datos no etiquetados $\{x_i\}$
# MAGIC
# MAGIC **Objetivo:** Descubrir estructura oculta en los datos
# MAGIC
# MAGIC **Tipos:**
# MAGIC
# MAGIC **a) Clustering** (agrupamiento):
# MAGIC * Particionar datos en $k$ grupos: $C_1, C_2, ..., C_k$
# MAGIC * Minimizar distancia intra-cluster, maximizar distancia inter-cluster
# MAGIC
# MAGIC **Ejemplos:**
# MAGIC * Segmentación de clientes
# MAGIC * Compresión de imágenes
# MAGIC * Detección de anomalías
# MAGIC
# MAGIC **b) Reducción de dimensionalidad:**
# MAGIC * Proyectar datos de alta dimensión a baja dimensión
# MAGIC * $x \in \mathbb{R}^d \rightarrow z \in \mathbb{R}^k$ donde $k \ll d$
# MAGIC
# MAGIC **Ejemplos:**
# MAGIC * PCA (Principal Component Analysis)
# MAGIC * t-SNE (visualización)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2.3 Aprendizaje por Refuerzo
# MAGIC
# MAGIC **Definición:** Un agente aprende a tomar decisiones mediante interacción con un ambiente
# MAGIC
# MAGIC **Componentes:**
# MAGIC * **Estado** ($s_t$): Situación actual
# MAGIC * **Acción** ($a_t$): Decisión del agente
# MAGIC * **Recompensa** ($r_t$): Feedback del ambiente
# MAGIC
# MAGIC **Objetivo:** Maximizar recompensa acumulada
# MAGIC
# MAGIC $$\max \sum_{t=0}^{\infty} \gamma^t r_t$$
# MAGIC
# MAGIC donde $\gamma \in [0,1]$ es el factor de descuento.
# MAGIC
# MAGIC **Ejemplos:**
# MAGIC * Juegos (AlphaGo, Chess)
# MAGIC * Robótica
# MAGIC * Sistemas de recomendación

# COMMAND ----------

# DBTITLE 1,Teoría del Aprendizaje Estadístico
# MAGIC %md
# MAGIC ## 3. Teoría del Aprendizaje Estadístico
# MAGIC
# MAGIC ### 3.1 Formulación Matemática
# MAGIC
# MAGIC **Problema:** Dado un conjunto de entrenamiento $D = \{(x_1, y_1), ..., (x_n, y_n)\}$, encontrar una función $\hat{f}$ que minimice el **error de generalización**.
# MAGIC
# MAGIC **Error de Generalización (Riesgo Real):**
# MAGIC
# MAGIC $$R(f) = \mathbb{E}_{(x,y) \sim P} [L(y, f(x))]$$
# MAGIC
# MAGIC donde:
# MAGIC * $P$: Distribución de probabilidad real (desconocida) de $(X, Y)$
# MAGIC * $L$: Función de pérdida (loss function)
# MAGIC
# MAGIC **Error Empírico (Riesgo Empírico):**
# MAGIC
# MAGIC $$\hat{R}(f) = \frac{1}{n} \sum_{i=1}^{n} L(y_i, f(x_i))$$
# MAGIC
# MAGIC **Objetivo de ML:** Minimizar $\hat{R}(f)$ esperando que $R(f)$ también sea pequeño.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3.2 Funciones de Pérdida Comunes
# MAGIC
# MAGIC **Regresión:**
# MAGIC
# MAGIC * **Error Cuadrático (MSE):**
# MAGIC   $$L(y, \hat{y}) = (y - \hat{y})^2$$
# MAGIC
# MAGIC * **Error Absoluto (MAE):**
# MAGIC   $$L(y, \hat{y}) = |y - \hat{y}|$$
# MAGIC
# MAGIC **Clasificación:**
# MAGIC
# MAGIC * **0-1 Loss:**
# MAGIC   $$L(y, \hat{y}) = \mathbb{1}_{y \neq \hat{y}} = \begin{cases} 0 & \text{si } y = \hat{y} \\ 1 & \text{si } y \neq \hat{y} \end{cases}$$
# MAGIC
# MAGIC * **Cross-Entropy (clasificación binaria):**
# MAGIC   $$L(y, \hat{p}) = -[y \log(\hat{p}) + (1-y) \log(1-\hat{p})]$$
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3.3 Descomposición Bias-Variance
# MAGIC
# MAGIC **Teorema fundamental:** El error esperado de un modelo se descompone en:
# MAGIC
# MAGIC $$\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{\text{Bias}^2(\hat{f}(x))}_{\text{Error sistemático}} + \underbrace{\text{Var}(\hat{f}(x))}_{\text{Sensibilidad}} + \underbrace{\sigma^2}_{\text{Ruido irreducible}}$$
# MAGIC
# MAGIC **Bias (Sesgo):**
# MAGIC $$\text{Bias}(\hat{f}(x)) = \mathbb{E}[\hat{f}(x)] - f(x)$$
# MAGIC
# MAGIC * Mide cuán lejos está el modelo promedio del valor verdadero
# MAGIC * **Alto bias** → **Underfitting** (modelo demasiado simple)
# MAGIC
# MAGIC **Variance (Varianza):**
# MAGIC $$\text{Var}(\hat{f}(x)) = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$$
# MAGIC
# MAGIC * Mide cuánto varía el modelo con diferentes datos de entrenamiento
# MAGIC * **Alta varianza** → **Overfitting** (modelo demasiado complejo)
# MAGIC
# MAGIC **Trade-off:**
# MAGIC
# MAGIC ```
# MAGIC Complejidad ↑  →  Bias ↓, Variance ↑
# MAGIC Complejidad ↓  →  Bias ↑, Variance ↓
# MAGIC ```
# MAGIC
# MAGIC **Objetivo:** Encontrar el balance óptimo

# COMMAND ----------

# DBTITLE 1,Overfitting y Underfitting
# MAGIC %md
# MAGIC ## 4. El Problema del Sobreajuste (Overfitting)
# MAGIC
# MAGIC ### 4.1 Definiciones
# MAGIC
# MAGIC **Underfitting (Subajuste):**
# MAGIC * El modelo es **demasiado simple** para capturar la estructura de los datos
# MAGIC * **Alto bias**, bajo variance
# MAGIC * Error alto en entrenamiento Y en prueba
# MAGIC
# MAGIC **Overfitting (Sobreajuste):**
# MAGIC * El modelo **memoriza** los datos de entrenamiento, incluyendo ruido
# MAGIC * **Alta varianza**, bajo bias
# MAGIC * Error bajo en entrenamiento, error alto en prueba
# MAGIC
# MAGIC **Buen ajuste (Good fit):**
# MAGIC * Balance óptimo entre bias y variance
# MAGIC * Generaliza bien a datos nuevos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4.2 Detección de Overfitting
# MAGIC
# MAGIC **Señales de alarma:**
# MAGIC
# MAGIC 1. **Gap grande entre train y test:**
# MAGIC    ```
# MAGIC    Accuracy train = 99%
# MAGIC    Accuracy test  = 65%  ← Overfitting!
# MAGIC    ```
# MAGIC
# MAGIC 2. **Curvas de aprendizaje divergentes:**
# MAGIC    * Error de entrenamiento continúa bajando
# MAGIC    * Error de validación se estanca o sube
# MAGIC
# MAGIC 3. **Modelo muy complejo:**
# MAGIC    * Árbol de decisión con profundidad = 50
# MAGIC    * Red neuronal con 1000 capas en dataset pequeño
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4.3 Técnicas para Combatir Overfitting
# MAGIC
# MAGIC **1. Más datos de entrenamiento**
# MAGIC * Regla empírica: $n > 10 \times p$ (10 muestras por feature)
# MAGIC
# MAGIC **2. Regularización**
# MAGIC
# MAGIC **Ridge (L2):**
# MAGIC $$\min_{\beta} \sum_{i=1}^{n} (y_i - \beta^T x_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2$$
# MAGIC
# MAGIC **Lasso (L1):**
# MAGIC $$\min_{\beta} \sum_{i=1}^{n} (y_i - \beta^T x_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j|$$
# MAGIC
# MAGIC * $\lambda$: Parámetro de regularización ($\lambda \uparrow$ → penalización ↑)
# MAGIC
# MAGIC **3. Validación cruzada (Cross-validation)**
# MAGIC
# MAGIC **K-Fold CV:**
# MAGIC 1. Dividir datos en $k$ folds (típicamente $k=5$ o $k=10$)
# MAGIC 2. Para cada fold:
# MAGIC    * Entrenar en $k-1$ folds
# MAGIC    * Validar en 1 fold
# MAGIC 3. Promediar resultados
# MAGIC
# MAGIC $$\text{CV Error} = \frac{1}{k} \sum_{i=1}^{k} \text{Error}_i$$
# MAGIC
# MAGIC **4. Early stopping**
# MAGIC * Detener entrenamiento cuando error de validación deja de mejorar
# MAGIC
# MAGIC **5. Simplificar el modelo**
# MAGIC * Reducir profundidad de árboles
# MAGIC * Eliminar features irrelevantes
# MAGIC * Usar menos parámetros

# COMMAND ----------

# DBTITLE 1,Proceso de Desarrollo ML
# MAGIC %md
# MAGIC ## 5. Proceso de Desarrollo de Modelos ML
# MAGIC
# MAGIC ### Pipeline Completo
# MAGIC
# MAGIC ```
# MAGIC 1. Definición del Problema
# MAGIC    ↓
# MAGIC 2. Recolección de Datos
# MAGIC    ↓
# MAGIC 3. Análisis Exploratorio (EDA)
# MAGIC    ↓
# MAGIC 4. Feature Engineering
# MAGIC    ↓
# MAGIC 5. División de Datos (Train/Validation/Test)
# MAGIC    ↓
# MAGIC 6. Selección de Modelo
# MAGIC    ↓
# MAGIC 7. Entrenamiento
# MAGIC    ↓
# MAGIC 8. Evaluación
# MAGIC    ↓
# MAGIC 9. Tuning de Hiperparámetros
# MAGIC    ↓
# MAGIC 10. Evaluación Final (Test Set)
# MAGIC    ↓
# MAGIC 11. Deployment
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5.1 División de Datos
# MAGIC
# MAGIC **Proporción típica:**
# MAGIC
# MAGIC ```
# MAGIC Todos los datos (100%)
# MAGIC ├── Train (60-70%)     → Entrenar modelos
# MAGIC ├── Validation (15-20%) → Seleccionar hiperparámetros
# MAGIC └── Test (15-20%)       → Evaluación final (una sola vez)
# MAGIC ```
# MAGIC
# MAGIC **Reglas de oro:**
# MAGIC
# MAGIC 1. **Test set:** Nunca usar hasta la evaluación final
# MAGIC 2. **Validation set:** Usar para comparar modelos y ajustar hiperparámetros
# MAGIC 3. **Train set:** Único conjunto usado para entrenar
# MAGIC
# MAGIC **Ejemplo con 10,000 datos:**
# MAGIC * Train: 7,000 (70%)
# MAGIC * Validation: 1,500 (15%)
# MAGIC * Test: 1,500 (15%)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5.2 Selección de Modelo
# MAGIC
# MAGIC **Factores a considerar:**
# MAGIC
# MAGIC | Factor | Consideración |
# MAGIC |---|---|
# MAGIC | **Tipo de problema** | Clasificación / Regresión / Clustering |
# MAGIC | **Tamaño de datos** | $n$ pequeño → modelos simples<br>$n$ grande → modelos complejos |
# MAGIC | **Número de features** | $p$ grande → regularización, selección de features |
# MAGIC | **Interpretabilidad** | Stakeholders → árboles, regresión lineal<br>Técnicos → redes neuronales, ensemble |
# MAGIC | **Tiempo de entrenamiento** | Producción → modelos rápidos<br>Investigación → modelos complejos |
# MAGIC | **Relación señal/ruido** | Ruido alto → modelos robustos (ensemble) |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5.3 Métricas de Evaluación
# MAGIC
# MAGIC **Clasificación:**
# MAGIC
# MAGIC * **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
# MAGIC * **Precision:** $\frac{TP}{TP + FP}$
# MAGIC * **Recall:** $\frac{TP}{TP + FN}$
# MAGIC * **F1-Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
# MAGIC
# MAGIC **Regresión:**
# MAGIC
# MAGIC * **RMSE:** $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$
# MAGIC * **MAE:** $\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
# MAGIC * **R²:** $1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$

# COMMAND ----------

# DBTITLE 1,Comparación de Enfoques
# MAGIC %md
# MAGIC ## 6. Comparación de Enfoques de Machine Learning
# MAGIC
# MAGIC ### 6.1 Modelos Lineales vs No Lineales
# MAGIC
# MAGIC | Aspecto | Lineales | No Lineales |
# MAGIC |---|---|---|
# MAGIC | **Complejidad** | Baja | Alta |
# MAGIC | **Interpretabilidad** | ✅ Excelente | ❌ Difícil |
# MAGIC | **Overfitting** | Riesgo bajo | Riesgo alto |
# MAGIC | **Datos necesarios** | Pocos | Muchos |
# MAGIC | **Relaciones capturadas** | Solo lineales | Lineales + no lineales |
# MAGIC | **Ejemplos** | Regresión lineal<br>Regresión logística | Árboles de decisión<br>Redes neuronales<br>SVM con kernel |
# MAGIC | **Cuándo usar** | Relación lineal clara<br>Interpretabilidad crítica<br>Pocos datos | Relaciones complejas<br>Muchos datos<br>Accuracy > interpretabilidad |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 6.2 Modelos Individuales vs Ensemble
# MAGIC
# MAGIC **Modelos Individuales:**
# MAGIC * Un solo algoritmo
# MAGIC * Ejemplos: Árbol de decisión, regresión lineal
# MAGIC
# MAGIC **Ensemble (Ensambles):**
# MAGIC * Combinan múltiples modelos
# MAGIC * **Bagging:** Entrenar en muestras aleatorias (Random Forest)
# MAGIC * **Boosting:** Entrenar secuencialmente corrigiendo errores (XGBoost, AdaBoost)
# MAGIC * **Stacking:** Combinar diferentes tipos de modelos
# MAGIC
# MAGIC **Comparación:**
# MAGIC
# MAGIC | Aspecto | Individual | Ensemble |
# MAGIC |---|---|---|
# MAGIC | **Accuracy** | Moderado | ✅ Alto |
# MAGIC | **Overfitting** | Riesgo variable | Riesgo reducido |
# MAGIC | **Tiempo entrenamiento** | ✅ Rápido | ❌ Lento |
# MAGIC | **Interpretabilidad** | ✅ Mejor | ❌ Peor |
# MAGIC | **Robustez** | Moderada | ✅ Alta |
# MAGIC
# MAGIC **Cuándo usar Ensemble:**
# MAGIC * Competiciones Kaggle
# MAGIC * Producción de alta criticidad
# MAGIC * Cuando accuracy es prioridad #1
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 6.3 Modelos Paramétricos vs No Paramétricos
# MAGIC
# MAGIC **Paramétricos:**
# MAGIC * Asumen una forma funcional específica
# MAGIC * Número fijo de parámetros
# MAGIC * Ejemplos: Regresión lineal ($y = \beta_0 + \beta_1 x$)
# MAGIC
# MAGIC **No Paramétricos:**
# MAGIC * No asumen forma funcional
# MAGIC * Flexibilidad máxima
# MAGIC * Ejemplos: K-Nearest Neighbors, Árboles de decisión
# MAGIC
# MAGIC | Aspecto | Paramétricos | No Paramétricos |
# MAGIC |---|---|---|
# MAGIC | **Flexibilidad** | ❌ Baja | ✅ Alta |
# MAGIC | **Datos necesarios** | ✅ Pocos | ❌ Muchos |
# MAGIC | **Velocidad inferencia** | ✅ Rápida | ❌ Lenta |
# MAGIC | **Riesgo underfitting** | Alto | Bajo |
# MAGIC | **Riesgo overfitting** | Bajo | Alto |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 6.4 Guía de Selección de Algoritmo
# MAGIC
# MAGIC **Para Clasificación:**
# MAGIC
# MAGIC ```
# MAGIC ¿Interpretabilidad crítica?
# MAGIC ├─ SÍ → Árbol de decisión, Regresión logística
# MAGIC └─ NO
# MAGIC    ├─ Datos pequeños (< 10k) → SVM, Regresión logística
# MAGIC    └─ Datos grandes (> 10k)
# MAGIC       ├─ Features numéricas → Random Forest, XGBoost
# MAGIC       └─ Texto/Imágenes → Redes neuronales
# MAGIC ```
# MAGIC
# MAGIC **Para Regresión:**
# MAGIC
# MAGIC ```
# MAGIC ¿Relación lineal?
# MAGIC ├─ SÍ → Regresión lineal
# MAGIC └─ NO
# MAGIC    ├─ Pocos datos → Regresión con regularización (Ridge, Lasso)
# MAGIC    └─ Muchos datos → Random Forest, XGBoost, Redes neuronales
# MAGIC ```
# MAGIC
# MAGIC **Para Clustering:**
# MAGIC
# MAGIC ```
# MAGIC ¿Conoces el número de clusters?
# MAGIC ├─ SÍ → K-Means
# MAGIC └─ NO
# MAGIC    ├─ Clusters esféricos → K-Means con método del codo
# MAGIC    └─ Clusters arbitrarios → DBSCAN, Hierarchical
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Conclusiones y Siguientes Pasos
# MAGIC %md
# MAGIC ## 7. Conclusiones y Recursos
# MAGIC
# MAGIC ### 📚 Resumen de Conceptos Clave
# MAGIC
# MAGIC 1. **Machine Learning** es aprendizaje automático de patrones desde datos
# MAGIC 2. **Tres paradigmas principales**: Supervisado, No supervisado, Refuerzo
# MAGIC 3. **Teoría estadística**: Minimizar riesgo empírico esperando generalización
# MAGIC 4. **Bias-Variance tradeoff**: Balance entre simplicidad y complejidad
# MAGIC 5. **Overfitting**: Enemigo #1 - combatir con regularización, CV, más datos
# MAGIC 6. **Proceso sistemático**: EDA → Feature Engineering → Train → Evaluate → Deploy
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Próximos Notebooks en esta Serie
# MAGIC
# MAGIC **En Fundamentos:**
# MAGIC * `02_Matematicas_Esenciales.ipynb` - Álgebra lineal, cálculo, optimización
# MAGIC
# MAGIC **En Clasificación:**
# MAGIC * `Teoria_Arboles_Decision.ipynb` - Entropía, ganancia información, CART
# MAGIC * `Arbol_Decision_Clasificacion.ipynb` - Implementación práctica (churn)
# MAGIC
# MAGIC **En Regresión:**
# MAGIC * `Teoria_Regresion.ipynb` - Mínimos cuadrados, supuestos, diagnósticos
# MAGIC * `Arbol_Decision_Regresion.ipynb` - Árboles para regresión
# MAGIC * `Regresion_Lineal_Multiple.ipynb` - Implementación práctica (precios)
# MAGIC
# MAGIC **En Clustering:**
# MAGIC * `Teoria_Clustering.ipynb` - K-Means, métricas, comparación algoritmos
# MAGIC * `KMeans_Clustering.ipynb` - Implementación práctica (segmentación)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📖 Referencias Académicas
# MAGIC
# MAGIC **Libros:**
# MAGIC 1. **Hastie, Tibshirani, Friedman** - "The Elements of Statistical Learning" (2009)
# MAGIC 2. **Bishop** - "Pattern Recognition and Machine Learning" (2006)
# MAGIC 3. **Murphy** - "Machine Learning: A Probabilistic Perspective" (2012)
# MAGIC 4. **James et al.** - "An Introduction to Statistical Learning" (2021)
# MAGIC
# MAGIC **Papers fundamentales:**
# MAGIC * Vapnik & Chervonenkis (1971) - VC Theory
# MAGIC * Breiman (2001) - Random Forests
# MAGIC * Friedman (2001) - Gradient Boosting Machines
# MAGIC
# MAGIC **Cursos online:**
# MAGIC * Andrew Ng - Machine Learning (Coursera)
# MAGIC * Stanford CS229 - Machine Learning
# MAGIC * MIT 6.034 - Artificial Intelligence
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔬 Ejercicios Propuestos
# MAGIC
# MAGIC 1. **Derivación matemática:** Demostrar la descomposición bias-variance
# MAGIC 2. **Análisis teórico:** ¿Por qué K-Fold CV reduce varianza de la estimación?
# MAGIC 3. **Comparación:** Construir tabla comparativa de 5 algoritmos de clasificación
# MAGIC 4. **Implementación:** Programar validación cruzada desde cero (sin sklearn)

# COMMAND ----------

