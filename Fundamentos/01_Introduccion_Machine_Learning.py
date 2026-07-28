# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Introducción al Machine Learning
# MAGIC %md
# MAGIC # Introducción al Machine Learning
# MAGIC
# MAGIC ## 1. ¿Qué es Machine Learning?
# MAGIC
# MAGIC **Machine Learning (Aprendizaje Automático)** es una rama de la Inteligencia Artificial que permite a los sistemas aprender y mejorar automáticamente a partir de la experiencia sin ser explícitamente programados.
# MAGIC
# MAGIC ### Definición Formal
# MAGIC
# MAGIC > "Un programa de computadora aprende de la experiencia E con respecto a alguna clase de tareas T y medida de rendimiento P, si su rendimiento en las tareas T, medido por P, mejora con la experiencia E."
# MAGIC >
# MAGIC > — Tom Mitchell (1997)
# MAGIC
# MAGIC ### Ejemplo
# MAGIC
# MAGIC * **Tarea (T)**: Clasificar correos como spam o no spam
# MAGIC * **Experiencia (E)**: Base de datos de correos etiquetados
# MAGIC * **Rendimiento (P)**: Porcentaje de correos clasificados correctamente
# MAGIC
# MAGIC ### Programación Tradicional vs Machine Learning
# MAGIC
# MAGIC | Programación Tradicional | Machine Learning |
# MAGIC |-------------------------|------------------|
# MAGIC | Reglas + Datos → Respuestas | Datos + Respuestas → Reglas |
# MAGIC | Experto define lógica | Algoritmo aprende patrones |
# MAGIC | Difícil para problemas complejos | Eficaz en problemas complejos |

# COMMAND ----------

# DBTITLE 1,Tipos de Aprendizaje
# MAGIC %md
# MAGIC ## 2. Tipos de Aprendizaje
# MAGIC
# MAGIC ### 2.1 Aprendizaje Supervisado
# MAGIC
# MAGIC El modelo aprende de **datos etiquetados**: se conocen las entradas y salidas correctas.
# MAGIC
# MAGIC $$\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), ..., (\mathbf{x}_n, y_n)\}$$
# MAGIC
# MAGIC **Objetivo:** Aprender función $f: \mathbb{X} \rightarrow \mathbb{Y}$ tal que $f(\mathbf{x}_i) \approx y_i$
# MAGIC
# MAGIC **Tipos de problemas:**
# MAGIC
# MAGIC * **Clasificación**: $y$ es categórico (spam/no spam, gato/perro/pájaro)
# MAGIC   - Ejemplos: Naive Bayes, Árboles de Decisión, SVM, Random Forest
# MAGIC * **Regresión**: $y$ es continuo (precio, temperatura, edad)
# MAGIC   - Ejemplos: Regresión Lineal, Árboles de Regresión, Gradient Boosting
# MAGIC
# MAGIC ### 2.2 Aprendizaje No Supervisado
# MAGIC
# MAGIC El modelo aprende de **datos sin etiquetas**: solo se conocen las entradas.
# MAGIC
# MAGIC $$\mathcal{D} = \{\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n\}$$
# MAGIC
# MAGIC **Objetivo:** Encontrar estructura oculta en los datos
# MAGIC
# MAGIC **Tipos de problemas:**
# MAGIC
# MAGIC * **Clustering**: Agrupar datos similares
# MAGIC   - Ejemplos: K-Means, DBSCAN, Hierarchical Clustering
# MAGIC * **Reducción de dimensionalidad**: Comprimir datos manteniendo información
# MAGIC   - Ejemplos: PCA, t-SNE, UMAP
# MAGIC * **Detección de anomalías**: Identificar datos atípicos
# MAGIC
# MAGIC ### 2.3 Aprendizaje por Refuerzo
# MAGIC
# MAGIC Un **agente** aprende a tomar decisiones mediante interacción con un **entorno**.
# MAGIC
# MAGIC * Recibe **recompensas** o **penalizaciones** por sus acciones
# MAGIC * Objetivo: Maximizar recompensa acumulada a largo plazo
# MAGIC * Ejemplos: Juegos (AlphaGo), robótica, vehículos autónomos

# COMMAND ----------

# DBTITLE 1,Proceso de Machine Learning
# MAGIC %md
# MAGIC ## 3. Proceso de Machine Learning
# MAGIC
# MAGIC ### Pipeline de ML
# MAGIC
# MAGIC ```
# MAGIC 1. Definir Problema
# MAGIC         ↓
# MAGIC 2. Recolectar Datos
# MAGIC         ↓
# MAGIC 3. Explorar y Limpiar Datos (EDA)
# MAGIC         ↓
# MAGIC 4. Ingeniería de Características
# MAGIC         ↓
# MAGIC 5. Dividir Datos (Train/Validation/Test)
# MAGIC         ↓
# MAGIC 6. Seleccionar y Entrenar Modelo
# MAGIC         ↓
# MAGIC 7. Evaluar Modelo
# MAGIC         ↓
# MAGIC 8. Ajustar Hiperparámetros
# MAGIC         ↓
# MAGIC 9. Implementar (Deploy)
# MAGIC         ↓
# MAGIC 10. Monitorear y Mantener
# MAGIC ```
# MAGIC
# MAGIC ### División de Datos
# MAGIC
# MAGIC * **Training set (60-80%)**: Para entrenar el modelo
# MAGIC * **Validation set (10-20%)**: Para ajustar hiperparámetros
# MAGIC * **Test set (10-20%)**: Para evaluación final
# MAGIC
# MAGIC **Importante:** El test set **nunca** se usa durante el entrenamiento.
# MAGIC
# MAGIC ### Cross-Validation (Validación Cruzada)
# MAGIC
# MAGIC Técnica para evaluar modelos dividiendo datos en $k$ particiones:
# MAGIC
# MAGIC 1. Dividir datos en $k$ folds
# MAGIC 2. Para cada fold $i$:
# MAGIC    - Entrenar con $k-1$ folds
# MAGIC    - Validar con fold $i$
# MAGIC 3. Promediar resultados
# MAGIC
# MAGIC **K-Fold CV** con $k=5$:
# MAGIC
# MAGIC ```
# MAGIC Fold 1: [Test][Train][Train][Train][Train]
# MAGIC Fold 2: [Train][Test][Train][Train][Train]
# MAGIC Fold 3: [Train][Train][Test][Train][Train]
# MAGIC Fold 4: [Train][Train][Train][Test][Train]
# MAGIC Fold 5: [Train][Train][Train][Train][Test]
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Métricas de Evaluación
# MAGIC %md
# MAGIC ## 4. Métricas de Evaluación
# MAGIC
# MAGIC ### 4.1 Clasificación
# MAGIC
# MAGIC **Matriz de Confusión:**
# MAGIC
# MAGIC |                  | Predicho Positivo | Predicho Negativo |
# MAGIC |------------------|-------------------|-------------------|
# MAGIC | **Real Positivo** | True Positive (TP) | False Negative (FN) |
# MAGIC | **Real Negativo** | False Positive (FP) | True Negative (TN) |
# MAGIC
# MAGIC **Métricas:**
# MAGIC
# MAGIC $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
# MAGIC
# MAGIC $$\text{Precision} = \frac{TP}{TP + FP}$$ (De los predichos positivos, ¿cuántos son correctos?)
# MAGIC
# MAGIC $$\text{Recall (Sensitivity)} = \frac{TP}{TP + FN}$$ (De los reales positivos, ¿cuántos detectamos?)
# MAGIC
# MAGIC $$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
# MAGIC
# MAGIC **Curva ROC y AUC:**
# MAGIC
# MAGIC * ROC (Receiver Operating Characteristic): Grafica TPR vs FPR
# MAGIC * AUC (Area Under Curve): Área bajo la curva ROC
# MAGIC * AUC = 1.0: Clasificador perfecto
# MAGIC * AUC = 0.5: Clasificador aleatorio
# MAGIC
# MAGIC ### 4.2 Regresión
# MAGIC
# MAGIC **Mean Absolute Error (MAE):**
# MAGIC
# MAGIC $$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$
# MAGIC
# MAGIC **Mean Squared Error (MSE):**
# MAGIC
# MAGIC $$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
# MAGIC
# MAGIC **Root Mean Squared Error (RMSE):**
# MAGIC
# MAGIC $$RMSE = \sqrt{MSE}$$
# MAGIC
# MAGIC **R² (Coeficiente de Determinación):**
# MAGIC
# MAGIC $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$
# MAGIC
# MAGIC Donde $0 \leq R^2 \leq 1$ (1 = ajuste perfecto)

# COMMAND ----------

# DBTITLE 1,Overfitting y Underfitting
# MAGIC %md
# MAGIC ## 5. Overfitting y Underfitting
# MAGIC
# MAGIC ### 5.1 Underfitting (Sesgo Alto)
# MAGIC
# MAGIC * Modelo **demasiado simple**
# MAGIC * No captura patrones en los datos
# MAGIC * **Alto error** en training y test
# MAGIC * **Solución**: Modelo más complejo, más características
# MAGIC
# MAGIC ### 5.2 Overfitting (Varianza Alta)
# MAGIC
# MAGIC * Modelo **demasiado complejo**
# MAGIC * Memoriza ruido en lugar de patrones generales
# MAGIC * **Bajo error** en training, **alto error** en test
# MAGIC * **Solución**: Regularización, más datos, simplificar modelo
# MAGIC
# MAGIC ### 5.3 Compromiso Bias-Variance
# MAGIC
# MAGIC $$\text{Error Total} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$
# MAGIC
# MAGIC * **Bias (Sesgo)**: Error por suposiciones incorrectas del modelo
# MAGIC * **Variance (Varianza)**: Sensibilidad a fluctuaciones en datos de entrenamiento
# MAGIC * **Irreducible Error**: Ruido inherente en los datos
# MAGIC
# MAGIC ### Trade-off Visual
# MAGIC
# MAGIC ```
# MAGIC Error ↑
# MAGIC       |
# MAGIC       |        Underfitting      Sweet Spot    Overfitting
# MAGIC       |             \              /     \       /
# MAGIC       |              \            /       \     /
# MAGIC       |               \          /         \   /
# MAGIC       |    Test Error  \________/___________\_/
# MAGIC       |                         /
# MAGIC       |    Train Error _______/___________________
# MAGIC       |______________________________________→ Complejidad
# MAGIC ```
# MAGIC
# MAGIC ### Técnicas de Regularización
# MAGIC
# MAGIC * **L1 (Lasso)**: Penaliza suma absoluta de pesos
# MAGIC * **L2 (Ridge)**: Penaliza suma cuadrática de pesos
# MAGIC * **Dropout**: Desactiva neuronas aleatoriamente (redes neuronales)
# MAGIC * **Early Stopping**: Detener entrenamiento antes de overfitting
# MAGIC * **Data Augmentation**: Aumentar datos de entrenamiento

# COMMAND ----------

# DBTITLE 1,Ingeniería de Características
# MAGIC %md
# MAGIC ## 6. Ingeniería de Características
# MAGIC
# MAGIC Transformar datos crudos en características útiles para el modelo.
# MAGIC
# MAGIC ### 6.1 Escalamiento
# MAGIC
# MAGIC **Normalización (Min-Max Scaling):**
# MAGIC
# MAGIC $$x' = \frac{x - x_{min}}{x_{max} - x_{min}} \in [0, 1]$$
# MAGIC
# MAGIC **Estandarización (Z-score):**
# MAGIC
# MAGIC $$x' = \frac{x - \mu}{\sigma}$$
# MAGIC
# MAGIC ### 6.2 Codificación de Variables Categóricas
# MAGIC
# MAGIC **One-Hot Encoding:**
# MAGIC
# MAGIC ```
# MAGIC Color: [Rojo, Verde, Azul]
# MAGIC → Rojo:  [1, 0, 0]
# MAGIC → Verde: [0, 1, 0]
# MAGIC → Azul:  [0, 0, 1]
# MAGIC ```
# MAGIC
# MAGIC **Label Encoding:**
# MAGIC
# MAGIC ```
# MAGIC Tamaño: [Pequeño, Mediano, Grande]
# MAGIC → [0, 1, 2]
# MAGIC ```
# MAGIC
# MAGIC ### 6.3 Manejo de Datos Faltantes
# MAGIC
# MAGIC * **Eliminar**: Remover filas/columnas con valores faltantes
# MAGIC * **Imputación**: Llenar con media, mediana, moda
# MAGIC * **Modelo predictivo**: Predecir valores faltantes
# MAGIC
# MAGIC ### 6.4 Creación de Características
# MAGIC
# MAGIC * **Polinomiales**: $x^2, x^3, \sqrt{x}$
# MAGIC * **Interacciones**: $x_1 \cdot x_2$
# MAGIC * **Transformaciones**: $\log(x), e^x$
# MAGIC * **Binning**: Discretizar variables continuas

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## 7. Conclusiones
# MAGIC
# MAGIC ### Conceptos Clave
# MAGIC
# MAGIC 1. **Aprendizaje Supervisado**: Datos etiquetados (clasificación, regresión)
# MAGIC 2. **Aprendizaje No Supervisado**: Datos sin etiquetas (clustering, reducción)
# MAGIC 3. **Pipeline de ML**: Datos → Preprocesamiento → Modelo → Evaluación
# MAGIC 4. **Evaluación**: Métricas apropiadas según el problema
# MAGIC 5. **Overfitting/Underfitting**: Balance entre complejidad y generalización
# MAGIC 6. **Ingeniería de características**: Transformar datos para mejor performance
# MAGIC
# MAGIC ### Próximos Pasos
# MAGIC
# MAGIC * **Fundamentos matemáticos**: Álgebra lineal, cálculo, probabilidad
# MAGIC * **Algoritmos específicos**: Árboles, SVM, redes neuronales
# MAGIC * **Frameworks**: Scikit-learn, TensorFlow, PyTorch
# MAGIC * **MLOps**: Despliegue, monitoreo, mantenimiento
# MAGIC
# MAGIC ### Recursos
# MAGIC
# MAGIC * **Cursos**: Coursera (Andrew Ng), Fast.ai
# MAGIC * **Libros**: "Hands-On Machine Learning" (Géron), "Pattern Recognition" (Bishop)
# MAGIC * **Práctica**: Kaggle, UCI ML Repository