# Databricks notebook source
# DBTITLE 1,Algoritmos de Clasificación para Negocios
# MAGIC %md
# MAGIC # Algoritmos de Clasificación para Problemas de Negocios
# MAGIC
# MAGIC ## 🎯 Objetivo
# MAGIC
# MAGIC Este notebook presenta los **principales algoritmos de clasificación** utilizados en Machine Learning, con énfasis en sus **aplicaciones prácticas en el mundo de los negocios**.
# MAGIC
# MAGIC ### 📌 ¿Qué es Clasificación?
# MAGIC
# MAGIC > **Clasificación** es el problema de predecir a qué categoría o clase pertenece una observación, basándose en sus características.
# MAGIC
# MAGIC **Definición formal:**
# MAGIC
# MAGIC Dado un dataset $\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), ..., (\mathbf{x}_n, y_n)\}$
# MAGIC
# MAGIC Donde:
# MAGIC * $\mathbf{x}_i \in \mathbb{R}^d$ son las características (features)
# MAGIC * $y_i \in \{1, 2, ..., K\}$ son las clases/categorías
# MAGIC
# MAGIC **Objetivo:** Aprender función $f: \mathbb{R}^d \rightarrow \{1, 2, ..., K\}$ que minimice el error de clasificación.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Casos de Uso Empresariales
# MAGIC
# MAGIC ### Clasificación Binaria (2 clases)
# MAGIC
# MAGIC | Problema de Negocio | Clase 0 | Clase 1 | Features Típicas |
# MAGIC |---------------------|---------|---------|------------------|
# MAGIC | **Detección de Fraude** | Transacción legítima | Transacción fraudulenta | Monto, ubicación, hora, historial |
# MAGIC | **Churn Prediction** | Cliente se queda | Cliente abandona | Uso del producto, tickets, antigüedad |
# MAGIC | **Credit Scoring** | No pagará | Pagará | Ingresos, deudas, historial crediticio |
# MAGIC | **Email Marketing** | No abre | Abre el email | Hora de envío, asunto, historial |
# MAGIC | **Aprobación de Préstamos** | Rechazar | Aprobar | Ingresos, edad, empleo, score |
# MAGIC
# MAGIC ### Clasificación Multiclase (3+ clases)
# MAGIC
# MAGIC | Problema de Negocio | Clases | Features Típicas |
# MAGIC |---------------------|--------|------------------|
# MAGIC | **Segmentación de Clientes** | Premium, Regular, Básico | Gasto, frecuencia, antigüedad |
# MAGIC | **Priorización de Tickets** | Baja, Media, Alta, Crítica | Tipo, cliente, impacto |
# MAGIC | **Categorización de Productos** | Electrónica, Ropa, Hogar... | Descripción, precio, marca |
# MAGIC | **Análisis de Sentimientos** | Positivo, Neutral, Negativo | Texto del review |
# MAGIC | **Riesgo de Inversión** | Bajo, Medio, Alto | Volatilidad, sector, métricas |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Contenido
# MAGIC
# MAGIC 1. **Regresión Logística** - El modelo lineal para clasificación
# MAGIC 2. **Árboles de Decisión** - Reglas interpretables
# MAGIC 3. **Random Forest** - Ensemble de árboles
# MAGIC 4. **Gradient Boosting** - XGBoost, LightGBM, CatBoost
# MAGIC 5. **Support Vector Machines (SVM)** - Máxima separación
# MAGIC 6. **Naive Bayes** - Probabilidad bayesiana
# MAGIC 7. **K-Nearest Neighbors (KNN)** - Clasificación por vecindad
# MAGIC 8. **Redes Neuronales** - Deep Learning
# MAGIC 9. **Comparación y Selección** - ¿Cuándo usar cada algoritmo?
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,1. Regresión Logística
# MAGIC %md
# MAGIC ## 1. Regresión Logística
# MAGIC
# MAGIC ### 🎯 Concepto
# MAGIC
# MAGIC **Regresión Logística** es un modelo lineal que predice la **probabilidad** de que una observación pertenezca a una clase.
# MAGIC
# MAGIC ### 📐 Formulación Matemática
# MAGIC
# MAGIC **Para clasificación binaria:**
# MAGIC
# MAGIC $$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$
# MAGIC
# MAGIC Donde:
# MAGIC * $\sigma(z)$ es la **función sigmoide** (logística)
# MAGIC * $\mathbf{w}$ son los pesos/coeficientes
# MAGIC * $b$ es el bias/intercepto
# MAGIC * Output: probabilidad $\in [0, 1]$
# MAGIC
# MAGIC **Función de decisión:**
# MAGIC
# MAGIC $$\hat{y} = \begin{cases} 1 & \text{si } P(y=1|\mathbf{x}) \geq 0.5 \\ 0 & \text{si } P(y=1|\mathbf{x}) < 0.5 \end{cases}$$
# MAGIC
# MAGIC **Función de pérdida (Log Loss):**
# MAGIC
# MAGIC $$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$$
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC * **Interpretabilidad**: Coeficientes muestran impacto de cada feature
# MAGIC * **Probabilidades calibradas**: Output es probabilidad real
# MAGIC * **Rápido**: Entrenamiento e inferencia eficientes
# MAGIC * **Funciona bien con datos linealmente separables**
# MAGIC * **Regularización**: L1 (Lasso) y L2 (Ridge) previenen overfitting
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC * **Asume linealidad**: No captura relaciones no-lineales complejas
# MAGIC * **Sensible a outliers**: Puede afectar los coeficientes
# MAGIC * **Requiere features escaladas**: Para convergencia óptima
# MAGIC * **No maneja automáticamente interacciones**: Hay que crearlas manualmente
# MAGIC
# MAGIC ### 💼 Casos de Uso Empresariales
# MAGIC
# MAGIC | Industria | Aplicación | Por qué Regresión Logística |
# MAGIC |-----------|------------|-----------------------------|  
# MAGIC | **Finanzas** | Credit scoring | Interpretabilidad para cumplimiento regulatorio |
# MAGIC | **Marketing** | Predicción de conversión | Necesitas probabilidades para optimizar campañas |
# MAGIC | **Salud** | Diagnóstico de enfermedades | Médicos necesitan entender factores de riesgo |
# MAGIC | **E-commerce** | Predicción de compra | Rápido para scoring en tiempo real |
# MAGIC | **Seguros** | Evaluación de riesgo | Transparencia para justificar primas |
# MAGIC
# MAGIC ### 🔧 Hiperparámetros Clave
# MAGIC
# MAGIC * **C (inverso de regularización)**: Más bajo = más regularización
# MAGIC * **penalty**: 'l1' (Lasso) o 'l2' (Ridge)
# MAGIC * **solver**: Algoritmo de optimización ('lbfgs', 'saga', 'liblinear')
# MAGIC * **class_weight**: 'balanced' para datasets desbalanceados
# MAGIC
# MAGIC ### 📊 Ejemplo de Interpretación
# MAGIC
# MAGIC **Modelo de Churn:**
# MAGIC
# MAGIC $$P(\text{Churn}) = \sigma(-2.5 + 0.3 \times \text{Tickets} - 0.8 \times \text{Satisfacción} + 0.5 \times \text{Precio})$$
# MAGIC
# MAGIC **Interpretación:**
# MAGIC * Cada ticket adicional aumenta **30%** el log-odds de churn
# MAGIC * Cada punto de satisfacción reduce **80%** el log-odds de churn
# MAGIC * Precio más alto aumenta probabilidad de churn

# COMMAND ----------

# DBTITLE 1,2. Árboles de Decisión
# MAGIC %md
# MAGIC ## 2. Árboles de Decisión (Decision Trees)
# MAGIC
# MAGIC ### 🎯 Concepto
# MAGIC
# MAGIC **Árbol de Decisión** es un modelo que toma decisiones mediante una serie de preguntas sobre las features, similar a un diagrama de flujo.
# MAGIC
# MAGIC ### 🌳 Estructura
# MAGIC
# MAGIC ```
# MAGIC            [Edad < 30?]
# MAGIC               /    \
# MAGIC             Sí     No
# MAGIC             /        \
# MAGIC     [Salario < 50K?]  [Tiene casa?]
# MAGIC        /    \          /      \
# MAGIC      Sí    No        Sí       No
# MAGIC      /      \        /         \
# MAGIC   Clase 0  Clase 1  Clase 1   Clase 0
# MAGIC ```
# MAGIC
# MAGIC ### 📐 Algoritmo de Construcción
# MAGIC
# MAGIC **Criterios de división:**
# MAGIC
# MAGIC 1. **Gini Impurity (CART):**
# MAGIC
# MAGIC $$\text{Gini}(S) = 1 - \sum_{i=1}^{K} p_i^2$$
# MAGIC
# MAGIC Donde $p_i$ es la proporción de clase $i$ en el set $S$.
# MAGIC
# MAGIC * **Gini = 0**: Nodo puro (una sola clase)
# MAGIC * **Gini = 0.5**: Máxima impureza (binario, 50/50)
# MAGIC
# MAGIC 2. **Entropy (ID3, C4.5):**
# MAGIC
# MAGIC $$\text{Entropy}(S) = -\sum_{i=1}^{K} p_i \log_2(p_i)$$
# MAGIC
# MAGIC **Information Gain:**
# MAGIC
# MAGIC $$IG(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)$$
# MAGIC
# MAGIC Se elige la feature $A$ que maximiza $IG$.
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC * **Altamente interpretable**: Fácil de visualizar y explicar
# MAGIC * **No requiere normalización**: Funciona con features de diferentes escalas
# MAGIC * **Maneja datos categóricos y numéricos**: Sin preprocessing
# MAGIC * **Captura interacciones**: Automáticamente encuentra combinaciones de features
# MAGIC * **Feature importance**: Mide importancia de cada variable
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC * **Overfitting**: Árboles profundos memorizan ruido
# MAGIC * **Inestabilidad**: Pequeños cambios en datos → árbol muy diferente
# MAGIC * **Sesgo hacia features con muchos valores**: Puede preferirlas injustamente
# MAGIC * **No extrapola**: No puede predecir fuera del rango de entrenamiento
# MAGIC * **Boundaries ortogonales**: Divisiones paralelas a ejes, no diagonal
# MAGIC
# MAGIC ### 💼 Casos de Uso Empresariales
# MAGIC
# MAGIC | Industria | Aplicación | Por qué Árboles |
# MAGIC |-----------|------------|----------------|
# MAGIC | **Retail** | Segmentación de clientes | Interpretabilidad para marketing |
# MAGIC | **Salud** | Diagnóstico médico | Los médicos pueden seguir la lógica |
# MAGIC | **Finanzas** | Aprobación de préstamos | Transparencia regulatoria |
# MAGIC | **Recursos Humanos** | Predicción de renuncia | Identificar factores clave de retención |
# MAGIC | **Manufactura** | Control de calidad | Reglas claras para operadores |
# MAGIC
# MAGIC ### 🔧 Hiperparámetros Clave
# MAGIC
# MAGIC * **max_depth**: Profundidad máxima del árbol (limita overfitting)
# MAGIC * **min_samples_split**: Mínimo de muestras para dividir un nodo
# MAGIC * **min_samples_leaf**: Mínimo de muestras en hoja
# MAGIC * **max_features**: Número de features a considerar en cada división
# MAGIC * **criterion**: 'gini' o 'entropy'
# MAGIC
# MAGIC ### 📊 Ejemplo de Negocio
# MAGIC
# MAGIC **Predicción de Abandono de Clientes:**
# MAGIC
# MAGIC ```
# MAGIC Raíz: 1000 clientes (30% churn)
# MAGIC     |
# MAGIC     ├─ [Tickets de soporte > 5?]
# MAGIC     │   ├─ Sí: 200 clientes (70% churn) → ALTO RIESGO
# MAGIC     │   └─ No: 800 clientes (20% churn)
# MAGIC     │       |
# MAGIC     │       ├─ [Antigüedad < 1 año?]
# MAGIC     │       │   ├─ Sí: 300 clientes (35% churn) → RIESGO MEDIO
# MAGIC     │       │   └─ No: 500 clientes (10% churn) → BAJO RIESGO
# MAGIC ```
# MAGIC
# MAGIC **Insights accionables:**
# MAGIC * Priorizar clientes con >5 tickets
# MAGIC * Programa de retención para nuevos clientes

# COMMAND ----------

# DBTITLE 1,3. Random Forest
# MAGIC %md
# MAGIC ## 3. Random Forest
# MAGIC
# MAGIC ### 🎯 Concepto
# MAGIC
# MAGIC **Random Forest** es un **ensemble** de múltiples árboles de decisión que votan para la predicción final.
# MAGIC
# MAGIC > **"La sabiduría de las multitudes"** - Muchos árboles diversos superan a un árbol individual.
# MAGIC
# MAGIC ### 🌲🌳🌴 Arquitectura
# MAGIC
# MAGIC ```
# MAGIC               Training Data
# MAGIC                     |
# MAGIC         ┌───────────┼───────────┐
# MAGIC         │           │           │
# MAGIC    Bootstrap     Bootstrap   Bootstrap
# MAGIC    Sample 1      Sample 2    Sample N
# MAGIC         │           │           │
# MAGIC    [Árbol 1]   [Árbol 2]   [Árbol N]
# MAGIC         │           │           │
# MAGIC         └───────────┼───────────┘
# MAGIC                     |
# MAGIC              Votación Mayoritaria
# MAGIC              (Clasificación)
# MAGIC                     |
# MAGIC               Predicción Final
# MAGIC ```
# MAGIC
# MAGIC ### 📐 Algoritmo
# MAGIC
# MAGIC **Bagging + Random Feature Selection:**
# MAGIC
# MAGIC 1. **Bootstrap Aggregating (Bagging):**
# MAGIC    - Crear $N$ datasets mediante muestreo con reemplazo
# MAGIC    - Cada árbol se entrena con ∼63% de datos únicos
# MAGIC
# MAGIC 2. **Random Feature Selection:**
# MAGIC    - En cada división, considerar solo $\sqrt{d}$ features aleatorias (clasificación)
# MAGIC    - Aumenta diversidad entre árboles
# MAGIC
# MAGIC 3. **Votación:**
# MAGIC    - **Clasificación**: Voto mayoritario (hard voting) o promedio de probabilidades (soft voting)
# MAGIC    - **Regresión**: Promedio de predicciones
# MAGIC
# MAGIC $$\hat{y} = \text{mode}\{h_1(\mathbf{x}), h_2(\mathbf{x}), ..., h_N(\mathbf{x})\}$$
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC * **Reduce overfitting**: Comparado con árbol individual
# MAGIC * **Robusto**: Maneja outliers y ruido
# MAGIC * **Feature importance**: Identifica variables más relevantes
# MAGIC * **Out-of-Bag (OOB) error**: Validación automática sin necesidad de validation set
# MAGIC * **Parallelizable**: Árboles se entrenan independientemente
# MAGIC * **Funciona bien "out of the box"**: Requiere poco tuning
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC * **Menos interpretable**: No puedes visualizar 100 árboles
# MAGIC * **Más lento**: En inferencia (debe consultar todos los árboles)
# MAGIC * **Mayor memoria**: Almacena múltiples modelos
# MAGIC * **No extrapola**: Hereda limitación de árboles base
# MAGIC
# MAGIC ### 💼 Casos de Uso Empresariales
# MAGIC
# MAGIC | Industria | Aplicación | Por qué Random Forest |
# MAGIC |-----------|------------|----------------------|
# MAGIC | **Finanzas** | Detección de fraude | Robusto, alta precisión |
# MAGIC | **E-commerce** | Sistema de recomendación | Maneja features heterogéneas |
# MAGIC | **Telecomunicaciones** | Predicción de churn | Feature importance para estrategias |
# MAGIC | **Seguros** | Pricing de pólizas | Captura interacciones complejas |
# MAGIC | **Energía** | Predicción de demanda | Robusto a valores atípicos |
# MAGIC
# MAGIC ### 🔧 Hiperparámetros Clave
# MAGIC
# MAGIC * **n_estimators**: Número de árboles (100-500 típico)
# MAGIC * **max_depth**: Profundidad de cada árbol
# MAGIC * **max_features**: Features por división ('sqrt', 'log2', o número)
# MAGIC * **min_samples_split**: Mínimo para dividir nodo
# MAGIC * **bootstrap**: True para bagging
# MAGIC * **oob_score**: True para usar OOB como validación
# MAGIC
# MAGIC ### 📊 Feature Importance
# MAGIC
# MAGIC **Cálculo:**
# MAGIC
# MAGIC Importancia de feature $f$ = Promedio de reducción de impureza cuando $f$ se usa en divisiones.
# MAGIC
# MAGIC **Ejemplo - Predicción de Conversión:**
# MAGIC
# MAGIC ```
# MAGIC 1. Tiempo en sitio:        0.28  ████████████████████████████
# MAGIC 2. Páginas visitadas:      0.22  ██████████████████████
# MAGIC 3. Fuente de tráfico:      0.15  ███████████████
# MAGIC 4. Hora del día:           0.12  ████████████
# MAGIC 5. Dispositivo:            0.10  ██████████
# MAGIC 6. Historial de compras:   0.08  ████████
# MAGIC 7. Ubicación:              0.05  █████
# MAGIC ```
# MAGIC
# MAGIC **Insight:** Optimizar experiencia de navegación (tiempo + páginas) tiene mayor impacto que cambiar canales de adquisición.

# COMMAND ----------

# DBTITLE 1,4. Gradient Boosting
# MAGIC %md
# MAGIC ## 4. Gradient Boosting (XGBoost, LightGBM, CatBoost)
# MAGIC
# MAGIC ### 🎯 Concepto
# MAGIC
# MAGIC **Gradient Boosting** construye árboles **secuencialmente**, donde cada nuevo árbol corrige errores del anterior.
# MAGIC
# MAGIC > **"Aprende de los errores"** - Cada modelo mejora donde el anterior falló.
# MAGIC
# MAGIC ### 📐 Algoritmo
# MAGIC
# MAGIC **Boosting vs Bagging:**
# MAGIC
# MAGIC | Característica | Bagging (Random Forest) | Boosting (XGBoost) |
# MAGIC |----------------|------------------------|--------------------|
# MAGIC | Construcción | Paralela | Secuencial |
# MAGIC | Árboles | Independientes | Dependientes |
# MAGIC | Peso de árboles | Igual | Ponderado |
# MAGIC | Objetivo | Reducir varianza | Reducir sesgo |
# MAGIC
# MAGIC **Proceso de Boosting:**
# MAGIC
# MAGIC 1. Entrenar modelo base $h_1$ con datos originales
# MAGIC 2. Calcular residuos/errores $r_1 = y - h_1(x)$
# MAGIC 3. Entrenar $h_2$ para predecir $r_1$
# MAGIC 4. Predicción combinada: $F_2(x) = h_1(x) + \alpha h_2(x)$
# MAGIC 5. Repetir $N$ veces
# MAGIC
# MAGIC **Fórmula final:**
# MAGIC
# MAGIC $$F(x) = \sum_{m=1}^{M} \alpha_m h_m(x)$$
# MAGIC
# MAGIC Donde $\alpha_m$ es el peso/learning rate del modelo $m$.
# MAGIC
# MAGIC ### 🚀 Implementaciones Populares
# MAGIC
# MAGIC #### **XGBoost** (eXtreme Gradient Boosting)
# MAGIC
# MAGIC * **Ventaja**: Rápido, parallelizable, regularización L1/L2
# MAGIC * **Innovación**: Aproximación de segundo orden (Hessian)
# MAGIC * **Usado por**: Ganadores de Kaggle (2015-2017)
# MAGIC
# MAGIC #### **LightGBM** (Microsoft)
# MAGIC
# MAGIC * **Ventaja**: **Más rápido** que XGBoost, usa menos memoria
# MAGIC * **Innovación**: Leaf-wise tree growth (vs level-wise)
# MAGIC * **Mejor para**: Datasets grandes (millones de filas)
# MAGIC
# MAGIC #### **CatBoost** (Yandex)
# MAGIC
# MAGIC * **Ventaja**: Maneja **categóricas nativas** (sin one-hot encoding)
# MAGIC * **Innovación**: Ordered boosting (previene overfitting)
# MAGIC * **Mejor para**: Datos con muchas variables categóricas
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC * **State-of-the-art accuracy**: Mejor performance en muchos problemas
# MAGIC * **Robusto a outliers**: Menos sensible que modelos lineales
# MAGIC * **Maneja datos faltantes**: Built-in
# MAGIC * **Feature importance**: Similar a Random Forest
# MAGIC * **Flexible**: Funciones de pérdida custom
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC * **Overfitting**: Si no se controlan hiperparámetros
# MAGIC * **Tuning complejo**: Muchos hiperparámetros
# MAGIC * **Secuencial**: No parallelizable como Random Forest
# MAGIC * **Sensible a ruido**: Puede aprender patrones espurios
# MAGIC * **Menos interpretable**: Más complejo que árbol simple
# MAGIC
# MAGIC ### 💼 Casos de Uso Empresariales
# MAGIC
# MAGIC | Industria | Aplicación | Por qué Gradient Boosting |
# MAGIC |-----------|------------|----------------------------|
# MAGIC | **Fintech** | Credit scoring | Máxima accuracy, alta competencia |
# MAGIC | **E-commerce** | Ranking de productos | Captura interacciones complejas |
# MAGIC | **Ad Tech** | CTR prediction | Performance es crítico (revenue) |
# MAGIC | **Seguros** | Pricing dinámico | Mejor estimación de riesgo |
# MAGIC | **Retail** | Demand forecasting | Supera a modelos tradicionales |
# MAGIC
# MAGIC ### 🔧 Hiperparámetros Clave
# MAGIC
# MAGIC **Prevención de Overfitting:**
# MAGIC * **n_estimators**: Número de árboles (100-1000)
# MAGIC * **learning_rate** ($\alpha$): Peso de cada árbol (0.01-0.3)
# MAGIC * **max_depth**: Profundidad de árboles (3-10, más bajo que RF)
# MAGIC * **subsample**: Fracción de datos por árbol (0.5-1.0)
# MAGIC * **colsample_bytree**: Fracción de features (0.5-1.0)
# MAGIC
# MAGIC **Regularización:**
# MAGIC * **reg_alpha**: L1 regularization
# MAGIC * **reg_lambda**: L2 regularization
# MAGIC * **min_child_weight**: Mínimo peso en hoja
# MAGIC
# MAGIC **Trade-off clásico:**
# MAGIC * Más árboles + learning rate bajo = Mejor generalización, más lento
# MAGIC * Menos árboles + learning rate alto = Más rápido, riesgo de underfitting
# MAGIC
# MAGIC ### 📊 Comparación XGBoost vs LightGBM vs CatBoost
# MAGIC
# MAGIC | Característica | XGBoost | LightGBM | CatBoost |
# MAGIC |----------------|---------|----------|----------|
# MAGIC | **Velocidad** | Rápido | **Más rápido** | Medio |
# MAGIC | **Memoria** | Media | **Baja** | Alta |
# MAGIC | **Accuracy** | Alto | Alto | **Muy alto** |
# MAGIC | **Categorías** | One-hot | One-hot | **Nativas** |
# MAGIC | **Overfitting** | Medio | Alto | **Bajo** |
# MAGIC | **Tuning** | Complejo | Complejo | **Simple** |
# MAGIC
# MAGIC **Recomendación:**
# MAGIC * **Datos grandes**: LightGBM
# MAGIC * **Muchas categóricas**: CatBoost  
# MAGIC * **Balance general**: XGBoost

# COMMAND ----------

# DBTITLE 1,5. Support Vector Machines (SVM)
# MAGIC %md
# MAGIC ## 5. Support Vector Machines (SVM)
# MAGIC
# MAGIC ### 🎯 Concepto
# MAGIC
# MAGIC **SVM** encuentra el **hiperplano óptimo** que maximiza la separación (margen) entre clases.
# MAGIC
# MAGIC ### 📐 Formulación Matemática
# MAGIC
# MAGIC **Para clasificación binaria linealmente separable:**
# MAGIC
# MAGIC Encontrar $\mathbf{w}$ y $b$ que maximicen el margen:
# MAGIC
# MAGIC $$\max_{\mathbf{w}, b} \frac{2}{||\mathbf{w}||} \quad \text{sujeto a } \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \forall i$$
# MAGIC
# MAGIC Equivalente a minimizar:
# MAGIC
# MAGIC $$\min_{\mathbf{w}, b} \frac{1}{2}||\mathbf{w}||^2$$
# MAGIC
# MAGIC **SVM con margen suave (soft margin):**
# MAGIC
# MAGIC $$\min_{\mathbf{w}, b, \xi} \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i=1}^{n}\xi_i$$
# MAGIC
# MAGIC Donde:
# MAGIC * $\xi_i$ son variables de holgura (permiten errores)
# MAGIC * $C$ es el parámetro de regularización
# MAGIC
# MAGIC ### 🎉 Kernel Trick
# MAGIC
# MAGIC **Para datos no linealmente separables**, se mapean a espacio dimensional más alto usando **kernels**:
# MAGIC
# MAGIC **Kernels comunes:**
# MAGIC
# MAGIC 1. **Lineal**: $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T\mathbf{x}_j$
# MAGIC
# MAGIC 2. **Polinomial**: $K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T\mathbf{x}_j + r)^d$
# MAGIC
# MAGIC 3. **RBF (Radial Basis Function)**:
# MAGIC    $$K(\mathbf{x}_i, \mathbf{x}_j) = e^{-\gamma||\mathbf{x}_i - \mathbf{x}_j||^2}$$
# MAGIC    * Más popular
# MAGIC    * Puede aproximar cualquier frontera de decisión
# MAGIC
# MAGIC 4. **Sigmoid**: $K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma \mathbf{x}_i^T\mathbf{x}_j + r)$
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC * **Efectivo en espacios de alta dimensión**: Funciona bien cuando $d > n$
# MAGIC * **Memory efficient**: Solo almacena support vectors (subconjunto de datos)
# MAGIC * **Versatilidad**: Diferentes kernels para diferentes problemas
# MAGIC * **Robusto a overfitting**: Especialmente en alta dimensión
# MAGIC * **Teóricamente sólido**: Basado en teoría de aprendizaje estadístico
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC * **Lento en datasets grandes**: Complejidad $O(n^2)$ a $O(n^3)$
# MAGIC * **Sensible a escala**: Requiere normalización de features
# MAGIC * **Elección de kernel**: No trivial, requiere experimentación
# MAGIC * **No da probabilidades directamente**: Necesita calibración (Platt scaling)
# MAGIC * **Interpretación limitada**: Especialmente con kernels no lineales
# MAGIC
# MAGIC ### 💼 Casos de Uso Empresariales
# MAGIC
# MAGIC | Industria | Aplicación | Por qué SVM |
# MAGIC |-----------|------------|-------------|
# MAGIC | **Bioinformática** | Clasificación de proteínas | Alta dimensión, pocas muestras |
# MAGIC | **Reconocimiento de texto** | OCR, clasificación de documentos | Funciona bien con features TF-IDF |
# MAGIC | **Visión por computadora** | Detección facial | Kernel RBF captura patrones complejos |
# MAGIC | **Finanzas** | Trading algorithmico | Features numéricas, alta dimensión |
# MAGIC | **Telecomunicaciones** | Detección de intrusos | Boundaries complejas entre normal/anomalía |
# MAGIC
# MAGIC ### 🔧 Hiperparámetros Clave
# MAGIC
# MAGIC * **C**: Parámetro de regularización
# MAGIC   - **C alto**: Margen estrecho, menos errores en train (overfitting)
# MAGIC   - **C bajo**: Margen amplio, más errores permitidos (underfitting)
# MAGIC
# MAGIC * **kernel**: Tipo de kernel ('linear', 'rbf', 'poly', 'sigmoid')
# MAGIC
# MAGIC * **gamma** (para RBF/poly/sigmoid):
# MAGIC   - **Gamma alto**: Influencia local, modelos complejos (overfitting)
# MAGIC   - **Gamma bajo**: Influencia global, modelos simples
# MAGIC
# MAGIC * **degree** (para poly): Grado del polinomio
# MAGIC
# MAGIC ### 📊 Selección de Kernel
# MAGIC
# MAGIC **Reglas generales:**
# MAGIC
# MAGIC 1. **Empezar con kernel lineal** si:
# MAGIC    - Muchas features (text classification: $d$ > 10,000)
# MAGIC    - Datos linealmente separables
# MAGIC    - Dataset muy grande (velocidad)
# MAGIC
# MAGIC 2. **RBF (default)** si:
# MAGIC    - No estás seguro
# MAGIC    - Features numéricas
# MAGIC    - Relaciones no lineales
# MAGIC
# MAGIC 3. **Polinomial** si:
# MAGIC    - Interacciones polinomiales conocidas del dominio
# MAGIC    - Ej: área = largo × ancho
# MAGIC
# MAGIC ### ⚠️ Importante: Escalamiento
# MAGIC
# MAGIC **SVM es extremadamente sensible a la escala de features.**
# MAGIC
# MAGIC **Ejemplo:**
# MAGIC
# MAGIC ```python
# MAGIC # Antes de SVM, SIEMPRE escalar:
# MAGIC from sklearn.preprocessing import StandardScaler
# MAGIC scaler = StandardScaler()
# MAGIC X_train_scaled = scaler.fit_transform(X_train)
# MAGIC X_test_scaled = scaler.transform(X_test)
# MAGIC ```
# MAGIC
# MAGIC **Sin escalar**: Feature con rango [0, 1000] dominará sobre feature con rango [0, 1].

# COMMAND ----------

# DBTITLE 1,6. Naive Bayes
# MAGIC %md
# MAGIC ## 6. Naive Bayes
# MAGIC
# MAGIC ### 🎯 Concepto
# MAGIC
# MAGIC **Naive Bayes** aplica el **Teorema de Bayes** con la suposición "naive" (ingenua) de que todas las features son **condicionalmente independientes** dado la clase.
# MAGIC
# MAGIC ### 📐 Formulación Matemática
# MAGIC
# MAGIC **Teorema de Bayes:**
# MAGIC
# MAGIC $$P(y|\mathbf{x}) = \frac{P(\mathbf{x}|y)P(y)}{P(\mathbf{x})}$$
# MAGIC
# MAGIC Donde:
# MAGIC * $P(y|\mathbf{x})$: Probabilidad posterior (lo que queremos)
# MAGIC * $P(\mathbf{x}|y)$: Likelihood (verosimilitud)
# MAGIC * $P(y)$: Probabilidad a priori
# MAGIC * $P(\mathbf{x})$: Evidencia (constante para todas las clases)
# MAGIC
# MAGIC **Suposición Naive:**
# MAGIC
# MAGIC $$P(\mathbf{x}|y) = P(x_1, x_2, ..., x_d|y) = \prod_{i=1}^{d} P(x_i|y)$$
# MAGIC
# MAGIC **Clasificación:**
# MAGIC
# MAGIC $$\hat{y} = \arg\max_y P(y) \prod_{i=1}^{d} P(x_i|y)$$
# MAGIC
# MAGIC ### 📚 Tipos de Naive Bayes
# MAGIC
# MAGIC #### 1️⃣ **Gaussian Naive Bayes**
# MAGIC
# MAGIC * **Uso**: Features continuas que siguen distribución normal
# MAGIC * **Likelihood**:
# MAGIC
# MAGIC $$P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} e^{-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}}$$
# MAGIC
# MAGIC #### 2️⃣ **Multinomial Naive Bayes**
# MAGIC
# MAGIC * **Uso**: Conteo de features (text classification, bolsa de palabras)
# MAGIC * **Likelihood**:
# MAGIC
# MAGIC $$P(\mathbf{x}|y) = \frac{(\sum_i x_i)!}{\prod_i x_i!} \prod_i p_{iy}^{x_i}$$
# MAGIC
# MAGIC * **Ejemplo**: Clasificación de emails (spam/no spam) basado en frecuencia de palabras
# MAGIC
# MAGIC #### 3️⃣ **Bernoulli Naive Bayes**
# MAGIC
# MAGIC * **Uso**: Features binarias (presencia/ausencia)
# MAGIC * **Likelihood**:
# MAGIC
# MAGIC $$P(x_i|y) = p_{iy}^{x_i}(1-p_{iy})^{(1-x_i)}$$
# MAGIC
# MAGIC * **Ejemplo**: Documento contiene palabra "oferta": Sí/No
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC * **Extremadamente rápido**: Entrenamiento e inferencia en $O(nd)$
# MAGIC * **Funciona con pocos datos**: No requiere muchas muestras
# MAGIC * **Escalable**: Maneja millones de features (NLP)
# MAGIC * **Multiclase nativo**: No necesita One-vs-Rest
# MAGIC * **Probabilidades calibradas**: Buenas estimaciones de confianza
# MAGIC * **Robusto a features irrelevantes**: La suposición naive ayuda
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC * **Suposición naive poco realista**: Features raramente son independientes
# MAGIC * **Zero probability problem**: Si feature nunca aparece con clase en train
# MAGIC * **No captura interacciones**: Por diseño
# MAGIC * **Sesgo**: Alto bias, bajo variance
# MAGIC
# MAGIC ### 💼 Casos de Uso Empresariales
# MAGIC
# MAGIC | Industria | Aplicación | Por qué Naive Bayes |
# MAGIC |-----------|------------|---------------------|
# MAGIC | **Email** | Filtro de spam | Rápido, alta dimensión (palabras) |
# MAGIC | **E-commerce** | Categorización de productos | Texto de descripción |
# MAGIC | **Redes sociales** | Análisis de sentimientos | Velocidad para millones de posts |
# MAGIC | **Atención al cliente** | Enrutamiento de tickets | Clasificación en tiempo real |
# MAGIC | **Medios** | Clasificación de noticias | Features de texto (bag-of-words) |
# MAGIC
# MAGIC ### 🔧 Hiperparámetros Clave
# MAGIC
# MAGIC * **alpha** (Laplace smoothing): Evita probabilidades cero
# MAGIC   - $\alpha = 0$: No smoothing
# MAGIC   - $\alpha = 1$: Laplace smoothing (default)
# MAGIC   - $\alpha > 1$: Mayor smoothing
# MAGIC
# MAGIC * **fit_prior**: Usar frecuencias de clases o asumir uniforme
# MAGIC
# MAGIC ### 📊 Ejemplo de Negocio
# MAGIC
# MAGIC **Clasificación de Emails de Clientes:**
# MAGIC
# MAGIC **Entrenamiento:**
# MAGIC
# MAGIC | Email | "urgente" | "gracias" | "problema" | Clase |
# MAGIC |-------|-----------|-----------|------------|-------|
# MAGIC | 1 | 2 | 0 | 3 | Queja |
# MAGIC | 2 | 0 | 5 | 0 | Agradecimiento |
# MAGIC | 3 | 1 | 0 | 2 | Queja |
# MAGIC | 4 | 0 | 3 | 0 | Agradecimiento |
# MAGIC
# MAGIC **Nuevo email**: "urgente problema problema"
# MAGIC
# MAGIC **Cálculo:**
# MAGIC
# MAGIC $$P(\text{Queja}|\text{email}) \propto P(\text{Queja}) \times P(\text{urgente}|\text{Queja}) \times P(\text{problema}|\text{Queja})^2$$
# MAGIC
# MAGIC $$P(\text{Agradecimiento}|\text{email}) \propto P(\text{Agradecimiento}) \times P(\text{urgente}|\text{Agradecimiento}) \times P(\text{problema}|\text{Agradecimiento})^2$$
# MAGIC
# MAGIC **Predicción**: Queja (mayor probabilidad)
# MAGIC
# MAGIC ### ⚠️ Cuándo NO usar Naive Bayes
# MAGIC
# MAGIC * Features altamente correlacionadas (edad/ingresos, precio/descuento)
# MAGIC * Relaciones complejas entre features
# MAGIC * Necesitas máxima accuracy (sacrifica velocidad)

# COMMAND ----------

# DBTITLE 1,7. K-Nearest Neighbors (KNN)
# MAGIC %md
# MAGIC ## 7. K-Nearest Neighbors (KNN)
# MAGIC
# MAGIC ### 🎯 Concepto
# MAGIC
# MAGIC **KNN** es un algoritmo **instance-based** (lazy learning): no aprende modelo explícito, sino que **memoriza** datos de entrenamiento y clasifica basado en los $k$ vecinos más cercanos.
# MAGIC
# MAGIC > **"Dime con quién andas y te diré quién eres."**
# MAGIC
# MAGIC ### 📐 Algoritmo
# MAGIC
# MAGIC **Proceso de clasificación:**
# MAGIC
# MAGIC 1. Calcular distancia entre punto nuevo $\mathbf{x}$ y todos los puntos de entrenamiento
# MAGIC 2. Seleccionar los $k$ puntos más cercanos
# MAGIC 3. **Votación mayoritaria**: Asignar clase más frecuente entre los $k$ vecinos
# MAGIC
# MAGIC $$\hat{y} = \text{mode}\{y_i : \mathbf{x}_i \in \mathcal{N}_k(\mathbf{x})\}$$
# MAGIC
# MAGIC Donde $\mathcal{N}_k(\mathbf{x})$ son los $k$ vecinos más cercanos.
# MAGIC
# MAGIC ### 📏 Métricas de Distancia
# MAGIC
# MAGIC #### 1️⃣ **Distancia Euclidiana** (default)
# MAGIC
# MAGIC $$d(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum_{d=1}^{D}(x_{id} - x_{jd})^2}$$
# MAGIC
# MAGIC * **Uso**: Features numéricas continuas
# MAGIC
# MAGIC #### 2️⃣ **Distancia Manhattan**
# MAGIC
# MAGIC $$d(\mathbf{x}_i, \mathbf{x}_j) = \sum_{d=1}^{D}|x_{id} - x_{jd}|$$
# MAGIC
# MAGIC * **Uso**: Features en grilla (coordenadas de ciudad)
# MAGIC
# MAGIC #### 3️⃣ **Distancia Minkowski** (generalización)
# MAGIC
# MAGIC $$d(\mathbf{x}_i, \mathbf{x}_j) = \left(\sum_{d=1}^{D}|x_{id} - x_{jd}|^p\right)^{1/p}$$
# MAGIC
# MAGIC * $p=1$: Manhattan
# MAGIC * $p=2$: Euclidiana
# MAGIC * $p=\infty$: Chebyshev
# MAGIC
# MAGIC #### 4️⃣ **Distancia de Coseno**
# MAGIC
# MAGIC $$d(\mathbf{x}_i, \mathbf{x}_j) = 1 - \frac{\mathbf{x}_i \cdot \mathbf{x}_j}{||\mathbf{x}_i|| \cdot ||\mathbf{x}_j||}$$
# MAGIC
# MAGIC * **Uso**: Text classification, sistemas de recomendación
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC * **Simple**: Fácil de entender e implementar
# MAGIC * **No requiere entrenamiento**: Lazy learning
# MAGIC * **Adapta a nuevos datos**: Solo agregar al conjunto
# MAGIC * **Multiclase natural**: Sin modificaciones
# MAGIC * **No asume distribución**: Non-parametric
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC * **Lento en inferencia**: Debe calcular distancia a TODOS los puntos ($O(n)$)
# MAGIC * **Curse of dimensionality**: Falla en alta dimensión (>20 features)
# MAGIC * **Sensible a escala**: Requiere normalización
# MAGIC * **Sensible a features irrelevantes**: Todas contribuyen a distancia
# MAGIC * **Memory intensive**: Debe almacenar todo el training set
# MAGIC * **Sensible a desbalance**: Clases mayoritarias dominan votación
# MAGIC
# MAGIC ### 💼 Casos de Uso Empresariales
# MAGIC
# MAGIC | Industria | Aplicación | Por qué KNN |
# MAGIC |-----------|------------|-------------|
# MAGIC | **E-commerce** | Sistema de recomendación | "Usuarios similares compraron..." |
# MAGIC | **Retail** | Detección de anomalías | Transacciones atípicas = lejos de vecinos |
# MAGIC | **Biométrica** | Reconocimiento facial | Matching contra base de datos |
# MAGIC | **Real Estate** | Valoración de propiedades | Basado en propiedades similares |
# MAGIC | **Salud** | Diagnóstico | Pacientes similares tienen misma condición |
# MAGIC
# MAGIC ### 🔧 Hiperparámetros Clave
# MAGIC
# MAGIC #### **k (número de vecinos)**
# MAGIC
# MAGIC * **k pequeño** (k=1, k=3):
# MAGIC   - Más flexible, captura patterns locales
# MAGIC   - **Alto variance**, sensible a ruido
# MAGIC   - Boundaries complejas
# MAGIC
# MAGIC * **k grande** (k=50, k=100):
# MAGIC   - Más suave, promedia más
# MAGIC   - **Alto bias**, puede perder detalles
# MAGIC   - Boundaries simples
# MAGIC
# MAGIC **Regla general**: $k = \sqrt{n}$ o cross-validation
# MAGIC
# MAGIC #### **weights**
# MAGIC
# MAGIC * **'uniform'**: Todos los vecinos votan igual
# MAGIC * **'distance'**: Vecinos más cercanos tienen más peso
# MAGIC
# MAGIC $$w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}$$
# MAGIC
# MAGIC ### 📊 Elección de k
# MAGIC
# MAGIC **Visualización:**
# MAGIC
# MAGIC ```
# MAGIC    Error  |
# MAGIC           |     Test Error
# MAGIC    High   |       /\
# MAGIC           |      /  \
# MAGIC           |     /    \\_____
# MAGIC           |    /
# MAGIC           |___/__________________  Train Error
# MAGIC    Low    |
# MAGIC           |
# MAGIC           +------------------------> k
# MAGIC               1   5   10  20  50
# MAGIC ```
# MAGIC
# MAGIC **Patrón:**
# MAGIC * k=1: Overfitting (train error = 0)
# MAGIC * k → n: Underfitting (todos predicen clase mayoritaria)
# MAGIC * **Sweet spot**: Validation error mínimo
# MAGIC
# MAGIC ### ⚠️ Importante: Escalamiento
# MAGIC
# MAGIC **KNN es MUY sensible a la escala.**
# MAGIC
# MAGIC **Ejemplo sin escalar:**
# MAGIC
# MAGIC ```
# MAGIC Feature 1: Edad (20-80)       → Contribución: 60
# MAGIC Feature 2: Salario (20K-200K) → Contribución: 180,000
# MAGIC ```
# MAGIC
# MAGIC ¡Salario domina completamente!
# MAGIC
# MAGIC **Solución:**
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.preprocessing import StandardScaler
# MAGIC scaler = StandardScaler()
# MAGIC X_scaled = scaler.fit_transform(X)
# MAGIC ```
# MAGIC
# MAGIC ### 🚀 Optimizaciones
# MAGIC
# MAGIC **Para datasets grandes:**
# MAGIC
# MAGIC * **KD-Tree / Ball-Tree**: Estructuras de datos para búsqueda rápida ($O(\log n)$)
# MAGIC * **Approximate Nearest Neighbors**: Librerías como Annoy, FAISS
# MAGIC * **Dimensionality Reduction**: PCA antes de KNN

# COMMAND ----------

# DBTITLE 1,8. Redes Neuronales
# MAGIC %md
# MAGIC ## 8. Redes Neuronales (Neural Networks)
# MAGIC
# MAGIC ### 🎯 Concepto
# MAGIC
# MAGIC **Redes Neuronales** son modelos inspirados en el cerebro humano, compuestos por capas de neuronas artificiales interconectadas que aprenden representaciones jerárquicas de features.
# MAGIC
# MAGIC ### 🧠 Arquitectura
# MAGIC
# MAGIC **Perceptron Multicapa (MLP):**
# MAGIC
# MAGIC ```
# MAGIC Input Layer    Hidden Layer 1   Hidden Layer 2   Output Layer
# MAGIC
# MAGIC    x1 ●────────●───────────●───────────●
# MAGIC                 │             │             │
# MAGIC    x2 ●────────●───────────●───────────●  y1 (Clase A)
# MAGIC                 │             │             │
# MAGIC    x3 ●────────●───────────●───────────●  y2 (Clase B)
# MAGIC                 │             │
# MAGIC    x4 ●────────●───────────●
# MAGIC ```
# MAGIC
# MAGIC ### 📐 Formulación Matemática
# MAGIC
# MAGIC **Forward Propagation:**
# MAGIC
# MAGIC Para cada capa $l$:
# MAGIC
# MAGIC $$\mathbf{z}^{[l]} = \mathbf{W}^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$
# MAGIC
# MAGIC $$\mathbf{a}^{[l]} = g(\mathbf{z}^{[l]})$$
# MAGIC
# MAGIC Donde:
# MAGIC * $\mathbf{W}^{[l]}$ son los pesos de la capa $l$
# MAGIC * $\mathbf{b}^{[l]}$ son los bias
# MAGIC * $g$ es la **función de activación**
# MAGIC * $\mathbf{a}^{[l]}$ son las activaciones (outputs)
# MAGIC
# MAGIC **Funciones de Activación:**
# MAGIC
# MAGIC 1. **ReLU** (Rectified Linear Unit):
# MAGIC    $$g(z) = \max(0, z)$$
# MAGIC    * Más popular para capas ocultas
# MAGIC    * Rápida, no sufre vanishing gradient
# MAGIC
# MAGIC 2. **Sigmoid**:
# MAGIC    $$g(z) = \frac{1}{1 + e^{-z}}$$
# MAGIC    * Output layer para clasificación binaria
# MAGIC    * Output $\in [0, 1]$ (probabilidad)
# MAGIC
# MAGIC 3. **Softmax** (multiclase):
# MAGIC    $$g(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$
# MAGIC    * Output layer para clasificación multiclase
# MAGIC    * Outputs suman 1 (distribuciones de probabilidad)
# MAGIC
# MAGIC 4. **Tanh**:
# MAGIC    $$g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
# MAGIC    * Output $\in [-1, 1]$
# MAGIC    * Centrada en cero
# MAGIC
# MAGIC **Backpropagation:**
# MAGIC
# MAGIC Algoritmo para calcular gradientes y actualizar pesos:
# MAGIC
# MAGIC $$\mathbf{W}^{[l]} := \mathbf{W}^{[l]} - \alpha \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$$
# MAGIC
# MAGIC Donde $\alpha$ es el learning rate.
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC * **Feature learning automático**: Aprende representaciones jerárquicas
# MAGIC * **Relaciones no lineales**: Captura patrones complejos
# MAGIC * **Versatilidad**: Funciona en imágenes, texto, audio, tabular
# MAGIC * **State-of-the-art**: Mejor performance en muchos dominios
# MAGIC * **Transfer learning**: Pre-entrenamiento reutilizable
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC * **Black box**: Difícil de interpretar
# MAGIC * **Requiere muchos datos**: Miles/millones de muestras
# MAGIC * **Caro computacionalmente**: GPU recomendada
# MAGIC * **Muchos hiperparámetros**: Arquitectura, learning rate, regularización...
# MAGIC * **Sensible a inicialización**: Resultados pueden variar
# MAGIC * **Overfitting**: Fácil sin regularización
# MAGIC
# MAGIC ### 💼 Casos de Uso Empresariales
# MAGIC
# MAGIC | Industria | Aplicación | Por qué Neural Networks |
# MAGIC |-----------|------------|---------------------------|
# MAGIC | **Visión** | Reconocimiento facial, OCR | CNNs son state-of-the-art |
# MAGIC | **NLP** | Chatbots, traducción | Transformers (BERT, GPT) |
# MAGIC | **Finanzas** | Trading algorithmico | Captura patterns temporales (LSTM) |
# MAGIC | **E-commerce** | Recomendaciones | Deep learning supera collaborative filtering |
# MAGIC | **Salud** | Detección de cáncer en imágenes | Performance crítico |
# MAGIC | **Publicidad** | CTR prediction | Google/Facebook usan DL |
# MAGIC
# MAGIC ### 🔧 Hiperparámetros Clave
# MAGIC
# MAGIC **Arquitectura:**
# MAGIC * **hidden_layer_sizes**: Tupla con neuronas por capa
# MAGIC   - Ej: (100, 50) = 2 capas ocultas, 100 y 50 neuronas
# MAGIC * **activation**: 'relu', 'tanh', 'logistic'
# MAGIC
# MAGIC **Optimización:**
# MAGIC * **learning_rate_init**: Tasa de aprendizaje inicial (0.001-0.01)
# MAGIC * **solver**: 'adam' (default, adaptativo), 'sgd', 'lbfgs'
# MAGIC * **batch_size**: Muestras por actualización (32, 64, 128, 256)
# MAGIC * **max_iter**: Número máximo de épocas
# MAGIC
# MAGIC **Regularización:**
# MAGIC * **alpha**: L2 regularization (0.0001 default)
# MAGIC * **early_stopping**: Detener si validation no mejora
# MAGIC * **validation_fraction**: % datos para early stopping (0.1)
# MAGIC
# MAGIC ### 🚀 Tipos de Redes para Clasificación
# MAGIC
# MAGIC #### **1. MLP (Multilayer Perceptron)**
# MAGIC
# MAGIC * **Uso**: Datos tabulares, features numéricas/categóricas
# MAGIC * **Ejemplo**: Predicción de churn, credit scoring
# MAGIC
# MAGIC #### **2. CNN (Convolutional Neural Network)**
# MAGIC
# MAGIC * **Uso**: Imágenes, señales 1D
# MAGIC * **Ejemplo**: Clasificación de productos por foto, detección de defectos
# MAGIC
# MAGIC #### **3. RNN/LSTM (Recurrent Neural Networks)**
# MAGIC
# MAGIC * **Uso**: Secuencias temporales, texto
# MAGIC * **Ejemplo**: Predicción de series de tiempo, análisis de sentimientos
# MAGIC
# MAGIC #### **4. Transformers**
# MAGIC
# MAGIC * **Uso**: NLP avanzado
# MAGIC * **Ejemplo**: Clasificación de tickets de soporte, extracción de entidades
# MAGIC
# MAGIC ### 📊 Reglas de Diseño
# MAGIC
# MAGIC **Número de capas ocultas:**
# MAGIC
# MAGIC * **1 capa**: Problemas simples, linealmente separables
# MAGIC * **2-3 capas**: Mayoría de problemas tabulares
# MAGIC * **4+ capas**: Imágenes, audio, NLP (deep learning)
# MAGIC
# MAGIC **Neuronas por capa:**
# MAGIC
# MAGIC * Empezar con: Entre input size y output size
# MAGIC * Ejemplo: Input=100, Output=5 → Hidden=(64, 32)
# MAGIC
# MAGIC **Forma de la red:**
# MAGIC
# MAGIC * **Pyramid**: Cada capa más pequeña (100 → 64 → 32 → 10)
# MAGIC * **Constant**: Todas las capas iguales (64 → 64 → 64)
# MAGIC * **Hourglass**: Compresión y expansión (100 → 32 → 100)
# MAGIC
# MAGIC ### ⚠️ Cuando NO usar Neural Networks
# MAGIC
# MAGIC * Dataset pequeño (<1000 muestras) → Usar tree-based
# MAGIC * Interpretabilidad crítica → Usar regresión logística/árboles
# MAGIC * Recursos limitados → Usar modelos más simples
# MAGIC * Prototipado rápido → Empezar con Random Forest/XGBoost

# COMMAND ----------

# DBTITLE 1,9. Comparación y Selección de Algoritmos
# MAGIC %md
# MAGIC ## 9. Comparación y Selección de Algoritmos
# MAGIC
# MAGIC ### 📊 Tabla Comparativa General
# MAGIC
# MAGIC | Algoritmo | Interpretabilidad | Velocidad Train | Velocidad Predict | Accuracy | Escalabilidad | Requiere Escalar |
# MAGIC |-----------|-------------------|-----------------|-------------------|----------|---------------|------------------|
# MAGIC | **Regresión Logística** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ |
# MAGIC | **Árbol de Decisión** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ❌ |
# MAGIC | **Random Forest** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ |
# MAGIC | **XGBoost/LightGBM** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ |
# MAGIC | **SVM** | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ✅ |
# MAGIC | **Naive Bayes** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ |
# MAGIC | **KNN** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐ | ✅ |
# MAGIC | **Neural Networks** | ⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🧐 Árbol de Decisión para Selección
# MAGIC
# MAGIC ```
# MAGIC                     ¿Qué algoritmo usar?
# MAGIC                              |
# MAGIC          ┌─────────────────────┼─────────────────────┐
# MAGIC          |                        |                        |
# MAGIC     ¿Interpretabilidad       ¿Máxima Accuracy       ¿Velocidad
# MAGIC        crítica?                  sin importar             crítica?
# MAGIC          |                    complejidad?                  |
# MAGIC          |                        |                        |
# MAGIC     ┌────┼────┐                   |                   ┌────┼────┐
# MAGIC     |         |                   |                   |         |
# MAGIC   Linear   Árbol          ┌─────┼─────┐         Pocos    Millones
# MAGIC    o no?                  |           |          datos     datos
# MAGIC     |                     |           |           |          |
# MAGIC   Lineal               Tabular   Imágenes/    Naive     Naive
# MAGIC     |                     |        NLP          Bayes     Bayes
# MAGIC  Regresión          XGBoost/      |         (Gaussian) (Multinomial)
# MAGIC  Logística           LightGBM   Neural
# MAGIC                                 Networks
# MAGIC                                  (CNN/RNN)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 👥 Selección por Tamaño de Dataset
# MAGIC
# MAGIC | Tamaño | N Muestras | Algoritmos Recomendados |
# MAGIC |---------|------------|------------------------|
# MAGIC | **Muy pequeño** | < 500 | Naive Bayes, Regresión Logística |
# MAGIC | **Pequeño** | 500 - 10K | Árboles de Decisión, SVM, KNN |
# MAGIC | **Mediano** | 10K - 100K | Random Forest, XGBoost |
# MAGIC | **Grande** | 100K - 1M | LightGBM, Neural Networks (MLP) |
# MAGIC | **Muy grande** | > 1M | LightGBM, Neural Networks (GPU), Mini-batch SGD |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💼 Selección por Tipo de Problema de Negocio
# MAGIC
# MAGIC #### **1️⃣ Finanzas (Crédito, Fraude, Riesgo)**
# MAGIC
# MAGIC **Prioridades:** Interpretabilidad, cumplimiento regulatorio, alta precisión
# MAGIC
# MAGIC **Algoritmos:**
# MAGIC 1. **Regresión Logística**: Baseline, explicable a reguladores
# MAGIC 2. **XGBoost/LightGBM**: Máxima accuracy, feature importance
# MAGIC 3. **Random Forest**: Balance entre accuracy e interpretabilidad
# MAGIC
# MAGIC **Evitar:** KNN (lento), Neural Networks (black box)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **2️⃣ Marketing (Churn, Conversión, Segmentación)**
# MAGIC
# MAGIC **Prioridades:** Probabilidades calibradas, feature importance, rapidez
# MAGIC
# MAGIC **Algoritmos:**
# MAGIC 1. **Regresión Logística**: Probabilidades directas para scoring
# MAGIC 2. **Random Forest / XGBoost**: Feature importance → Insights accionables
# MAGIC 3. **Árboles de Decisión**: Segmentos claros para campañas
# MAGIC
# MAGIC **Uso:** Feature importance identifica palancas de retención/conversión
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **3️⃣ E-commerce (Recomendaciones, Categorización)**
# MAGIC
# MAGIC **Prioridades:** Escalabilidad, velocidad de inferencia, manejo de texto
# MAGIC
# MAGIC **Algoritmos:**
# MAGIC 1. **Naive Bayes**: Categorización de productos (texto)
# MAGIC 2. **LightGBM**: Predicción de compra (features mixtas)
# MAGIC 3. **Neural Networks**: Recomendaciones (embeddings)
# MAGIC
# MAGIC **Nota:** Millones de usuarios requieren inferencia rápida
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **4️⃣ Salud (Diagnóstico, Predicción de Riesgo)**
# MAGIC
# MAGIC **Prioridades:** Alta precisión (recall), interpretabilidad, confianza
# MAGIC
# MAGIC **Algoritmos:**
# MAGIC 1. **Random Forest**: Balance accuracy/interpretabilidad
# MAGIC 2. **Regresión Logística**: Coeficientes = factores de riesgo
# MAGIC 3. **Neural Networks (CNN)**: Imágenes médicas (rayos X, MRI)
# MAGIC
# MAGIC **Importante:** Sesgo hacia recall (no perder casos positivos)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **5️⃣ Atención al Cliente (Enrutamiento, Priorización)**
# MAGIC
# MAGIC **Prioridades:** Tiempo real, manejo de texto, multiclase
# MAGIC
# MAGIC **Algoritmos:**
# MAGIC 1. **Naive Bayes**: Clasificación rápida de tickets
# MAGIC 2. **Random Forest**: Priorización (urgencia, impacto)
# MAGIC 3. **Neural Networks (BERT)**: Entendimiento avanzado de texto
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔧 Estrategia de Desarrollo
# MAGIC
# MAGIC **Pipeline recomendado:**
# MAGIC
# MAGIC 1. **Baseline Simple** (1-2 horas)
# MAGIC    - Regresión Logística o Naive Bayes
# MAGIC    - Establece métrica mínima a superar
# MAGIC
# MAGIC 2. **Modelo Robusto** (1-2 días)
# MAGIC    - Random Forest o XGBoost
# MAGIC    - Tuning básico de hiperparámetros
# MAGIC    - Cross-validation
# MAGIC
# MAGIC 3. **Optimización** (1 semana)
# MAGIC    - Ensemble de modelos
# MAGIC    - Feature engineering avanzado
# MAGIC    - Tuning exhaustivo (Grid Search, Bayesian)
# MAGIC
# MAGIC 4. **Producción** (ongoing)
# MAGIC    - Monitoreo de drift
# MAGIC    - Reentrenamiento periódico
# MAGIC    - A/B testing
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚖️ Trade-offs Clave
# MAGIC
# MAGIC | Trade-off | Cuándo priorizar A | Cuándo priorizar B |
# MAGIC |-----------|-------------------|-------------------|
# MAGIC | **Accuracy vs Interpretabilidad** | Competencias Kaggle, revenue crítico | Regulación, stakeholders no-técnicos |
# MAGIC | **Complejidad vs Simplicidad** | Performance crítico, muchos datos | Mantenimiento, pocos datos |
# MAGIC | **Velocidad Train vs Predict** | Reentrenamiento raro | Inferencia tiempo real |
# MAGIC | **Generalización vs Fit Perfecto** | Nuevos datos frecuentes | Datos estables, históricos |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ❗ Errores Comunes
# MAGIC
# MAGIC 1. **Usar Neural Networks para todo**: Overkill en datos tabulares pequeños
# MAGIC 2. **No probar baseline simple**: Siempre empezar con Regresión Logística
# MAGIC 3. **Ignorar interpretabilidad**: Stakeholders necesitan entender decisiones
# MAGIC 4. **Optimizar métrica incorrecta**: Accuracy en datos desbalanceados
# MAGIC 5. **No considerar costos operacionales**: GPU caro, inferencia lenta
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🏆 Recomendación Final
# MAGIC
# MAGIC **Si solo puedes elegir DOS algoritmos:**
# MAGIC
# MAGIC 1. **Regresión Logística**: Baseline rápido, interpretable
# MAGIC 2. **XGBoost/LightGBM**: Máxima accuracy, robusto
# MAGIC
# MAGIC **Estos dos cubren ~80% de problemas de clasificación en negocios.**

# COMMAND ----------

# DBTITLE 1,Conclusiones y Próximos Pasos
# MAGIC %md
# MAGIC ## 📍 Conclusiones y Próximos Pasos
# MAGIC
# MAGIC ### 💡 Key Takeaways
# MAGIC
# MAGIC 1. **No existe el "mejor" algoritmo universal**
# MAGIC    - El mejor algoritmo depende del problema, datos y restricciones
# MAGIC    - "No Free Lunch Theorem"
# MAGIC
# MAGIC 2. **Siempre empezar simple**
# MAGIC    - Baseline con Regresión Logística o Naive Bayes
# MAGIC    - Iterar hacia complejidad solo si es necesario
# MAGIC
# MAGIC 3. **Los datos importan más que el algoritmo**
# MAGIC    - "Garbage in, garbage out"
# MAGIC    - Feature engineering > algoritmo sofisticado
# MAGIC
# MAGIC 4. **Interpretabilidad vs Accuracy es un trade-off real**
# MAGIC    - Finanzas/salud: Interpretabilidad crítica
# MAGIC    - Ad tech/recomendaciones: Accuracy lo es todo
# MAGIC
# MAGIC 5. **Producción != Kaggle**
# MAGIC    - Considera velocidad, mantenibilidad, costos operacionales
# MAGIC    - Modelo simple que se ejecuta gana a modelo perfecto que no escala
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Resumen Rápido por Algoritmo
# MAGIC
# MAGIC | Algoritmo | Cuándo es la MEJOR opción |
# MAGIC |-----------|-------------------------------|
# MAGIC | **Regresión Logística** | Necesitas interpretabilidad + probabilidades calibradas |
# MAGIC | **Árbol de Decisión** | Necesitas reglas simples que un humano pueda seguir |
# MAGIC | **Random Forest** | Quieres accuracy + interpretabilidad (feature importance) |
# MAGIC | **XGBoost/LightGBM** | Máxima accuracy en datos tabulares es prioridad #1 |
# MAGIC | **SVM** | Alta dimensión con pocas muestras (text, bioinformática) |
# MAGIC | **Naive Bayes** | Velocidad extrema + datos de texto |
# MAGIC | **KNN** | Dataset pequeño, fronteras de decisión muy irregulares |
# MAGIC | **Neural Networks** | Imágenes, audio, texto (NLP avanzado), >100K muestras |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📚 Recursos Adicionales
# MAGIC
# MAGIC #### **Libros:**
# MAGIC
# MAGIC * **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** - Aurélien Géron
# MAGIC   - Cubre todos los algoritmos con implementaciones prácticas
# MAGIC
# MAGIC * **"The Elements of Statistical Learning"** - Hastie, Tibshirani, Friedman
# MAGIC   - Fundamentos matemáticos profundos
# MAGIC
# MAGIC * **"Pattern Recognition and Machine Learning"** - Christopher Bishop
# MAGIC   - Perspectiva bayesiana
# MAGIC
# MAGIC #### **Cursos:**
# MAGIC
# MAGIC * **Coursera**: Machine Learning (Andrew Ng)
# MAGIC * **Fast.ai**: Practical Deep Learning for Coders
# MAGIC * **Kaggle Learn**: Hands-on micro-courses
# MAGIC
# MAGIC #### **Herramientas:**
# MAGIC
# MAGIC * **scikit-learn**: Implementación de todos los algoritmos clásicos
# MAGIC * **XGBoost / LightGBM / CatBoost**: Gradient boosting state-of-the-art
# MAGIC * **TensorFlow / PyTorch**: Deep learning
# MAGIC * **MLflow**: Tracking de experimentos
# MAGIC
# MAGIC #### **Práctica:**
# MAGIC
# MAGIC * **Kaggle**: Competencias y datasets reales
# MAGIC * **UCI ML Repository**: Datasets clásicos
# MAGIC * **OpenML**: Plataforma de benchmarking
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚀 Próximos Pasos
# MAGIC
# MAGIC 1. **Implementación Práctica**
# MAGIC    - Notebook de código con ejemplos ejecutables
# MAGIC    - Comparación empírica en datasets reales
# MAGIC    - Visualización de fronteras de decisión
# MAGIC
# MAGIC 2. **Técnicas Avanzadas**
# MAGIC    - Ensemble methods (stacking, blending)
# MAGIC    - Manejo de datos desbalanceados (SMOTE, class weights)
# MAGIC    - Calibración de probabilidades
# MAGIC    - Explainability (SHAP, LIME)
# MAGIC
# MAGIC 3. **Producción**
# MAGIC    - Model serving (REST APIs, batch)
# MAGIC    - Monitoreo de performance
# MAGIC    - Reentrenamiento automático
# MAGIC    - A/B testing
# MAGIC
# MAGIC 4. **Casos de Estudio**
# MAGIC    - Proyecto end-to-end: Predicción de Churn
# MAGIC    - Proyecto end-to-end: Detección de Fraude
# MAGIC    - Proyecto end-to-end: Credit Scoring
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ✅ Checklist antes de Elegir Algoritmo
# MAGIC
# MAGIC **Datos:**
# MAGIC - [ ] Tamaño del dataset (n muestras)
# MAGIC - [ ] Número de features (d)
# MAGIC - [ ] Tipo de features (numéricas, categóricas, texto, imágenes)
# MAGIC - [ ] Balance de clases
# MAGIC - [ ] Presencia de outliers
# MAGIC - [ ] Valores faltantes
# MAGIC
# MAGIC **Requisitos de Negocio:**
# MAGIC - [ ] Interpretabilidad requerida
# MAGIC - [ ] Latencia máxima de predicción
# MAGIC - [ ] Recursos computacionales (CPU, GPU, memoria)
# MAGIC - [ ] Métrica de evaluación (accuracy, precision, recall, F1, AUC)
# MAGIC - [ ] Costos de false positives vs false negatives
# MAGIC
# MAGIC **Restricciones:**
# MAGIC - [ ] Tiempo de desarrollo
# MAGIC - [ ] Experiencia del equipo
# MAGIC - [ ] Mantenibilidad
# MAGIC - [ ] Cumplimiento regulatorio
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎉 ¡Felicidades!
# MAGIC
# MAGIC Ahora tienes una sólida comprensión de los **principales algoritmos de clasificación** y cómo aplicarlos a **problemas de negocios reales**.
# MAGIC
# MAGIC **Recuerda:**
# MAGIC > "The best model is the one you actually ship." 
# MAGIC
# MAGIC No busques la perfección. Busca **valor de negocio** y **mejora continua**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC 🚀 **¡A construir modelos que generen impacto!**