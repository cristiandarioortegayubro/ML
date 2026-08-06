# Databricks notebook source
# DBTITLE 1,Algoritmos de Regresión para Negocios
# MAGIC %md
# MAGIC # Algoritmos de Regresión para Problemas de Negocios
# MAGIC
# MAGIC ## 🎯 Objetivo
# MAGIC
# MAGIC Este notebook presenta los **principales algoritmos de regresión** utilizados en Machine Learning, con énfasis en sus **aplicaciones prácticas en el mundo de los negocios**.
# MAGIC
# MAGIC ### 📌 ¿Qué es Regresión?
# MAGIC
# MAGIC > **Regresión** es el problema de predecir un **valor numérico continuo** basándose en las características de una observación.
# MAGIC
# MAGIC **Definición formal:**
# MAGIC
# MAGIC Dado un dataset $\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), ..., (\mathbf{x}_n, y_n)\}$
# MAGIC
# MAGIC Donde:
# MAGIC * $\mathbf{x}_i \in \mathbb{R}^d$ son las características (features)
# MAGIC * $y_i \in \mathbb{R}$ son los valores objetivo continuos
# MAGIC
# MAGIC **Objetivo:** Aprender función $f: \mathbb{R}^d \rightarrow \mathbb{R}$ que minimice el error de predicción.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Casos de Uso Empresariales
# MAGIC
# MAGIC ### Predicción de Valores Continuos
# MAGIC
# MAGIC | Problema de Negocio | Variable a Predecir | Features Típicas |
# MAGIC |---------------------|---------------------|------------------|
# MAGIC | **Precio de Viviendas** | Precio de venta | Tamaño, ubicación, habitaciones, año |
# MAGIC | **Demanda de Producto** | Unidades vendidas | Precio, temporada, competencia, marketing |
# MAGIC | **Revenue Forecasting** | Ingresos mensuales | Histórico, estacionalidad, inversión |
# MAGIC | **Valoración de Activos** | Precio de acciones | Ratios financieros, sector, mercado |
# MAGIC | **Customer Lifetime Value** | Valor futuro del cliente | Compras, frecuencia, engagement |
# MAGIC | **Optimización de Precios** | Precio óptimo | Elasticidad, costos, competencia |
# MAGIC | **Forecast de Ventas** | Ventas proyectadas | Histórico, marketing, economía |
# MAGIC | **Estimación de Costos** | Costo de proyecto | Alcance, recursos, duración |
# MAGIC
# MAGIC ### Diferencias con Clasificación
# MAGIC
# MAGIC | Aspecto | Clasificación | Regresión |
# MAGIC |---------|---------------|------------|
# MAGIC | **Output** | Categoría/Clase | Valor numérico |
# MAGIC | **Métrica** | Accuracy, Precision, Recall | MAE, RMSE, R² |
# MAGIC | **Ejemplo** | ¿Fraude o no? | ¿Cuánto vale? |
# MAGIC | **Función de pérdida** | Cross-entropy | MSE, MAE |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Contenido
# MAGIC
# MAGIC 1. **Regresión Lineal** - El modelo base interpretable
# MAGIC 2. **Regresión Polinomial** - Capturando no-linealidad
# MAGIC 3. **Ridge, Lasso, Elastic Net** - Regularización para prevenir overfitting
# MAGIC 4. **Árboles de Decisión** - Modelos interpretables no-lineales
# MAGIC 5. **Random Forest** - Ensemble robusto de árboles
# MAGIC 6. **Gradient Boosting** - XGBoost, LightGBM para máxima accuracy
# MAGIC 7. **Support Vector Regression (SVR)** - Kernel trick para no-linealidad
# MAGIC 8. **Redes Neuronales** - Deep Learning para patrones complejos
# MAGIC 9. **Comparación y Selección** - ¿Cuándo usar cada algoritmo?
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,1. Regresión Lineal
# MAGIC %md
# MAGIC ## 1. Regresión Lineal
# MAGIC
# MAGIC ### 🎯 Concepto
# MAGIC
# MAGIC **Regresión Lineal** modela la relación entre variables mediante una **ecuación lineal**. Es el algoritmo más simple e interpretable para regresión.
# MAGIC
# MAGIC ### 📐 Formulación Matemática
# MAGIC
# MAGIC **Regresión Lineal Simple (una feature):**
# MAGIC
# MAGIC $$y = \beta_0 + \beta_1 x + \epsilon$$
# MAGIC
# MAGIC **Regresión Lineal Múltiple:**
# MAGIC
# MAGIC $$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_d x_d + \epsilon$$
# MAGIC
# MAGIC **Forma matricial:**
# MAGIC
# MAGIC $$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$
# MAGIC
# MAGIC **Función de costo (MSE - Mean Squared Error):**
# MAGIC
# MAGIC $$J(\boldsymbol{\beta}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{1}{n}\sum_{i=1}^{n}(y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2$$
# MAGIC
# MAGIC **Solución de Mínimos Cuadrados:**
# MAGIC
# MAGIC $$\boldsymbol{\hat{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC * **Máxima interpretabilidad**: Coeficientes muestran impacto directo de cada feature
# MAGIC * **Rápido**: Entrenamiento e inferencia muy eficientes
# MAGIC * **Solución analítica**: No requiere iteraciones (en casos simples)
# MAGIC * **Extrapolación**: Puede predecir fuera del rango de entrenamiento
# MAGIC * **Base estadística sólida**: Tests de significancia, intervalos de confianza
# MAGIC * **Poco propenso a overfitting**: Con pocas features
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC * **Asume linealidad**: No captura relaciones no-lineales
# MAGIC * **Sensible a outliers**: Valores extremos afectan mucho el modelo
# MAGIC * **Multicolinealidad**: Features correlacionadas causan inestabilidad
# MAGIC * **Homoscedasticidad**: Asume varianza constante de errores
# MAGIC * **Requiere features escaladas**: Para interpretación correcta de coeficientes
# MAGIC
# MAGIC ### 💼 Casos de Uso Empresariales
# MAGIC
# MAGIC | Industria | Aplicación | Por qué Regresión Lineal |
# MAGIC |-----------|------------|--------------------------|
# MAGIC | **Finanzas** | Valoración de activos | Interpretabilidad para inversores |
# MAGIC | **Retail** | Elasticidad precio-demanda | Coeficientes muestran sensibilidad |
# MAGIC | **Recursos Humanos** | Predicción de salarios | Transparencia para equidad salarial |
# MAGIC | **Marketing** | ROI de campañas | Cuantificar impacto de cada canal |
# MAGIC | **Manufactura** | Relación costo-producción | Planificación basada en coeficientes |
# MAGIC | **Inmobiliario** | Avalúo de propiedades | Explicar factores que afectan precio |
# MAGIC
# MAGIC ### 🔧 Métricas Clave
# MAGIC
# MAGIC **Coeficiente de Determinación (R²):**
# MAGIC
# MAGIC $$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$
# MAGIC
# MAGIC * $R^2 = 1$: Ajuste perfecto
# MAGIC * $R^2 = 0$: Modelo no mejor que la media
# MAGIC * Rango: $[0, 1]$ (puede ser negativo en validación si el modelo es muy malo)
# MAGIC
# MAGIC **R² Ajustado** (penaliza complejidad):
# MAGIC
# MAGIC $$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-d-1}$$
# MAGIC
# MAGIC ### 📊 Ejemplo de Interpretación
# MAGIC
# MAGIC **Modelo de Precio de Viviendas:**
# MAGIC
# MAGIC $$\text{Precio} = 50,000 + 200 \times \text{m}^2 + 15,000 \times \text{Habitaciones} - 1,000 \times \text{Edad}$$
# MAGIC
# MAGIC **Interpretación:**
# MAGIC * Cada m² adicional aumenta el precio en **$200**
# MAGIC * Cada habitación adicional suma **$15,000**
# MAGIC * Cada año de antigüedad reduce **$1,000**
# MAGIC * Una casa de 0m², 0 habitaciones, nueva valdría $50,000 (intercepto)

# COMMAND ----------

# DBTITLE 1,2. Regresión Polinomial
# MAGIC %md
# MAGIC ## 2. Regresión Polinomial
# MAGIC
# MAGIC ### 🎯 Concepto
# MAGIC
# MAGIC **Regresión Polinomial** extiende regresión lineal al transformar features a **potencias** (polinomios), capturando relaciones no-lineales.
# MAGIC
# MAGIC ### 📐 Formulación Matemática
# MAGIC
# MAGIC **Polinomio de grado 2:**
# MAGIC
# MAGIC $$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \epsilon$$
# MAGIC
# MAGIC **Polinomio de grado $d$:**
# MAGIC
# MAGIC $$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_d x^d + \epsilon$$
# MAGIC
# MAGIC **Para múltiples features:**
# MAGIC
# MAGIC $$\mathbf{x} = [x_1, x_2] \rightarrow [1, x_1, x_2, x_1^2, x_1x_2, x_2^2, x_1^3, ...]$$
# MAGIC
# MAGIC ### 🔄 Transformación de Features
# MAGIC
# MAGIC **Original → Polinomial (grado 2):**
# MAGIC
# MAGIC | Feature Original | Features Polinomiales (grado 2) |
# MAGIC |------------------|----------------------------------|
# MAGIC | $x_1$ | $1, x_1, x_1^2$ |
# MAGIC | $x_1, x_2$ | $1, x_1, x_2, x_1^2, x_1x_2, x_2^2$ |
# MAGIC | $x_1, x_2, x_3$ | $1, x_1, x_2, x_3, x_1^2, x_1x_2, x_1x_3, x_2^2, x_2x_3, x_3^2$ |
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC * **Captura no-linealidad**: Con modelo lineal (en coeficientes)
# MAGIC * **Sigue siendo interpretable**: Cada término tiene significado
# MAGIC * **Flexible**: Grado controla complejidad
# MAGIC * **Solución analítica**: Igual que regresión lineal estándar
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC * **Overfitting**: Grados altos memorizan ruido
# MAGIC * **Explosión de features**: $d$ features → $O(d^p)$ features polinomiales
# MAGIC * **Extrapolación peligrosa**: Polinomios divergen fuera del rango
# MAGIC * **Multicolinealidad**: $x$ y $x^2$ pueden estar muy correlacionados
# MAGIC
# MAGIC ### 💼 Casos de Uso Empresariales
# MAGIC
# MAGIC | Industria | Aplicación | Por qué Polinomial |
# MAGIC |-----------|------------|--------------------|
# MAGIC | **E-commerce** | Curva precio-demanda | Relación no-lineal (elasticidad variable) |
# MAGIC | **Marketing** | Retornos decrecientes de publicidad | A más gasto, menor ROI marginal |
# MAGIC | **Manufactura** | Costos de producción | Economías/deseconomías de escala |
# MAGIC | **Energía** | Consumo vs temperatura | Relación en forma de U (calefacción/AC) |
# MAGIC | **Finanzas** | Riesgo-retorno | Curva cóncava (diversificación) |
# MAGIC
# MAGIC ### 🔧 Selección del Grado
# MAGIC
# MAGIC **Reglas generales:**
# MAGIC
# MAGIC * **Grado 1** (lineal): Relación lineal simple
# MAGIC * **Grado 2** (cuadrático): Curvas suaves, un máximo/mínimo
# MAGIC * **Grado 3** (cúbico): Forma de S, dos puntos de inflexión
# MAGIC * **Grado 4+**: Raramente útil, alto riesgo de overfitting
# MAGIC
# MAGIC **Estrategia:** Usar validación cruzada para seleccionar grado óptimo.
# MAGIC
# MAGIC ### 📊 Ejemplo de Negocio
# MAGIC
# MAGIC **Relación Precio-Demanda (elasticidad no-constante):**
# MAGIC
# MAGIC $$\text{Demanda} = 10,000 - 50 \times \text{Precio} - 0.5 \times \text{Precio}^2$$
# MAGIC
# MAGIC **Interpretación:**
# MAGIC * A precio bajo: reducción lineal de demanda
# MAGIC * A precio alto: caída acelerada (elasticidad aumenta)
# MAGIC * **Insight:** Incrementos de precio tienen mayor impacto cuando el precio ya es alto

# COMMAND ----------

# DBTITLE 1,3. Ridge, Lasso, Elastic Net - Regularización
# MAGIC %md
# MAGIC ## 3. Ridge, Lasso, Elastic Net - Regularización
# MAGIC
# MAGIC ### 🎯 Concepto
# MAGIC
# MAGIC **Regularización** añade una penalización a la función de costo para **reducir overfitting** al limitar la magnitud de los coeficientes.
# MAGIC
# MAGIC ### 📐 Formulaciones Matemáticas
# MAGIC
# MAGIC #### Ridge Regression (L2 Regularization)
# MAGIC
# MAGIC $$J(\boldsymbol{\beta}) = MSE + \lambda \sum_{j=1}^{d}\beta_j^2 = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda ||\boldsymbol{\beta}||_2^2$$
# MAGIC
# MAGIC **Efecto:** Reduce magnitud de coeficientes, **no los elimina** (coeficientes pequeños pero no cero).
# MAGIC
# MAGIC #### Lasso Regression (L1 Regularization)
# MAGIC
# MAGIC $$J(\boldsymbol{\beta}) = MSE + \lambda \sum_{j=1}^{d}|\beta_j| = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda ||\boldsymbol{\beta}||_1$$
# MAGIC
# MAGIC **Efecto:** Puede forzar $\beta_j = 0$ → **Selección automática de features**.
# MAGIC
# MAGIC #### Elastic Net (L1 + L2)
# MAGIC
# MAGIC $$J(\boldsymbol{\beta}) = MSE + \lambda_1 ||\boldsymbol{\beta}||_1 + \lambda_2 ||\boldsymbol{\beta}||_2^2$$
# MAGIC
# MAGIC **Efecto:** Combina ventajas de Ridge y Lasso.
# MAGIC
# MAGIC ### 📊 Comparación Visual
# MAGIC
# MAGIC | Método | Penalización | Selección Features | Coeficientes | Estabilidad |
# MAGIC |---------|--------------|---------------------|--------------|-------------|
# MAGIC | **OLS** | Ninguna | No | Grandes | Baja |
# MAGIC | **Ridge** | L2 ($\beta^2$) | No | Pequeños | Alta |
# MAGIC | **Lasso** | L1 ($|\beta|$) | Sí (algunos = 0) | Sparse | Media |
# MAGIC | **Elastic Net** | L1 + L2 | Sí | Sparse + pequeños | Alta |
# MAGIC
# MAGIC ### ✅ Ventajas por Método
# MAGIC
# MAGIC **Ridge:**
# MAGIC * Maneja multicolinealidad
# MAGIC * Mantiene todas las features
# MAGIC * Solución única
# MAGIC * Funciona bien con muchas features correlacionadas
# MAGIC
# MAGIC **Lasso:**
# MAGIC * Selección automática de features
# MAGIC * Modelos interpretables (sparse)
# MAGIC * Útil para feature importance
# MAGIC * Descarta features irrelevantes
# MAGIC
# MAGIC **Elastic Net:**
# MAGIC * Mejor de ambos mundos
# MAGIC * Robusto a grupos de features correlacionadas
# MAGIC * Selección de features + estabilidad
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC **Ridge:**
# MAGIC * No selecciona features (todas se mantienen)
# MAGIC * Interpretación más difícil con muchas features
# MAGIC
# MAGIC **Lasso:**
# MAGIC * Inestable con features muy correlacionadas
# MAGIC * Puede seleccionar arbitrariamente una de varias correlacionadas
# MAGIC
# MAGIC **Elastic Net:**
# MAGIC * Dos hiperparámetros para tunear
# MAGIC * Más complejo de explicar
# MAGIC
# MAGIC ### 💼 Casos de Uso Empresariales
# MAGIC
# MAGIC | Industria | Aplicación | Método Recomendado | Razón |
# MAGIC |-----------|------------|--------------------| ------|
# MAGIC | **Marketing** | Atribución multicanal | **Lasso** | Identifica canales más efectivos |
# MAGIC | **Finanzas** | Predicción de riesgo | **Ridge** | Mantiene todas las variables regulatorias |
# MAGIC | **E-commerce** | Predicción de ventas | **Elastic Net** | Muchas features correlacionadas (productos) |
# MAGIC | **Salud** | Diagnóstico médico | **Lasso** | Selecciona biomarcadores clave |
# MAGIC | **Real Estate** | Valoración de propiedades | **Ridge** | Mantiene todas las características |
# MAGIC
# MAGIC ### 🔧 Selección del Hiperparámetro λ
# MAGIC
# MAGIC **λ (lambda) o α (alpha):** Controla la fuerza de la regularización
# MAGIC
# MAGIC * **λ = 0**: Sin regularización (OLS estándar)
# MAGIC * **λ pequeño**: Poca regularización
# MAGIC * **λ grande**: Mucha regularización (coeficientes → 0)
# MAGIC
# MAGIC **Estrategia:** Usar **Cross-Validation** para encontrar λ óptimo.
# MAGIC
# MAGIC ### 📊 Ejemplo de Negocio
# MAGIC
# MAGIC **Predicción de Ventas con 50 features:**
# MAGIC
# MAGIC **Ridge:** Mantiene todas las 50 variables con coeficientes pequeños
# MAGIC ```
# MAGIC Precio: 0.15, Publicidad_TV: 0.08, Publicidad_Radio: 0.05, ...
# MAGIC (todas las features con valores pequeños)
# MAGIC ```
# MAGIC
# MAGIC **Lasso:** Selecciona solo 10 features más relevantes
# MAGIC ```
# MAGIC Precio: 0.25, Publicidad_TV: 0.18, Estación: 0.12, ...
# MAGIC (40 features eliminadas, coeficientes = 0)
# MAGIC ```
# MAGIC
# MAGIC **Insight:** Lasso identifica que solo 10 de 50 features realmente importan, simplificando el modelo y mejorando interpretabilidad.

# COMMAND ----------

# DBTITLE 1,4. Árboles de Decisión para Regresión
# MAGIC %md
# MAGIC ## 4. Árboles de Decisión para Regresión
# MAGIC
# MAGIC ### 🎯 Concepto
# MAGIC
# MAGIC **Árboles de Decisión para Regresión** particionan el espacio de features en regiones y asignan un **valor promedio** a cada región.
# MAGIC
# MAGIC ### 🌳 Estructura
# MAGIC
# MAGIC ```
# MAGIC            [Tamaño < 100m²?]
# MAGIC               /    \
# MAGIC             Sí     No
# MAGIC             /        \
# MAGIC    [¿Ubicación = Centro?]  [Habitaciones < 3?]
# MAGIC        /    \          /      \
# MAGIC      Sí    No        Sí       No
# MAGIC      /      \        /         \
# MAGIC   $150K  $180K    $250K     $300K
# MAGIC ```
# MAGIC
# MAGIC ### 📐 Algoritmo
# MAGIC
# MAGIC **Criterio de división:** Minimizar MSE o MAE en nodos hijos
# MAGIC
# MAGIC $$MSE = \frac{1}{n}\sum_{i \in \text{nodo}}(y_i - \bar{y}_{\text{nodo}})^2$$
# MAGIC
# MAGIC **Proceso:**
# MAGIC 1. Para cada feature y punto de corte posible, calcular MSE resultante
# MAGIC 2. Elegir división que minimiza MSE
# MAGIC 3. Repetir recursivamente en nodos hijos
# MAGIC 4. Parar según criterios (profundidad, min_samples)
# MAGIC
# MAGIC **Predicción:** Para nueva observación $\mathbf{x}$, navegar el árbol hasta hoja y retornar $\bar{y}_{\text{hoja}}$
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC * **Altamente interpretable**: Visualizable, fácil de explicar
# MAGIC * **No asume linealidad**: Captura relaciones complejas
# MAGIC * **No requiere normalización**: Invariante a escala de features
# MAGIC * **Maneja categorícas y numéricas**: Sin preprocessing especial
# MAGIC * **Captura interacciones**: Automáticamente
# MAGIC * **Feature importance**: Identifica variables más relevantes
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC * **Overfitting**: Árboles profundos memorizan ruido
# MAGIC * **Inestabilidad**: Pequeños cambios → árbol muy diferente
# MAGIC * **Predicciones escalonadas**: No interpola suavemente
# MAGIC * **No extrapola**: No puede predecir fuera de rango de entrenamiento
# MAGIC * **Sesgo hacia features continuas con muchos valores únicos**
# MAGIC
# MAGIC ### 💼 Casos de Uso Empresariales
# MAGIC
# MAGIC | Industria | Aplicación | Por qué Árboles |
# MAGIC |-----------|------------|-------------------|
# MAGIC | **Finanzas** | Estimación de tasas de interés | Interpretabilidad para clientes |
# MAGIC | **Inmobiliario** | Valuación de propiedades | Reglas claras para tasadores |
# MAGIC | **Recursos Humanos** | Predicción de salarios | Transparencia para equidad |
# MAGIC | **Manufactura** | Estimación de tiempo de producción | Operadores entienden las reglas |
# MAGIC | **Retail** | Precio dinámico | Segmentación clara de clientes |
# MAGIC
# MAGIC ### 🔧 Hiperparámetros Clave
# MAGIC
# MAGIC * **max_depth**: Profundidad máxima del árbol (controla overfitting)
# MAGIC * **min_samples_split**: Mínimo de muestras para dividir nodo
# MAGIC * **min_samples_leaf**: Mínimo de muestras en hoja
# MAGIC * **max_features**: Número de features a considerar por división
# MAGIC * **criterion**: 'squared_error' (MSE) o 'absolute_error' (MAE)
# MAGIC
# MAGIC ### 📊 Ejemplo de Negocio
# MAGIC
# MAGIC **Predicción de Precio de Vivienda:**
# MAGIC
# MAGIC ```
# MAGIC Raíz: 1000 viviendas (precio promedio = $200K)
# MAGIC     |
# MAGIC     ├─ [Tamaño < 80m²?]
# MAGIC     │   ├─ Sí: 400 viviendas (promedio = $150K)
# MAGIC     │   └─ No: 600 viviendas (promedio = $230K)
# MAGIC     │       |
# MAGIC     │       ├─ [¿Ubicación = Centro?]
# MAGIC     │       │   ├─ Sí: 200 viviendas (promedio = $280K)
# MAGIC     │       │   └─ No: 400 viviendas (promedio = $205K)
# MAGIC ```
# MAGIC
# MAGIC **Insights accionables:**
# MAGIC * Tamaño es el factor más importante
# MAGIC * En viviendas >80m², ubicación centro agrega $75K

# COMMAND ----------

# DBTITLE 1,5. Random Forest para Regresión
# MAGIC %md
# MAGIC ## 5. Random Forest para Regresión
# MAGIC
# MAGIC ### 🎯 Concepto
# MAGIC
# MAGIC **Random Forest Regressor** es un **ensemble** de múltiples árboles de decisión que **promedian** sus predicciones.
# MAGIC
# MAGIC > **"La sabiduría de las multitudes"** aplicada a regresión.
# MAGIC
# MAGIC ### 🌲🌳🌴 Arquitectura
# MAGIC
# MAGIC ```
# MAGIC               Training Data
# MAGIC                     |
# MAGIC         ┌───────────┤───────────┐
# MAGIC         │           │           │
# MAGIC    Bootstrap     Bootstrap   Bootstrap
# MAGIC    Sample 1      Sample 2    Sample N
# MAGIC         │           │           │
# MAGIC    [Árbol 1]   [Árbol 2]   [Árbol N]
# MAGIC    Pred: $150K  Pred: $155K Pred: $148K
# MAGIC         │           │           │
# MAGIC         └───────────┤───────────┘
# MAGIC                     |
# MAGIC               Promedio
# MAGIC                     |
# MAGIC          Predicción: $151K
# MAGIC ```
# MAGIC
# MAGIC ### 📐 Algoritmo
# MAGIC
# MAGIC **Predicción:**
# MAGIC
# MAGIC $$\hat{y} = \frac{1}{N}\sum_{i=1}^{N} f_i(\mathbf{x})$$
# MAGIC
# MAGIC Donde $f_i$ es la predicción del árbol $i$.
# MAGIC
# MAGIC **Bagging + Random Features:**
# MAGIC 1. Bootstrap sampling: Cada árbol con ∼63% datos únicos
# MAGIC 2. Random feature selection: Considerar solo $\sqrt{d}$ o $d/3$ features por división
# MAGIC 3. Entrenar árboles profundos (sin poda)
# MAGIC 4. Promediar predicciones
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC * **Reduce overfitting**: Comparado con árbol individual
# MAGIC * **Muy robusto**: Maneja outliers y ruido
# MAGIC * **Feature importance**: Ranking de variables relevantes
# MAGIC * **Out-of-Bag (OOB) error**: Validación automática
# MAGIC * **Parallelizable**: Árboles independientes
# MAGIC * **Funciona bien "out of the box"**: Poco tuning requerido
# MAGIC * **Captura no-linealidad e interacciones**
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC * **Menos interpretable**: No visualizable (100+ árboles)
# MAGIC * **Más lento en inferencia**: Consulta todos los árboles
# MAGIC * **Mayor memoria**: Almacena múltiples modelos
# MAGIC * **No extrapola**: Hereda limitación de árboles
# MAGIC * **Predicciones suavizadas hacia la media**: En extremos
# MAGIC
# MAGIC ### 💼 Casos de Uso Empresariales
# MAGIC
# MAGIC | Industria | Aplicación | Por qué Random Forest |
# MAGIC |-----------|------------|------------------------|
# MAGIC | **Finanzas** | Predicción de precios de acciones | Robusto, captura interacciones |
# MAGIC | **Inmobiliario** | Valuación automática | Alta precisión, feature importance |
# MAGIC | **E-commerce** | Predicción de demanda | Maneja features heterogéneas |
# MAGIC | **Energía** | Forecast de consumo | Robusto a outliers (picos) |
# MAGIC | **Seguros** | Estimación de reclamaciones | Captura patrones complejos |
# MAGIC
# MAGIC ### 🔧 Hiperparámetros Clave
# MAGIC
# MAGIC * **n_estimators**: Número de árboles (100-500 típico, más = mejor pero más lento)
# MAGIC * **max_depth**: Profundidad de cada árbol (None = sin límite)
# MAGIC * **max_features**: Features por división ('sqrt', 1.0, o número)
# MAGIC * **min_samples_split**: Mínimo para dividir nodo (2-10)
# MAGIC * **min_samples_leaf**: Mínimo en hoja (1-5)
# MAGIC * **bootstrap**: True para bagging
# MAGIC * **oob_score**: True para usar OOB como validación
# MAGIC
# MAGIC ### 📊 Feature Importance
# MAGIC
# MAGIC **Cálculo:**
# MAGIC
# MAGIC Importancia = Promedio de reducción de MSE cuando feature se usa en divisiones, across todos los árboles.
# MAGIC
# MAGIC **Ejemplo - Predicción de Revenue:**
# MAGIC
# MAGIC ```
# MAGIC 1. Inversión Marketing:     0.32  ████████████████████████████████
# MAGIC 2. Estacionalidad:          0.25  █████████████████████████
# MAGIC 3. Precio Promedio:         0.18  ██████████████████
# MAGIC 4. Competencia:             0.12  ████████████
# MAGIC 5. Tráfico Web:             0.08  ████████
# MAGIC 6. Reviews:                 0.05  █████
# MAGIC ```
# MAGIC
# MAGIC **Insight:** Enfocar esfuerzos en Marketing (32%) y aprovechar estacionalidad (25%) tiene mayor ROI que mejorar reviews (5%).

# COMMAND ----------

# DBTITLE 1,6. Gradient Boosting para Regresión
# MAGIC %md
# MAGIC ## 6. Gradient Boosting para Regresión (XGBoost, LightGBM)
# MAGIC
# MAGIC ### 🎯 Concepto
# MAGIC
# MAGIC **Gradient Boosting** construye árboles **secuencialmente**, donde cada nuevo árbol corrige los **errores** (residuos) del anterior.
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
# MAGIC | Peso | Igual | Ponderado |
# MAGIC | Objetivo | Reducir varianza | Reducir sesgo |
# MAGIC
# MAGIC **Proceso:**
# MAGIC
# MAGIC 1. Inicializar: $F_0(x) = \bar{y}$ (predicción = media)
# MAGIC 2. Para $m = 1$ a $M$:
# MAGIC    - Calcular residuos: $r_i = y_i - F_{m-1}(x_i)$
# MAGIC    - Entrenar árbol $h_m$ para predecir $r_i$
# MAGIC    - Actualizar: $F_m(x) = F_{m-1}(x) + \alpha h_m(x)$
# MAGIC 3. Predicción final: $\hat{y} = F_M(x)$
# MAGIC
# MAGIC Donde $\alpha$ es el **learning rate** (0.01-0.3).
# MAGIC
# MAGIC ### 🚀 Implementaciones Populares
# MAGIC
# MAGIC #### **XGBoost** (eXtreme Gradient Boosting)
# MAGIC
# MAGIC * **Ventaja**: Rápido, regularización L1/L2, maneja missing values
# MAGIC * **Innovación**: Aproximación de segundo orden (Hessian)
# MAGIC * **Mejor para**: Balance general, competencias
# MAGIC
# MAGIC #### **LightGBM** (Microsoft)
# MAGIC
# MAGIC * **Ventaja**: **Más rápido** que XGBoost, menos memoria
# MAGIC * **Innovación**: Leaf-wise growth (vs level-wise)
# MAGIC * **Mejor para**: Datasets grandes (millones de filas)
# MAGIC
# MAGIC #### **CatBoost** (Yandex)
# MAGIC
# MAGIC * **Ventaja**: Maneja **categóricas nativas**
# MAGIC * **Innovación**: Ordered boosting
# MAGIC * **Mejor para**: Muchas features categóricas
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC * **State-of-the-art accuracy**: Máxima precisión en muchos benchmarks
# MAGIC * **Captura interacciones complejas**: No-linealidad, interacciones de orden alto
# MAGIC * **Feature importance**: Similar a Random Forest
# MAGIC * **Maneja missing values**: Built-in
# MAGIC * **Flexible**: Funciones de pérdida custom
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC * **Overfitting**: Si no se regularizan hiperparámetros
# MAGIC * **Tuning complejo**: Muchos hiperparámetros
# MAGIC * **Secuencial**: No parallelizable como Random Forest
# MAGIC * **Sensible a ruido**: Puede aprender patrones espúreos
# MAGIC * **Menos interpretable**: Comparado con árbol simple
# MAGIC
# MAGIC ### 💼 Casos de Uso Empresariales
# MAGIC
# MAGIC | Industria | Aplicación | Por qué Gradient Boosting |
# MAGIC |-----------|------------|----------------------------|
# MAGIC | **Fintech** | Scoring de crédito | Máxima accuracy = menos riesgo |
# MAGIC | **E-commerce** | Pricing dinámico | Captura elasticidades complejas |
# MAGIC | **Ad Tech** | Predicción de CTR/CPC | Performance es crítico (revenue) |
# MAGIC | **Retail** | Demand forecasting | Supera modelos tradicionales |
# MAGIC | **Inmobiliario** | Valuación automatizada | Mejor que modelos lineales |
# MAGIC
# MAGIC ### 🔧 Hiperparámetros Clave
# MAGIC
# MAGIC **Control de Overfitting:**
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
# MAGIC ### 📊 Comparación de Implementaciones
# MAGIC
# MAGIC | Característica | XGBoost | LightGBM | CatBoost |
# MAGIC |----------------|---------|----------|----------|
# MAGIC | **Velocidad** | Rápido | **Más rápido** | Medio |
# MAGIC | **Memoria** | Media | **Baja** | Alta |
# MAGIC | **Accuracy** | Alto | Alto | **Muy alto** |
# MAGIC | **Categóricas** | One-hot | One-hot | **Nativas** |
# MAGIC | **Overfitting** | Medio | Alto | **Bajo** |
# MAGIC | **Tuning** | Complejo | Complejo | **Simple** |
# MAGIC
# MAGIC **Recomendación general:**
# MAGIC * **Datasets grandes**: LightGBM
# MAGIC * **Muchas categóricas**: CatBoost
# MAGIC * **Balance general**: XGBoost

# COMMAND ----------

# DBTITLE 1,7. Support Vector Regression (SVR)
# MAGIC %md
# MAGIC ## 7. Support Vector Regression (SVR)
# MAGIC
# MAGIC ### 🎯 Concepto
# MAGIC
# MAGIC **SVR** encuentra una función que tenga a lo sumo $\epsilon$ desviación de los valores objetivo, mientras es lo más **plana** posible.
# MAGIC
# MAGIC ### 📐 Formulación Matemática
# MAGIC
# MAGIC **Objetivo:** Minimizar $||\mathbf{w}||^2$ sujeto a:
# MAGIC
# MAGIC $$|y_i - f(\mathbf{x}_i)| \leq \epsilon$$
# MAGIC
# MAGIC **Con margen suave (permite errores):**
# MAGIC
# MAGIC $$\min_{\mathbf{w}, b, \xi} \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i=1}^{n}(\xi_i + \xi_i^*)$$
# MAGIC
# MAGIC Sujeto a:
# MAGIC * $y_i - (\mathbf{w}^T\mathbf{x}_i + b) \leq \epsilon + \xi_i$
# MAGIC * $(\mathbf{w}^T\mathbf{x}_i + b) - y_i \leq \epsilon + \xi_i^*$
# MAGIC * $\xi_i, \xi_i^* \geq 0$
# MAGIC
# MAGIC **Tubo $\epsilon$-insensitivo:**
# MAGIC
# MAGIC Errores dentro de $\epsilon$ no se penalizan. Solo errores fuera del tubo contribuyen a la pérdida.
# MAGIC
# MAGIC ### 🎩 Kernel Trick
# MAGIC
# MAGIC **Para relaciones no-lineales**, mapear a espacio dimensional superior:
# MAGIC
# MAGIC **Kernels comunes:**
# MAGIC
# MAGIC 1. **Lineal**: $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T\mathbf{x}_j$
# MAGIC
# MAGIC 2. **Polinomial**: $K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T\mathbf{x}_j + r)^d$
# MAGIC
# MAGIC 3. **RBF (Radial Basis Function)**:
# MAGIC    $$K(\mathbf{x}_i, \mathbf{x}_j) = e^{-\gamma||\mathbf{x}_i - \mathbf{x}_j||^2}$$
# MAGIC    * Más popular para SVR
# MAGIC    * Puede aproximar cualquier función continua
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC * **Efectivo en alta dimensión**: Funciona cuando $d > n$
# MAGIC * **Memory efficient**: Solo almacena support vectors
# MAGIC * **Versatilidad**: Diferentes kernels para diferentes problemas
# MAGIC * **Robusto a outliers**: Dentro del tubo $\epsilon$
# MAGIC * **Regularización**: Parámetro $C$ controla complejidad
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC * **Lento en datasets grandes**: Complejidad $O(n^2)$ a $O(n^3)$
# MAGIC * **Sensible a escala**: **REQUIERE normalización**
# MAGIC * **Selección de hiperparámetros**: $C$, $\epsilon$, $\gamma$ no trivial
# MAGIC * **Interpretación limitada**: Especialmente con kernels no-lineales
# MAGIC
# MAGIC ### 💼 Casos de Uso Empresariales
# MAGIC
# MAGIC | Industria | Aplicación | Por qué SVR |
# MAGIC |-----------|------------|-------------|
# MAGIC | **Finanzas** | Predicción de volatilidad | Captura patrones complejos |
# MAGIC | **Energía** | Forecast de carga eléctrica | Kernel RBF para no-linealidad |
# MAGIC | **Manufactura** | Control de calidad | Robusto a outliers |
# MAGIC | **Telecomunicaciones** | Predicción de tráfico de red | Alta dimensión |
# MAGIC | **Retail** | Predicción de precios | Cuando pocos datos pero muchas features |
# MAGIC
# MAGIC ### 🔧 Hiperparámetros Clave
# MAGIC
# MAGIC * **C**: Parámetro de regularización
# MAGIC   - **C alto**: Menos errores permitidos (overfitting)
# MAGIC   - **C bajo**: Más tolerante a errores (underfitting)
# MAGIC
# MAGIC * **epsilon** ($\epsilon$): Ancho del tubo
# MAGIC   - Más grande = más robusto, menos support vectors
# MAGIC   - Más pequeño = más preciso, más support vectors
# MAGIC
# MAGIC * **kernel**: Tipo ('linear', 'rbf', 'poly')
# MAGIC
# MAGIC * **gamma** (para RBF/poly):
# MAGIC   - **Gamma alto**: Influencia local (overfitting)
# MAGIC   - **Gamma bajo**: Influencia global (underfitting)
# MAGIC
# MAGIC ### ⚠️ IMPORTANTE: Escalamiento
# MAGIC
# MAGIC **SVR es EXTREMADAMENTE sensible a la escala.**
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.preprocessing import StandardScaler
# MAGIC
# MAGIC # SIEMPRE escalar antes de SVR:
# MAGIC scaler = StandardScaler()
# MAGIC X_train_scaled = scaler.fit_transform(X_train)
# MAGIC X_test_scaled = scaler.transform(X_test)
# MAGIC ```
# MAGIC
# MAGIC ### 📊 Ejemplo
# MAGIC
# MAGIC **Sin escalar:** Feature A [0-1] vs Feature B [0-10000]
# MAGIC * Feature B domina, resultados pobres
# MAGIC
# MAGIC **Con escalar:** Ambas features [-1, 1]
# MAGIC * Ambas contribuyen equitativamente, mejor performance

# COMMAND ----------

# DBTITLE 1,8. Redes Neuronales para Regresión
# MAGIC %md
# MAGIC ## 8. Redes Neuronales para Regresión
# MAGIC
# MAGIC ### 🎯 Concepto
# MAGIC
# MAGIC **Redes Neuronales** (Deep Learning) son modelos inspirados en el cerebro que pueden aprender representaciones jerárquicas complejas de los datos.
# MAGIC
# MAGIC ### 🧠 Arquitectura Básica
# MAGIC
# MAGIC ```
# MAGIC     Input Layer      Hidden Layers       Output Layer
# MAGIC         
# MAGIC     x1 ● ────┐
# MAGIC              │───● ──┐
# MAGIC     x2 ● ───┤    │───● ──┐
# MAGIC              │───● ──┤    │
# MAGIC     x3 ● ───┤    │───● ──┤───● ──▶ ŷ (precio)
# MAGIC              │───● ──┤    │
# MAGIC     xd ● ───┘    │───● ──┘
# MAGIC              │───● ──┘
# MAGIC ```
# MAGIC
# MAGIC ### 📐 Formulación Matemática
# MAGIC
# MAGIC **Forward propagation:**
# MAGIC
# MAGIC $$z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$$
# MAGIC $$a^{[l]} = g^{[l]}(z^{[l]})$$
# MAGIC
# MAGIC Donde:
# MAGIC * $W^{[l]}$: Pesos de capa $l$
# MAGIC * $b^{[l]}$: Bias
# MAGIC * $g^{[l]}$: Función de activación (ReLU, sigmoid, tanh)
# MAGIC * $a^{[l]}$: Activaciones
# MAGIC
# MAGIC **Output para regresión:**
# MAGIC
# MAGIC $$\hat{y} = a^{[L]}$$ (sin activación en última capa, o lineal)
# MAGIC
# MAGIC **Función de pérdida:** MSE o MAE
# MAGIC
# MAGIC $$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
# MAGIC
# MAGIC **Optimización:** Gradient Descent, Adam, RMSprop
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC * **Máxima capacidad de modelado**: Aproximador universal de funciones
# MAGIC * **Feature learning**: Aprende representaciones automáticamente
# MAGIC * **Maneja datos no estructurados**: Imágenes, texto, audio
# MAGIC * **Escalabilidad**: Mejora con más datos
# MAGIC * **Transfer learning**: Reutilizar modelos pre-entrenados
# MAGIC * **Flexible**: Arquitecturas custom para problemas específicos
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC * **Requiere MUCHOS datos**: Miles a millones de ejemplos
# MAGIC * **Computacionalmente costoso**: GPUs necesarias para datasets grandes
# MAGIC * **Caja negra**: Difícil de interpretar
# MAGIC * **Muchos hiperparámetros**: Arquitectura, learning rate, regularización, etc.
# MAGIC * **Propenso a overfitting**: Especialmente con pocos datos
# MAGIC * **Largo tiempo de entrenamiento**: Comparado con otros métodos
# MAGIC
# MAGIC ### 💼 Casos de Uso Empresariales
# MAGIC
# MAGIC | Industria | Aplicación | Por qué Redes Neuronales |
# MAGIC |-----------|------------|-----------------------------|
# MAGIC | **Finance** | Predicción de series temporales | Captura patrones temporales complejos |
# MAGIC | **E-commerce** | Predicción de demanda con imágenes | Combina datos estructurados + imágenes |
# MAGIC | **Real Estate** | Valuación con fotos de propiedades | Extrae features visuales |
# MAGIC | **Marketing** | Predicción de LTV con secuencias | Secuencias de comportamiento complejas |
# MAGIC | **Energy** | Load forecasting | Patrones temporales no-lineales |
# MAGIC
# MAGIC ### 🔧 Componentes Clave
# MAGIC
# MAGIC **Arquitectura:**
# MAGIC * **Input layer**: Tamaño = número de features
# MAGIC * **Hidden layers**: 1-5 capas (deep), 10-1000 neuronas por capa
# MAGIC * **Output layer**: 1 neurona (regresión), sin activación
# MAGIC
# MAGIC **Funciones de activación:**
# MAGIC * **ReLU**: $f(x) = \max(0, x)$ (más popular)
# MAGIC * **Leaky ReLU**: $f(x) = \max(0.01x, x)$
# MAGIC * **tanh**: $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
# MAGIC
# MAGIC **Regularización:**
# MAGIC * **Dropout**: Apagar neuronas aleatoriamente (0.2-0.5)
# MAGIC * **L2 regularization**: Penalizar pesos grandes
# MAGIC * **Early stopping**: Parar cuando validación empeora
# MAGIC * **Batch normalization**: Normalizar activaciones
# MAGIC
# MAGIC ### 📊 Cuándo Usar Redes Neuronales
# MAGIC
# MAGIC **Usar cuando:**
# MAGIC ✅ Tienes **muchos datos** (>10K ejemplos, idealmente >100K)
# MAGIC ✅ Alta no-linealidad y interacciones complejas
# MAGIC ✅ Datos no estructurados (imágenes, texto, audio)
# MAGIC ✅ La interpretabilidad no es crítica
# MAGIC ✅ Tienes recursos computacionales (GPU)
# MAGIC
# MAGIC **NO usar cuando:**
# MAGIC ❌ Pocos datos (<1K ejemplos)
# MAGIC ❌ Se requiere alta interpretabilidad
# MAGIC ❌ Recursos computacionales limitados
# MAGIC ❌ Gradient Boosting ya da buenos resultados
# MAGIC
# MAGIC ### 🔥 Frameworks Populares
# MAGIC
# MAGIC * **TensorFlow/Keras**: Ecosistema completo, producción
# MAGIC * **PyTorch**: Research, flexibilidad
# MAGIC * **scikit-learn MLPRegressor**: Simple, datasets pequeños

# COMMAND ----------

# DBTITLE 1,9. Comparación y Selección de Algoritmos
# MAGIC %md
# MAGIC ## 9. Comparación y Selección de Algoritmos
# MAGIC
# MAGIC ### 📈 Tabla Comparativa Completa
# MAGIC
# MAGIC | Algoritmo | Interpretabilidad | Accuracy | Velocidad Entrenamiento | Velocidad Inferencia | Maneja No-linealidad | Requiere Escala | Robusto Outliers | Selección Features |
# MAGIC |-----------|-------------------|----------|-------------------------|----------------------|---------------------|-----------------|------------------|---------------------|
# MAGIC | **Regresión Lineal** | ⭐️⭐️⭐️⭐️⭐️ | ⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ | ❌ | Sí | ❌ | No |
# MAGIC | **Polinomial** | ⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ | ✅ | Sí | ❌ | No |
# MAGIC | **Ridge** | ⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ | ❌ | Sí | ❌ | No |
# MAGIC | **Lasso** | ⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ | ❌ | Sí | ❌ | Sí |
# MAGIC | **Árbol Decisión** | ⭐️⭐️⭐️⭐️⭐️ | ⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ | ✅ | No | ✅ | Sí |
# MAGIC | **Random Forest** | ⭐️⭐️ | ⭐️⭐️⭐️⭐️ | ⭐️⭐️ | ⭐️⭐️ | ✅ | No | ✅ | Sí |
# MAGIC | **Gradient Boosting** | ⭐️ | ⭐️⭐️⭐️⭐️⭐️ | ⭐️ | ⭐️⭐️ | ✅ | No | ✅ | Sí |
# MAGIC | **SVR** | ⭐️ | ⭐️⭐️⭐️ | ⭐️ | ⭐️⭐️ | ✅ | Sí | ✅ | No |
# MAGIC | **Redes Neuronales** | ⭐️ | ⭐️⭐️⭐️⭐️⭐️ | ⭐️ | ⭐️⭐️⭐️ | ✅ | Sí | ✅ | Automático |
# MAGIC
# MAGIC ### 🧭 Árbol de Decisión para Selección
# MAGIC
# MAGIC ```
# MAGIC                     [¿Tienes muchos datos (>100K)?]
# MAGIC                           /              \
# MAGIC                         NO               SÍ
# MAGIC                         /                  \
# MAGIC         [¿Necesitas interpretabilidad?]   [¿Tienes GPUs?]
# MAGIC               /           \                  /        \
# MAGIC             SÍ            NO               SÍ         NO
# MAGIC             /              \                /           \
# MAGIC   [¿Lineal?]    [¿Muchas features?]   Redes      Gradient
# MAGIC      /    \           /      \        Neuronales    Boosting
# MAGIC    SÍ    NO        SÍ       NO
# MAGIC    /      \         /        \
# MAGIC Lineal  Árbol    Lasso   Random
# MAGIC                (Elastic)  Forest
# MAGIC ```
# MAGIC
# MAGIC ### 👥 Guía de Selección por Prioridad
# MAGIC
# MAGIC #### **1° PRIORIDAD: Interpretabilidad**
# MAGIC
# MAGIC ➡️ **Regresión Lineal** o **Árbol de Decisión**
# MAGIC
# MAGIC **Cuándo:**
# MAGIC * Regulación requiere explicar cada predicción
# MAGIC * Stakeholders no-técnicos necesitan entender
# MAGIC * Finanzas, salud, recursos humanos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **2° PRIORIDAD: Máxima Accuracy**
# MAGIC
# MAGIC ➡️ **Gradient Boosting** (XGBoost/LightGBM) o **Redes Neuronales**
# MAGIC
# MAGIC **Cuándo:**
# MAGIC * Competencia Kaggle
# MAGIC * Performance es crítico para el negocio
# MAGIC * Pricing, forecasting, ad tech
# MAGIC
# MAGIC **Elección:**
# MAGIC * Datos tabulares → **Gradient Boosting**
# MAGIC * Datos no estructurados (imágenes/texto) → **Redes Neuronales**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **3° PRIORIDAD: Velocidad / Simplicidad**
# MAGIC
# MAGIC ➡️ **Regresión Lineal** o **Ridge/Lasso**
# MAGIC
# MAGIC **Cuándo:**
# MAGIC * Prototipado rápido
# MAGIC * Inferencia en tiempo real (ms)
# MAGIC * Recursos computacionales limitados
# MAGIC * Baseline para comparar
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **4° PRIORIDAD: Robustez**
# MAGIC
# MAGIC ➡️ **Random Forest** o **Gradient Boosting**
# MAGIC
# MAGIC **Cuándo:**
# MAGIC * Datos con outliers y ruido
# MAGIC * Features heterogéneas (numéricas + categóricas)
# MAGIC * Mantenimiento bajo (modelo estable)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **5° PRIORIDAD: Selección de Features**
# MAGIC
# MAGIC ➡️ **Lasso** o **Elastic Net**
# MAGIC
# MAGIC **Cuándo:**
# MAGIC * Muchas features, pocas relevantes
# MAGIC * Necesitas identificar variables clave
# MAGIC * Reducir dimensionalidad para interpretabilidad
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 Estrategia Práctica
# MAGIC
# MAGIC **Workflow recomendado:**
# MAGIC
# MAGIC 1. **Baseline simple**: Regresión Lineal
# MAGIC    - Establece métrica de referencia
# MAGIC    - Identifica relaciones básicas
# MAGIC
# MAGIC 2. **Regularización**: Ridge/Lasso/Elastic Net
# MAGIC    - Si baseline overfits o hay multicolinealidad
# MAGIC
# MAGIC 3. **Ensemble**: Random Forest
# MAGIC    - Balance robustez/accuracy
# MAGIC    - Poco tuning requerido
# MAGIC
# MAGIC 4. **State-of-the-art**: Gradient Boosting
# MAGIC    - Si necesitas máxima accuracy
# MAGIC    - Requiere tuning cuidadoso
# MAGIC
# MAGIC 5. **Deep Learning**: Redes Neuronales
# MAGIC    - Solo si tienes muchos datos y recursos
# MAGIC    - O datos no estructurados
# MAGIC
# MAGIC ### ⚠️ Errores Comunes
# MAGIC
# MAGIC ❌ **NO** usar Redes Neuronales con <1K datos
# MAGIC ❌ **NO** usar SVR sin escalar features
# MAGIC ❌ **NO** usar Árbol simple sin regularizar (max_depth)
# MAGIC ❌ **NO** asumir que más complejo = mejor
# MAGIC ✅ **SÍ** empezar simple, agregar complejidad solo si es necesario
# MAGIC ✅ **SÍ** usar validación cruzada para comparar modelos
# MAGIC ✅ **SÍ** considerar trade-off interpretabilidad vs accuracy

# COMMAND ----------

# DBTITLE 1,10. Conclusiones y Próximos Pasos
# MAGIC %md
# MAGIC ## 10. Conclusiones y Próximos Pasos
# MAGIC
# MAGIC ### 🎯 Resumen Ejecutivo
# MAGIC
# MAGIC #### **Modelos Lineales**
# MAGIC * **Regresión Lineal**: Base interpretable, rápida
# MAGIC * **Regresión Polinomial**: Captura curvas simples
# MAGIC * **Ridge/Lasso/Elastic Net**: Regularización para prevenir overfitting
# MAGIC
# MAGIC **Cuándo usarlos:** Interpretabilidad crítica, baseline, velocidad
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **Modelos Basados en Árboles**
# MAGIC * **Árbol de Decisión**: Interpretable, no-lineal
# MAGIC * **Random Forest**: Robusto, ensemble paralelo
# MAGIC * **Gradient Boosting**: Máxima accuracy, ensemble secuencial
# MAGIC
# MAGIC **Cuándo usarlos:** Balance interpretabilidad/accuracy, robustez, datos tabulares
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **Modelos Avanzados**
# MAGIC * **SVR**: Kernel trick para no-linealidad, alta dimensión
# MAGIC * **Redes Neuronales**: Máxima capacidad, datos no estructurados
# MAGIC
# MAGIC **Cuándo usarlos:** Muchos datos, máxima accuracy, datos complejos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🛤️ Framework de Selección Rápida
# MAGIC
# MAGIC **Pregunta 1:** ¿Necesitas explicar cada predicción al negocio?
# MAGIC * **Sí** → Regresión Lineal o Árbol de Decisión
# MAGIC * **No** → Pregunta 2
# MAGIC
# MAGIC **Pregunta 2:** ¿Tienes >10K datos?
# MAGIC * **Sí** → Gradient Boosting o Redes Neuronales
# MAGIC * **No** → Random Forest o Ridge/Lasso
# MAGIC
# MAGIC **Pregunta 3:** ¿Tienes datos no estructurados (imágenes/texto)?
# MAGIC * **Sí** → Redes Neuronales
# MAGIC * **No** → Gradient Boosting (XGBoost/LightGBM)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Métricas de Evaluación
# MAGIC
# MAGIC **Métricas principales para regresión:**
# MAGIC
# MAGIC 1. **MAE (Mean Absolute Error)**
# MAGIC    $$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$
# MAGIC    * Interpretable (mismas unidades que $y$)
# MAGIC    * Robusto a outliers
# MAGIC
# MAGIC 2. **RMSE (Root Mean Squared Error)**
# MAGIC    $$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$
# MAGIC    * Penaliza errores grandes más
# MAGIC    * Más popular
# MAGIC
# MAGIC 3. **R² (Coeficiente de Determinación)**
# MAGIC    $$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$
# MAGIC    * Proporción de varianza explicada
# MAGIC    * 0-1 (puede ser negativo si modelo es muy malo)
# MAGIC
# MAGIC 4. **MAPE (Mean Absolute Percentage Error)**
# MAGIC    $$MAPE = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$
# MAGIC    * Error porcentual
# MAGIC    * Comparación entre datasets
# MAGIC
# MAGIC **Elegir métrica:**
# MAGIC * **Outliers problemáticos:** MAE
# MAGIC * **Penalizar errores grandes:** RMSE
# MAGIC * **Interpretabilidad para negocio:** R² o MAPE
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚀 Próximos Pasos
# MAGIC
# MAGIC #### **1. Preprocesamiento**
# MAGIC * Manejo de valores faltantes
# MAGIC * Encoding de variables categóricas
# MAGIC * Feature scaling (StandardScaler, MinMaxScaler)
# MAGIC * Feature engineering (creación de nuevas variables)
# MAGIC * Detección y manejo de outliers
# MAGIC
# MAGIC #### **2. Validación**
# MAGIC * Train/Test split (70/30 o 80/20)
# MAGIC * Cross-validation (K-fold, 5 o 10 folds)
# MAGIC * Evitar data leakage
# MAGIC * Métricas apropiadas para el problema
# MAGIC
# MAGIC #### **3. Optimización de Hiperparámetros**
# MAGIC * Grid Search: Búsqueda exhaustiva
# MAGIC * Random Search: Más eficiente
# MAGIC * Bayesian Optimization: Estado del arte
# MAGIC * Usar validación cruzada en búsqueda
# MAGIC
# MAGIC #### **4. Ensemble y Stacking**
# MAGIC * Combinar múltiples modelos
# MAGIC * Voting/Promedio de predicciones
# MAGIC * Stacking: Meta-modelo sobre predicciones
# MAGIC
# MAGIC #### **5. Deployment**
# MAGIC * Persistencia del modelo (pickle, joblib)
# MAGIC * API para servir predicciones
# MAGIC * Monitoreo de performance en producción
# MAGIC * Re-entrenamiento periódico
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📚 Recursos Adicionales
# MAGIC
# MAGIC **Librerías Python:**
# MAGIC * `scikit-learn`: Todos los modelos tradicionales
# MAGIC * `xgboost`, `lightgbm`, `catboost`: Gradient Boosting
# MAGIC * `tensorflow`, `pytorch`: Deep Learning
# MAGIC * `statsmodels`: Modelos estadísticos clásicos
# MAGIC
# MAGIC **Práctica:**
# MAGIC * **Kaggle**: Competencias y datasets
# MAGIC * **UCI ML Repository**: Datasets clásicos
# MAGIC * **Papers with Code**: Benchmarks y código
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Recuerda
# MAGIC
# MAGIC > **"El mejor modelo no es el más complejo, sino el que mejor resuelve tu problema de negocio considerando interpretabilidad, accuracy, velocidad y mantenimiento."**
# MAGIC
# MAGIC ✅ Empieza simple, agrega complejidad solo si es necesario
# MAGIC ✅ Valida con datos que el modelo nunca vio
# MAGIC ✅ Interpreta y comunica resultados al negocio
# MAGIC ✅ Monitorea performance en producción
# MAGIC ❌ No uses Deep Learning para todo
# MAGIC ❌ No ignores la interpretabilidad sin razón
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ¡Éxito en tus proyectos de Machine Learning! 🚀