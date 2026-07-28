# Databricks notebook source
# DBTITLE 1,Título y Objetivos
# MAGIC %md
# MAGIC # Teoría de Regresión: Matemáticas y Fundamentos
# MAGIC
# MAGIC ## 🎯 Objetivos
# MAGIC
# MAGIC * **Matemáticas de regresión lineal**: Mínimos cuadrados ordinarios (OLS), derivación
# MAGIC * **Supuestos y diagnósticos**: Linealidad, homocedasticidad, normalidad
# MAGIC * **Interpretación**: Coeficientes, R², significancia
# MAGIC * **Comparación**: Regresión lineal vs árboles de regresión
# MAGIC
# MAGIC ### Contenido
# MAGIC
# MAGIC 1. Regresión lineal: Derivación matemática
# MAGIC 2. Supuestos del modelo
# MAGIC 3. Inferencia estadística
# MAGIC 4. Diagnósticos y validación
# MAGIC 5. Árboles de regresión vs lineal

# COMMAND ----------

# DBTITLE 1,Regresión Lineal Simple
# MAGIC %md
# MAGIC ## 1. Regresión Lineal Simple
# MAGIC
# MAGIC ### 1.1 Modelo
# MAGIC
# MAGIC $$y_i = \beta_0 + \beta_1 x_i + \epsilon_i$$
# MAGIC
# MAGIC donde:
# MAGIC * $y_i$: Variable dependiente (respuesta)
# MAGIC * $x_i$: Variable independiente (predictor)
# MAGIC * $\beta_0$: Intercepto
# MAGIC * $\beta_1$: Pendiente
# MAGIC * $\epsilon_i$: Error aleatorio
# MAGIC
# MAGIC ### 1.2 Mínimos Cuadrados Ordinarios (OLS)
# MAGIC
# MAGIC **Objetivo:** Minimizar la suma de errores cuadrados
# MAGIC
# MAGIC $$\min_{\beta_0, \beta_1} SSE = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2$$
# MAGIC
# MAGIC **Derivadas parciales:**
# MAGIC
# MAGIC $$\frac{\partial SSE}{\partial \beta_0} = -2\sum(y_i - \beta_0 - \beta_1 x_i) = 0$$
# MAGIC
# MAGIC $$\frac{\partial SSE}{\partial \beta_1} = -2\sum x_i(y_i - \beta_0 - \beta_1 x_i) = 0$$
# MAGIC
# MAGIC **Solución:**
# MAGIC
# MAGIC $$\hat{\beta}_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2} = \frac{Cov(x,y)}{Var(x)}$$
# MAGIC
# MAGIC $$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

# COMMAND ----------

# DBTITLE 1,Regresión Lineal Múltiple
# MAGIC %md
# MAGIC ## 2. Regresión Lineal Múltiple
# MAGIC
# MAGIC ### 2.1 Modelo Matricial
# MAGIC
# MAGIC $$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$
# MAGIC
# MAGIC donde:
# MAGIC * $\mathbf{y} \in \mathbb{R}^{n \times 1}$: Vector de respuestas
# MAGIC * $\mathbf{X} \in \mathbb{R}^{n \times (p+1)}$: Matriz de diseño
# MAGIC * $\boldsymbol{\beta} \in \mathbb{R}^{(p+1) \times 1}$: Vector de coeficientes
# MAGIC * $\boldsymbol{\epsilon} \in \mathbb{R}^{n \times 1}$: Vector de errores
# MAGIC
# MAGIC ### 2.2 Solución de Forma Cerrada
# MAGIC
# MAGIC **Ecuaciones normales:**
# MAGIC $$\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$
# MAGIC
# MAGIC **Solución:**
# MAGIC $$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$
# MAGIC
# MAGIC **Predicciones:**
# MAGIC $$\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} = \mathbf{H}\mathbf{y}$$
# MAGIC
# MAGIC donde $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ es la matriz "hat".
# MAGIC
# MAGIC ### 2.3 Interpretación de Coeficientes
# MAGIC
# MAGIC $$\hat{\beta}_j = \frac{\partial \hat{y}}{\partial x_j}$$
# MAGIC
# MAGIC **Ejemplo:**
# MAGIC $$\text{Precio} = 50000 + 1000 \cdot \text{m}^2 + 5000 \cdot \text{habitaciones}$$
# MAGIC
# MAGIC * $\beta_0 = 50000$: Precio base
# MAGIC * $\beta_1 = 1000$: Cada m² adicional aumenta precio en $1000 (manteniendo habitaciones constante)
# MAGIC * $\beta_2 = 5000$: Cada habitación adicional aumenta precio en $5000 (manteniendo m² constante)

# COMMAND ----------

# DBTITLE 1,Supuestos del Modelo
# MAGIC %md
# MAGIC ## 3. Supuestos del Modelo Lineal
# MAGIC
# MAGIC ### 3.1 Los 5 Supuestos Clave
# MAGIC
# MAGIC **1. Linealidad**
# MAGIC * $\mathbb{E}[y|x] = \beta_0 + \beta_1 x$
# MAGIC * **Diagnóstico:** Gráfico residuos vs predichos (debe ser aleatorio)
# MAGIC
# MAGIC **2. Independencia de errores**
# MAGIC * $Cov(\epsilon_i, \epsilon_j) = 0$ para $i \neq j$
# MAGIC * **Problema si falla:** Series temporales, datos agrupados
# MAGIC
# MAGIC **3. Homocedasticidad (varianza constante)**
# MAGIC * $Var(\epsilon_i) = \sigma^2$ para todo $i$
# MAGIC * **Diagnóstico:** Gráfico residuos vs predichos (amplitud constante)
# MAGIC * **Si falla:** Heterocedasticidad → usar errores robustos o transformación
# MAGIC
# MAGIC **4. Normalidad de errores**
# MAGIC * $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$
# MAGIC * **Diagnóstico:** QQ-plot
# MAGIC * **Nota:** Menos crítico con $n$ grande (TCL)
# MAGIC
# MAGIC **5. No multicolinealidad**
# MAGIC * Predictores no deben estar altamente correlacionados
# MAGIC * **Diagnóstico:** VIF (Variance Inflation Factor)
# MAGIC * **Si falla:** Inestabilidad de coeficientes, errores estándar grandes
# MAGIC
# MAGIC ### 3.2 Diagnósticos Gráficos
# MAGIC
# MAGIC ```
# MAGIC 1. Residuos vs Predichos: Linealidad, homocedasticidad
# MAGIC 2. QQ-Plot: Normalidad
# MAGIC 3. Scale-Location: Homocedasticidad
# MAGIC 4. Residuos vs Leverage: Valores influyentes
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Métricas e Inferencia
# MAGIC %md
# MAGIC ## 4. Métricas e Inferencia Estadística
# MAGIC
# MAGIC ### 4.1 Métricas de Bondad de Ajuste
# MAGIC
# MAGIC **R² (Coeficiente de determinación):**
# MAGIC $$R^2 = 1 - \frac{SSE}{SST} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$
# MAGIC
# MAGIC * $R^2 \in [0, 1]$
# MAGIC * **Interpretación:** % de varianza de $y$ explicada por el modelo
# MAGIC * **Problema:** Siempre aumenta con más predictores
# MAGIC
# MAGIC **R² ajustado:**
# MAGIC $$R^2_{adj} = 1 - \frac{SSE/(n-p-1)}{SST/(n-1)}$$
# MAGIC
# MAGIC * Penaliza por número de predictores
# MAGIC * Usar para comparar modelos
# MAGIC
# MAGIC **RMSE (Root Mean Squared Error):**
# MAGIC $$RMSE = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$$
# MAGIC
# MAGIC **MAE (Mean Absolute Error):**
# MAGIC $$MAE = \frac{1}{n}\sum|y_i - \hat{y}_i|$$
# MAGIC
# MAGIC ### 4.2 Inferencia sobre Coeficientes
# MAGIC
# MAGIC **Error estándar de $\hat{\beta}_j$:**
# MAGIC $$SE(\hat{\beta}_j) = \sqrt{\hat{\sigma}^2 [(\mathbf{X}^T\mathbf{X})^{-1}]_{jj}}$$
# MAGIC
# MAGIC donde $\hat{\sigma}^2 = \frac{SSE}{n-p-1}$ (varianza residual)
# MAGIC
# MAGIC **Test t:**
# MAGIC $$t = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)} \sim t_{n-p-1}$$
# MAGIC
# MAGIC **Hipótesis:**
# MAGIC * $H_0: \beta_j = 0$ (no hay efecto)
# MAGIC * $H_1: \beta_j \neq 0$ (hay efecto)
# MAGIC
# MAGIC **Intervalo de confianza (95%):**
# MAGIC $$\hat{\beta}_j \pm t_{0.025, n-p-1} \cdot SE(\hat{\beta}_j)$$

# COMMAND ----------

# DBTITLE 1,Árboles de Regresión vs Lineal
# MAGIC %md
# MAGIC ## 5. Comparación: Regresión Lineal vs Árboles
# MAGIC
# MAGIC ### 5.1 Tabla Comparativa
# MAGIC
# MAGIC | Aspecto | Regresión Lineal | Árboles de Regresión |
# MAGIC |---|---|---|
# MAGIC | **Supuestos** | Muchos (linealidad, etc.) | Ninguno |
# MAGIC | **Interpretabilidad** | ⭐⭐⭐ Coeficientes claros | ⭐⭐ Reglas IF-THEN |
# MAGIC | **Relaciones no lineales** | ❌ Requiere transformación | ✅ Automático |
# MAGIC | **Interacciones** | ❌ Manual (crear términos) | ✅ Automático |
# MAGIC | **Extrapolación** | ✅ Razonable | ❌ Pobre |
# MAGIC | **Continuidad** | ✅ Suave | ❌ Escalonada |
# MAGIC | **Overfitting** | Bajo | Alto (sin poda) |
# MAGIC | **Eficiencia con $n$ pequeño** | ✅ Bien | ❌ Mal |
# MAGIC
# MAGIC ### 5.2 ¿Cuándo usar qué?
# MAGIC
# MAGIC **Usa Regresión Lineal cuando:**
# MAGIC * Relación es aproximadamente lineal
# MAGIC * Necesitas inferencia estadística (p-valores, IC)
# MAGIC * Extrapolación es importante
# MAGIC * Interpretabilidad de coeficientes es crítica
# MAGIC * Tienes pocos datos (< 100)
# MAGIC
# MAGIC **Usa Árboles de Regresión cuando:**
# MAGIC * Relaciones son claramente no lineales
# MAGIC * Muchas interacciones entre variables
# MAGIC * No cumples supuestos de regresión lineal
# MAGIC * Datos mixtos (categóricos + numéricos)
# MAGIC * Solo te importa predicción, no inferencia
# MAGIC
# MAGIC **Usa Ensemble (Random Forest, XGBoost) cuando:**
# MAGIC * Accuracy es prioridad #1
# MAGIC * Tienes muchos datos (> 1000)
# MAGIC * Relaciones son muy complejas
# MAGIC
# MAGIC ### 5.3 Ejemplo: Relación Cuadrática
# MAGIC
# MAGIC **Datos:** $y = x^2 + \epsilon$
# MAGIC
# MAGIC **Regresión Lineal Simple:**
# MAGIC $$\hat{y} = \beta_0 + \beta_1 x$$
# MAGIC * R² bajo (~0.3)
# MAGIC * Residuos muestran patrón cuadrático
# MAGIC
# MAGIC **Regresión Lineal con Transformación:**
# MAGIC $$\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2$$
# MAGIC * R² alto (~0.95)
# MAGIC * Requiere conocer forma funcional
# MAGIC
# MAGIC **Árbol de Regresión:**
# MAGIC * Divide en múltiples regiones
# MAGIC * R² alto (~0.90)
# MAGIC * No requiere especificar forma
# MAGIC * Predicción escalonada (no suave)

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## 6. Conclusiones
# MAGIC
# MAGIC ### 📚 Resumen
# MAGIC
# MAGIC 1. **OLS:** Minimiza SSE → ecuaciones normales → $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$
# MAGIC 2. **Supuestos:** Linealidad, homocedasticidad, normalidad, independencia, no multicolinealidad
# MAGIC 3. **Inferencia:** Tests t, intervalos de confianza para coeficientes
# MAGIC 4. **Métricas:** R², RMSE, MAE
# MAGIC 5. **Trade-off:** Lineal (interpretable, supuestos) vs Árboles (flexible, sin supuestos)
# MAGIC
# MAGIC ### 🎯 Siguientes Notebooks
# MAGIC
# MAGIC **Prácticos:**
# MAGIC * `Arbol_Decision_Regresion.ipynb` - Árboles aplicados a precios
# MAGIC * `Regresion_Lineal_Multiple.ipynb` - Implementación OLS
# MAGIC
# MAGIC ### 📖 Referencias
# MAGIC
# MAGIC * Freedman et al. - "Statistics" (2007)
# MAGIC * James et al. - "An Introduction to Statistical Learning" (2021), Cap. 3
# MAGIC * Montgomery et al. - "Introduction to Linear Regression Analysis" (2012)

# COMMAND ----------

