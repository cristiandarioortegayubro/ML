# Databricks notebook source
# DBTITLE 1,Teoría de Regresión
# MAGIC %md
# MAGIC # Teoría de Regresión
# MAGIC
# MAGIC ## 1. Introducción
# MAGIC
# MAGIC La **Regresión** es una técnica de aprendizaje supervisado para predecir valores continuos.
# MAGIC
# MAGIC ### Definición Formal
# MAGIC
# MAGIC Dado un conjunto de entrenamiento $\\mathcal{D} = \\{(\\mathbf{x}_i, y_i)\\}_{i=1}^{n}$ donde:
# MAGIC - $\\mathbf{x}_i \\in \\mathbb{R}^d$ es el vector de características
# MAGIC - $y_i \\in \\mathbb{R}$ es el valor objetivo continuo
# MAGIC
# MAGIC El objetivo es aprender una función:
# MAGIC
# MAGIC $$f: \\mathbb{R}^d \\rightarrow \\mathbb{R}$$
# MAGIC
# MAGIC Tal que $f(\\mathbf{x}_i) \\approx y_i$ para nuevos datos.
# MAGIC
# MAGIC ### Ejemplos de Aplicación
# MAGIC
# MAGIC * **Precio de viviendas**: $f$(tamaño, ubicación, año) = precio
# MAGIC * **Ventas**: $f$(publicidad, temporada) = ingresos
# MAGIC * **Temperatura**: $f$(hora, mes, ubicación) = temperatura
# MAGIC * **Demanda**: $f$(precio, competencia) = unidades vendidas

# COMMAND ----------

# DBTITLE 1,Regresión Lineal Simple
# MAGIC %md
# MAGIC ## 2. Regresión Lineal Simple
# MAGIC
# MAGIC ### Modelo
# MAGIC
# MAGIC Una variable independiente $x$ predice una variable dependiente $y$:
# MAGIC
# MAGIC $$y = \\beta_0 + \\beta_1 x + \\epsilon$$
# MAGIC
# MAGIC Donde:
# MAGIC - $\\beta_0$: Intercepto (valor cuando $x=0$)
# MAGIC - $\\beta_1$: Pendiente (cambio en $y$ por unidad de $x$)
# MAGIC - $\\epsilon$: Error aleatorio
# MAGIC
# MAGIC ### Función de Costo: Error Cuadrático Medio (MSE)
# MAGIC
# MAGIC $$J(\\beta_0, \\beta_1) = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\beta_0 - \\beta_1 x_i)^2$$
# MAGIC
# MAGIC ### Solución: Mínimos Cuadrados
# MAGIC
# MAGIC **Derivadas parciales = 0:**
# MAGIC
# MAGIC $$\\beta_1 = \\frac{\\sum_{i=1}^{n}(x_i - \\bar{x})(y_i - \\bar{y})}{\\sum_{i=1}^{n}(x_i - \\bar{x})^2} = \\frac{\\text{Cov}(x,y)}{\\text{Var}(x)}$$
# MAGIC
# MAGIC $$\\beta_0 = \\bar{y} - \\beta_1 \\bar{x}$$
# MAGIC
# MAGIC ### Interpretación
# MAGIC
# MAGIC * $\\beta_1 > 0$: Relación positiva
# MAGIC * $\\beta_1 < 0$: Relación negativa
# MAGIC * $\\beta_1 = 0$: Sin relación lineal

# COMMAND ----------

# DBTITLE 1,Regresión Lineal Múltiple
# MAGIC %md
# MAGIC ## 3. Regresión Lineal Múltiple
# MAGIC
# MAGIC ### Modelo
# MAGIC
# MAGIC Múltiples variables independientes:
# MAGIC
# MAGIC $$y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\cdots + \\beta_d x_d + \\epsilon$$
# MAGIC
# MAGIC **Forma matricial:**
# MAGIC
# MAGIC $$\\mathbf{y} = \\mathbf{X}\\boldsymbol{\\beta} + \\boldsymbol{\\epsilon}$$
# MAGIC
# MAGIC Donde:
# MAGIC - $\\mathbf{y} = [y_1, y_2, ..., y_n]^T$ (vector $n \\times 1$)
# MAGIC - $\\mathbf{X} = [\\mathbf{1}, \\mathbf{x}_1, \\mathbf{x}_2, ..., \\mathbf{x}_d]$ (matriz $n \\times (d+1)$)
# MAGIC - $\\boldsymbol{\\beta} = [\\beta_0, \\beta_1, ..., \\beta_d]^T$ (parámetros)
# MAGIC
# MAGIC ### Solución de Mínimos Cuadrados
# MAGIC
# MAGIC $$\\boldsymbol{\\hat{\\beta}} = (\\mathbf{X}^T\\mathbf{X})^{-1}\\mathbf{X}^T\\mathbf{y}$$
# MAGIC
# MAGIC **Condiciones:**
# MAGIC - $\\mathbf{X}^T\\mathbf{X}$ debe ser invertible
# MAGIC - No multicolinealidad severa
# MAGIC - $n > d$ (más datos que variables)
# MAGIC
# MAGIC ### Predicción
# MAGIC
# MAGIC $$\\hat{y} = \\mathbf{X}\\boldsymbol{\\hat{\\beta}}$$

# COMMAND ----------

# DBTITLE 1,Supuestos y Métricas
# MAGIC %md
# MAGIC ## 4. Supuestos del Modelo Lineal
# MAGIC
# MAGIC ### Supuestos Clave
# MAGIC
# MAGIC 1. **Linealidad**: Relación lineal entre $x$ e $y$
# MAGIC 2. **Independencia**: Observaciones independientes
# MAGIC 3. **Homoscedasticidad**: Varianza constante de errores
# MAGIC 4. **Normalidad**: Errores $\\epsilon \\sim N(0, \\sigma^2)$
# MAGIC 5. **No multicolinealidad**: Variables independientes no correlacionadas
# MAGIC
# MAGIC ### Métricas de Evaluación
# MAGIC
# MAGIC **Coeficiente de Determinación ($R^2$):**
# MAGIC
# MAGIC $$R^2 = 1 - \\frac{SS_{res}}{SS_{tot}} = 1 - \\frac{\\sum(y_i - \\hat{y}_i)^2}{\\sum(y_i - \\bar{y})^2}$$
# MAGIC
# MAGIC * $R^2 = 1$: Ajuste perfecto
# MAGIC * $R^2 = 0$: Modelo no mejor que la media
# MAGIC * $0 \\leq R^2 \\leq 1$
# MAGIC
# MAGIC **$R^2$ Ajustado** (penaliza complejidad):
# MAGIC
# MAGIC $$R^2_{adj} = 1 - \\frac{(1-R^2)(n-1)}{n-d-1}$$
# MAGIC
# MAGIC **RMSE** (Error Cuadrático Medio Raíz):
# MAGIC
# MAGIC $$RMSE = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2}$$

# COMMAND ----------

# DBTITLE 1,Regularización
# MAGIC %md
# MAGIC ## 5. Regularización
# MAGIC
# MAGIC ### Problema del Overfitting
# MAGIC
# MAGIC Modelos complejos memorizan ruido.
# MAGIC
# MAGIC ### Ridge Regression (L2)
# MAGIC
# MAGIC Añade penalización L2 a la función de costo:
# MAGIC
# MAGIC $$J(\\boldsymbol{\\beta}) = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\mathbf{x}_i^T\\boldsymbol{\\beta})^2 + \\lambda\\sum_{j=1}^{d}\\beta_j^2$$
# MAGIC
# MAGIC $$J(\\boldsymbol{\\beta}) = MSE + \\lambda ||\\boldsymbol{\\beta}||_2^2$$
# MAGIC
# MAGIC **Efecto:** Reduce magnitud de coeficientes, no los elimina.
# MAGIC
# MAGIC ### Lasso Regression (L1)
# MAGIC
# MAGIC Penalización L1:
# MAGIC
# MAGIC $$J(\\boldsymbol{\\beta}) = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\mathbf{x}_i^T\\boldsymbol{\\beta})^2 + \\lambda\\sum_{j=1}^{d}|\\beta_j|$$
# MAGIC
# MAGIC **Efecto:** Puede forzar $\\beta_j = 0$ (selección de características).
# MAGIC
# MAGIC ### Elastic Net
# MAGIC
# MAGIC Combina L1 y L2:
# MAGIC
# MAGIC $$J(\\boldsymbol{\\beta}) = MSE + \\lambda_1 ||\\boldsymbol{\\beta}||_1 + \\lambda_2 ||\\boldsymbol{\\beta}||_2^2$$
# MAGIC
# MAGIC | Método | Penalización | Selección Features | Estabilidad |
# MAGIC |---------|--------------|---------------------|-------------|
# MAGIC | Ridge | L2 | No | Alta |
# MAGIC | Lasso | L1 | Sí | Media |
# MAGIC | Elastic Net | L1 + L2 | Sí | Alta |

# COMMAND ----------

# DBTITLE 1,Regresión No Lineal
# MAGIC %md
# MAGIC ## 6. Regresión No Lineal
# MAGIC
# MAGIC ### Regresión Polinomial
# MAGIC
# MAGIC Transforma características a potencias:
# MAGIC
# MAGIC $$y = \\beta_0 + \\beta_1 x + \\beta_2 x^2 + \\beta_3 x^3 + \\cdots + \\epsilon$$
# MAGIC
# MAGIC **Ejemplo grado 2:**
# MAGIC
# MAGIC $$\\mathbf{x} = [x_1, x_2] \\rightarrow [1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]$$
# MAGIC
# MAGIC ### Árboles de Decisión para Regresión
# MAGIC
# MAGIC Particionan espacio y asignan valor promedio por región.
# MAGIC
# MAGIC **Criterio de división:** MSE o MAE
# MAGIC
# MAGIC ### Random Forest Regressor
# MAGIC
# MAGIC Ensemble de árboles:
# MAGIC
# MAGIC $$\\hat{y} = \\frac{1}{T}\\sum_{t=1}^{T} f_t(\\mathbf{x})$$
# MAGIC
# MAGIC ### Gradient Boosting Regressor
# MAGIC
# MAGIC Combina modelos débiles secuencialmente:
# MAGIC
# MAGIC $$f_m(\\mathbf{x}) = f_{m-1}(\\mathbf{x}) + \\alpha h_m(\\mathbf{x})$$
# MAGIC
# MAGIC Donde $h_m$ corrige errores de $f_{m-1}$.

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## 7. Conclusiones
# MAGIC
# MAGIC ### Resumen
# MAGIC
# MAGIC 1. **Regresión Lineal**: Base interpretable
# MAGIC 2. **Regularización**: Control de overfitting (Ridge, Lasso)
# MAGIC 3. **Métricas**: $R^2$, RMSE, MAE
# MAGIC 4. **No lineal**: Polinomial, árboles, ensembles
# MAGIC
# MAGIC ### Cuándo Usar Cada Método
# MAGIC
# MAGIC * **Lineal simple/múltiple**: Relaciones lineales, interpretabilidad
# MAGIC * **Ridge**: Multicolinealidad, mantener todas las variables
# MAGIC * **Lasso**: Selección automática de features
# MAGIC * **Árboles/RF/GB**: No linealidad, interacciones complejas

# COMMAND ----------

