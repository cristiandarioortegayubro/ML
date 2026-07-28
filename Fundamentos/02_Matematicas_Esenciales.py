# Databricks notebook source
# DBTITLE 1,Título y Objetivos
# MAGIC %md
# MAGIC # Matemáticas Esenciales para Machine Learning
# MAGIC
# MAGIC ## 🎯 Objetivos de Aprendizaje
# MAGIC
# MAGIC Este notebook proporciona los fundamentos matemáticos rigurosos necesarios para entender Machine Learning:
# MAGIC
# MAGIC * **Álgebra lineal**: Vectores, matrices, operaciones, espacios vectoriales
# MAGIC * **Cálculo multivariable**: Derivadas parciales, gradientes, optimización
# MAGIC * **Probabilidad y estadística**: Distribuciones, esperanza, varianza, inferencia
# MAGIC * **Optimización**: Métodos de descenso de gradiente
# MAGIC
# MAGIC ### Prerrequisitos
# MAGIC
# MAGIC * Álgebra básica
# MAGIC * Trigonometría
# MAGIC * Conceptos de función y derivada
# MAGIC
# MAGIC ### Contenido
# MAGIC
# MAGIC 1. Álgebra lineal para ML
# MAGIC 2. Cálculo multivariable
# MAGIC 3. Probabilidad y estadística
# MAGIC 4. Teoría de optimización
# MAGIC 5. Aplicaciones en ML

# COMMAND ----------

# DBTITLE 1,Álgebra Lineal - Vectores
# MAGIC %md
# MAGIC ## 1. Álgebra Lineal para Machine Learning
# MAGIC
# MAGIC ### 1.1 Vectores
# MAGIC
# MAGIC **Definición:** Un vector es una lista ordenada de números.
# MAGIC
# MAGIC **Notación:**
# MAGIC $$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n$$
# MAGIC
# MAGIC **En ML:** Cada observación es un vector de features.
# MAGIC
# MAGIC **Ejemplo:**
# MAGIC ```
# MAGIC Cliente 1: [antigüedad=24, edad=35, gasto_mensual=89.50]
# MAGIC          → x = [24, 35, 89.50] ∈ ℝ³
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Operaciones con Vectores
# MAGIC
# MAGIC **1. Suma de vectores:**
# MAGIC $$\mathbf{x} + \mathbf{y} = \begin{bmatrix} x_1 + y_1 \\ x_2 + y_2 \\ \vdots \\ x_n + y_n \end{bmatrix}$$
# MAGIC
# MAGIC **2. Multiplicación por escalar:**
# MAGIC $$\alpha \mathbf{x} = \begin{bmatrix} \alpha x_1 \\ \alpha x_2 \\ \vdots \\ \alpha x_n \end{bmatrix}$$
# MAGIC
# MAGIC **3. Producto punto (dot product):**
# MAGIC $$\mathbf{x}^T \mathbf{y} = \sum_{i=1}^{n} x_i y_i = x_1 y_1 + x_2 y_2 + \cdots + x_n y_n$$
# MAGIC
# MAGIC **Propiedad importante:**
# MAGIC $$\mathbf{x}^T \mathbf{y} = |\mathbf{x}| \cdot |\mathbf{y}| \cdot \cos(\theta)$$
# MAGIC
# MAGIC donde $\theta$ es el ángulo entre los vectores.
# MAGIC
# MAGIC **Aplicación en ML:** Similitud de coseno para medir distancia entre observaciones.
# MAGIC
# MAGIC **4. Norma (magnitud):**
# MAGIC
# MAGIC **Norma L2 (Euclidiana):**
# MAGIC $$||\mathbf{x}||_2 = \sqrt{\sum_{i=1}^{n} x_i^2} = \sqrt{\mathbf{x}^T \mathbf{x}}$$
# MAGIC
# MAGIC **Norma L1 (Manhattan):**
# MAGIC $$||\mathbf{x}||_1 = \sum_{i=1}^{n} |x_i|$$
# MAGIC
# MAGIC **Aplicación en ML:**
# MAGIC * L2: Regularización Ridge
# MAGIC * L1: Regularización Lasso (promueve sparsity)

# COMMAND ----------

# DBTITLE 1,Álgebra Lineal - Matrices
# MAGIC %md
# MAGIC ### 1.2 Matrices
# MAGIC
# MAGIC **Definición:** Una matriz es un arreglo rectangular de números.
# MAGIC
# MAGIC **Notación:**
# MAGIC $$\mathbf{X} = \begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1p} \\ x_{21} & x_{22} & \cdots & x_{2p} \\ \vdots & \vdots & \ddots & \vdots \\ x_{n1} & x_{n2} & \cdots & x_{np} \end{bmatrix} \in \mathbb{R}^{n \times p}$$
# MAGIC
# MAGIC **En ML:** Dataset completo
# MAGIC * $n$ filas = observaciones
# MAGIC * $p$ columnas = features
# MAGIC
# MAGIC **Ejemplo:**
# MAGIC ```
# MAGIC         antiguedad  edad  gasto
# MAGIC Cliente1    24      35    89.50
# MAGIC Cliente2    12      28    45.20
# MAGIC Cliente3    36      42    120.00
# MAGIC
# MAGIC X = [[24, 35, 89.50],
# MAGIC      [12, 28, 45.20],
# MAGIC      [36, 42, 120.00]]  ∈ ℝ³ˣ³
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Operaciones con Matrices
# MAGIC
# MAGIC **1. Multiplicación matriz-vector:**
# MAGIC
# MAGIC $$\mathbf{X} \boldsymbol{\beta} = \begin{bmatrix} \mathbf{x}_1^T \boldsymbol{\beta} \\ \mathbf{x}_2^T \boldsymbol{\beta} \\ \vdots \\ \mathbf{x}_n^T \boldsymbol{\beta} \end{bmatrix}$$
# MAGIC
# MAGIC **Aplicación en ML:** Predicción en regresión lineal
# MAGIC $$\hat{\mathbf{y}} = \mathbf{X} \boldsymbol{\beta}$$
# MAGIC
# MAGIC **2. Multiplicación matriz-matriz:**
# MAGIC
# MAGIC Si $\mathbf{A} \in \mathbb{R}^{n \times m}$ y $\mathbf{B} \in \mathbb{R}^{m \times p}$, entonces:
# MAGIC
# MAGIC $$[\mathbf{C}]_{ij} = \sum_{k=1}^{m} a_{ik} b_{kj}$$
# MAGIC
# MAGIC **3. Transpuesta:**
# MAGIC $$[\mathbf{X}^T]_{ij} = [\mathbf{X}]_{ji}$$
# MAGIC
# MAGIC **Propiedades:**
# MAGIC * $(\mathbf{X}^T)^T = \mathbf{X}$
# MAGIC * $(\mathbf{X} \mathbf{Y})^T = \mathbf{Y}^T \mathbf{X}^T$
# MAGIC
# MAGIC **4. Inversa:**
# MAGIC
# MAGIC Si $\mathbf{X}$ es cuadrada y no singular:
# MAGIC $$\mathbf{X} \mathbf{X}^{-1} = \mathbf{X}^{-1} \mathbf{X} = \mathbf{I}$$
# MAGIC
# MAGIC **Aplicación en ML:** Solución de forma cerrada en regresión lineal
# MAGIC $$\boldsymbol{\hat{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 1.3 Conceptos Avanzados
# MAGIC
# MAGIC **Autovalores y autovectores:**
# MAGIC
# MAGIC Dado $\mathbf{A} \in \mathbb{R}^{n \times n}$:
# MAGIC $$\mathbf{A} \mathbf{v} = \lambda \mathbf{v}$$
# MAGIC
# MAGIC * $\lambda$: autovalor (eigenvalue)
# MAGIC * $\mathbf{v}$: autovector (eigenvector)
# MAGIC
# MAGIC **Aplicación en ML:**
# MAGIC * PCA (Principal Component Analysis)
# MAGIC * Análisis de covarianza
# MAGIC
# MAGIC **Descomposición en valores singulares (SVD):**
# MAGIC
# MAGIC $$\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$
# MAGIC
# MAGIC **Aplicación en ML:**
# MAGIC * Reducción de dimensionalidad
# MAGIC * Sistemas de recomendación
# MAGIC * Compresión de datos

# COMMAND ----------

# DBTITLE 1,Cálculo Multivariable
# MAGIC %md
# MAGIC ## 2. Cálculo Multivariable
# MAGIC
# MAGIC ### 2.1 Derivadas Parciales
# MAGIC
# MAGIC **Función de una variable:**
# MAGIC $$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$
# MAGIC
# MAGIC **Función de múltiples variables:**
# MAGIC
# MAGIC Si $f(x_1, x_2, ..., x_n)$, la derivada parcial con respecto a $x_i$ es:
# MAGIC
# MAGIC $$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, ..., x_i+h, ..., x_n) - f(x_1, ..., x_i, ..., x_n)}{h}$$
# MAGIC
# MAGIC **Ejemplo:** Error cuadrático en regresión lineal
# MAGIC
# MAGIC $$L(\beta_0, \beta_1) = \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2$$
# MAGIC
# MAGIC **Derivadas parciales:**
# MAGIC $$\frac{\partial L}{\partial \beta_0} = -2 \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)$$
# MAGIC
# MAGIC $$\frac{\partial L}{\partial \beta_1} = -2 \sum_{i=1}^{n} x_i(y_i - \beta_0 - \beta_1 x_i)$$
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2.2 Gradiente
# MAGIC
# MAGIC **Definición:** El gradiente es el vector de todas las derivadas parciales.
# MAGIC
# MAGIC $$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$
# MAGIC
# MAGIC **Propiedades:**
# MAGIC 1. **Dirección de máximo crecimiento**: $\nabla f$ apunta hacia donde $f$ crece más rápido
# MAGIC 2. **Ortogonal a curvas de nivel**: $\nabla f$ es perpendicular a $f(\mathbf{x}) = c$
# MAGIC
# MAGIC **Aplicación en ML:** Descenso de gradiente para minimizar funciones de pérdida
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2.3 Matriz Hessiana
# MAGIC
# MAGIC **Definición:** Matriz de segundas derivadas parciales.
# MAGIC
# MAGIC $$\mathbf{H}_f = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}$$
# MAGIC
# MAGIC **Aplicación en ML:**
# MAGIC * Métodos de optimización de segundo orden (Newton-Raphson)
# MAGIC * Análisis de convergencia
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2.4 Regla de la Cadena
# MAGIC
# MAGIC **Una variable:**
# MAGIC $$\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$$
# MAGIC
# MAGIC **Múltiples variables:**
# MAGIC
# MAGIC Si $z = f(y)$ y $y = g(x)$:
# MAGIC $$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}$$
# MAGIC
# MAGIC **Aplicación en ML:** Backpropagation en redes neuronales
# MAGIC
# MAGIC **Ejemplo:** Red neuronal simple
# MAGIC
# MAGIC $$z = f(w_2 \cdot g(w_1 x))$$
# MAGIC
# MAGIC $$\frac{\partial z}{\partial w_1} = \frac{\partial z}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial w_1}$$
# MAGIC
# MAGIC donde $h_1 = w_1 x$, $h_2 = g(h_1)$

# COMMAND ----------

# DBTITLE 1,Probabilidad y Estadística
# MAGIC %md
# MAGIC ## 3. Probabilidad y Estadística
# MAGIC
# MAGIC ### 3.1 Conceptos Fundamentales
# MAGIC
# MAGIC **Espacio muestral ($\Omega$):** Conjunto de todos los resultados posibles
# MAGIC
# MAGIC **Evento ($A$):** Subconjunto del espacio muestral
# MAGIC
# MAGIC **Probabilidad:** Función $P: \mathcal{F} \rightarrow [0, 1]$ que satisface:
# MAGIC
# MAGIC 1. **No negatividad**: $P(A) \geq 0$ para todo $A$
# MAGIC 2. **Normalización**: $P(\Omega) = 1$
# MAGIC 3. **Aditividad**: Si $A \cap B = \emptyset$, entonces $P(A \cup B) = P(A) + P(B)$
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3.2 Variables Aleatorias
# MAGIC
# MAGIC **Variable aleatoria discreta:**
# MAGIC
# MAGIC **Función de masa de probabilidad (PMF):**
# MAGIC $$P(X = x) = p(x)$$
# MAGIC
# MAGIC **Ejemplo:** Lanzamiento de dado
# MAGIC $$P(X = k) = \frac{1}{6}, \quad k \in \{1, 2, 3, 4, 5, 6\}$$
# MAGIC
# MAGIC **Variable aleatoria continua:**
# MAGIC
# MAGIC **Función de densidad de probabilidad (PDF):**
# MAGIC $$P(a \leq X \leq b) = \int_a^b f(x) dx$$
# MAGIC
# MAGIC **Ejemplo:** Distribución Normal
# MAGIC $$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3.3 Esperanza y Varianza
# MAGIC
# MAGIC **Esperanza (Media):**
# MAGIC
# MAGIC **Discreta:**
# MAGIC $$\mathbb{E}[X] = \sum_{x} x \cdot P(X = x)$$
# MAGIC
# MAGIC **Continua:**
# MAGIC $$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f(x) dx$$
# MAGIC
# MAGIC **Propiedades:**
# MAGIC * $\mathbb{E}[aX + b] = a\mathbb{E}[X] + b$ (linealidad)
# MAGIC * $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$
# MAGIC
# MAGIC **Varianza:**
# MAGIC
# MAGIC $$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$
# MAGIC
# MAGIC **Desviación estándar:**
# MAGIC $$\sigma = \sqrt{\text{Var}(X)}$$
# MAGIC
# MAGIC **Aplicación en ML:**
# MAGIC * Normalización de features: $z = \frac{x - \mu}{\sigma}$
# MAGIC * Análisis de incertidumbre en predicciones
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3.4 Distribuciones Importantes
# MAGIC
# MAGIC **1. Distribución Bernoulli** (éxito/fracaso):
# MAGIC $$P(X = 1) = p, \quad P(X = 0) = 1-p$$
# MAGIC
# MAGIC **Aplicación:** Clasificación binaria
# MAGIC
# MAGIC **2. Distribución Normal (Gaussiana):**
# MAGIC $$X \sim \mathcal{N}(\mu, \sigma^2)$$
# MAGIC $$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
# MAGIC
# MAGIC **Teorema Central del Límite:** La suma de muchas variables aleatorias independientes tiende a una distribución normal.
# MAGIC
# MAGIC **Aplicación:**
# MAGIC * Suposición en regresión lineal: errores $\epsilon \sim \mathcal{N}(0, \sigma^2)$
# MAGIC * Inicialización de pesos en redes neuronales
# MAGIC
# MAGIC **3. Distribución Exponencial:**
# MAGIC $$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$
# MAGIC
# MAGIC **Aplicación:** Modelado de tiempos de espera
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3.5 Probabilidad Condicional
# MAGIC
# MAGIC **Definición:**
# MAGIC $$P(A|B) = \frac{P(A \cap B)}{P(B)}$$
# MAGIC
# MAGIC **Teorema de Bayes:**
# MAGIC $$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$
# MAGIC
# MAGIC **Aplicación en ML:** Naive Bayes Classifier
# MAGIC
# MAGIC **Ejemplo:** Clasificación de spam
# MAGIC $$P(\text{spam} | \text{palabras}) = \frac{P(\text{palabras} | \text{spam}) \cdot P(\text{spam})}{P(\text{palabras})}$$
# MAGIC
# MAGIC **Independencia:**
# MAGIC
# MAGIC Dos eventos $A$ y $B$ son independientes si:
# MAGIC $$P(A \cap B) = P(A) \cdot P(B)$$
# MAGIC
# MAGIC Equivalentemente:
# MAGIC $$P(A|B) = P(A)$$

# COMMAND ----------

# DBTITLE 1,Optimización
# MAGIC %md
# MAGIC ## 4. Teoría de Optimización
# MAGIC
# MAGIC ### 4.1 Problema de Optimización
# MAGIC
# MAGIC **Forma general:**
# MAGIC
# MAGIC $$\min_{\mathbf{x}} f(\mathbf{x})$$
# MAGIC
# MAGIC sujeto a:
# MAGIC * $g_i(\mathbf{x}) \leq 0$ (restricciones de desigualdad)
# MAGIC * $h_j(\mathbf{x}) = 0$ (restricciones de igualdad)
# MAGIC
# MAGIC **En ML (sin restricciones):**
# MAGIC $$\min_{\boldsymbol{\theta}} L(\boldsymbol{\theta}; \mathcal{D})$$
# MAGIC
# MAGIC donde $L$ es la función de pérdida y $\mathcal{D}$ son los datos.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4.2 Condiciones de Optimalidad
# MAGIC
# MAGIC **Condición necesaria de primer orden:**
# MAGIC
# MAGIC Si $\mathbf{x}^*$ es un mínimo local, entonces:
# MAGIC $$\nabla f(\mathbf{x}^*) = \mathbf{0}$$
# MAGIC
# MAGIC **Condición suficiente de segundo orden:**
# MAGIC
# MAGIC Si además la Hessiana es positiva definida:
# MAGIC $$\mathbf{H}_f(\mathbf{x}^*) \succ 0$$
# MAGIC
# MAGIC entonces $\mathbf{x}^*$ es un mínimo local.
# MAGIC
# MAGIC **Convexidad:**
# MAGIC
# MAGIC Una función $f$ es **convexa** si para todo $\mathbf{x}, \mathbf{y}$ y $\lambda \in [0,1]$:
# MAGIC $$f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y})$$
# MAGIC
# MAGIC **Ventaja:** En funciones convexas, todo mínimo local es global.
# MAGIC
# MAGIC **Ejemplos de funciones convexas en ML:**
# MAGIC * MSE (Mean Squared Error) en regresión lineal
# MAGIC * Cross-entropy en regresión logística
# MAGIC * Normas L1, L2
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4.3 Descenso de Gradiente (Gradient Descent)
# MAGIC
# MAGIC **Idea:** Moverse en la dirección opuesta al gradiente (dirección de mayor descenso).
# MAGIC
# MAGIC **Algoritmo:**
# MAGIC
# MAGIC ```
# MAGIC Inicializar: θ₀
# MAGIC Para t = 0, 1, 2, ...:
# MAGIC     θₜ₊₁ = θₜ - α · ∇L(θₜ)
# MAGIC ```
# MAGIC
# MAGIC donde $\alpha > 0$ es el **learning rate** (tasa de aprendizaje).
# MAGIC
# MAGIC **Derivación matemática:**
# MAGIC
# MAGIC Expansión de Taylor de primer orden:
# MAGIC $$L(\boldsymbol{\theta} + \boldsymbol{\Delta}) \approx L(\boldsymbol{\theta}) + \nabla L(\boldsymbol{\theta})^T \boldsymbol{\Delta}$$
# MAGIC
# MAGIC Para minimizar localmente, elegimos:
# MAGIC $$\boldsymbol{\Delta} = -\alpha \nabla L(\boldsymbol{\theta})$$
# MAGIC
# MAGIC **Variantes:**
# MAGIC
# MAGIC **1. Batch Gradient Descent:**
# MAGIC * Usa todos los datos en cada iteración
# MAGIC * Actualización: $\nabla L(\boldsymbol{\theta}) = \frac{1}{n} \sum_{i=1}^{n} \nabla L_i(\boldsymbol{\theta})$
# MAGIC * Ventaja: Convergencia estable
# MAGIC * Desventaja: Lento con datos grandes
# MAGIC
# MAGIC **2. Stochastic Gradient Descent (SGD):**
# MAGIC * Usa una muestra aleatoria en cada iteración
# MAGIC * Actualización: $\nabla L(\boldsymbol{\theta}) \approx \nabla L_i(\boldsymbol{\theta})$
# MAGIC * Ventaja: Rápido, puede escapar mínimos locales
# MAGIC * Desventaja: Convergencia ruidosa
# MAGIC
# MAGIC **3. Mini-batch Gradient Descent:**
# MAGIC * Usa un subconjunto aleatorio (batch) de tamaño $b$
# MAGIC * Actualización: $\nabla L(\boldsymbol{\theta}) = \frac{1}{b} \sum_{i \in \text{batch}} \nabla L_i(\boldsymbol{\theta})$
# MAGIC * **Estándar en ML moderno** (típicamente $b = 32, 64, 128$)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4.4 Learning Rate y Convergencia
# MAGIC
# MAGIC **Elección de $\alpha$:**
# MAGIC
# MAGIC * **Muy pequeño** ($\alpha \ll 1$): Convergencia lenta
# MAGIC * **Muy grande** ($\alpha \gg 1$): Divergencia, oscilaciones
# MAGIC * **Óptimo**: $\alpha$ suficientemente grande para converger rápido, pero no tan grande que diverja
# MAGIC
# MAGIC **Regla de convergencia:**
# MAGIC
# MAGIC Para funciones convexas con $\alpha$ apropiado:
# MAGIC $$||\nabla L(\boldsymbol{\theta}_t)|| \rightarrow 0 \text{ cuando } t \rightarrow \infty$$
# MAGIC
# MAGIC **Learning rate adaptativo:**
# MAGIC
# MAGIC **Adam (Adaptive Moment Estimation):**
# MAGIC * Ajusta $\alpha$ individualmente para cada parámetro
# MAGIC * Usa promedios móviles de gradientes (momentum)
# MAGIC * **Optimizador más popular en deep learning**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4.5 Optimización con Regularización
# MAGIC
# MAGIC **Ridge (L2):**
# MAGIC $$\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \lambda ||\boldsymbol{\beta}||_2^2$$
# MAGIC
# MAGIC **Gradiente:**
# MAGIC $$\nabla L = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) + 2\lambda \boldsymbol{\beta}$$
# MAGIC
# MAGIC **Lasso (L1):**
# MAGIC $$\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \lambda ||\boldsymbol{\beta}||_1$$
# MAGIC
# MAGIC **Subgradiente** (L1 no es diferenciable en 0):
# MAGIC $$\partial ||\boldsymbol{\beta}||_1 = \text{sign}(\boldsymbol{\beta})$$

# COMMAND ----------

# DBTITLE 1,Aplicaciones en ML
# MAGIC %md
# MAGIC ## 5. Aplicaciones Matemáticas en ML
# MAGIC
# MAGIC ### 5.1 Regresión Lineal - Derivación Completa
# MAGIC
# MAGIC **Problema:**
# MAGIC $$\min_{\boldsymbol{\beta}} L(\boldsymbol{\beta}) = \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2$$
# MAGIC
# MAGIC **Forma matricial:**
# MAGIC $$L(\boldsymbol{\beta}) = ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||_2^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$
# MAGIC
# MAGIC **Expandir:**
# MAGIC $$L(\boldsymbol{\beta}) = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$$
# MAGIC
# MAGIC **Derivar con respecto a $\boldsymbol{\beta}$:**
# MAGIC $$\nabla_\boldsymbol{\beta} L = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$$
# MAGIC
# MAGIC **Igualar a cero (condición de primer orden):**
# MAGIC $$\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$
# MAGIC
# MAGIC **Solución de forma cerrada (Normal Equations):**
# MAGIC $$\boldsymbol{\hat{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$
# MAGIC
# MAGIC **Nota:** Requiere que $\mathbf{X}^T\mathbf{X}$ sea invertible (no multicolinealidad).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5.2 Regresión Logística - Derivación
# MAGIC
# MAGIC **Problema:** Clasificación binaria $y \in \{0, 1\}$
# MAGIC
# MAGIC **Modelo:**
# MAGIC $$P(y=1|\mathbf{x}) = \sigma(\boldsymbol{\beta}^T \mathbf{x}) = \frac{1}{1 + e^{-\boldsymbol{\beta}^T \mathbf{x}}}$$
# MAGIC
# MAGIC **Función sigmoide:**
# MAGIC $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
# MAGIC
# MAGIC **Propiedades:**
# MAGIC * $\sigma(-z) = 1 - \sigma(z)$
# MAGIC * $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
# MAGIC
# MAGIC **Función de pérdida (negative log-likelihood):**
# MAGIC $$L(\boldsymbol{\beta}) = -\sum_{i=1}^{n} [y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i)]$$
# MAGIC
# MAGIC donde $\hat{p}_i = \sigma(\boldsymbol{\beta}^T \mathbf{x}_i)$.
# MAGIC
# MAGIC **Gradiente:**
# MAGIC $$\nabla L = \sum_{i=1}^{n} (\hat{p}_i - y_i) \mathbf{x}_i = \mathbf{X}^T(\boldsymbol{\hat{p}} - \mathbf{y})$$
# MAGIC
# MAGIC **No tiene solución cerrada** → usar descenso de gradiente.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5.3 K-Means - Derivación Matemática
# MAGIC
# MAGIC **Problema:**
# MAGIC $$\min_{\{C_k\}_{k=1}^{K}} \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} ||\mathbf{x}_i - \boldsymbol{\mu}_k||_2^2$$
# MAGIC
# MAGIC **Algoritmo:**
# MAGIC
# MAGIC 1. **Asignación:** Para cada $\mathbf{x}_i$, asignar al cluster más cercano:
# MAGIC    $$c_i = \arg\min_k ||\mathbf{x}_i - \boldsymbol{\mu}_k||_2^2$$
# MAGIC
# MAGIC 2. **Actualización de centroides:**
# MAGIC    $$\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i$$
# MAGIC
# MAGIC **Demostración de optimalidad de centroides:**
# MAGIC
# MAGIC Minimizar:
# MAGIC $$J_k = \sum_{\mathbf{x}_i \in C_k} ||\mathbf{x}_i - \boldsymbol{\mu}_k||_2^2$$
# MAGIC
# MAGIC Derivada con respecto a $\boldsymbol{\mu}_k$:
# MAGIC $$\frac{\partial J_k}{\partial \boldsymbol{\mu}_k} = \sum_{\mathbf{x}_i \in C_k} -2(\mathbf{x}_i - \boldsymbol{\mu}_k) = 0$$
# MAGIC
# MAGIC Despejando:
# MAGIC $$|C_k| \boldsymbol{\mu}_k = \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i$$
# MAGIC $$\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i \quad \text{(media)}$$
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5.4 Entropía e Información (Árboles de Decisión)
# MAGIC
# MAGIC **Entropía de Shannon:**
# MAGIC $$H(Y) = -\sum_{i=1}^{k} p_i \log_2(p_i)$$
# MAGIC
# MAGIC **Interpretación:** Medida de incertidumbre o "sorpresa"
# MAGIC
# MAGIC **Propiedades:**
# MAGIC * $H(Y) \geq 0$ (no negativa)
# MAGIC * $H(Y) = 0$ si $p_i = 1$ para algún $i$ (certeza total)
# MAGIC * $H(Y)$ es máxima cuando $p_i = \frac{1}{k}$ para todo $i$ (máxima incertidumbre)
# MAGIC
# MAGIC **Ganancia de información:**
# MAGIC $$IG(Y, X) = H(Y) - H(Y|X)$$
# MAGIC
# MAGIC donde:
# MAGIC $$H(Y|X) = \sum_{x} P(X=x) H(Y|X=x)$$
# MAGIC
# MAGIC **Aplicación:** Selección de la mejor feature para dividir en árboles de decisión.
# MAGIC
# MAGIC **Ejemplo:** Clasificación Churn
# MAGIC
# MAGIC * Antes de dividir: $H(\text{Churn}) = -[0.3 \log_2(0.3) + 0.7 \log_2(0.7)] = 0.88$
# MAGIC * Después de dividir por "antigüedad > 12 meses":
# MAGIC   * Grupo 1 (antigüedad ≤ 12): $H = 0.97$
# MAGIC   * Grupo 2 (antigüedad > 12): $H = 0.54$
# MAGIC   * Promedio ponderado: $H(\text{Churn} | \text{antigüedad}) = 0.72$
# MAGIC * **Ganancia de información:** $IG = 0.88 - 0.72 = 0.16$

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## 6. Conclusiones y Recursos
# MAGIC
# MAGIC ### 📚 Resumen de Conceptos Clave
# MAGIC
# MAGIC **Álgebra Lineal:**
# MAGIC * Vectores y matrices representan datos y modelos
# MAGIC * Producto punto, normas, inversas, autovalores
# MAGIC * Aplicaciones: regresión lineal, PCA, SVD
# MAGIC
# MAGIC **Cálculo:**
# MAGIC * Derivadas parciales, gradiente, Hessiana
# MAGIC * Regla de la cadena para backpropagation
# MAGIC * Optimización de funciones de pérdida
# MAGIC
# MAGIC **Probabilidad:**
# MAGIC * Variables aleatorias, distribuciones
# MAGIC * Esperanza, varianza, distribución normal
# MAGIC * Teorema de Bayes para clasificación
# MAGIC
# MAGIC **Optimización:**
# MAGIC * Descenso de gradiente y variantes (SGD, Adam)
# MAGIC * Convexidad garantiza óptimo global
# MAGIC * Regularización L1/L2
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Próximos Pasos
# MAGIC
# MAGIC **Notebooks teóricos:**
# MAGIC * `Teoria_Arboles_Decision.ipynb` - Aplicar entropía, ganancia de información
# MAGIC * `Teoria_Regresion.ipynb` - Aplicar ecuaciones normales, supuestos
# MAGIC * `Teoria_Clustering.ipynb` - Aplicar K-Means matemático, métricas
# MAGIC
# MAGIC **Notebooks prácticos:**
# MAGIC * Implementar desde cero: regresión lineal, logística, K-Means
# MAGIC * Comparar soluciones analíticas vs numéricas
# MAGIC * Visualizar descenso de gradiente
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📖 Referencias
# MAGIC
# MAGIC **Álgebra Lineal:**
# MAGIC * Gilbert Strang - "Linear Algebra and Its Applications"
# MAGIC * 3Blue1Brown - "Essence of Linear Algebra" (videos)
# MAGIC
# MAGIC **Cálculo:**
# MAGIC * James Stewart - "Calculus: Early Transcendentals"
# MAGIC * Khan Academy - Multivariable Calculus
# MAGIC
# MAGIC **Probabilidad:**
# MAGIC * Sheldon Ross - "A First Course in Probability"
# MAGIC * MIT 6.041 - Probabilistic Systems Analysis
# MAGIC
# MAGIC **Optimización:**
# MAGIC * Stephen Boyd - "Convex Optimization"
# MAGIC * Stanford CS229 - Machine Learning (notas de clase)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔬 Ejercicios Propuestos
# MAGIC
# MAGIC 1. **Álgebra:** Demostrar que $(\mathbf{X}^T\mathbf{X})^{-1}$ es simétrica
# MAGIC 2. **Cálculo:** Derivar el gradiente de MSE con regularización Ridge
# MAGIC 3. **Probabilidad:** Calcular $P(\text{spam}|\text{"gratis"})$ usando Bayes
# MAGIC 4. **Optimización:** Implementar SGD desde cero y comparar con solución cerrada
# MAGIC 5. **Entropía:** Calcular ganancia de información en un dataset de ejemplo

# COMMAND ----------

