# Databricks notebook source
# DBTITLE 1,Matemáticas Esenciales para ML
# MAGIC %md
# MAGIC # Matemáticas Esenciales para Machine Learning
# MAGIC
# MAGIC ## 1. Introducción
# MAGIC
# MAGIC Las matemáticas son la base del Machine Learning. Comprender estos conceptos permite:
# MAGIC
# MAGIC * Entender cómo funcionan los algoritmos
# MAGIC * Diseñar nuevos modelos
# MAGIC * Depurar problemas de entrenamiento
# MAGIC * Optimizar hiperparámetros
# MAGIC
# MAGIC ### Áreas Clave
# MAGIC
# MAGIC 1. **Álgebra Lineal**: Representación de datos y transformaciones
# MAGIC 2. **Cálculo**: Optimización y gradientes
# MAGIC 3. **Probabilidad y Estadística**: Inferencia y incertidumbre
# MAGIC 4. **Optimización**: Encontrar parámetros óptimos

# COMMAND ----------

# DBTITLE 1,Álgebra Lineal
# MAGIC %md
# MAGIC ## 2. Álgebra Lineal
# MAGIC
# MAGIC ### 2.1 Vectores
# MAGIC
# MAGIC Un **vector** es un arreglo ordenado de números:
# MAGIC
# MAGIC $$\mathbf{v} = \begin{bmatrix} v_1 \\\\ v_2 \\\\ \vdots \\\\ v_n \end{bmatrix} \in \mathbb{R}^n$$
# MAGIC
# MAGIC **Operaciones:**
# MAGIC
# MAGIC * **Suma**: $\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\\\ u_2 + v_2 \\\\ \vdots \end{bmatrix}$
# MAGIC
# MAGIC * **Producto escalar**: $c \cdot \mathbf{v} = \begin{bmatrix} c \cdot v_1 \\\\ c \cdot v_2 \\\\ \vdots \end{bmatrix}$
# MAGIC
# MAGIC * **Producto punto (dot product)**: $\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i$
# MAGIC
# MAGIC * **Norma**: $||\mathbf{v}|| = \sqrt{\sum_{i=1}^{n} v_i^2}$
# MAGIC
# MAGIC ### 2.2 Matrices
# MAGIC
# MAGIC Una **matriz** es un arreglo rectangular de números:
# MAGIC
# MAGIC $$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\\\ a_{21} & a_{22} & \cdots & a_{2n} \\\\ \vdots & \vdots & \ddots & \vdots \\\\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}$$
# MAGIC
# MAGIC **Operaciones:**
# MAGIC
# MAGIC * **Transpuesta**: $(\mathbf{A}^T)_{ij} = A_{ji}$
# MAGIC
# MAGIC * **Multiplicación matriz-vector**: $\mathbf{A}\mathbf{v} = \mathbf{b}$
# MAGIC
# MAGIC * **Multiplicación matriz-matriz**: $(\mathbf{AB})_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}$
# MAGIC
# MAGIC * **Inversa**: $\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$ (si existe)
# MAGIC
# MAGIC ### 2.3 Aplicaciones en ML
# MAGIC
# MAGIC * **Datos**: Matriz $\mathbf{X} \in \mathbb{R}^{n \times d}$ (n ejemplos, d features)
# MAGIC * **Regresión lineal**: $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$
# MAGIC * **Transformaciones**: Rotación, proyección, PCA
# MAGIC * **Redes neuronales**: Cada capa es $\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$

# COMMAND ----------

# DBTITLE 1,Cálculo
# MAGIC %md
# MAGIC ## 3. Cálculo
# MAGIC
# MAGIC ### 3.1 Derivadas
# MAGIC
# MAGIC La **derivada** mide la tasa de cambio instantánea:
# MAGIC
# MAGIC $$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$
# MAGIC
# MAGIC **Reglas:**
# MAGIC
# MAGIC * **Potencias**: $\frac{d}{dx}x^n = nx^{n-1}$
# MAGIC * **Exponencial**: $\frac{d}{dx}e^x = e^x$
# MAGIC * **Logarítmica**: $\frac{d}{dx}\ln(x) = \frac{1}{x}$
# MAGIC * **Cadena**: $\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$
# MAGIC * **Producto**: $\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$
# MAGIC
# MAGIC ### 3.2 Derivadas Parciales
# MAGIC
# MAGIC Para funciones de múltiples variables $f(x_1, x_2, ..., x_n)$:
# MAGIC
# MAGIC $$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, ..., x_i+h, ..., x_n) - f(x_1, ..., x_i, ..., x_n)}{h}$$
# MAGIC
# MAGIC ### 3.3 Gradiente
# MAGIC
# MAGIC El **gradiente** es el vector de derivadas parciales:
# MAGIC
# MAGIC $$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\\\ \frac{\partial f}{\partial x_2} \\\\ \vdots \\\\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$
# MAGIC
# MAGIC **Propiedades:**
# MAGIC
# MAGIC * Apunta en la dirección de **mayor crecimiento**
# MAGIC * Su negativo $-\nabla f$ apunta hacia el **mínimo**
# MAGIC
# MAGIC ### 3.4 Descenso por Gradiente
# MAGIC
# MAGIC Algoritmo de optimización fundamental en ML:
# MAGIC
# MAGIC $$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla_{\mathbf{w}} J(\mathbf{w}_t)$$
# MAGIC
# MAGIC Donde:
# MAGIC * $\mathbf{w}$: Parámetros del modelo
# MAGIC * $\alpha$: Tasa de aprendizaje (learning rate)
# MAGIC * $J(\mathbf{w})$: Función de costo
# MAGIC
# MAGIC ### 3.5 Backpropagation
# MAGIC
# MAGIC **Regla de la cadena** en redes neuronales:
# MAGIC
# MAGIC $$\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial z_j} \cdot \frac{\partial z_j}{\partial w_{ij}}$$
# MAGIC
# MAGIC Permite calcular gradientes eficientemente en redes profundas.

# COMMAND ----------

# DBTITLE 1,Probabilidad
# MAGIC %md
# MAGIC ## 4. Probabilidad y Estadística
# MAGIC
# MAGIC ### 4.1 Probabilidad
# MAGIC
# MAGIC **Probabilidad** mide la certeza de que ocurra un evento:
# MAGIC
# MAGIC $$0 \leq P(A) \leq 1$$
# MAGIC
# MAGIC **Reglas:**
# MAGIC
# MAGIC * **Suma**: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
# MAGIC * **Producto**: $P(A \cap B) = P(A) \cdot P(B|A)$
# MAGIC * **Complemento**: $P(A^c) = 1 - P(A)$
# MAGIC
# MAGIC ### 4.2 Probabilidad Condicional
# MAGIC
# MAGIC $$P(A|B) = \frac{P(A \cap B)}{P(B)}$$
# MAGIC
# MAGIC **Teorema de Bayes:**
# MAGIC
# MAGIC $$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$
# MAGIC
# MAGIC **Aplicación en ML**: Naive Bayes classifier
# MAGIC
# MAGIC ### 4.3 Variables Aleatorias
# MAGIC
# MAGIC **Esperanza (Media):**
# MAGIC
# MAGIC $$E[X] = \mu = \sum_{i} x_i P(X=x_i) \quad \text{(discreta)}$$
# MAGIC
# MAGIC $$E[X] = \int_{-\infty}^{\infty} x f(x) dx \quad \text{(continua)}$$
# MAGIC
# MAGIC **Varianza:**
# MAGIC
# MAGIC $$\text{Var}(X) = \sigma^2 = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$
# MAGIC
# MAGIC **Desviación estándar:**
# MAGIC
# MAGIC $$\sigma = \sqrt{\text{Var}(X)}$$
# MAGIC
# MAGIC ### 4.4 Distribuciones Comunes
# MAGIC
# MAGIC **Gaussiana (Normal):**
# MAGIC
# MAGIC $$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
# MAGIC
# MAGIC Notación: $X \sim \mathcal{N}(\mu, \sigma^2)$
# MAGIC
# MAGIC **Bernoulli:** Experimento binario (0/1)
# MAGIC
# MAGIC $$P(X=1) = p, \quad P(X=0) = 1-p$$
# MAGIC
# MAGIC **Binomial:** n experimentos Bernoulli
# MAGIC
# MAGIC $$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$$

# COMMAND ----------

# DBTITLE 1,Estadística
# MAGIC %md
# MAGIC ## 5. Estadística Inferencial
# MAGIC
# MAGIC ### 5.1 Estimación de Parámetros
# MAGIC
# MAGIC **Máxima Verosimilitud (MLE):**
# MAGIC
# MAGIC Dados datos $\mathcal{D} = \{x_1, x_2, ..., x_n\}$, encontrar $\theta$ que maximiza:
# MAGIC
# MAGIC $$\hat{\theta}_{MLE} = \arg\max_{\theta} P(\mathcal{D}|\theta) = \arg\max_{\theta} \prod_{i=1}^{n} P(x_i|\theta)$$
# MAGIC
# MAGIC En la práctica, maximizamos el **log-likelihood**:
# MAGIC
# MAGIC $$\hat{\theta}_{MLE} = \arg\max_{\theta} \sum_{i=1}^{n} \log P(x_i|\theta)$$
# MAGIC
# MAGIC ### 5.2 Intervalos de Confianza
# MAGIC
# MAGIC Rango que probablemente contiene el parámetro verdadero:
# MAGIC
# MAGIC $$\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$
# MAGIC
# MAGIC Para 95% de confianza: $z_{0.025} = 1.96$
# MAGIC
# MAGIC ### 5.3 Pruebas de Hipótesis
# MAGIC
# MAGIC 1. **Hipótesis nula** $H_0$: No hay efecto
# MAGIC 2. **Hipótesis alternativa** $H_1$: Hay efecto
# MAGIC 3. Calcular **estadístico de prueba**
# MAGIC 4. Calcular **p-valor**
# MAGIC 5. Si $p < \alpha$ (ej. 0.05), rechazar $H_0$
# MAGIC
# MAGIC ### 5.4 Correlación
# MAGIC
# MAGIC **Coeficiente de Pearson:**
# MAGIC
# MAGIC $$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$
# MAGIC
# MAGIC * $r = 1$: Correlación positiva perfecta
# MAGIC * $r = 0$: Sin correlación lineal
# MAGIC * $r = -1$: Correlación negativa perfecta
# MAGIC
# MAGIC **Importante:** Correlación ≠ Causalidad

# COMMAND ----------

# DBTITLE 1,Optimización
# MAGIC %md
# MAGIC ## 6. Optimización
# MAGIC
# MAGIC ### 6.1 Problema de Optimización
# MAGIC
# MAGIC Encontrar:
# MAGIC
# MAGIC $$\mathbf{x}^* = \arg\min_{\mathbf{x}} f(\mathbf{x})$$
# MAGIC
# MAGIC Sujeto a restricciones (opcional).
# MAGIC
# MAGIC ### 6.2 Condiciones de Optimalidad
# MAGIC
# MAGIC **Mínimo local:** $\nabla f(\mathbf{x}^*) = \mathbf{0}$
# MAGIC
# MAGIC **Mínimo global:** $f(\mathbf{x}^*) \leq f(\mathbf{x})$ para todo $\mathbf{x}$
# MAGIC
# MAGIC ### 6.3 Algoritmos de Optimización
# MAGIC
# MAGIC **Gradient Descent (GD):**
# MAGIC
# MAGIC $$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla J(\mathbf{w}_t)$$
# MAGIC
# MAGIC **Stochastic Gradient Descent (SGD):**
# MAGIC
# MAGIC Usa un ejemplo aleatorio por iteración:
# MAGIC
# MAGIC $$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla J_i(\mathbf{w}_t)$$
# MAGIC
# MAGIC **Mini-batch GD:**
# MAGIC
# MAGIC Usa un subconjunto de ejemplos:
# MAGIC
# MAGIC $$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \frac{1}{|B|} \sum_{i \in B} \nabla J_i(\mathbf{w}_t)$$
# MAGIC
# MAGIC **Momentum:**
# MAGIC
# MAGIC $$\mathbf{v}_{t+1} = \beta \mathbf{v}_t - \alpha \nabla J(\mathbf{w}_t)$$
# MAGIC $$\mathbf{w}_{t+1} = \mathbf{w}_t + \mathbf{v}_{t+1}$$
# MAGIC
# MAGIC **Adam (Adaptive Moment Estimation):**
# MAGIC
# MAGIC Combina momentum y RMSProp:
# MAGIC
# MAGIC $$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\nabla J$$
# MAGIC $$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)(\nabla J)^2$$
# MAGIC $$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \frac{\mathbf{m}_t}{\sqrt{\mathbf{v}_t} + \epsilon}$$
# MAGIC
# MAGIC ### 6.4 Hiperparámetros
# MAGIC
# MAGIC * **Learning rate** $\alpha$: Muy importante
# MAGIC   - Demasiado pequeño: Convergencia lenta
# MAGIC   - Demasiado grande: Divergencia
# MAGIC * **Batch size**: Trade-off velocidad vs estabilidad
# MAGIC * **Epochs**: Número de pasadas sobre los datos

# COMMAND ----------

# DBTITLE 1,Teoría de la Información
# MAGIC %md
# MAGIC ## 7. Teoría de la Información
# MAGIC
# MAGIC ### 7.1 Entropía
# MAGIC
# MAGIC Mide la **incertidumbre** de una variable aleatoria:
# MAGIC
# MAGIC $$H(X) = -\sum_{i} P(x_i) \log_2 P(x_i)$$
# MAGIC
# MAGIC **Propiedades:**
# MAGIC
# MAGIC * $H(X) \geq 0$
# MAGIC * Máxima cuando todas las probabilidades son iguales
# MAGIC * Mínima (0) cuando hay certeza
# MAGIC
# MAGIC ### 7.2 Entropía Cruzada (Cross-Entropy)
# MAGIC
# MAGIC $$H(p, q) = -\sum_{i} p(x_i) \log q(x_i)$$
# MAGIC
# MAGIC **Aplicación en ML:** Función de costo para clasificación
# MAGIC
# MAGIC **Binary Cross-Entropy:**
# MAGIC
# MAGIC $$BCE = -\frac{1}{n}\sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$
# MAGIC
# MAGIC ### 7.3 Divergencia KL (Kullback-Leibler)
# MAGIC
# MAGIC Mide la diferencia entre dos distribuciones:
# MAGIC
# MAGIC $$D_{KL}(p||q) = \sum_{i} p(x_i) \log \frac{p(x_i)}{q(x_i)}$$
# MAGIC
# MAGIC **Propiedades:**
# MAGIC
# MAGIC * $D_{KL}(p||q) \geq 0$
# MAGIC * $D_{KL}(p||q) = 0$ si y solo si $p = q$
# MAGIC * No es simétrica: $D_{KL}(p||q) \neq D_{KL}(q||p)$
# MAGIC
# MAGIC ### 7.4 Información Mutua
# MAGIC
# MAGIC Mide dependencia entre variables:
# MAGIC
# MAGIC $$I(X;Y) = H(X) + H(Y) - H(X,Y)$$
# MAGIC
# MAGIC **Aplicación:** Selección de características

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## 8. Conclusiones
# MAGIC
# MAGIC ### Resumen de Conceptos
# MAGIC
# MAGIC | Área | Conceptos Clave | Aplicación en ML |
# MAGIC |------|-----------------|-------------------|
# MAGIC | **Álgebra Lineal** | Vectores, matrices, transformaciones | Representación de datos, redes neuronales |
# MAGIC | **Cálculo** | Derivadas, gradientes | Optimización, backpropagation |
# MAGIC | **Probabilidad** | Distribuciones, Bayes | Clasificación, incertidumbre |
# MAGIC | **Estadística** | MLE, intervalos, correlación | Inferencia, evaluación |
# MAGIC | **Optimización** | Gradient descent, Adam | Entrenamiento de modelos |
# MAGIC | **Información** | Entropía, cross-entropy | Funciones de costo |
# MAGIC
# MAGIC ### Fórmulas Esenciales
# MAGIC
# MAGIC 1. **Gradiente**: $\nabla f = [\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ...]^T$
# MAGIC 2. **Gradient Descent**: $\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla J(\mathbf{w}_t)$
# MAGIC 3. **Bayes**: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
# MAGIC 4. **Cross-Entropy**: $-\sum_i y_i \log(\hat{y}_i)$
# MAGIC
# MAGIC ### Próximos Pasos
# MAGIC
# MAGIC * Practicar cálculos a mano con ejemplos simples
# MAGIC * Implementar algoritmos desde cero (sin librerías)
# MAGIC * Estudiar demostraciones matemáticas de algoritmos
# MAGIC * Aplicar conceptos en notebooks prácticos
# MAGIC
# MAGIC ### Recursos
# MAGIC
# MAGIC * **Libros**: "Mathematics for Machine Learning" (Deisenroth et al.)
# MAGIC * **Cursos**: 3Blue1Brown (YouTube), Khan Academy
# MAGIC * **Práctica**: Ejercicios en Jupyter notebooks

# COMMAND ----------

