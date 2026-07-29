# Fundamentos de Machine Learning

## 📂 Contenido

Esta carpeta contiene los **notebooks teóricos fundamentales** que establecen las bases matemáticas y conceptuales de Machine Learning.

### Notebooks:

1. **01_Introduccion_Machine_Learning.ipynb** ✅ *Actualizado 2026*
   - **7 celdas markdown** (formato moderno, sin legacy markup)
   - **¿Qué es Machine Learning?**: Definición formal, programación tradicional vs ML
   - **Tipos de aprendizaje**: Supervisado, no supervisado, por refuerzo
   - **Proceso de ML**: Pipeline completo desde definición del problema hasta implementación
   - **Métricas de evaluación**: Clasificación (accuracy, precision, recall, F1, ROC-AUC) y regresión (MAE, MSE, RMSE, R²)
   - **Overfitting y Underfitting**: Bias-variance tradeoff, técnicas de regularización
   - **Ingeniería de características**: Escalamiento, codificación, manejo de datos faltantes
   - **Conclusiones**: Recursos y próximos pasos

2. **02_Matematicas_Esenciales.ipynb** ✅ *Actualizado 2026*
   - **8 celdas markdown** (formato moderno, con ecuaciones LaTeX)
   - **Álgebra Lineal**: Vectores, matrices, operaciones, aplicaciones en ML (redes neuronales)
   - **Cálculo**: Derivadas, derivadas parciales, gradiente, descenso por gradiente, backpropagation
   - **Probabilidad**: Probabilidad condicional, Teorema de Bayes, variables aleatorias, distribuciones comunes
   - **Estadística Inferencial**: MLE, intervalos de confianza, pruebas de hipótesis, correlación
   - **Optimización**: Gradient Descent, SGD, Mini-Batch GD, Adam, convexidad
   - **Teoría de la Información**: Entropía, cross-entropy, divergencia KL, información mutua
   - **Conclusiones**: Resumen conceptual y recursos recomendados

3. **03_Preprocesamiento_y_Feature_Engineering.ipynb** ✅ *Actualizado 2026*
   - **12 celdas** (10 markdown + 2 código Python ejecutable)
   - **Limpieza de datos**: Valores faltantes (estrategias de imputación), outliers (IQR, Z-Score), duplicados
   - **Encoding categ��rico**: Label Encoding (ordinales), One-Hot (nominales), Target Encoding (alta cardinalidad)
   - **Scaling numérico**: StandardScaler (Z-score), MinMaxScaler, RobustScaler
   - **Feature Engineering**: Interacciones, features de fecha/tiempo, binning, agregaciones
   - **Selección de Features**: Filter methods, Wrapper methods, Embedded methods, Feature Importance
   - **Ejemplos ejecutables**: sklearn, pandas con datasets sintéticos realistas
   - **Mejores prácticas**: Checklist de preprocesamiento, errores comunes

4. **04_Evaluacion_y_Validacion.ipynb** ✅ *Actualizado 2026*
   - **8 celdas** (5 markdown + 3 código Python ejecutable)
   - **Métricas de Clasificación**: Matriz de confusión, Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - **Validación Cruzada**: K-Fold CV, Stratified K-Fold, cuándo usar cada uno
   - **Overfitting y Underfitting**: Síntomas, diagnóstico con curvas, soluciones prácticas
   - **Bias-Variance Tradeoff**: Teoría y aplicación en problemas reales
   - **Ejemplos ejecutables**: Clasificación desbalanceada (detección de fraude), diagnóstico de modelos (regresión)
   - **Mejores prácticas**: Checklist de evaluación, errores comunes, métricas por tipo de problema

## 🎯 Objetivo

Estos notebooks proporcionan la **base teórica rigurosa** necesaria para comprender cómo funcionan los algoritmos de Machine Learning, por qué funcionan, y cuándo aplicarlos.

## 📊 Por qué son importantes

### Sin estos fundamentos:
* Aplicarás algoritmos como "cajas negras"
* No entenderás por qué un modelo falla
* No podrás depurar problemas de entrenamiento
* No podrás optimizar hiperparámetros efectivamente

### Con estos fundamentos:
* Comprenderás el "por qué" detrás de cada algoritmo
* Podrás diagnosticar y resolver problemas de convergencia
* Sabrás qué técnicas aplicar en cada situación
* Podrás diseñar soluciones personalizadas

## 🚀 Orden de Estudio Recomendado

1. **Primero**: Lee `01_Introduccion_Machine_Learning`
   - Comprende los conceptos generales
   - Familiarízate con el pipeline de ML
   - Aprende las métricas básicas

2. **Segundo**: Lee `02_Matematicas_Esenciales`
   - Repasa álgebra lineal y cálculo
   - Comprende la base matemática del ML
   - Estudia las fórmulas de optimización

3. **Luego**: Pasa a los notebooks teóricos específicos
   - Clasificación: Teoría de Árboles de Decisión
   - Regresión: Teoría de Regresión
   - Clustering: Teoría de Clustering

4. **Finalmente**: Practica con los notebooks prácticos
   - Aplica lo aprendido en problemas reales
   - Experimenta con diferentes algoritmos
   - Compara resultados

## 📚 Conceptos Clave por Notebook

### 01_Introduccion_Machine_Learning

**Fórmulas importantes:**

* Accuracy: $\frac{TP + TN}{TP + TN + FP + FN}$
* Precision: $\frac{TP}{TP + FP}$
* Recall: $\frac{TP}{TP + FN}$
* F1-Score: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
* RMSE: $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$
* R²: $1 - \frac{SS_{res}}{SS_{tot}}$

**Conceptos:**
* Train/Validation/Test split
* Cross-validation
* Overfitting vs Underfitting
* Regularización (L1, L2)

### 02_Matematicas_Esenciales

**Fórmulas fundamentales:**

* Gradiente: $\nabla f = [\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ...]^T$
* Gradient Descent: $\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla J(\mathbf{w}_t)$
* Teorema de Bayes: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
* Cross-Entropy: $-\sum_i y_i \log(\hat{y}_i)$
* MLE: $\hat{\theta}_{MLE} = \arg\max_{\theta} \prod_{i=1}^{n} P(x_i|\theta)$

**Conceptos:**
* Vectores y matrices en ML
* Backpropagation (regla de la cadena)
* Optimizadores (SGD, Adam, Momentum)
* Distribuciones (Gaussiana, Bernoulli, Binomial)

## 💡 Cómo usar estos notebooks

1. **Lee secuencialmente** celda por celda
2. **Toma notas** de las fórmulas clave
3. **Practica** los cálculos a mano con ejemplos simples
4. **No te saltes** las demostraciones matemáticas
5. **Relaciona** con los notebooks prácticos

## 📚 Recursos Complementarios

### Libros
* **"Pattern Recognition and Machine Learning"** - Christopher Bishop
* **"The Elements of Statistical Learning"** - Hastie, Tibshirani, Friedman
* **"Mathematics for Machine Learning"** - Deisenroth, Faisal, Ong
* **"Hands-On Machine Learning"** - Aurélien Géron

### Cursos Online
* **Coursera**: Machine Learning (Andrew Ng)
* **Fast.ai**: Practical Deep Learning
* **3Blue1Brown (YouTube)**: Visualizaciones de álgebra lineal y cálculo
* **Khan Academy**: Matemáticas básicas

### Datasets para Practicar
* **Kaggle**: Competencias y datasets
* **UCI ML Repository**: Datasets clásicos
* **Databricks Datasets**: Integrados en la plataforma

## 🔄 Relación con Notebooks Prácticos

Estos fundamentos se aplican directamente en:

* **Clasificación**: Entropía y Gini (de Teoría de la Información) para árboles de decisión
* **Regresión**: Álgebra lineal y cálculo para regresión lineal y gradient descent
* **Clustering**: Álgebra lineal (distancias) y optimización para K-Means

## ✅ Checklist de Comprensión

Después de estudiar estos notebooks, deberías poder:

- [ ] Explicar la diferencia entre aprendizaje supervisado y no supervisado
- [ ] Describir el pipeline completo de Machine Learning
- [ ] Calcular métricas de clasificación dada una matriz de confusión
- [ ] Interpretar valores de RMSE, MAE y R²
- [ ] Explicar qué es overfitting y cómo prevenirlo
- [ ] Multiplicar matrices y vectores
- [ ] Calcular el gradiente de una función simple
- [ ] Aplicar el Teorema de Bayes en un problema simple
- [ ] Explicar cómo funciona gradient descent
- [ ] Distinguir entre diferentes optimizadores (SGD, Adam)

---

**Siguiente paso**: Después de dominar estos fundamentos, explora las carpetas `Clasificacion`, `Regresion` y `Clustering` para ver aplicaciones específicas.

**¡Éxito en tu aprendizaje de Machine Learning!** 🎓🚀