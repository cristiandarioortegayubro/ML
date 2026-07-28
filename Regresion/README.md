# Regresión - Predicción de Precios Inmobiliarios

## 📂 Contenido

Esta carpeta contiene notebooks **teóricos y prácticos** de **Machine Learning supervisado para Regresión** usando PySpark ML.

### Notebooks:

#### 📖 Teoría

**Teoria_Regresion.ipynb** (7 celdas, 336 líneas)
- **Introducción**: Definición formal, ejemplos de aplicación
- **Regresión Lineal Simple**: Modelo, función de costo (MSE), solución de mínimos cuadrados
- **Regresión Lineal Múltiple**: Forma matricial, solución analítica
- **Supuestos y Métricas**: Supuestos del modelo lineal, R², RMSE, MAE
- **Regularización**: Ridge (L2), Lasso (L1), Elastic Net
- **Regresión No Lineal**: Polinomial, árboles, Random Forest, Gradient Boosting
- **Conclusiones**: Resumen y cuándo usar cada método

**Fórmulas clave:**
- Regresión simple: $y = \beta_0 + \beta_1 x + \epsilon$
- Solución matricial: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$
- Ridge: $J(\boldsymbol{\beta}) = MSE + \lambda ||\boldsymbol{\beta}||_2^2$
- Lasso: $J(\boldsymbol{\beta}) = MSE + \lambda ||\boldsymbol{\beta}||_1$
- R²: $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$

#### 💻 Práctica

**1. Arbol_Decision_Regresion.ipynb**
- **Algoritmo**: Decision Tree Regressor
- **Problema**: Predicción de precios de propiedades inmobiliarias
- **Dataset**: 10,000 propiedades (sintético)
- **Variables**: Área, habitaciones, baños, antigüedad, distancia al centro, barrio, etc.
- **Métricas**: RMSE, MAE, R²
- **Caso de uso**: Tasación automática de propiedades para agencia inmobiliaria

**2. Regresion_Lineal_Multiple.ipynb**
- **Algoritmo**: Linear Regression
- **Problema**: Mismo dataset de propiedades
- **Diferencia clave**: Modelo lineal (asume relaciones constantes)
- **Ventaja**: Máxima interpretabilidad (coeficientes directos)
- **Comparación**: vs Decision Tree para evaluar cuál funciona mejor

## 🎯 Objetivo

Aprender a construir modelos de regresión que predicen valores **numéricos continuos** (precio, temperatura, ventas, etc.).

## 📊 Concepto: Regresión

**Regresión** es una tarea de Machine Learning supervisado donde el objetivo es predecir un **valor numérico continuo**.

### Diferencia con Clasificación:
* **Regresión**: Predice números (ej: Precio = $350,000)
* **Clasificación**: Predice categorías (ej: Churn = Sí/No)

### Algoritmos comunes:
* **Linear Regression**: Modelo lineal simple y rápido
* **Decision Tree Regressor**: Captura relaciones no lineales
* **Random Forest Regressor**: Ensemble de árboles
* **Gradient Boosted Trees (GBT)**: Árboles secuenciales optimizados

### Comparación:

| Aspecto | Regresión Lineal | Decision Tree |
|---------|-------------------|----------------|
| **Relación** | Lineal (constante) | No lineal |
| **Interpretación** | Coeficientes directos | Feature importance |
| **Velocidad** | Muy rápido | Rápido |
| **Accuracy** | Menor (si no lineal) | Mayor |
| **Normalización** | Requerida | No requerida |

## 🚀 Orden de Estudio Recomendado

1. **Primero**: Lee el notebook teórico `Teoria_Regresion`
   - Comprende la regresión lineal simple y múltiple
   - Estudia las fórmulas de mínimos cuadrados
   - Aprende sobre regularización (Ridge, Lasso)

2. **Segundo**: Ejecuta ambos notebooks prácticos
   - Compara Regresión Lineal vs Árbol de Decisión
   - Observa las diferencias en interpretabilidad y performance
   - Decide cuál es mejor para este problema

## 💡 Cómo usar estos notebooks

### Notebook Teórico:
1. Lee todas las secciones secuencialmente
2. Toma notas de las fórmulas clave
3. Comprende cuándo usar cada método
4. No ejecutes (es contenido markdown)

### Notebooks Prácticos:
1. Ejecuta ambos notebooks con el mismo dataset
2. Compara RMSE, MAE y R²
3. ¿Cuál tiene mejor rendimiento?
4. Experimenta:
   - Ajusta hiperparámetros
   - Prueba con tus propios datos
   - Añade nuevas features (ej: distancia a transporte público)

## 📚 Recursos

* [PySpark ML Regression](https://spark.apache.org/docs/latest/ml-classification-regression.html#regression)
* [Linear Regression Guide](https://spark.apache.org/docs/latest/ml-classification-regression.html#linear-regression)
* [Decision Tree Regressor](https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-regression)

## 💼 Aplicaciones reales

* **Real Estate**: Tasación de propiedades
* **Finance**: Predicción de precios de acciones
* **Retail**: Forecasting de ventas
* **Energy**: Predicción de consumo eléctrico
* **Agriculture**: Estimación de rendimiento de cultivos

---

**Siguiente paso**: Explora la carpeta `Clustering` para aprender sobre aprendizaje no supervisado.
