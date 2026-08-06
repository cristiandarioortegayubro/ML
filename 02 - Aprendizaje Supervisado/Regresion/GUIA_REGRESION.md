# Regresión - Predicción de Precios Inmobiliarios

## 📂 Contenido

Esta carpeta contiene notebooks **teóricos y prácticos** de **Machine Learning supervisado para Regresión** usando PySpark ML.

### Notebooks:

#### 📚 Guía Completa de Algoritmos

**01_Algoritmos_Regresion_Negocios.ipynb** (11 celdas) - ⭐️ **RECOMENDADO PARA EMPEZAR**
- **Enfoque**: Aplicaciones empresariales y casos de uso reales
- **Contenido completo**:
  1. **Introducción**: Definición formal, casos de uso empresariales por industria
  2. **Regresión Lineal**: Formulación matemática, ventajas/desventajas, métricas (R², RMSE)
  3. **Regresión Polinomial**: Transformaciones, selección de grado, ejemplos de negocio
  4. **Ridge, Lasso, Elastic Net**: Regularización L1/L2, selección de features
  5. **Árboles de Decisión**: Algoritmo, criterios de división, interpretabilidad
  6. **Random Forest**: Ensemble, feature importance, casos de uso
  7. **Gradient Boosting**: XGBoost, LightGBM, CatBoost, hiperparámetros
  8. **SVR**: Kernel trick, tubo ε-insensitivo, escalamiento obligatorio
  9. **Redes Neuronales**: Deep Learning, arquitecturas, cuándo usar
  10. **Comparación**: Tabla comparativa completa, árbol de decisión para selección
  11. **Conclusiones**: Framework de selección, métricas, próximos pasos
- **Características únicas**:
  - Casos de uso por industria (Finanzas, Retail, E-commerce, Real Estate, etc.)
  - Ejemplos de interpretación para stakeholders
  - Guía de selección según prioridades de negocio
  - Comparación de implementaciones (XGBoost vs LightGBM vs CatBoost)
  - Trade-offs interpretabilidad vs accuracy

#### 📖 Teoría Técnica

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

**3. Random_Forest_Regresion.ipynb**
- **Algoritmo**: Random Forest Regressor (100 árboles)
- **Problema**: Mismo dataset de propiedades
- **Comparación**: vs Decision Tree y Linear Regression
- **Feature Importance**: Identificación de variables más relevantes
- **Resultados**: R² ≈ 0.95-0.97 (mejor que Decision Tree y Linear Regression)
- **Trade-off**: Mayor accuracy a cambio de velocidad e interpretabilidad

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

### 🌟 Para Aprender sobre Algoritmos (Principiantes y Avanzados):

1. **PRIMERO**: Lee `01_Algoritmos_Regresion_Negocios` ⭐️
   - Revisión completa de todos los algoritmos
   - Enfoque en aplicaciones empresariales reales
   - Ventajas/desventajas de cada método
   - Guía de selección según tu problema
   - **Ideal para**: Entender cuándo usar cada algoritmo

2. **Opcional - Profundización Teórica**: Lee `Teoria_Regresion`
   - Fórmulas matemáticas detalladas
   - Demostraciones de mínimos cuadrados
   - Formulación rigurosa de regularización
   - **Ideal para**: Comprensión matemática profunda

### 💻 Para Implementar (Práctica):

3. **SEGUNDO**: Ejecuta `Regresion_Lineal_Multiple`
   - Implementa regresión lineal con múltiples variables
   - Observa la interpretabilidad de los coeficientes
   - Evalúa las métricas (RMSE, MAE, R²)

4. **TERCERO**: Ejecuta `Arbol_Decision_Regresion`
   - Compara con regresión lineal
   - Observa cómo captura relaciones no lineales
   - Analiza feature importance

5. **CUARTO**: Ejecuta `Random_Forest_Regresion`
   - Compara los 3 modelos (Linear, Tree, Random Forest)
   - Observa la mejora en R² y reducción de RMSE
   - Analiza feature importance más robusta
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
