# Regresión - Predicción de Precios Inmobiliarios

## 📂 Contenido

Esta carpeta contiene ejemplos de **Machine Learning supervisado para Regresión** usando PySpark ML.

### Notebooks:

1. **Arbol_Decision_Regresion.py**
   - **Algoritmo**: Decision Tree Regressor
   - **Problema**: Predicción de precios de propiedades inmobiliarias
   - **Dataset**: 10,000 propiedades (sintético)
   - **Variables**: Área, habitaciones, baños, antigüedad, distancia al centro, barrio, etc.
   - **Métricas**: RMSE, MAE, R²
   - **Caso de uso**: Tasación automática de propiedades para agencia inmobiliaria

2. **Regresion_Lineal_Multiple.py**
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

## 🚀 Cómo usar estos notebooks

1. **Abrir en Databricks**:
   - Navega a `ML/Regresion`
   - Abre ambos notebooks

2. **Ejecutar y Comparar**:
   - Ejecuta ambos notebooks con el mismo dataset
   - Compara RMSE, MAE y R²
   - ¿Cuál tiene mejor rendimiento?

3. **Experimentar**:
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