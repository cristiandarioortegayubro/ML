# Aprendizaje Supervisado

## 🎯 Definición

**Aprendizaje Supervisado** es un paradigma de Machine Learning donde el modelo aprende a partir de datos **etiquetados** (labeled data). Cada ejemplo de entrenamiento consiste en un par `(entrada, salida deseada)`, y el objetivo es aprender una función que mapee entradas a salidas correctas.

### Características Clave

* 📊 **Datos etiquetados**: Cada muestra tiene su respuesta correcta conocida
* 🎯 **Objetivo claro**: Predecir la etiqueta correcta para nuevos datos
* 📊 **Métricas definidas**: Accuracy, RMSE, Precision, Recall, etc.
* 🔄 **Proceso iterativo**: Entrenamiento → Validación → Ajuste → Test

## 📁 Contenido

Esta carpeta contiene dos tipos principales de problemas supervisados:

### 1️⃣ Clasificación

**Objetivo**: Predecir una **categoría** o **clase** discreta.

* **Carpeta**: `Clasificacion/`
* **Notebook principal**: `01_Algoritmos_Clasificacion_Negocios` - Guía completa de algoritmos con enfoque empresarial
* **Ejemplos**:
  - Predicción de churn (Sí/No)
  - Detección de fraude (Fraudulento/Legítimo)
  - Clasificación de imágenes (Gato/Perro/Pájaro)
* **Algoritmos cubiertos**:
  - Regresión Logística
  - Árboles de Decisión
  - Random Forest
  - Gradient Boosting (XGBoost, LightGBM, CatBoost)
  - Support Vector Machines (SVM)
  - Naive Bayes
  - K-Nearest Neighbors (KNN)
  - Redes Neuronales
  - Comparación y selección de algoritmos

### 2️⃣ Regresión

**Objetivo**: Predecir un **valor numérico continuo**.

* **Carpeta**: `Regresion/`
* **Notebook principal**: `01_Algoritmos_Regresion_Negocios` - Guía completa de algoritmos con enfoque empresarial
* **Ejemplos**:
  - Predicción de precios (casas, acciones, productos)
  - Previsión de demanda
  - Estimación de temperatura
  - Predicción de ventas
  - Customer Lifetime Value (CLV)
  - Revenue Forecasting
* **Algoritmos cubiertos**:
  - Regresión Lineal
  - Regresión Polinomial
  - Ridge, Lasso, Elastic Net (Regularización)
  - Árboles de Decisión
  - Random Forest
  - Gradient Boosting (XGBoost, LightGBM, CatBoost)
  - Support Vector Regression (SVR)
  - Redes Neuronales
  - Comparación y selección de algoritmos

## 🔄 Proceso de Aprendizaje Supervisado

```
1. Recolección de Datos
   ↓
2. Preprocesamiento y Limpieza
   ↓
3. División: Train / Validation / Test
   ↓
4. Entrenamiento del Modelo
   ↓
5. Evaluación en Validation Set
   ↓
6. Ajuste de Hiperparámetros (tuning)
   ↓
7. Evaluación Final en Test Set
   ↓
8. Despliegue en Producción
```

## 📊 Comparación: Clasificación vs Regresión

| Aspecto | Clasificación | Regresión |
|---------|----------------|------------|
| **Output** | Categoría discreta | Valor numérico continuo |
| **Ejemplos** | Spam/No Spam, Tipo A/B/C | Precio \$250k, Temperatura 23.5°C |
| **Métricas** | Accuracy, Precision, Recall, F1, AUC | RMSE, MAE, R², MAPE |
| **Función de costo** | Cross-Entropy, Hinge Loss | MSE, MAE, Huber Loss |
| **Activación final** | Softmax, Sigmoid | Linear, ReLU |

## 🚀 Orden de Estudio Recomendado

### Para Principiantes:

1. **Fundamentos** (carpeta `../Fundamentos/`)
   - Introducción a Machine Learning
   - Matemáticas esenciales (vectores, matrices, derivadas)

2. **Clasificación** (`./Clasificacion/`)
   - Comenzar con Decision Trees (más intuitivos)
   - Avanzar a Random Forest
   - Problema: Predicción de churn

3. **Regresión** (`./Regresion/`)
   - Regresión lineal (base teórica)
   - Decision Tree Regressor
   - Random Forest Regressor
   - Problema: Predicción de precios inmobiliarios

### Para Avanzados:

1. Explorar algoritmos de ensemble (Gradient Boosting, XGBoost)
2. Deep Learning para clasificación y regresión
3. Transfer Learning y Fine-Tuning

## 📚 Conceptos Clave

### Overfitting y Underfitting

* **Overfitting** (🔴 Sobreajuste): El modelo memoriza los datos de entrenamiento pero falla en datos nuevos
  - **Solución**: Regularización, más datos, validación cruzada
  
* **Underfitting** (🔵 Subajuste): El modelo es demasiado simple y no captura patrones
  - **Solución**: Modelo más complejo, más features, menos regularización

### Trade-off Bias-Variance

* **High Bias**: Modelo muy simple (underfitting)
* **High Variance**: Modelo muy complejo (overfitting)
* **Óptimo**: Balance entre ambos

### Validación Cruzada (Cross-Validation)

Técnica para evaluar la robustez del modelo:
* **K-Fold CV**: Dividir datos en K subconjuntos, entrenar K veces
* **Stratified K-Fold**: Mantener proporción de clases en cada fold
* **Leave-One-Out (LOO)**: K = N (un dato para test en cada iteración)

## 💼 Aplicaciones Reales

### Clasificación
* 📞 **Telecom**: Predicción de churn, segmentación de clientes
* 🏦 **Banking**: Detección de fraude, aprobación de créditos
* 🏪 **Retail**: Recomendación de productos, predicción de compra
* 🏥 **Healthcare**: Diagnóstico de enfermedades, predicción de readmisión
* 📧 **Marketing**: Clasificación de emails (spam), respuesta a campañas

### Regresión
* 🏡 **Real Estate**: Valoración de propiedades, predicción de alquileres
* 📹 **Finance**: Predicción de precios de acciones, riesgo de cartera
* 🛒 **E-commerce**: Forecasting de demanda, pricing dinámico
* ⚡ **Energy**: Predicción de consumo eléctrico
* 🚗 **Transportation**: Estimación de tiempo de viaje, predicción de tráfico

## 🛠️ Herramientas en Databricks

* **PySpark ML**: Pipeline unificado para ML a gran escala
* **MLflow**: Tracking de experimentos, registro de modelos
* **Feature Store**: Gestión centralizada de features
* **AutoML**: Automated Machine Learning
* **Model Serving**: Despliegue de modelos en producción

## 📚 Recursos

* [PySpark ML Guide](https://spark.apache.org/docs/latest/ml-guide.html)
* [Supervised Learning - Coursera](https://www.coursera.org/learn/machine-learning)
* [Databricks ML in Production](https://docs.databricks.com/machine-learning/index.html)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/supervised_learning.html)

---

**Siguiente paso**: Explora `../Aprendizaje No Supervisado/` para aprender sobre clustering y reducción de dimensionalidad.
