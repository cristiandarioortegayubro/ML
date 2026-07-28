# Clasificación - Predicción de Churn

## 📂 Contenido

Esta carpeta contiene ejemplos de **Machine Learning supervisado para Clasificación** usando PySpark ML.

### Notebooks:

1. **Arbol_Decision_Clasificacion.py**
   - **Algoritmo**: Decision Tree Classifier
   - **Problema**: Predicción de churn (abandono de clientes)
   - **Dataset**: 10,000 clientes de telecomunicaciones (sintético)
   - **Variables**: Antigüedad, gasto mensual, tipo contrato, llamadas soporte, etc.
   - **Métricas**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
   - **Caso de uso**: Identificar clientes en riesgo de abandono para campañas de retención

## 🎯 Objetivo

Aprender a construir modelos de clasificación binaria que predicen una categoría (Churn Sí/No) basándose en características del cliente.

## 📊 Concepto: Clasificación

**Clasificación** es una tarea de Machine Learning supervisado donde el objetivo es predecir una **categoría** o **clase** discreta.

### Tipos:
* **Binaria**: 2 clases (ej: Churn Sí/No, Spam/No Spam)
* **Multiclase**: 3+ clases (ej: Tipo de producto: A/B/C)

### Algoritmos comunes:
* **Decision Trees**: Reglas de decisión interpretables
* **Random Forest**: Ensemble de árboles (mayor accuracy)
* **Gradient Boosting**: Árboles secuenciales optimizados
* **Logistic Regression**: Modelo lineal probabilístico
* **Support Vector Machines**: Hiperplanos de separación

## 🚀 Cómo usar estos notebooks

1. **Abrir en Databricks**:
   - Navega a la carpeta `ML/Clasificacion` en tu workspace
   - Abre `Arbol_Decision_Clasificacion`

2. **Ejecutar**:
   - Asegúrate de tener un cluster activo (Serverless o standard)
   - Ejecuta todas las celdas secuencialmente

3. **Experimentar**:
   - Cambia hiperparámetros (`maxDepth`, `minInstancesPerNode`)
   - Prueba con tus propios datos reales
   - Compara con otros algoritmos (Random Forest, GBT)

## 📚 Recursos

* [PySpark ML Classification](https://spark.apache.org/docs/latest/ml-classification-regression.html#classification)
* [Churn Prediction Guide](https://www.databricks.com/glossary/churn-prediction)
* [Decision Trees Explained](https://scikit-learn.org/stable/modules/tree.html)

## 💼 Aplicaciones reales

* **Telecom**: Predicción de churn
* **Banking**: Detección de fraude
* **Retail**: Predicción de compra
* **Healthcare**: Diagnóstico de enfermedades
* **Marketing**: Respuesta a campañas

---

**Siguiente paso**: Explora la carpeta `Regresion` para ver cómo predecir valores numéricos continuos.