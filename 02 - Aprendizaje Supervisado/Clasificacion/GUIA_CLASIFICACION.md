# Clasificación - Predicción de Churn

## 📂 Contenido

Esta carpeta contiene notebooks **teóricos y prácticos** de **Machine Learning supervisado para Clasificación** usando PySpark ML.

### Notebooks:

#### 📖 Teoría

**1. Teoria_Arboles_Decision.ipynb** (7 celdas, 311 líneas)
- **Estructura de árboles**: Nodos, ramas, hojas, definición formal
- **Algoritmo ID3**: Pseudocódigo y proceso de construcción
- **Entropía y Ganancia de Información**: Fórmulas, propiedades, selección de atributos
- **Índice Gini**: Impureza de Gini, comparación con entropía
- **Poda (Pruning)**: Pre-pruning, post-pruning, control de overfitting
- **Ventajas y Desventajas**: Interpretabilidad, inestabilidad, fronteras de decisión
- **Conclusiones**: Conceptos clave y próximos pasos

**2. Teoria_Random_Forest.ipynb** (7 celdas, ~400 líneas)
- **Ensemble Learning**: Concepto de bosques aleatorios, historia
- **Algoritmo**: Pseudocódigo, bagging, feature randomness, agregación
- **Ventajas y Desventajas**: Comparación con árboles individuales
- **Hiperparámetros**: numTrees, featureSubsetStrategy, maxDepth
- **Feature Importance**: Cálculo e interpretación
- **Out-Of-Bag (OOB) Error**: Validación implícita
- **Fórmulas clave**: $m = \sqrt{p}$, predicción por voto mayoritario

**Fórmulas clave:**
- Entropía: $H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$
- Ganancia de Información: $IG(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)$
- Índice Gini: $\text{Gini}(S) = 1 - \sum_{i=1}^{c} p_i^2$

#### 💻 Práctica

**1. Arbol_Decision_Clasificacion.ipynb**
- **Algoritmo**: Decision Tree Classifier
- **Problema**: Predicción de churn (abandono de clientes)
- **Dataset**: 10,000 clientes de telecomunicaciones (sintético)
- **Variables**: Antigüedad, gasto mensual, tipo contrato, llamadas soporte, etc.
- **Métricas**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Caso de uso**: Identificar clientes en riesgo de abandono para campañas de retención

**2. Random_Forest_Clasificacion.ipynb**
- **Algoritmo**: Random Forest Classifier (100 árboles)
- **Problema**: Mismo dataset de churn
- **Comparación**: vs Decision Tree (mejora en todas las métricas)
- **Feature Importance**: Identificación de variables más relevantes
- **Resultados**: ~92% accuracy vs ~88% de Decision Tree
- **Trade-off**: Mayor accuracy a cambio de velocidad e interpretabilidad

## 🎯 Objetivo

Aprender a construir modelos de clasificación binaria que predicen una **categoría** (Churn Sí/No) basándose en características del cliente.

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

## 🚀 Orden de Estudio Recomendado

1. **Primero**: Lee `Teoria_Arboles_Decision`
   - Comprende cómo funcionan los árboles de decisión
   - Estudia las fórmulas de entropía y Gini
   - Aprende sobre poda y overfitting

2. **Segundo**: Ejecuta `Arbol_Decision_Clasificacion`
   - Aplica lo aprendido en un problema real
   - Observa cómo se calculan las métricas
   - Experimenta con diferentes hiperparámetros

3. **Tercero**: Lee `Teoria_Random_Forest`
   - Comprende ensemble learning y bagging
   - Estudia cómo se combinan múltiples árboles
   - Aprende sobre feature importance y OOB error

4. **Cuarto**: Ejecuta `Random_Forest_Clasificacion`
   - Compara Random Forest vs Decision Tree
   - Analiza feature importance
   - Observa la mejora en accuracy y robustez

## 💡 Cómo usar estos notebooks

### Notebook Teórico:
1. Lee secuencialmente todas las celdas
2. Toma notas de las fórmulas clave
3. Comprende las ventajas y limitaciones
4. No ejecutes (es contenido markdown)

### Notebook Práctico:
1. Asegúrate de tener un cluster activo
2. Ejecuta todas las celdas secuencialmente
3. Observa resultados y visualizaciones
4. Experimenta:
   - Cambia `maxDepth` (profundidad máxima)
   - Ajusta `minInstancesPerNode` (mínimo de ejemplos)
   - Prueba con tus propios datos

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
