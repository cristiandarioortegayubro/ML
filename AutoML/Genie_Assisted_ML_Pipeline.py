# Databricks notebook source
# DBTITLE 1,# Genie Code: Asistente IA para ML
# MAGIC %md
# MAGIC # Genie Code - Asistente IA para Machine Learning
# MAGIC
# MAGIC ## 🤖 ¿Qué es Genie Code?
# MAGIC
# MAGIC **Genie Code** es el asistente de IA de Databricks que ayuda a desarrollar pipelines de ML más rápido.
# MAGIC
# MAGIC ### 🎯 Capacidades
# MAGIC
# MAGIC * ✅ **Generar código**: "Create a Random Forest for classification"
# MAGIC * ✅ **Feature engineering**: "Add polynomial features for this dataset"
# MAGIC * ✅ **Debugging**: "Why is my model overfitting?"
# MAGIC * ✅ **Explicar código**: "Explain this PySpark transformation"
# MAGIC * ✅ **Optimizar**: "How can I improve model performance?"
# MAGIC
# MAGIC ### 💡 Workflow
# MAGIC
# MAGIC 1. Describir lo que necesitas en lenguaje natural
# MAGIC 2. Genie genera código
# MAGIC 3. Ejecutar y ajustar
# MAGIC 4. Iterar con Genie

# COMMAND ----------

# DBTITLE 1,Ejemplo 1: Generar dataset
# MAGIC %md
# MAGIC ## Ejemplo 1: Generar Dataset
# MAGIC
# MAGIC ### 💬 Prompt para Genie:
# MAGIC
# MAGIC ```
# MAGIC Generate a synthetic dataset for customer churn prediction with:
# MAGIC - 1000 rows
# MAGIC - Features: age, tenure, monthly_charges, total_charges, contract_type
# MAGIC - Target: churn (binary)
# MAGIC ```
# MAGIC
# MAGIC ### 🤖 Genie generaría:
# MAGIC
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC import numpy as np
# MAGIC
# MAGIC np.random.seed(42)
# MAGIC n = 1000
# MAGIC
# MAGIC df_churn = pd.DataFrame({
# MAGIC     'age': np.random.randint(18, 80, n),
# MAGIC     'tenure': np.random.randint(1, 72, n),
# MAGIC     'monthly_charges': np.random.uniform(20, 120, n),
# MAGIC     'total_charges': lambda x: x['monthly_charges'] * x['tenure'],
# MAGIC     'contract_type': np.random.choice(['Monthly', 'Annual', 'Biennial'], n),
# MAGIC     'churn': np.random.binomial(1, 0.25, n)
# MAGIC })
# MAGIC ```
# MAGIC
# MAGIC ✅ Ahorra tiempo en setup de datos

# COMMAND ----------

# DBTITLE 1,Ejemplo 2: Feature Engineering
# MAGIC %md
# MAGIC ## Ejemplo 2: Feature Engineering
# MAGIC
# MAGIC ### 💬 Prompt:
# MAGIC
# MAGIC ```
# MAGIC Create these features:
# MAGIC 1. Interaction: monthly_charges * tenure
# MAGIC 2. Binning: age into groups (Young, Adult, Senior)
# MAGIC 3. One-hot encode contract_type
# MAGIC ```
# MAGIC
# MAGIC ### 🤖 Genie generaría:
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.preprocessing import LabelEncoder
# MAGIC
# MAGIC # Interaction
# MAGIC df_churn['total_spend'] = df_churn['monthly_charges'] * df_churn['tenure']
# MAGIC
# MAGIC # Binning
# MAGIC df_churn['age_group'] = pd.cut(
# MAGIC     df_churn['age'], 
# MAGIC     bins=[0, 30, 60, 100], 
# MAGIC     labels=['Young', 'Adult', 'Senior']
# MAGIC )
# MAGIC
# MAGIC # One-hot encoding
# MAGIC df_encoded = pd.get_dummies(df_churn, columns=['contract_type'], prefix='contract')
# MAGIC ```
# MAGIC
# MAGIC ✅ Genera transformaciones comunes instantáneamente

# COMMAND ----------

# DBTITLE 1,Ejemplo 3: Entrenar modelo
# MAGIC %md
# MAGIC ## Ejemplo 3: Entrenar Modelo
# MAGIC
# MAGIC ### 💬 Prompt:
# MAGIC
# MAGIC ```
# MAGIC Train a Random Forest classifier with:
# MAGIC - Train/test split 80/20
# MAGIC - Hyperparameters: n_estimators=100, max_depth=10
# MAGIC - Evaluate with accuracy, precision, recall, F1
# MAGIC ```
# MAGIC
# MAGIC ### 🤖 Genie generaría:
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.model_selection import train_test_split
# MAGIC from sklearn.ensemble import RandomForestClassifier
# MAGIC from sklearn.metrics import classification_report
# MAGIC
# MAGIC X = df_encoded.drop('churn', axis=1)
# MAGIC y = df_encoded['churn']
# MAGIC
# MAGIC X_train, X_test, y_train, y_test = train_test_split(
# MAGIC     X, y, test_size=0.2, random_state=42, stratify=y
# MAGIC )
# MAGIC
# MAGIC model = RandomForestClassifier(
# MAGIC     n_estimators=100, 
# MAGIC     max_depth=10, 
# MAGIC     random_state=42
# MAGIC )
# MAGIC model.fit(X_train, y_train)
# MAGIC
# MAGIC y_pred = model.predict(X_test)
# MAGIC print(classification_report(y_test, y_pred))
# MAGIC ```
# MAGIC
# MAGIC ✅ Setup completo de entrenamiento en segundos

# COMMAND ----------

# DBTITLE 1,Ejemplo 4: Debugging
# MAGIC %md
# MAGIC ## Ejemplo 4: Debugging con Genie
# MAGIC
# MAGIC ### 🐛 Problema:
# MAGIC
# MAGIC ```python
# MAGIC # Este código da error
# MAGIC model.fit(X_train_scaled, y_train)
# MAGIC # ValueError: Input contains NaN
# MAGIC ```
# MAGIC
# MAGIC ### 💬 Prompt:
# MAGIC
# MAGIC ```
# MAGIC My model training fails with "Input contains NaN". 
# MAGIC How do I find and handle missing values?
# MAGIC ```
# MAGIC
# MAGIC ### 🤖 Genie respondería:
# MAGIC
# MAGIC ```python
# MAGIC # 1. Detectar valores faltantes
# MAGIC print(X_train.isnull().sum())
# MAGIC
# MAGIC # 2. Estrategias de manejo:
# MAGIC
# MAGIC # Opción A: Imputar con media
# MAGIC from sklearn.impute import SimpleImputer
# MAGIC imputer = SimpleImputer(strategy='mean')
# MAGIC X_train_imputed = imputer.fit_transform(X_train)
# MAGIC X_test_imputed = imputer.transform(X_test)
# MAGIC
# MAGIC # Opción B: Eliminar filas con NaN
# MAGIC X_train_clean = X_train.dropna()
# MAGIC y_train_clean = y_train[X_train.index]
# MAGIC
# MAGIC # Recomendación: Opción A si <10% missing
# MAGIC ```
# MAGIC
# MAGIC ✅ Diagnostica y soluciona errores

# COMMAND ----------

# DBTITLE 1,Ejemplo 5: Optimización
# MAGIC %md
# MAGIC ## Ejemplo 5: Optimización
# MAGIC
# MAGIC ### 💬 Prompt:
# MAGIC
# MAGIC ```
# MAGIC My model has 75% accuracy but 30% recall. 
# MAGIC How can I improve recall for churn prediction?
# MAGIC ```
# MAGIC
# MAGIC ### 🤖 Genie sugeriría:
# MAGIC
# MAGIC ```python
# MAGIC # 1. Ajustar threshold de clasificación
# MAGIC y_pred_proba = model.predict_proba(X_test)[:, 1]
# MAGIC y_pred_custom = (y_pred_proba > 0.3).astype(int)  # Threshold más bajo
# MAGIC
# MAGIC # 2. Balancear clases con class_weight
# MAGIC model_balanced = RandomForestClassifier(
# MAGIC     n_estimators=100,
# MAGIC     class_weight='balanced',  # Penaliza más clase minoritaria
# MAGIC     random_state=42
# MAGIC )
# MAGIC
# MAGIC # 3. SMOTE (oversampling de clase minoritaria)
# MAGIC from imblearn.over_sampling import SMOTE
# MAGIC smote = SMOTE(random_state=42)
# MAGIC X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# MAGIC
# MAGIC # 4. Usar F1 o Recall como métrica de optimización
# MAGIC from sklearn.model_selection import GridSearchCV
# MAGIC grid_search = GridSearchCV(
# MAGIC     model, 
# MAGIC     param_grid, 
# MAGIC     scoring='recall'  # Optimizar recall
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ✅ Sugiere múltiples estrategias

# COMMAND ----------

# DBTITLE 1,Mejores Prácticas con Genie
# MAGIC %md
# MAGIC ## 📚 Mejores Prácticas con Genie
# MAGIC
# MAGIC ### ✅ Cómo hacer buenos prompts
# MAGIC
# MAGIC 1. **Ser específico**
# MAGIC    * ❌ "Train a model"
# MAGIC    * ✅ "Train a Random Forest with 100 trees for binary classification"
# MAGIC
# MAGIC 2. **Dar contexto**
# MAGIC    * ❌ "Fix this error"
# MAGIC    * ✅ "My RandomForest gives ValueError: Input contains NaN during fit()"
# MAGIC
# MAGIC 3. **Especificar formato de salida**
# MAGIC    * ✅ "Generate code using sklearn"
# MAGIC    * ✅ "Explain in 3 steps"
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚀 Casos de Uso Principales
# MAGIC
# MAGIC | Tarea | Prompt Ejemplo |
# MAGIC |-------|----------------|
# MAGIC | **Generar código base** | "Create a classification pipeline with scaling and Random Forest" |
# MAGIC | **Feature engineering** | "Create polynomial features of degree 2 for X1 and X2" |
# MAGIC | **Debugging** | "Why is my model giving 100% accuracy? Check for data leakage" |
# MAGIC | **Optimización** | "My model is overfitting (train 95%, test 70%). Suggest solutions" |
# MAGIC | **Explicar código** | "Explain this GridSearchCV code line by line" |
# MAGIC | **Comparar opciones** | "Compare RandomForest vs XGBoost for imbalanced classification" |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚠️ Limitaciones
# MAGIC
# MAGIC * **No reemplaza expertise**: Genie acelera, pero necesitas entender ML
# MAGIC * **Revisar código generado**: Siempre validar antes de usar
# MAGIC * **Iterar**: Primer intento puede no ser perfecto, refinar prompt
# MAGIC * **Datos sensibles**: No compartir datos confidenciales en prompts

# COMMAND ----------

# DBTITLE 1,## Conclusiones
# MAGIC %md
# MAGIC ## 📝 Conclusiones
# MAGIC
# MAGIC ### 🎯 Key Takeaways
# MAGIC
# MAGIC 1. **Genie Code acelera desarrollo de ML**
# MAGIC    - Genera código boilerplate instantáneamente
# MAGIC    - Sugiere mejores prácticas
# MAGIC    - Debuggea errores
# MAGIC
# MAGIC 2. **Mejores prompts = Mejores resultados**
# MAGIC    - Ser específico
# MAGIC    - Dar contexto
# MAGIC    - Iterar y refinar
# MAGIC
# MAGIC 3. **Casos de uso principales**
# MAGIC    - Setup rápido de pipelines
# MAGIC    - Feature engineering
# MAGIC    - Debugging y optimización
# MAGIC    - Aprendizaje de nuevas técnicas
# MAGIC
# MAGIC 4. **Complementa, no reemplaza**
# MAGIC    - Genie acelera, pero necesitas expertise ML
# MAGIC    - Revisar código generado
# MAGIC    - Validar resultados
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚀 Próximos Pasos
# MAGIC
# MAGIC * **MLflow_Experiment_Tracking**: Tracking end-to-end
# MAGIC * **Producción**: Desplegar modelos
# MAGIC * **Monitoreo**: Model drift y retraining
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎉 ¡Felicidades!
# MAGIC
# MAGIC Ahora sabes cómo usar **Genie Code como asistente de ML**. 🚀
# MAGIC
# MAGIC **Prueba Genie ahora mismo en el chat de Databricks!**

# COMMAND ----------

