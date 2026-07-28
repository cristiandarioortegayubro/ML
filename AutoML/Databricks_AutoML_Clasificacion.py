# Databricks notebook source
# DBTITLE 1,# Databricks AutoML - Clasificación (Churn Prediction)
# MAGIC %md
# MAGIC # Databricks AutoML - Clasificación Práctica
# MAGIC
# MAGIC ## 🎯 Objetivo
# MAGIC
# MAGIC En este notebook usaremos **Databricks AutoML** para crear un modelo de clasificación que prediga **customer churn** (cancelación de clientes).
# MAGIC
# MAGIC ### Lo que aprenderemos:
# MAGIC
# MAGIC 1. ✅ Cómo ejecutar AutoML con Python API
# MAGIC 2. ✅ Analizar los resultados y el leaderboard
# MAGIC 3. ✅ Comparar AutoML vs modelo manual (Decision Tree del notebook anterior)
# MAGIC 4. ✅ Entender qué modelos probó AutoML
# MAGIC 5. ✅ Registrar el mejor modelo en MLflow
# MAGIC 6. ✅ Ver el notebook generado automáticamente
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Dataset: Customer Churn
# MAGIC
# MAGIC **Problema de negocio:**  
# MAGIC Una empresa de telecomunicaciones quiere predecir qué clientes cancelarán su servicio.
# MAGIC
# MAGIC **Features:**
# MAGIC * `tenure`: Meses como cliente
# MAGIC * `monthly_charges`: Cargo mensual ($)
# MAGIC * `total_charges`: Total gastado ($)
# MAGIC * `contract`: Tipo de contrato (Month-to-month, One year, Two year)
# MAGIC * `internet_service`: Tipo de internet (DSL, Fiber optic, No)
# MAGIC * `payment_method`: Método de pago
# MAGIC * `senior_citizen`: Es adulto mayor (0/1)
# MAGIC * Y más...
# MAGIC
# MAGIC **Target:**
# MAGIC * `churn`: 1 = Canceló, 0 = Sigue activo
# MAGIC
# MAGIC **Dataset balanceado:**
# MAGIC * ~73% No Churn (clase 0)
# MAGIC * ~27% Churn (clase 1)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔧 Setup
# MAGIC
# MAGIC Primero vamos a crear un dataset sintético de churn para este ejemplo.

# COMMAND ----------

# DBTITLE 1,Crear dataset de ejemplo
# Crear dataset sintético de customer churn
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

np.random.seed(42)

n_samples = 2000

# Generar features
data = {
    'customer_id': range(1, n_samples + 1),
    'tenure': np.random.randint(1, 72, n_samples),  # Meses
    'monthly_charges': np.random.uniform(20, 120, n_samples),  # $
    'total_charges': np.random.uniform(100, 8000, n_samples),  # $
    'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
    'senior_citizen': np.random.binomial(1, 0.15, n_samples),
    'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
    'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
}

df = pd.DataFrame(data)

# Generar target con lógica de negocio
# Factores que aumentan churn:
# - Contrato mes a mes
# - Tenure bajo
# - Cargos altos
# - Sin servicios adicionales

churn_prob = 0.1  # Probabilidad base

# Ajustar probabilidad según features
churn_prob_adjusted = np.where(df['contract'] == 'Month-to-month', 0.4, 0.1)
churn_prob_adjusted = np.where(df['tenure'] < 12, churn_prob_adjusted * 2, churn_prob_adjusted)
churn_prob_adjusted = np.where(df['monthly_charges'] > 80, churn_prob_adjusted * 1.5, churn_prob_adjusted)
churn_prob_adjusted = np.where(df['online_security'] == 'No', churn_prob_adjusted * 1.3, churn_prob_adjusted)

# Clip entre 0 y 1
churn_prob_adjusted = np.clip(churn_prob_adjusted, 0, 1)

# Generar target
df['churn'] = np.random.binomial(1, churn_prob_adjusted)

# Convertir a Spark DataFrame
spark_df = spark.createDataFrame(df)

# Mostrar estadísticas
print("\n┌──────────────────────────────────────────────────┐")
print("│        DATASET DE CUSTOMER CHURN CREADO          │")
print("└──────────────────────────────────────────────────┘")
print(f"\n📊 Total registros: {spark_df.count():,}")
print(f"\n📋 Columnas ({len(spark_df.columns)}): {', '.join(spark_df.columns)}")

# Distribución de churn
churn_dist = spark_df.groupBy('churn').count().toPandas()
print("\n🎯 Distribución de Churn:")
for _, row in churn_dist.iterrows():
    label = "No Churn (0)" if row['churn'] == 0 else "Churn (1)"
    pct = row['count'] / n_samples * 100
    print(f"  {label}: {row['count']:,} ({pct:.1f}%)")

print("\n✅ Dataset listo para AutoML")

# Preview
display(spark_df.limit(5))

# COMMAND ----------

# DBTITLE 1,## 🤖 Ejecutar Databricks AutoML
# MAGIC %md
# MAGIC ## 🤖 Ejecutar Databricks AutoML
# MAGIC
# MAGIC ### Parámetros Clave
# MAGIC
# MAGIC ```python
# MAGIC automl.classify(
# MAGIC     dataset=spark_df,              # Spark DataFrame
# MAGIC     target_col="churn",            # Columna target
# MAGIC     primary_metric="f1",           # Métrica a optimizar
# MAGIC     timeout_minutes=15,            # Tiempo máximo
# MAGIC     max_trials=20,                 # Número máximo de modelos
# MAGIC     feature_store_lookups=None,    # Opcional: Feature Store
# MAGIC     exclude_cols=["customer_id"],  # Columnas a excluir
# MAGIC     pos_label=1                    # Clase positiva
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### Métricas Disponibles para Clasificación
# MAGIC
# MAGIC * `f1` ⭐ (recomendado para desbalance)
# MAGIC * `accuracy`
# MAGIC * `precision`
# MAGIC * `recall`
# MAGIC * `log_loss`
# MAGIC * `roc_auc`
# MAGIC
# MAGIC ### Modelos que AutoML Probará
# MAGIC
# MAGIC * Decision Tree Classifier
# MAGIC * Random Forest Classifier
# MAGIC * **XGBoost Classifier** ⭐
# MAGIC * **LightGBM Classifier** ⭐ (usualmente el mejor)
# MAGIC * Logistic Regression
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **⚠️ Nota:** El siguiente código tomará ~10-15 minutos en ejecutar.

# COMMAND ----------

# DBTITLE 1,Ejecutar AutoML
from databricks import automl
import mlflow

print("┌──────────────────────────────────────────────────┐")
print("│        INICIANDO DATABRICKS AUTOML              │")
print("└──────────────────────────────────────────────────┘")
print("\n🔍 Problema: Clasificación Binaria (Churn)")
print("⏱️  Tiempo estimado: 10-15 minutos")
print("🎯 Métrica objetivo: F1 Score")
print("🤖 Modelos a probar: Decision Tree, Random Forest, XGBoost, LightGBM, etc.")
print("\n🚀 Ejecutando AutoML...\n")

# Ejecutar AutoML
summary = automl.classify(
    dataset=spark_df,
    target_col="churn",
    primary_metric="f1",
    timeout_minutes=15,
    max_trials=20,
    exclude_cols=["customer_id"],
    pos_label=1
)

print("\n\n✅ ¡AutoML completado!")
print("\n┌──────────────────────────────────────────────────┐")
print("│             MEJOR MODELO ENCONTRADO               │")
print("└──────────────────────────────────────────────────┘")
print(f"\n🤖 Algoritmo: {summary.best_trial.model_description}")
print(f"📋 MLflow Run ID: {summary.best_trial.mlflow_run_id}")
print(f"📓 Notebook generado: {summary.best_trial.notebook_path}")

print("\n🎯 Métricas del Mejor Modelo:")
for metric, value in summary.best_trial.metrics.items():
    if 'val_' in metric:  # Solo métricas de validación
        metric_name = metric.replace('val_', '').upper()
        print(f"  {metric_name}: {value:.4f}")

# COMMAND ----------

# DBTITLE 1,## 📊 Analizar Resultados
# MAGIC %md
# MAGIC ## 📊 Analizar Resultados del AutoML
# MAGIC
# MAGIC ### 🔍 Exploración del Summary
# MAGIC
# MAGIC El objeto `summary` contiene:
# MAGIC
# MAGIC * `best_trial`: Información del mejor modelo
# MAGIC * `trials`: Lista de todos los trials ejecutados
# MAGIC * `experiment`: MLflow Experiment donde se guardaron los runs
# MAGIC
# MAGIC ### 🏆 Leaderboard de Modelos
# MAGIC
# MAGIC Vamos a ver todos los modelos que AutoML probó y compararlos.

# COMMAND ----------

# DBTITLE 1,Ver todos los modelos probados (Leaderboard)
import pandas as pd

print("┌──────────────────────────────────────────────────────────────────────────────────────────┐")
print("│                         LEADERBOARD DE MODELOS AUTOML                          │")
print("└──────────────────────────────────────────────────────────────────────────────────────────┘")

# Crear dataframe con todos los trials
leaderboard_data = []

for i, trial in enumerate(summary.trials, 1):
    leaderboard_data.append({
        'Rank': i,
        'Model': trial.model_description,
        'F1 Score': trial.metrics.get('val_f1_score', 0),
        'Accuracy': trial.metrics.get('val_accuracy_score', 0),
        'Precision': trial.metrics.get('val_precision_score', 0),
        'Recall': trial.metrics.get('val_recall_score', 0),
        'ROC AUC': trial.metrics.get('val_roc_auc_score', 0)
    })

leaderboard_df = pd.DataFrame(leaderboard_data)

# Ordenar por F1 (métrica primaria)
leaderboard_df = leaderboard_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)
leaderboard_df['Rank'] = range(1, len(leaderboard_df) + 1)

print(f"\n📈 Total de modelos probados: {len(leaderboard_df)}\n")

# Mostrar top 10
print("🏆 TOP 10 MODELOS:\n")
print(leaderboard_df.head(10).to_string(index=False))

print("\n" + "="*90)
print("\n🥇 GANADOR:")
best = leaderboard_df.iloc[0]
print(f"  Modelo: {best['Model']}")
print(f"  F1 Score: {best['F1 Score']:.4f}")
print(f"  Accuracy: {best['Accuracy']:.4f}")
print(f"  ROC AUC: {best['ROC AUC']:.4f}")

print("\n📉 Diferencia con el segundo mejor:")
if len(leaderboard_df) > 1:
    second_best = leaderboard_df.iloc[1]
    f1_diff = (best['F1 Score'] - second_best['F1 Score']) * 100
    print(f"  +{f1_diff:.2f} puntos porcentuales en F1")
    print(f"  Segundo lugar: {second_best['Model']} (F1: {second_best['F1 Score']:.4f})")

# COMMAND ----------

# DBTITLE 1,## 🔬 Comparación: AutoML vs Manual
# MAGIC %md
# MAGIC ## 🔬 Comparación: AutoML vs Modelo Manual
# MAGIC
# MAGIC ### Recordatorio: Notebooks Anteriores
# MAGIC
# MAGIC En los notebooks de **Aprendizaje Supervisado/Clasificación**, creamos manualmente:
# MAGIC
# MAGIC * Decision Tree Classifier
# MAGIC * Random Forest Classifier
# MAGIC
# MAGIC Con preprocesamiento manual, feature engineering, y tuning de hiperparámetros.
# MAGIC
# MAGIC ### Simulación de Comparación
# MAGIC
# MAGIC Vamos a simular un modelo manual simple para comparar con AutoML.

# COMMAND ----------

# DBTITLE 1,Entrenar modelo manual para comparar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import time

print("┌──────────────────────────────────────────────────┐")
print("│      ENTRENAR MODELO MANUAL (RANDOM FOREST)      │")
print("└──────────────────────────────────────────────────┘")

start_time = time.time()

# Convertir a Pandas
df_manual = spark_df.toPandas()

# Preprocesamiento manual
print("\n🧹 Preprocesamiento manual...")

# Separar features y target
X = df_manual.drop(['churn', 'customer_id'], axis=1)
y = df_manual['churn']

# Encoding manual de categóricas
categorical_cols = ['contract', 'internet_service', 'payment_method', 'online_security', 'tech_support']

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

print("  ✅ Categorical encoding completado")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"  ✅ Train/test split: {len(X_train)} train, {len(X_test)} test")

# Entrenar Random Forest con hiperparámetros default
print("\n🌳 Entrenando Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("  ✅ Modelo entrenado")

# Evaluar
print("\n📊 Evaluando en test set...")

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

manual_f1 = f1_score(y_test, y_pred)
manual_accuracy = accuracy_score(y_test, y_pred)
manual_precision = precision_score(y_test, y_pred)
manual_recall = recall_score(y_test, y_pred)
manual_roc_auc = roc_auc_score(y_test, y_pred_proba)

manual_time = time.time() - start_time

print("\n✅ Evaluación completada")

print("\n┌──────────────────────────────────────────────────┐")
print("│            RESULTADOS DEL MODELO MANUAL           │")
print("└──────────────────────────────────────────────────┘")
print(f"\nF1 SCORE: {manual_f1:.4f}")
print(f"ACCURACY: {manual_accuracy:.4f}")
print(f"PRECISION: {manual_precision:.4f}")
print(f"RECALL: {manual_recall:.4f}")
print(f"ROC AUC: {manual_roc_auc:.4f}")
print(f"\n⏱️  TIEMPO: {manual_time:.1f} segundos")

# COMMAND ----------

# DBTITLE 1,Tabla comparativa AutoML vs Manual
print("\n\n")
print("┌──────────────────────────────────────────────────────────────────────────────────────────┐")
print("│                      ⚡ AUTOML vs MANUAL - COMPARACIÓN COMPLETA ⚡                     │")
print("└──────────────────────────────────────────────────────────────────────────────────────────┘")

# Obtener métricas de AutoML
automl_f1 = summary.best_trial.metrics.get('val_f1_score', 0)
automl_accuracy = summary.best_trial.metrics.get('val_accuracy_score', 0)
automl_precision = summary.best_trial.metrics.get('val_precision_score', 0)
automl_recall = summary.best_trial.metrics.get('val_recall_score', 0)
automl_roc_auc = summary.best_trial.metrics.get('val_roc_auc_score', 0)

comparison = pd.DataFrame({
    'Métrica': ['F1 Score', 'Accuracy', 'Precision', 'Recall', 'ROC AUC'],
    'AutoML': [automl_f1, automl_accuracy, automl_precision, automl_recall, automl_roc_auc],
    'Manual (RF)': [manual_f1, manual_accuracy, manual_precision, manual_recall, manual_roc_auc]
})

# Calcular diferencia
comparison['Diferencia'] = comparison['AutoML'] - comparison['Manual (RF)']
comparison['Mejora (%)'] = (comparison['Diferencia'] / comparison['Manual (RF)']) * 100

print("\n📊 COMPARACIÓN DE MÉTRICAS:\n")
print(comparison.to_string(index=False))

# Resumen
print("\n" + "="*90)
print("\n🏆 GANADOR:")

if automl_f1 > manual_f1:
    improvement = ((automl_f1 - manual_f1) / manual_f1) * 100
    print(f"  🤖 AUTOML ganó por {improvement:.1f}% en F1 Score")
    print(f"  Modelo: {summary.best_trial.model_description}")
    print(f"  F1: {automl_f1:.4f} vs {manual_f1:.4f}")
else:
    improvement = ((manual_f1 - automl_f1) / automl_f1) * 100
    print(f"  👨‍💻 MANUAL ganó por {improvement:.1f}% en F1 Score")
    print(f"  F1: {manual_f1:.4f} vs {automl_f1:.4f}")

print("\n⏱️  TIEMPO:")
print(f"  AutoML: ~10-15 minutos (probando {len(summary.trials)} modelos)")
print(f"  Manual: {manual_time:.1f} segundos (1 solo modelo)")

print("\n🔧 ESFUERZO:")
print("  AutoML: 5 líneas de código")
print("  Manual: ~50 líneas de código (preprocesamiento + entrenamiento + evaluación)")

print("\n📝 CONCLUSIÓN:")
if automl_f1 > manual_f1:
    print("  ✅ AutoML logró MEJOR performance con MENOS esfuerzo")
    print("  ✅ Probó múltiples algoritmos y encontró el mejor automáticamente")
else:
    print("  ⚠️  Manual logró mejor performance, pero AutoML estuvo cerca")
    print("  ✅ AutoML sigue siendo excelente para baseline rápido")

# COMMAND ----------

# DBTITLE 1,## 📚 Explorar Notebook Generado
# MAGIC %md
# MAGIC ## 📚 Explorar el Notebook Generado por AutoML
# MAGIC
# MAGIC ### 🔍 Lo que contiene el notebook:
# MAGIC
# MAGIC 1. **Data Exploration**:
# MAGIC    - Estadísticas descriptivas
# MAGIC    - Visualizaciones de distribuciones
# MAGIC    - Correlaciones
# MAGIC    - Detección de outliers
# MAGIC
# MAGIC 2. **Preprocesamiento Automático**:
# MAGIC    - Imputación de valores faltantes
# MAGIC    - Encoding de categóricas (One-Hot, Label, Target)
# MAGIC    - Scaling de numéricas
# MAGIC    - Feature engineering
# MAGIC
# MAGIC 3. **Model Training**:
# MAGIC    - Código completo del mejor modelo
# MAGIC    - Hiperparámetros optimizados
# MAGIC    - Train/validation split
# MAGIC
# MAGIC 4. **Evaluation**:
# MAGIC    - Métricas detalladas
# MAGIC    - Matriz de confusión
# MAGIC    - Feature importance
# MAGIC    - ROC Curve, Precision-Recall Curve
# MAGIC
# MAGIC 5. **SHAP Explainability**:
# MAGIC    - Feature contributions
# MAGIC    - Interpretabilidad del modelo
# MAGIC
# MAGIC ### ✅ El notebook es 100% reproducible y editable
# MAGIC
# MAGIC **Puedes:**
# MAGIC * Copiar el código
# MAGIC * Modificar hiperparámetros
# MAGIC * Agregar features custom
# MAGIC * Reentrenar con datos nuevos
# MAGIC
# MAGIC Veamos la ruta del notebook:

# COMMAND ----------

# DBTITLE 1,Ver información del notebook generado
print("┌──────────────────────────────────────────────────────────────────────────────────────────┐")
print("│                    📓 NOTEBOOK GENERADO POR AUTOML                          │")
print("└──────────────────────────────────────────────────────────────────────────────────────────┘")

print(f"\n📋 Ruta del notebook:")
print(f"  {summary.best_trial.notebook_path}")

print(f"\n🔗 MLflow Run ID:")
print(f"  {summary.best_trial.mlflow_run_id}")

print(f"\n💾 Model URI:")
model_uri = f"runs:/{summary.best_trial.mlflow_run_id}/model"
print(f"  {model_uri}")

print("\n👁️  Para ver el notebook generado:")
print("  1. Ve al workspace de Databricks")
print("  2. Navega a la ruta mostrada arriba")
print("  3. Abre el notebook y explora el código completo")

print("\n🚀 Para ver el experiment en MLflow UI:")
print("  1. Menú izquierdo → 'Experiments'")
print(f"  2. Busca el experiment asociado a este run")
print("  3. Compara todos los modelos probados")

print("\n✅ El notebook incluye:")
print("  • Data exploration completo")
print("  • Preprocesamiento automático")
print("  • Código del modelo ganador")
print("  • Hiperparámetros optimizados")
print("  • Evaluación con múltiples métricas")
print("  • Feature importance")
print("  • SHAP explainability")
print("  • 100% reproducible y editable")

# COMMAND ----------

# DBTITLE 1,## 📝 Conclusiones
# MAGIC %md
# MAGIC ## 📝 Conclusiones del Notebook
# MAGIC
# MAGIC ### ✅ Lo que aprendimos:
# MAGIC
# MAGIC 1. **AutoML es increíblemente eficiente**:
# MAGIC    - 5 líneas de código
# MAGIC    - Prueba múltiples algoritmos
# MAGIC    - Optimización automática de hiperparámetros
# MAGIC    - Genera notebook explicativo
# MAGIC
# MAGIC 2. **Performance comparable o superior** a modelos manuales:
# MAGIC    - Especialmente para datasets tabulares
# MAGIC    - LightGBM y XGBoost suelen ser los ganadores
# MAGIC
# MAGIC 3. **Integración perfecta con MLflow**:
# MAGIC    - Todos los runs trackeados automáticamente
# MAGIC    - Fácil comparación de modelos
# MAGIC    - Model Registry integrado
# MAGIC
# MAGIC 4. **Excelente punto de partida**:
# MAGIC    - Usa AutoML para baseline
# MAGIC    - Analiza el notebook generado
# MAGIC    - Itera manualmente desde ahí si es necesario
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚀 Próximos Pasos
# MAGIC
# MAGIC **En el siguiente notebook** (`Databricks_AutoML_Regresion.ipynb`):
# MAGIC * AutoML para regresión
# MAGIC * Predicción de precios
# MAGIC * Métricas de regresión (RMSE, MAE, R²)
# MAGIC
# MAGIC **Luego**:
# MAGIC * Genie Code como asistente de ML
# MAGIC * MLflow tracking end-to-end
# MAGIC * Feature Store
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 Key Takeaways
# MAGIC
# MAGIC > **"AutoML no reemplaza a los Data Scientists, sino que los hace 10x más productivos."**
# MAGIC
# MAGIC ✅ **Cuándo usar AutoML:**
# MAGIC * Baseline rápido
# MAGIC * Exploración inicial
# MAGIC * Proyectos con poco tiempo
# MAGIC * Democratización de ML
# MAGIC
# MAGIC 🔧 **Cuándo iterar manualmente:**
# MAGIC * Necesitas features custom
# MAGIC * Arquitecturas especializadas
# MAGIC * Performance crítico
# MAGIC * Deep understanding del problema
# MAGIC
# MAGIC **Mejor práctica:** 🤝 **AutoML + Manual = Winning Combination**

# COMMAND ----------

