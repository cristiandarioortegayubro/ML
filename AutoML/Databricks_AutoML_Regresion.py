# Databricks notebook source
# DBTITLE 1,# Databricks AutoML - Regresión
# MAGIC %md
# MAGIC # Databricks AutoML para Regresión
# MAGIC ## Predicción de Precios de Propiedades
# MAGIC
# MAGIC ### 🎯 Objetivo
# MAGIC Demostrar **Databricks AutoML** en un problema de **regresión**: predecir precio de propiedades.
# MAGIC
# MAGIC ### 📊 Workflow
# MAGIC 1. Generar dataset sintético de propiedades
# MAGIC 2. Ejecutar Databricks AutoML para regresión
# MAGIC 3. Comparar con modelo manual
# MAGIC 4. Analizar notebook generado
# MAGIC 5. Evaluar métricas (RMSE, MAE, R²)

# COMMAND ----------

# DBTITLE 1,Generar dataset de propiedades
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 1000

df_properties = pd.DataFrame({
    'sqft': np.random.uniform(500, 3000, n),
    'bedrooms': np.random.randint(1, 6, n),
    'bathrooms': np.random.randint(1, 4, n),
    'age_years': np.random.randint(0, 50, n),
    'location_score': np.random.uniform(1, 10, n),
    'has_garage': np.random.choice([0, 1], n),
    'has_pool': np.random.choice([0, 1], n, p=[0.7, 0.3])
})

df_properties['price'] = (
    50000 + 
    100 * df_properties['sqft'] + 
    20000 * df_properties['bedrooms'] +
    15000 * df_properties['bathrooms'] +
    (-500 * df_properties['age_years']) +
    10000 * df_properties['location_score'] +
    25000 * df_properties['has_garage'] +
    30000 * df_properties['has_pool'] +
    np.random.normal(0, 20000, n)
)

print("🏠 DATASET DE PROPIEDADES")
print(df_properties.head())
print(f"\n📊 Precio promedio: ${df_properties['price'].mean():,.0f}")

# COMMAND ----------

# DBTITLE 1,Código para ejecutar AutoML
print("""
🤖 DATABRICKS AutoML - REGRESIÓN
══════════════════════════════════

from databricks import automl

train_df = spark.createDataFrame(df_properties)

summary = automl.regress(
    dataset=train_df,
    target_col='price',
    primary_metric='rmse',
    timeout_minutes=15,
    max_trials=20
)

print(f"✅ Mejor modelo: {summary.best_trial.model_description}")
print(f"📊 RMSE: ${summary.best_trial.metrics['val_rmse']:,.0f}")

🔑 AutoML probará:
  • Linear Regression
  • Decision Tree
  • Random Forest
  • XGBoost / LightGBM
  • Hyperparameter tuning
""")

# COMMAND ----------

# DBTITLE 1,Modelo manual de comparación
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X = df_properties.drop('price', axis=1)
y = df_properties['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Modelo': name,
        'RMSE': f'${rmse:,.0f}',
        'MAE': f'${mae:,.0f}',
        'R²': f'{r2:.4f}'
    })

print("📈 RESULTADOS:\n")
print(pd.DataFrame(results).to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Qué genera AutoML
# MAGIC %md
# MAGIC ## 📝 Notebook Generado por AutoML
# MAGIC
# MAGIC ### Lo que AutoML genera automáticamente:
# MAGIC
# MAGIC 1. **Data Exploration**: Estadísticas, correlaciones, outliers
# MAGIC 2. **Feature Engineering**: One-hot encoding, scaling
# MAGIC 3. **Model Training**: Múltiples algoritmos + tuning
# MAGIC 4. **Evaluation**: Métricas, feature importance, residual plots
# MAGIC 5. **MLflow Registry**: Modelo guardado y listo para deployment
# MAGIC
# MAGIC ### Ventajas:
# MAGIC * ✅ Código reproducible
# MAGIC * ✅ Trazabilidad completa
# MAGIC * ✅ Hyperparameters documentados
# MAGIC * ✅ Fácil de modificar

# COMMAND ----------

# DBTITLE 1,Feature Importance
# Simular feature importance
feature_importance = pd.DataFrame({
    'Feature': ['sqft', 'location_score', 'bedrooms', 'bathrooms', 'has_pool', 'has_garage', 'age_years'],
    'Importance': [0.45, 0.22, 0.12, 0.09, 0.06, 0.04, 0.02]
}).sort_values('Importance', ascending=False)

print("📈 FEATURE IMPORTANCE:\n")
for idx, row in feature_importance.iterrows():
    bar = '█' * int(row['Importance'] * 100)
    print(f"  {row['Feature']:<18} {bar} {row['Importance']:.2%}")

print("\n💡 sqft es el factor más importante (45%)")

# COMMAND ----------

# DBTITLE 1,Comparación AutoML vs Manual
# MAGIC %md
# MAGIC ## ⚖️ AutoML vs Manual
# MAGIC
# MAGIC | Aspecto | AutoML | Manual |
# MAGIC |---------|--------|--------|
# MAGIC | **Tiempo** | 15-30 min | Horas/Días |
# MAGIC | **Modelos** | 10-20+ | 2-3 |
# MAGIC | **Tuning** | Automático | Manual |
# MAGIC | **MLflow** | Automático | Manual |
# MAGIC
# MAGIC ### ✅ Usar AutoML cuando:
# MAGIC * Prototipado rápido
# MAGIC * Baseline en minutos
# MAGIC * Equipos sin expertos ML
# MAGIC * Time-to-market crítico
# MAGIC
# MAGIC ### ⚠️ Usar manual cuando:
# MAGIC * Lógica de negocio compleja
# MAGIC * Custom features específicos
# MAGIC * Máximo performance crítico

# COMMAND ----------

# DBTITLE 1,## Conclusiones
# MAGIC %md
# MAGIC ## 📝 Conclusiones
# MAGIC
# MAGIC ### Key Takeaways
# MAGIC
# MAGIC 1. **AutoML acelera desarrollo**: Baseline en minutos, 10-20 modelos probados
# MAGIC 2. **Código reproducible**: Notebook generado + MLflow tracking
# MAGIC 3. **Métricas**: RMSE, MAE, R² para regresión
# MAGIC 4. **Feature importance**: Identifica features clave
# MAGIC
# MAGIC ### 🚀 Próximos Pasos
# MAGIC
# MAGIC * Genie_Assisted_ML_Pipeline
# MAGIC * MLflow_Experiment_Tracking
# MAGIC * Model Serving
# MAGIC
# MAGIC ## 🎉 ¡Felicidades!
# MAGIC
# MAGIC Dominas **Databricks AutoML para Regresión**. 🚀

# COMMAND ----------

