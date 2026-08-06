# AutoML y MLOps - Automatización de Machine Learning

## 📚 Descripción

Este módulo introduce **Automated Machine Learning (AutoML)** y **MLOps**, las prácticas modernas para desarrollar, desplegar y mantener modelos de Machine Learning en producción de forma eficiente y escalable.

---

## 📋 Contenido del Módulo

| Notebook | Tipo | Descripción | Celdas |
|----------|------|-------------|--------|
| **Teoria_AutoML.ipynb** | Teóría | Conceptos de AutoML, MLOps, herramientas, comparación | 3 |
| **Databricks_AutoML_Clasificacion.ipynb** | Práctica | AutoML para clasificación, comparación con modelo manual | 12 |
| **Databricks_AutoML_Regresion.ipynb** | Práctica | AutoML para regresión | En desarrollo |
| **Genie_Assisted_ML_Pipeline.ipynb** | Práctica | Pipelines completos con Genie Code | En desarrollo |
| **MLflow_Experiment_Tracking.ipynb** | Práctica | Tracking, Registry, deployment | En desarrollo |

---

## 🤖 AutoML (Automated Machine Learning)

**Definición:** Automatización del ciclo completo de ML.

### Flujo:
```
📥 Dataset + Target
    ↓
🧹 Preprocesamiento automático
    ↓
🔧 Feature Engineering automático
    ↓
🤖 Prueba múltiples algoritmos
    ↓
⚙️ Optimización de hiperparámetros
    ↓
🏆 Mejor modelo + Notebook explicativo
```

### Ventajas:
* ⚡ **Velocidad**: Horas vs días
* 🎯 **Consistencia**: Mejores prácticas automáticas
* 📊 **Performance**: Comparable a expertos (~95%)
* 🤝 **Democratización**: Accesible para no-expertos

---

## 🏭 MLOps (Machine Learning Operations)

**Definición:** DevOps aplicado a ML - gestión del ciclo de vida completo de modelos en producción.

### Stack:
```
Code → Train → Test → Deploy → Monitor → Retrain
  ↑                                          ↓
  └──────────────────────────────────────────┘
```

### Componentes:
| Componente | Herramienta | Propósito |
|------------|-------------|-----------|