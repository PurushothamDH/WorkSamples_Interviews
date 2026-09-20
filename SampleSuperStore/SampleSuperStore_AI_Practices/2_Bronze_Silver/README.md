# Superstore Agentic AI Workflow

This project demonstrates a complete **Agentic AI pipeline** applied to the `samplesuperstore.bronzedata.orders` dataset.  
It follows the clarified analytics roadmap: **EDA → Cleaning → ML → Visualization**, with each stage handled by a dedicated agent notebook.

---

## 📂 Project Structure

:/Workspace/Shared/AgenticAI/
│
├── Agent1_DataEDA.ipynb  
├── Agent2_DataCleaning.ipynb  
├── Agent3_MLWork.ipynb  
└── Agent4_Visualization.ipynb  

---

## 🧩 Notebook Roles

### Agent1_DataEDA
- Load dataset from `samplesuperstore.bronzedata.orders`.
- Inspect schema, nulls, distributions.
- Compute summary metrics (sales, profit, quantity).
- Generate initial plots (histograms, boxplots).

### Agent2_DataCleaning
- Handle missing values, duplicates, invalid entries.
- Normalize column names (`Ship Mode`, `Postal Code`).
- Save cleaned dataset → `samplesuperstore.silverdata.orders`.

### Agent3_MLWork
- Feature engineering (discount rate, profit ratio, region sales).
- Train predictive models (regression for sales, classification for delivery delays).
- Evaluate with RMSE, accuracy, F1.
- Track experiments with MLflow.

### Agent4_Visualization
- Build dashboards (sales vs profit by category, region heatmaps).
- Use Matplotlib/Seaborn for notebook visuals.
- Export BI‑style insights for presentation.

---

## 🚀 How to Run

1. Open notebooks in Databricks workspace:  
   `:/Workspace/Shared/AgenticAI/`
2. Run **Agent1_DataEDA** to explore raw data.  
3. Run **Agent2_DataCleaning** to prepare silver dataset.  
4. Run **Agent3_MLWork** to train and evaluate models.  
5. Run **Agent4_Visualization** to generate dashboards and insights.

---

## 🎯 Outcomes
- Modular, agent‑based workflow.  
- Clear separation of analytics phases.  
- MLflow tracking for reproducibility.  
- BI‑style visualizations for storytelling.  
