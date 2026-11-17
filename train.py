"""Train, compare and save the solar power models.

Run:  python train.py
Creates: models/best_model.joblib, results/*.csv|png, solar_power_analysis_excel.xlsx, sql/*.sql
"""
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config
from models_def import get_models, tune_knn
from utils import evaluate, load_data, prepare_data, time_split


def write_sql(info):
    """Create SQL files that match the real column names of the dataset."""
    os.makedirs("sql", exist_ok=True)
    t, tc = info["target"], info["time_col"] or "timestamp"
    cols = info["weather_cols"]
    schema = [f'    "{tc}" TIMESTAMP', *[f'    "{c}" REAL' for c in cols], f'    "{t}" REAL']
    with open("sql/schema.sql", "w") as f:
        f.write("-- Table for the Kaggle solar + weather data (SQLite / PostgreSQL friendly)\n")
        f.write("CREATE TABLE IF NOT EXISTS solar_weather (\n" + ",\n".join(schema) + "\n);\n")
    avg_cols = ",\n       ".join(f'ROUND(AVG("{c}"), 2) AS avg_{i}' for i, c in enumerate(cols[:4]))
    q = f"""-- Analysis queries (SQLite syntax, adjust date functions for other databases)

-- 1. Average generation by hour of day
SELECT CAST(strftime('%H', "{tc}") AS INTEGER) AS hour,
       ROUND(AVG("{t}"), 3) AS avg_power
FROM solar_weather GROUP BY hour ORDER BY hour;

-- 2. Average generation by month
SELECT CAST(strftime('%m', "{tc}") AS INTEGER) AS month,
       ROUND(AVG("{t}"), 3) AS avg_power
FROM solar_weather GROUP BY month ORDER BY month;

-- 3. Top 10 highest-generation records
SELECT * FROM solar_weather ORDER BY "{t}" DESC LIMIT 10;

-- 4. Weather averages when the plant is producing vs. not producing
SELECT CASE WHEN "{t}" > 0 THEN 'producing' ELSE 'idle' END AS status,
       COUNT(*) AS records,
       {avg_cols}
FROM solar_weather GROUP BY status;

-- 5. Daily total generation
SELECT date("{tc}") AS day, ROUND(SUM("{t}"), 2) AS total_power
FROM solar_weather GROUP BY day ORDER BY day;
"""
    with open("sql/analysis_queries.sql", "w") as f:
        f.write(q)


def write_excel(results, preds, X, y, info):
    desc = pd.concat([X, y], axis=1).describe().T.reset_index().rename(columns={"index": "feature"})
    summary = pd.DataFrame({
        "Item": ["Target", "Time column", "Rows", "Features used", "Best model",
                 "Best RMSE", "Best MAE", "Best R2"],
        "Value": [info["target"], info["time_col"], len(X), len(info["feature_names"]),
                  results.iloc[0]["Model"], round(results.iloc[0]["RMSE"], 4),
                  round(results.iloc[0]["MAE"], 4), round(results.iloc[0]["R2"], 4)],
    })
    with pd.ExcelWriter(config.EXCEL_PATH, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        results.to_excel(xw, sheet_name="Model Comparison", index=False)
        desc.to_excel(xw, sheet_name="Feature Statistics", index=False)
        pd.concat([X, y], axis=1).corr().round(3).to_excel(xw, sheet_name="Correlation")
        preds.head(1000).to_excel(xw, sheet_name="Test Predictions", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(width + 3, 40)


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = load_data()
    X, y, info = prepare_data(df)
    print(f"[info] target='{info['target']}' time='{info['time_col']}' rows={len(X)}")
    print(f"[info] features: {info['feature_names']}")
    if info["dropped_cols"]:
        print(f"[info] ignored columns: {info['dropped_cols']}")

    X_train, X_test, y_train, y_test = time_split(X, y)
    print(f"[info] train={len(X_train)} test={len(X_test)} (chronological split)")

    fitted, rows, preds = {}, [], pd.DataFrame({"actual": y_test.values})
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        fitted[name] = model
        p = model.predict(X_test)
        rows.append({"Model": name, **evaluate(y_test, p)})
        preds[name] = p
        print(f"  {name:<20} {rows[-1]}")

    grid = tune_knn(X_train, y_train)
    p = grid.best_estimator_.predict(X_test)
    knn_name = "KNN (tuned)"
    fitted[knn_name] = grid.best_estimator_
    rows.append({"Model": knn_name, **evaluate(y_test, p)})
    preds[knn_name] = p
    print(f"  best KNN params: {grid.best_params_}")

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    results.to_csv(f"{config.RESULTS_DIR}/model_comparison.csv", index=False)
    best_name = results.iloc[0]["Model"]
    print(f"\n[done] best model: {best_name}\n{results.round(4).to_string(index=False)}")

    # ---- save best model + everything the app needs
    joblib.dump({
        "model": fitted[best_name], "model_name": best_name,
        "feature_names": info["feature_names"], "weather_cols": info["weather_cols"],
        "time_col": info["time_col"], "target": info["target"],
        "feature_stats": X.describe().T[["min", "50%", "max"]].rename(columns={"50%": "median"}),
        "metrics": results.iloc[0].to_dict(),
    }, config.MODEL_PATH)

    # ---- plots
    plt.figure(figsize=(8, 4))
    sns.barplot(data=results, x="RMSE", y="Model", color="#4C72B0")
    plt.title("Model comparison (lower RMSE is better)"); plt.tight_layout()
    plt.savefig(f"{config.RESULTS_DIR}/model_comparison.png", dpi=150); plt.close()

    n = min(24 * 14, len(y_test))
    plt.figure(figsize=(12, 4))
    plt.plot(preds["actual"].values[:n], label="Actual", lw=1.5)
    plt.plot(preds[best_name].values[:n], label=f"Predicted ({best_name})", lw=1.2)
    plt.title("Actual vs predicted solar power (start of test period)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{config.RESULTS_DIR}/actual_vs_predicted.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.concat([X[info["weather_cols"]], y], axis=1).corr(),
                annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation heatmap"); plt.tight_layout()
    plt.savefig(f"{config.RESULTS_DIR}/correlation_heatmap.png", dpi=150); plt.close()

    rf = fitted["Random Forest"].named_steps["model"]
    imp = pd.Series(rf.feature_importances_, index=info["feature_names"]).sort_values()
    plt.figure(figsize=(8, 5)); imp.plot.barh(color="#55A868")
    plt.title("Feature importance (Random Forest)"); plt.tight_layout()
    plt.savefig(f"{config.RESULTS_DIR}/feature_importance.png", dpi=150); plt.close()

    write_excel(results, preds, X, y, info)
    write_sql(info)
    print(f"\n[saved] {config.MODEL_PATH}, {config.EXCEL_PATH}, results/, sql/")


if __name__ == "__main__":
    main()
