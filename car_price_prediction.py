"""
CodeAlpha Task 3: Car Price Prediction
Train and compare regression models for used-car selling-price estimation.
Run: python car_price_prediction.py
"""
from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "car data.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place 'car data.csv' inside the project's data folder."
        )

    df = pd.read_csv(DATA_PATH)
    print("=" * 65)
    print("CODEALPHA TASK 3 — CAR PRICE PREDICTION")
    print("=" * 65)
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")

    required = [
        "Car_Name", "Year", "Selling_Price", "Present_Price",
        "Driven_kms", "Fuel_Type", "Selling_type", "Transmission", "Owner"
    ]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing from dataset: {missing_columns}")

    df = df[required].copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=["Selling_Price"])
    df["Car_Age"] = pd.Timestamp.now().year - pd.to_numeric(df["Year"], errors="coerce")
    df["Car_Age"] = df["Car_Age"].clip(lower=0)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Selling_Price"] = pd.to_numeric(df["Selling_Price"], errors="coerce")
    df["Present_Price"] = pd.to_numeric(df["Present_Price"], errors="coerce")
    df["Driven_kms"] = pd.to_numeric(df["Driven_kms"], errors="coerce")
    df["Owner"] = pd.to_numeric(df["Owner"], errors="coerce")
    df = df.dropna(subset=["Selling_Price", "Year", "Present_Price", "Driven_kms", "Owner"])
    df = df[df["Selling_Price"] >= 0]
    df = df[df["Present_Price"] >= 0]
    df = df[df["Driven_kms"] >= 0]

    print(f"Rows after basic cleaning: {len(df)}")
    print(f"Missing values after cleaning: {int(df.isna().sum().sum())}")
    print(f"Target: Selling_Price (dataset price units, commonly lakh INR)")
    print(f"Mean selling price: {df['Selling_Price'].mean():.2f}")
    print(f"Median selling price: {df['Selling_Price'].median():.2f}")

    target = "Selling_Price"
    features = [
        "Car_Name", "Year", "Car_Age", "Present_Price", "Driven_kms",
        "Fuel_Type", "Selling_type", "Transmission", "Owner"
    ]
    X = df[features]
    y = df[target]

    categorical_features = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]
    numerical_features = ["Year", "Car_Age", "Present_Price", "Driven_kms", "Owner"]

    numeric_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_preprocessor, numerical_features),
        ("categorical", categorical_preprocessor, categorical_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, min_samples_leaf=1, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, random_state=RANDOM_STATE, min_samples_leaf=1, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    results = []
    fitted_models = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "R2": r2_score(y_test, predictions),
        })
        fitted_models[name] = (pipeline, predictions)

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    best_name = results_df.iloc[0]["Model"]
    best_model, best_predictions = fitted_models[best_name]
    best_metrics = results_df.iloc[0].to_dict()
    joblib.dump(best_model, MODEL_DIR / "best_car_price_model.joblib")

    prediction_output = X_test.copy()
    prediction_output["Actual_Selling_Price"] = y_test.values
    prediction_output["Predicted_Selling_Price"] = best_predictions
    prediction_output["Absolute_Error"] = np.abs(
        prediction_output["Actual_Selling_Price"] - prediction_output["Predicted_Selling_Price"]
    )
    prediction_output.to_csv(OUTPUT_DIR / "test_set_predictions.csv", index=False)

    plt.style.use("default")
    plt.figure(figsize=(9, 6))
    plt.hist(df["Selling_Price"], bins=30, edgecolor="black", alpha=0.8)
    plt.title("Distribution of Used-Car Selling Prices")
    plt.xlabel("Selling Price (dataset units)")
    plt.ylabel("Number of Cars")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "selling_price_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.bar(results_df["Model"], results_df["RMSE"], edgecolor="black")
    plt.title("Regression Model Comparison (Lower RMSE Is Better)")
    plt.xlabel("Regression Model")
    plt.ylabel("RMSE (dataset units)")
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison_rmse.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, best_predictions, alpha=0.8)
    low = min(float(y_test.min()), float(np.min(best_predictions)))
    high = max(float(y_test.max()), float(np.max(best_predictions)))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.title(f"Actual vs Predicted Selling Price — {best_name}")
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close()

    residuals = y_test.to_numpy() - best_predictions
    plt.figure(figsize=(9, 6))
    plt.scatter(best_predictions, residuals, alpha=0.8)
    plt.axhline(0, linestyle="--")
    plt.title(f"Residual Plot — {best_name}")
    plt.xlabel("Predicted Selling Price")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_plot.png", dpi=160)
    plt.close()

    print("\nModel comparison (sorted by RMSE):")
    print(results_df.to_string(index=False, formatters={
        "MAE": "{:.3f}".format, "RMSE": "{:.3f}".format, "R2": "{:.3f}".format
    }))
    print(f"\nBest model by test RMSE: {best_name}")
    print(f"MAE:  {best_metrics['MAE']:.3f}")
    print(f"RMSE: {best_metrics['RMSE']:.3f}")
    print(f"R²:   {best_metrics['R2']:.3f}")
    print("\nSaved files:")
    print("- outputs/model_comparison.csv")
    print("- outputs/test_set_predictions.csv")
    print("- outputs/selling_price_distribution.png")
    print("- outputs/model_comparison_rmse.png")
    print("- outputs/actual_vs_predicted.png")
    print("- outputs/residual_plot.png")
    print("- models/best_car_price_model.joblib")
    print("\nAnalysis completed successfully.")
    print("Note: Metrics are from one holdout split; this small dataset limits generalization.")


if __name__ == "__main__":
    main()
