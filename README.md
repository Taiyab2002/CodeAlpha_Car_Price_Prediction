# CodeAlpha Task 3: Car Price Prediction

A regression project that estimates used-car selling prices from the supplied `car data.csv` dataset.

## Dataset

Place the dataset at `data/car data.csv`. The supplied file contains 301 records and 9 original columns:
`Car_Name`, `Year`, `Selling_Price`, `Present_Price`, `Driven_kms`, `Fuel_Type`, `Selling_type`, `Transmission`, and `Owner`.

The target variable is `Selling_Price`. This dataset's price units are commonly described as lakh INR; confirm the dataset source/units before using predictions commercially.

## What the project does

- Checks required columns, removes duplicate rows and rows with invalid/missing target or core numeric values.
- Creates `Car_Age` as a derived feature.
- Uses a preprocessing pipeline: median imputation and scaling for numeric fields, most-frequent imputation and one-hot encoding for categorical fields.
- Compares Linear Regression, Random Forest, Extra Trees, and Gradient Boosting on a held-out test set.
- Reports MAE, RMSE, and R²; selects the model with the lowest test-set RMSE.
- Saves a trained pipeline, test-set predictions, model-comparison table, and four visualizations.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python car_price_prediction.py
```

If PowerShell blocks virtual-environment activation, run this once in the current terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Output files

- `outputs/model_comparison.csv` — test metrics for each model.
- `outputs/test_set_predictions.csv` — actual, predicted, and absolute error for test rows.
- `outputs/selling_price_distribution.png`
- `outputs/model_comparison_rmse.png`
- `outputs/actual_vs_predicted.png`
- `outputs/residual_plot.png`
- `models/best_car_price_model.joblib` — fitted preprocessing + regression pipeline.

## Evaluation

- **MAE**: average absolute prediction error (lower is better).
- **RMSE**: penalizes larger errors more strongly (lower is better).
- **R²**: fraction of target variance explained relative to a mean baseline (higher is generally better; it can be negative).

The dataset is small (301 rows), and the reported metrics come from one train/test split. Results may vary across splits and should not be treated as a guarantee of real-world car value.
