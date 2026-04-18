<div align="center">

# House Price Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)](https://matplotlib.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Status](https://img.shields.io/badge/Status-Complete-2ea44f?style=for-the-badge)]()

*A regression-based machine learning pipeline for predicting residential property sale prices, using structured tabular data covering location, size, condition, and amenity features.*

</div>

---

A machine learning project that models residential property sale prices from structured tabular data. The pipeline covers exploratory analysis, feature engineering, model training, and submission generation — structured around the well-known Kaggle House Prices competition format.

---

## Table of Contents

- [Background](#background)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Methodology](#methodology)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Background

Accurate property valuation is a core problem in real estate, finance, and urban planning. Traditional appraisal methods rely on expert judgment and comparable sales; data-driven approaches offer a scalable, consistent alternative by learning price relationships directly from historical transaction records.

This project trains regression models on a richly featured dataset of residential sales, capturing the nonlinear effects of location, size, age, and condition on final sale price. The output is a submission-ready predictions file compatible with the Kaggle evaluation pipeline.

---

## Dataset

| Property | Detail |
|----------|--------|
| Training set | `train.csv` — labelled samples with `SalePrice` target |
| Test set | `test.csv` — unlabelled samples for prediction |
| Feature reference | `data_description.txt` — full description of all 79 features |
| Submission format | `Submission.csv` — predicted sale prices for test set |

The dataset includes features across several categories:

- **Location** — neighbourhood, zoning classification, proximity to key infrastructure
- **Size** — total square footage, above-ground living area, basement area, lot size
- **Rooms** — bedroom count, bathroom count, kitchen quality
- **Condition & Age** — year built, year remodelled, overall quality and condition ratings
- **Amenities** — garage type and capacity, pool, fireplace, fence, porch

Full column definitions are documented in `data_description.txt`.

---

## Project Structure

```
.
├── HousePrice.ipynb          # Main analysis, training, and prediction notebook
├── train.csv                 # Labelled training data
├── test.csv                  # Unlabelled test data for submission
├── data_description.txt      # Feature documentation
├── Submission.csv            # Model predictions on test set
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.8+

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm tqdm
```

All experiments are self-contained within the Jupyter notebook. No additional configuration is required.

---

## Methodology

The pipeline follows a structured regression workflow:

**1. Exploratory Data Analysis**
- Distribution analysis of `SalePrice` (log-transformation for right skew)
- Correlation matrix to identify strongly predictive numerical features
- Neighbourhood-level and quality-rating breakdowns

**2. Data Preprocessing**
- Missing value imputation — median for numerics, mode/`None` for categoricals per feature semantics
- Ordinal encoding of quality/condition ratings (e.g. `Ex > Gd > TA > Fa > Po`)
- One-hot encoding of nominal categorical features

**3. Feature Engineering**
- Composite features: total square footage, total bathroom count, property age, years since remodel
- Interaction terms between quality ratings and size features
- Low-variance and near-duplicate feature removal

**4. Model Training**
- Baseline: Linear Regression and Ridge/Lasso for interpretability
- Ensemble: Random Forest, Gradient Boosting, XGBoost, LightGBM
- Cross-validated RMSE on log-transformed `SalePrice` used as the selection criterion

**5. Prediction & Submission**
- Best model applied to `test.csv`
- Predictions inverse-transformed and exported to `Submission.csv`

---

## Results

- **Primary metric:** Root Mean Squared Error on log-transformed sale price (RMSLE)
- Visualisations of actual vs. predicted prices on the validation set
- Feature importance analysis identifying top drivers of sale price (overall quality, above-ground living area, neighbourhood, and garage capacity consistently rank highest)

> Full evaluation metrics, confusion matrices, and residual plots are documented in the notebook.

---

## Usage

1. Clone the repository and install dependencies (see [Installation](#installation)).
2. Ensure `train.csv`, `test.csv`, and `data_description.txt` are present in the root directory.
3. Open `HousePrice.ipynb` in Jupyter.
4. Run all cells sequentially — predictions are written to `Submission.csv` at the final step.

---

## Future Work

- **Stacking & blending** — ensemble multiple base learners via a meta-regressor for improved RMSLE
- **Advanced imputation** — iterative imputation or KNN-based strategies for missing values
- **Geospatial features** — distance to city centre, schools, and transit using latitude/longitude
- **AutoML baseline** — benchmark against H2O or FLAML for rapid model selection
- **Deployment** — interactive property valuation tool via Streamlit

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request. Ensure any new code is accompanied by clear inline documentation.

---

## License

This project is released for academic and research purposes. See `LICENSE` for details.
