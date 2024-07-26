# House Price Prediction - Research & Engineering Notes

This document tracks engineering milestones, experiment logs, parameter optimization notes, and technical findings for the `HousePrice` project.

### Milestone Log - 2024-07-22 (10:45)
- **Focus**: docs: update regression model performance metrics
- **Technical Summary**: Added RMSE and R² comparison across Linear, Ridge, and Lasso models.

### Milestone Log - 2024-07-22 (14:34)
- **Focus**: refactor: streamline missing value imputation logic
- **Technical Summary**: Replaced iterative imputation with median strategy for numerical columns.

### Milestone Log - 2024-07-23 (10:55)
- **Focus**: docs: add feature importance analysis for Ames dataset
- **Technical Summary**: Identified OverallQual, GrLivArea, and Neighborhood as primary predictors.

### Milestone Log - 2024-07-23 (14:38)
- **Focus**: refactor: vectorize log-transformation for skewed target
- **Technical Summary**: Applied np.log1p on SalePrice to normalize residual distribution.

### Milestone Log - 2024-07-23 (19:20)
- **Focus**: perf: optimize cross-validation splits for grid search
- **Technical Summary**: Used KFold(n_splits=5, shuffle=True) with fixed random seed.

### Milestone Log - 2024-07-25 (19:23)
- **Focus**: docs: update XGBoost hyperparameter tuning documentation
- **Technical Summary**: Documented best parameters: learning_rate=0.05, max_depth=4, n_estimators=600.

### Milestone Log - 2024-07-26 (10:23)
- **Focus**: docs: update regression model performance metrics
- **Technical Summary**: Added RMSE and R² comparison across Linear, Ridge, and Lasso models.

### Milestone Log - 2024-07-26 (14:46)
- **Focus**: refactor: streamline missing value imputation logic
- **Technical Summary**: Replaced iterative imputation with median strategy for numerical columns.

