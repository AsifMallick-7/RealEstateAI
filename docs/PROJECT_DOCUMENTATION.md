# RealEstateAI — Project Documentation

## 1. Project Title

**AI-Based Real Estate Price Prediction & Market Analysis System**

## 2. Problem Statement

Property valuation is influenced by multiple numerical and geographic characteristics. A machine-learning system can learn relationships between historical housing attributes and a target property value and provide a consistent estimated output.

The project demonstrates this workflow using the California Housing dataset.

## 3. Objectives

1. Build a reproducible regression pipeline.
2. Validate and preprocess structured housing data.
3. Engineer additional ratio-based features.
4. Compare multiple regression algorithms.
5. Tune a Random Forest model using cross-validation.
6. Evaluate models using MAE, RMSE and R².
7. Provide model explainability using permutation importance.
8. Deploy the trained model through Streamlit.

## 4. Modules

### Module 1 — Data Acquisition
Loads the California Housing dataset using scikit-learn and optionally stores a CSV copy.

### Module 2 — Validation
Checks required columns, data types, missing values, duplicate rows and infinite values.

### Module 3 — Feature Engineering
Creates:
- RoomsPerBedroom
- BedroomsPerRoom

### Module 4 — Preprocessing
Uses median imputation and standard scaling inside an sklearn Pipeline.

### Module 5 — Model Training
Trains Linear Regression, Ridge, Random Forest and Gradient Boosting.

### Module 6 — Hyperparameter Tuning
Uses RandomizedSearchCV to tune Random Forest parameters.

### Module 7 — Evaluation
Computes:
- MAE
- RMSE
- R²
- Cross-validation metrics

### Module 8 — Explainability
Uses permutation importance to identify features whose shuffling changes model performance.

### Module 9 — Application
Streamlit provides:
- Home dashboard
- Prediction form
- Model comparison
- Data analysis
- Explainability
- Project information

## 5. Testing Strategy

Testing includes:
- Data validation tests
- Feature engineering tests
- Prediction workflow testing
- Invalid input handling
- Missing model artifact handling

## 6. Limitations

- Dataset is California-specific.
- Target is a block-group median value, not an individual property appraisal.
- Predictions should not be interpreted as guaranteed market prices.
- The current application does not use live property listings.

## 7. Future Scope

- Indian/local housing dataset
- Geospatial visualization
- FastAPI service
- Prediction database
- Authentication
- Docker
- CI/CD
- Model drift monitoring
- Periodic retraining
