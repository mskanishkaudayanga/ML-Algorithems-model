# Report Outline — Sri Lanka Mobile Phone Price Prediction using XGBoost

## 1. Introduction
- Problem statement: predicting mobile phone prices in the Sri Lanka market
- Motivation: helping buyers and sellers make informed pricing decisions
- Objective: build a regression model using XGBoost to predict prices accurately

## 2. Dataset Description
- Source: ikman.lk (Sri Lanka's largest online marketplace)
- Collection method: web scraping with BeautifulSoup
- Number of records: 5,000+
- Key features: brand, model, condition, storage, RAM, location
- Target variable: price (in LKR)

## 3. Data Preprocessing
- Currency cleaning: handled "Rs", commas, "Lakh", "Mn" formats
- Missing value treatment: median imputation for numerics, 'Unknown' for categoricals
- Feature engineering:
  - Phone model extraction from ad titles
  - Storage/RAM converted to numeric (GB)
  - Brand and location consolidation for rare categories
- Encoding: LabelEncoder for categoricals
- Normalization: StandardScaler for numerics
- Split: 70% train / 15% validation / 15% test (random_state=42)

## 4. Model Selection and Training
- Algorithm: XGBoost Regressor
- Justification: handles non-linear relationships, robust to outliers, built-in regularization
- Hyperparameter tuning: RandomizedSearchCV (50 iterations, 3-fold CV)
  - Why RandomizedSearchCV: computationally efficient for large search spaces
- Early stopping: used validation set with patience of 30 rounds
- Final model: retrained with best hyperparameters + early stopping

## 5. Evaluation Results
- Metrics:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R2 Score
- Plots:
  - Predicted vs Actual scatter plot
  - Residual histogram
- Interpretation of results

## 6. Model Explainability (SHAP)
- Method: SHAP TreeExplainer
- SHAP Summary Plot: overall feature importance with direction of impact
- SHAP Dependence Plot: relationship between most important feature and predictions
- Feature Importance Bar Chart: ranked feature impact
- Key findings from SHAP analysis

## 7. Streamlit Application
- User interface: sidebar inputs for phone specifications
- Prediction output: estimated market price
- Global explanations: SHAP summary and feature importance plots
- Local explanations: SHAP waterfall plot for individual predictions
- Design: clean, professional, card-style layout

## 8. Limitations and Future Work
- Limited to features available from listing pages
- Price predictions may not account for cosmetic condition or accessories
- Future improvements:
  - Add more features (color, seller ratings, listing age)
  - Try ensemble methods (stacking, blending)
  - Time-series analysis for price trends
  - Deploy as a public web service

## 9. Conclusion
- Summary of findings
- Model performance assessment
- Practical applications

## 10. References
- ikman.lk
- XGBoost documentation
- SHAP library documentation
- scikit-learn documentation
