# Credit Default Risk Prediction with LightGBM

## Project Overview
This project focuses on building a predictive model to assess credit risk. Credit risk, the chance of loss from a borrower's failure to repay a loan, is a key concern for lenders. Using LightGBM, we develop a classification model to predict loan defaulters and help minimize potential financial loss.

## Business Context
Credit risk assessment is essential for determining a borrower's likelihood of default, which allows lenders to make informed decisions. This project leverages demographic, employment, and credit history data to predict default risks.

## Data Description
The dataset contains information about **143,727 borrowers** and includes attributes like:
- Employment Type
- Work Experience
- Income
- Number of Dependents
- Total Loans
- Total Payment History

## Tech Stack
- **Programming Language:** Python
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit_learn`, `lightgbm`, `hyperopt`, `shap`

## Project Steps
1. **Data Preparation**
   - Data reading and processing
   - Dropping unnecessary columns
   - Data splitting

2. **Feature Engineering**
   - Roll Rate Analysis
   - Loan Repayment History Analysis
   - Label and target encoding

3. **Exploratory Data Analysis (EDA)**
   - **Univariate Analysis:** Numerical (min, max, mean) and Categorical summaries
   - **Bivariate Analysis:** Correlation plots and Box plots

4. **Model Development**
   - **Model Selection:** Using LightGBM
   - **Hyperparameter Tuning:** Optimized with `hyperopt`
   - **Evaluation Metrics:** ROC AUC, PR AUC

5. **Feature Importance**
   - Feature selection with SHAP and other techniques

6. **Threshold Analysis**
   - Class rate curve for optimal threshold selection to minimize credit loss

## Repository Structure
- `input/`: Contains raw data (`credit_risk_data.csv`)
- `documents/`: Supporting materials and references
- `lib/`: Reference folder with the original Jupyter notebooks
- `ml_pipeline/`: Contains modularized Python functions
- `output/`: Stores model outputs
- `engine.py`: Main script to run the full pipeline
- `requirements.txt`: Libraries required for the project
- `README.md`: Project documentation

## Installation
To install the required packages, run:
```bash
pip install -r requirements.txt
Usage
Run the engine.py script to execute the pipeline.
Trained models and results will be saved in the output folder.

Key Insights
Definition and importance of credit risk assessment
Understanding "default" and "days past due" (dpd)
Feature selection with Random Forest and Decision Tree
Model parameter optimization using Hyperopt
SHAP-based feature importance visualization
Authors
Developed by [Your Name/Team].

License
This project is licensed under the MIT License - see the LICENSE file for details.
