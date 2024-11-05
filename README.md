# Credit Risk Assessment with LightGBM

A machine learning project for predicting loan defaults using LightGBM with advanced feature selection and model optimization techniques.

## Business Overview

Credit Risk represents the potential loss from a borrower's failure to repay loans or meet contractual obligations. This project implements a sophisticated credit risk assessment model to:
- Evaluate borrower creditworthiness
- Predict default probability
- Minimize lending risks
- Optimize loan approval decisions

## Project Aim

Develop a classification model to predict loan defaulters using credit history, employment, and demographic data, ultimately minimizing financial loss risk.

## Dataset

- 143,727 borrower records
- Features include:
  - Employment type
  - Work experience
  - Income
  - Number of dependents
  - Total loans
  - Payment history
  - And more

## Tech Stack

- **Language:** Python
- **Core Libraries:**
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - lightgbm
  - hyperopt
  - shap

## Project Structure

```
├── input/
│   └── credit_risk_data.csv
├── documents/
│   └── learning_materials/
├── lib/
│   └── notebooks/
├── ml_pipeline/
│   ├── data_processing/
│   ├── feature_engineering/
│   ├── model_training/
│   └── evaluation/
├── output/
│   ├── models/
│   └── results/
├── engine.py
├── requirements.txt
└── README.md
```

## Methodology

### 1. Data Processing
- Column dropping
- Data splitting
- Label definition through Roll Rate Analysis
- Window Roll Analysis

### 2. Feature Engineering
- Label engineering
- Interest payment ratio calculation
- Historical default rate analysis
- Custom financial metrics

### 3. Exploratory Data Analysis
- **Univariate Analysis**
  - Numerical summaries
  - Categorical distributions
- **Bivariate Analysis**
  - Correlation analysis
  - Box plot visualizations

### 4. Model Development
- Target Encoding implementation
- Feature Selection using:
  - Random Forest
  - Decision Tree
- LightGBM model training
- Hyperparameter optimization with Hyperopt

### 5. Model Evaluation
- ROC AUC scoring
- PR AUC analysis
- Score distribution assessment
- Feature importance analysis:
  - Split and Gain metrics
  - SHAP values
- Class Rate Curve analysis
- Threshold optimization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/credit-risk-assessment.git

# Install requirements
pip install -r requirements.txt
```

## Usage

```bash
# Run the complete pipeline
python engine.py
```

## Key Learnings

1. Credit risk assessment fundamentals
2. Default prediction using Roll Rate Analysis
3. DPD (Days Past Due) significance
4. Target Encoding advantages
5. Feature Selection techniques
6. LightGBM optimization with Hyperopt
7. SHAP-based model explainability
8. Threshold optimization for credit loss minimization

## Model Performance

Results and model artifacts are saved in the `output` directory:
- Trained models
- Performance metrics
- Feature importance visualizations
- SHAP analysis plots
- Class rate curves

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- Financial domain experts
- Open-source ML community
- Project contributors and reviewers
