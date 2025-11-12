# 🎯 Customer Churn Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Rojin-Dhami/Customer_churn_prediction_model/graphs/commit-activity)

A production-ready machine learning pipeline for predicting customer churn in telecommunications using advanced ensemble methods and automated feature engineering.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## 📊 Model Performance

| Model | Cross-Val Score | ROC AUC | Accuracy |
|-------|----------------|---------|----------|
| **Stacking Ensemble** | **~87%** | **~87%** | **~86%** |
| LightGBM | ~86% | ~86% | ~85% |
| XGBoost | ~85% | ~85% | ~84% |
| Random Forest | ~83% | ~83% | ~82% |
| Decision Tree | ~78% | ~78% | ~77% |

## 🌟 Features

- ✅ **Automated Data Pipeline**: End-to-end preprocessing and feature engineering
- ✅ **Class Balancing**: SMOTE implementation for handling imbalanced datasets
- ✅ **Multiple ML Models**: XGBoost, LightGBM, Random Forest, Decision Trees
- ✅ **Ensemble Methods**: Advanced stacking classifier for optimal performance
- ✅ **Comprehensive Visualization**: Interactive plots and correlation analysis
- ✅ **Production Ready**: Object-oriented design with proper error handling
- ✅ **Jupyter Notebooks**: Exploratory analysis and model experiments included
- ✅ **Easy to Extend**: Modular architecture for adding new models

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Rojin-Dhami/Customer_churn_prediction_model.git
cd Customer_churn_prediction_model
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Run the Main Script

```bash
python Churn_predictor.py
```

#### Option 2: Use Python API

```python
from src.Churn_predictor import CustomerChurnPredictor

# Initialize the predictor
predictor = CustomerChurnPredictor("Data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Run the complete pipeline
predictor.run_full_pipeline()
```

#### Option 3: Explore with Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks:
# - notebook/Churn_predictor.ipynb (Full pipeline)
# - notebook/exploratory_analysis.ipynb (Detailed EDA)
```

## 📁 Project Structure

```
Customer_churn_prediction_model/
│
├── Churn_predictor.py          # Main executable script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── Data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
│
├── model/                      # Directory for saved models
│
├── notebook/
│   ├── Churn_predictor.ipynb           # Main analysis notebook
│   └── exploratory_analysis.ipynb      # Exploratory data analysis
│
└── src/
    ├── __init__.py             # Package initialization
    └── Churn_predictor.py      # Core pipeline implementation
```

## 📊 Dataset Information

The dataset contains customer information from a telecommunications company:

- **Total Samples**: 7,043 customers
- **Features**: 20 independent variables
- **Target**: Churn (Yes/No)
- **Target Distribution**: ~26% churn rate (imbalanced)

### Feature Categories:

**Demographics:**
- Gender, SeniorCitizen, Partner, Dependents

**Services:**
- PhoneService, MultipleLines, InternetService
- OnlineSecurity, OnlineBackup, DeviceProtection
- TechSupport, StreamingTV, StreamingMovies

**Account Information:**
- tenure, Contract, PaperlessBilling, PaymentMethod
- MonthlyCharges, TotalCharges

**Target Variable:**
- Churn (Yes/No)

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **XGBoost**: Extreme gradient boosting
- **LightGBM**: Light gradient boosting machine
- **Imbalanced-learn**: SMOTE for handling class imbalance
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development and analysis

## 🔬 Methodology

### 1. **Data Preprocessing**
- Handle missing values in TotalCharges column
- Remove duplicate entries
- Drop irrelevant features (customerID)
- Reset index after cleaning

### 2. **Feature Engineering**
- Normalize numerical features using MinMaxScaler
  - `tenure`, `MonthlyCharges`, `TotalCharges`
- Label encode categorical variables
- Correlation analysis for feature selection
- Remove low-correlation features (threshold < 0.1)

### 3. **Class Balancing**
- Apply SMOTE (Synthetic Minority Over-sampling Technique)
- Balance class distribution from 26%/74% to 50%/50%

### 4. **Model Training**
- Train multiple classifiers:
  - XGBoost Classifier
  - LightGBM Classifier
  - Random Forest Classifier
  - Decision Tree Classifier
  - Stacking Ensemble (combines all above)
- Cross-validation: Repeated Stratified K-Fold (10 splits, 3 repeats)
- Hyperparameters optimized for each model

### 5. **Model Evaluation**
- ROC AUC Score
- Cross-validation Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC Curve visualization

## 📈 Key Insights

Analysis reveals the following factors strongly influence customer churn:

### High Churn Indicators 🔴
1. **Contract Type**: Month-to-month contracts show ~3x higher churn
2. **Tenure**: Customers with < 12 months tenure are high-risk
3. **Monthly Charges**: Higher charges (>$70/month) correlate with increased churn
4. **Payment Method**: Electronic check users exhibit higher churn rates
5. **Internet Service**: Fiber optic users show elevated churn
6. **Tech Support**: Customers without tech support churn more

### Low Churn Indicators 🟢
- Long-term contracts (One year / Two year)
- Longer tenure (> 24 months)
- Lower monthly charges
- Bundled services with online security and tech support

## 🎯 Business Recommendations

1. **Target Retention**: Focus on month-to-month contract customers
2. **Onboarding**: Improve first 6 months experience (critical period)
3. **Pricing Strategy**: Review pricing for high monthly charge segments
4. **Service Quality**: Investigate fiber optic service issues
5. **Incentives**: Promote annual/2-year contracts with benefits
6. **Support Services**: Encourage tech support adoption

## 📚 Notebooks

### 1. **Churn_predictor.ipynb**
Complete machine learning pipeline including:
- Data loading and preprocessing
- Feature engineering
- Model training and evaluation
- Performance comparison

### 2. **exploratory_analysis.ipynb**
Comprehensive exploratory data analysis featuring:
- Statistical summaries
- Univariate and bivariate analysis
- Correlation analysis
- Visualization of patterns and trends
- Feature importance insights

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Rojin Dhami**

- GitHub: [@Rojin-Dhami](https://github.com/Rojin-Dhami)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/rojin-dhami-575730347)
- Email: rojindhami@gmail.com

## 🙏 Acknowledgments

- Dataset provided by IBM Sample Data Sets
- Inspired by real-world telecommunication churn prediction challenges
- Thanks to the open-source data science community

## 🗺️ Roadmap

Future enhancements planned:

- [ ] Web application deployment (Flask/FastAPI)
- [ ] Interactive dashboard (Streamlit/Plotly Dash)
- [ ] Hyperparameter tuning with Optuna
- [ ] Docker containerization
- [ ] Model monitoring and drift detection
- [ ] CI/CD pipeline with GitHub Actions
- [ ] REST API for real-time predictions
- [ ] Feature importance visualization with SHAP

## 📞 Support

For support, questions, or feedback:

- Open an issue in this repository
- Email: rojindhami@gmail.com
- Connect on LinkedIn

## ⭐ Show Your Support

Give a ⭐️ if this project helped you learn or build something amazing!

## 📸 Results Preview

### Model Comparison
All models trained and evaluated with comprehensive metrics including confusion matrices and ROC curves.

### Feature Correlations
Heatmaps showing relationships between features and their correlation with churn.

### Distribution Analysis
Visualizations of customer demographics, service usage, and churn patterns.

---

**Built with ❤️ using Python and Machine Learning**

**Last Updated**: November 2025
