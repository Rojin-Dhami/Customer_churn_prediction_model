"""
Customer Churn Prediction Model
================================
A comprehensive machine learning pipeline for predicting customer churn
in telecommunications using multiple classifiers and ensemble methods.

Author: Refactored Version
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
from typing import Tuple, List, Dict, Any

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, classification_report, 
    accuracy_score, RocCurveDisplay
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE

# Gradient boosting imports
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class CustomerChurnPredictor:
    """
    A complete pipeline for customer churn prediction including data loading,
    preprocessing, feature engineering, model training, and evaluation.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the predictor with dataset filepath.
        
        Args:
            filepath: Path to the CSV file containing customer data
        """
        self.filepath = filepath
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and return the dataset."""
        print("Loading dataset...")
        self.df = pd.read_csv(self.filepath)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def explore_data(self):
        """Perform initial data exploration and display key statistics."""
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"\nData Types:\n{self.df.dtypes.value_counts()}")
        print(f"\nMissing Values:\n{self.df.isna().sum().sum()} total")
        print(f"\nDuplicate Rows: {self.df.duplicated().sum()}")
        
        # Display target variable distribution
        churn_dist = self.df['Churn'].value_counts()
        churn_pct = (churn_dist / len(self.df) * 100).round(2)
        print(f"\nChurn Distribution:")
        print(f"  No:  {churn_dist['No']} ({churn_pct['No']}%)")
        print(f"  Yes: {churn_dist['Yes']} ({churn_pct['Yes']}%)")
        
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values, duplicates, and data type issues.
        
        Returns:
            Cleaned DataFrame
        """
        print("\n" + "="*50)
        print("DATA CLEANING")
        print("="*50)
        
        # Convert TotalCharges to numeric
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        missing_total_charges = self.df['TotalCharges'].isna().sum()
        print(f"Missing values in TotalCharges: {missing_total_charges}")
        
        # Drop rows with missing TotalCharges
        self.df.dropna(subset=['TotalCharges'], inplace=True)
        
        # Drop duplicates
        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"Rows removed (duplicates): {initial_rows - len(self.df)}")
        
        # Drop customerID (not needed for modeling)
        self.df.drop('customerID', axis=1, inplace=True)
        
        # Reset index
        self.df.reset_index(drop=True, inplace=True)
        
        print(f"Cleaned dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def visualize_distributions(self):
        """Create visualizations for feature distributions by churn status."""
        print("\n" + "="*50)
        print("GENERATING VISUALIZATIONS")
        print("="*50)
        
        # Churn by gender
        plt.figure(figsize=(10, 5))
        sns.countplot(x='Churn', hue='gender', data=self.df, palette='Set2')
        plt.title('Customer Churn by Gender', fontsize=16, fontweight='bold')
        plt.xlabel('Churn')
        plt.ylabel('Number of Customers')
        plt.legend(title='Gender')
        plt.tight_layout()
        plt.show()
        
        # Numerical features distribution
        numerical_features = ['MonthlyCharges', 'TotalCharges', 'tenure']
        for col in numerical_features:
            plt.figure(figsize=(10, 5))
            sns.histplot(data=self.df, x=col, hue='Churn', kde=True, palette='viridis')
            plt.title(f'Distribution of {col} by Churn', fontsize=14, fontweight='bold')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()
        
        print("✓ Visualizations complete")
    
    def visualize_categorical_features(self):
        """Visualize categorical features distribution by churn."""
        categorical_features = self.df.select_dtypes(include='object').columns.tolist()
        
        # Remove already processed features
        for feature in ['Churn', 'gender']:
            if feature in categorical_features:
                categorical_features.remove(feature)
        
        # Add SeniorCitizen
        if 'SeniorCitizen' not in categorical_features:
            categorical_features.append('SeniorCitizen')
        
        for col in categorical_features:
            plt.figure(figsize=(10, 5))
            sns.countplot(data=self.df, x=col, hue='Churn', palette='coolwarm')
            plt.title(f'Distribution of {col} by Churn', fontsize=14, fontweight='bold')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Churn')
            plt.tight_layout()
            plt.show()
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Perform feature engineering including encoding and scaling.
        
        Returns:
            DataFrame with engineered features
        """
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)
        
        # Encode target variable
        self.df['Churn'] = self.df['Churn'].map({'No': 0, 'Yes': 1})
        
        # Check correlation with target before scaling
        correlations = self.df.corr(numeric_only=True)['Churn'].sort_values(ascending=False)
        print(f"\nTop 5 Correlated Features:\n{correlations.head()}")
        
        # Normalize numerical features
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numerical_cols:
            self.df[col] = self.scaler.fit_transform(self.df[[col]])
        print(f"✓ Normalized features: {', '.join(numerical_cols)}")
        
        # Label encode categorical features
        categorical_cols = self.df.select_dtypes(include='object').columns
        for col in categorical_cols:
            self.df[col] = self.label_encoder.fit_transform(self.df[col])
        print(f"✓ Encoded {len(categorical_cols)} categorical features")
        
        return self.df
    
    def visualize_correlation(self):
        """Create correlation heatmaps."""
        print("\n" + "="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        
        # Full correlation matrix
        plt.figure(figsize=(18, 12))
        sns.heatmap(self.df.corr(), cmap='coolwarm', annot=True, fmt='.2f', 
                    linewidths=0.5, linecolor='black')
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Correlation with target
        corr_with_target = self.df.corrwith(self.df['Churn']).sort_values(ascending=False).to_frame()
        corr_with_target.columns = ['Correlations']
        
        plt.figure(figsize=(8, 10))
        sns.heatmap(corr_with_target, annot=True, cmap='RdYlGn', center=0,
                    linewidths=0.5, linecolor='black', cbar_kws={'label': 'Correlation'})
        plt.title('Feature Correlation with Churn', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def drop_low_correlation_features(self, threshold: float = 0.1) -> pd.DataFrame:
        """
        Drop features with correlation below threshold.
        
        Args:
            threshold: Minimum absolute correlation value to keep feature
            
        Returns:
            DataFrame with relevant features
        """
        # Identify features to drop
        correlations = self.df.corrwith(self.df['Churn']).abs()
        features_to_drop = correlations[
            (correlations < threshold) & (correlations.index != 'Churn')
        ].index.tolist()
        
        print(f"\nDropping {len(features_to_drop)} low-correlation features:")
        print(f"  {', '.join(features_to_drop)}")
        
        self.df.drop(columns=features_to_drop, inplace=True)
        print(f"Remaining features: {self.df.shape[1]}")
        
        return self.df
    
    def balance_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance the dataset using SMOTE (Synthetic Minority Over-sampling Technique).
        
        Returns:
            Tuple of (features, target) arrays
        """
        print("\n" + "="*50)
        print("BALANCING DATASET WITH SMOTE")
        print("="*50)
        
        # Separate features and target
        X = self.df.iloc[:, :-1].values
        y = self.df.iloc[:, -1].values
        
        print(f"Original class distribution: {Counter(y)}")
        
        # Apply SMOTE
        smote = SMOTE(sampling_strategy=1, random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        print(f"Balanced class distribution: {Counter(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.20, random_state: int = 42):
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature array
            y: Target array
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nTraining set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
    
    def train_and_evaluate_model(self, name: str, classifier: Any):
        """
        Train a model and evaluate its performance.
        
        Args:
            name: Model name for identification
            classifier: Sklearn-compatible classifier instance
        """
        print(f"\n{'='*50}")
        print(f"TRAINING: {name}")
        print(f"{'='*50}")
        
        # Train model
        classifier.fit(self.X_train, self.y_train)
        
        # Cross-validation score
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        cv_score = cross_val_score(
            classifier, self.X_train, self.y_train, 
            cv=cv, scoring='roc_auc'
        ).mean()
        
        # Test set predictions
        y_pred = classifier.predict(self.X_test)
        
        # ROC AUC Score
        roc_auc = roc_auc_score(self.y_test, y_pred)
        
        print(f"Cross Validation Score: {cv_score:.2%}")
        print(f"ROC AUC Score: {roc_auc:.2%}")
        
        # ROC Curve
        RocCurveDisplay.from_estimator(classifier, self.X_test, self.y_test)
        plt.title(f'ROC Curve - {name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Store model
        self.models[name] = {
            'classifier': classifier,
            'cv_score': cv_score,
            'roc_auc': roc_auc
        }
        
        return classifier
    
    def evaluate_model(self, name: str, classifier: Any):
        """
        Display detailed evaluation metrics for a trained model.
        
        Args:
            name: Model name
            classifier: Trained classifier
        """
        print(f"\n{'='*50}")
        print(f"EVALUATION: {name}")
        print(f"{'='*50}")
        
        y_pred = classifier.predict(self.X_test)
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        counts = [value for value in cm.flatten()]
        percentages = [f'{value:.2%}' for value in cm.flatten() / np.sum(cm)]
        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False,
                    xticklabels=['Predicted No', 'Predicted Yes'],
                    yticklabels=['Actual No', 'Actual Yes'])
        plt.title(f'Confusion Matrix - {name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['No Churn', 'Churn']))
    
    def compare_models(self):
        """Display comparison of all trained models."""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        if not self.models:
            print("No models trained yet!")
            return
        
        comparison_df = pd.DataFrame({
            'Model': list(self.models.keys()),
            'CV Score': [self.models[m]['cv_score'] for m in self.models],
            'ROC AUC': [self.models[m]['roc_auc'] for m in self.models]
        })
        
        comparison_df = comparison_df.sort_values('ROC AUC', ascending=False)
        print("\n", comparison_df.to_string(index=False))
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        comparison_df.plot(x='Model', y='CV Score', kind='bar', ax=axes[0], 
                          color='skyblue', legend=False)
        axes[0].set_title('Cross Validation Scores', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Score')
        axes[0].set_ylim([0.5, 1.0])
        axes[0].tick_params(axis='x', rotation=45)
        
        comparison_df.plot(x='Model', y='ROC AUC', kind='bar', ax=axes[1], 
                          color='coral', legend=False)
        axes[1].set_title('ROC AUC Scores', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Score')
        axes[1].set_ylim([0.5, 1.0])
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def run_full_pipeline(self):
        """Execute the complete churn prediction pipeline."""
        print("\n" + "="*60)
        print("CUSTOMER CHURN PREDICTION PIPELINE")
        print("="*60)
        
        # Step 1: Load and explore data
        self.load_data()
        self.explore_data()
        
        # Step 2: Clean data
        self.clean_data()
        
        # Step 3: Visualizations
        self.visualize_distributions()
        self.visualize_categorical_features()
        
        # Step 4: Feature engineering
        self.engineer_features()
        self.visualize_correlation()
        self.drop_low_correlation_features(threshold=0.1)
        
        # Step 5: Balance dataset
        X, y = self.balance_dataset()
        
        # Step 6: Split data
        self.split_data(X, y)
        
        # Step 7: Train models
        print("\n" + "="*60)
        print("MODEL TRAINING PHASE")
        print("="*60)
        
        # XGBoost
        xgb_clf = XGBClassifier(learning_rate=0.01, max_depth=3, 
                               n_estimators=1000, random_state=42)
        self.train_and_evaluate_model("XGBoost", xgb_clf)
        self.evaluate_model("XGBoost", xgb_clf)
        
        # LightGBM
        lgbm_clf = LGBMClassifier(learning_rate=0.01, max_depth=3,
                                 n_estimators=1000, verbose=-1,
                                 min_gain_to_split=0.0, random_state=42)
        self.train_and_evaluate_model("LightGBM", lgbm_clf)
        self.evaluate_model("LightGBM", lgbm_clf)
        
        # Random Forest
        rf_clf = RandomForestClassifier(max_depth=4, random_state=42)
        self.train_and_evaluate_model("Random Forest", rf_clf)
        self.evaluate_model("Random Forest", rf_clf)
        
        # Decision Tree
        dt_clf = DecisionTreeClassifier(random_state=42, max_depth=4, 
                                       min_samples_leaf=1)
        self.train_and_evaluate_model("Decision Tree", dt_clf)
        self.evaluate_model("Decision Tree", dt_clf)
        
        # Stacking Ensemble
        stack_clf = StackingClassifier(
            estimators=[
                ('xgb', xgb_clf),
                ('lgbm', lgbm_clf),
                ('rf', rf_clf),
                ('dt', dt_clf)
            ],
            final_estimator=lgbm_clf
        )
        self.train_and_evaluate_model("Stacking Ensemble", stack_clf)
        self.evaluate_model("Stacking Ensemble", stack_clf)
        
        # Step 8: Compare all models
        self.compare_models()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)


# Main execution
if __name__ == "__main__":
    # Initialize predictor
    predictor = CustomerChurnPredictor("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Run complete pipeline
    predictor.run_full_pipeline()