"""
Customer Churn Prediction Package
==================================

A production-ready machine learning pipeline for predicting customer churn.

Main Components:
    - CustomerChurnPredictor: Main pipeline class
    
Example:
    >>> from src.churn_predictor import CustomerChurnPredictor
    >>> predictor = CustomerChurnPredictor("Data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    >>> predictor.run_full_pipeline()
"""

__version__ = "1.0.0"
__author__ = "Rojin Dhami"
__email__ = "your.email@example.com"

from .churn_predictor import CustomerChurnPredictor

__all__ = ["CustomerChurnPredictor"]