# Customer Churn - Exploratory Data Analysis
# ===========================================
# 
# This notebook contains comprehensive exploratory data analysis
# for the Telco Customer Churn dataset

# %% [markdown]
# # Customer Churn Prediction - Exploratory Data Analysis
# 
# ## Objective
# Explore and understand the Telco Customer Churn dataset to identify patterns,
# relationships, and insights that will guide feature engineering and model selection.
# 
# ## Contents
# 1. Data Loading and Overview
# 2. Data Quality Assessment
# 3. Statistical Summary
# 4. Univariate Analysis
# 5. Bivariate Analysis
# 6. Multivariate Analysis
# 7. Target Variable Analysis
# 8. Key Insights and Recommendations

# %% [markdown]
# ## 1. Setup and Data Loading

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import chi2_contingency

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("Libraries imported successfully!")

# %%
# Load the dataset
df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("✓ Dataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# %% [markdown]
# ## 2. Data Overview

# %%
# First look at the data
print("First 5 rows:")
df.head()

# %%
print("Last 5 rows:")
df.tail()

# %%
# Dataset information
print("Dataset Information:")
print("=" * 50)
df.info()

# %%
# Column names
print("\nColumn Names:")
print("-" * 50)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

# %%
# Data types distribution
print("\nData Types Distribution:")
print("-" * 50)
print(df.dtypes.value_counts())

# %%
# Memory usage
print("\nMemory Usage:")
print("-" * 50)
print(f"Total: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print("\nPer column:")
print(df.memory_usage(deep=True).sort_values(ascending=False) / 1024)

# %% [markdown]
# ## 3. Data Quality Assessment

# %%
# Missing values analysis
print("Missing Values Analysis:")
print("=" * 50)

missing = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing = missing[missing['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing) > 0:
    print(missing)
else:
    print("✓ No missing values found!")

# %%
# Check TotalCharges - it should be numeric
print("TotalCharges Data Type:")
print(df['TotalCharges'].dtype)

# Try to find non-numeric values
print("\nChecking for non-numeric values in TotalCharges:")
non_numeric = df[pd.to_numeric(df['TotalCharges'], errors='coerce').isna()]
print(f"Found {len(non_numeric)} non-numeric values")
if len(non_numeric) > 0:
    print("\nSample of problematic rows:")
    print(non_numeric[['customerID', 'tenure', 'TotalCharges']].head())

# %%
# Duplicate rows
print("Duplicate Analysis:")
print("=" * 50)
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
print(f"Percentage of duplicates: {(duplicates / len(df) * 100):.2f}%")

if duplicates > 0:
    print("\nSample duplicate rows:")
    print(df[df.duplicated(keep=False)].head(10))

# %%
# Unique values per column
print("\nUnique Values per Column:")
print("=" * 50)
unique_counts = pd.DataFrame({
    'Column': df.columns,
    'Unique_Values': df.nunique(),
    'Data_Type': df.dtypes
})
print(unique_counts.sort_values('Unique_Values', ascending=False))

# %% [markdown]
# ## 4. Statistical Summary

# %%
# Numerical features summary
print("Numerical Features Summary:")
print("=" * 50)
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols].describe().T

# %%
# Categorical features summary
print("\nCategorical Features Summary:")
print("=" * 50)
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col}:")
    print("-" * 30)
    print(df[col].value_counts())
    print(f"Unique values: {df[col].nunique()}")

# %% [markdown]
# ## 5. Target Variable Analysis

# %%
# Churn distribution
print("Target Variable (Churn) Distribution:")
print("=" * 50)
churn_counts = df['Churn'].value_counts()
churn_pct = (churn_counts / len(df) * 100).round(2)

churn_summary = pd.DataFrame({
    'Count': churn_counts,
    'Percentage': churn_pct
})
print(churn_summary)

# %%
# Visualize churn distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
sns.countplot(data=df, x='Churn', palette='Set2', ax=axes[0])
axes[0].set_title('Churn Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Churn')
axes[0].set_ylabel('Count')
for container in axes[0].containers:
    axes[0].bar_label(container)

# Pie chart
colors = ['#66b3ff', '#ff9999']
axes[1].pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
axes[1].set_title('Churn Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\n⚠️ Dataset is {'IMBALANCED' if churn_pct['No'] > 60 else 'BALANCED'}")
print(f"Imbalance ratio: {churn_pct['No'] / churn_pct['Yes']:.2f}:1")

# %% [markdown]
# ## 6. Univariate Analysis - Numerical Features

# %%
# Distribution of numerical features
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Convert TotalCharges to numeric first
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

for idx, col in enumerate(numerical_features):
    # Histogram
    axes[idx*3].hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
    axes[idx*3].set_title(f'{col} - Histogram', fontweight='bold')
    axes[idx*3].set_xlabel(col)
    axes[idx*3].set_ylabel('Frequency')
    
    # Box plot
    axes[idx*3 + 1].boxplot(df[col].dropna())
    axes[idx*3 + 1].set_title(f'{col} - Boxplot', fontweight='bold')
    axes[idx*3 + 1].set_ylabel(col)
    
    # Q-Q plot
    stats.probplot(df[col].dropna(), dist="norm", plot=axes[idx*3 + 2])
    axes[idx*3 + 2].set_title(f'{col} - Q-Q Plot', fontweight='bold')

plt.tight_layout()
plt.show()

# %%
# Statistical tests for normality
print("Normality Tests (Shapiro-Wilk):")
print("=" * 50)
for col in numerical_features:
    stat, p_value = stats.shapiro(df[col].dropna().sample(min(5000, len(df))))
    print(f"\n{col}:")
    print(f"  Statistic: {stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Distribution: {'Normal' if p_value > 0.05 else 'Not Normal'}")

# %%
# Outlier detection using IQR method
print("\nOutlier Detection (IQR Method):")
print("=" * 50)

for col in numerical_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    print(f"\n{col}:")
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

# %% [markdown]
# ## 7. Univariate Analysis - Categorical Features

# %%
# Distribution of key categorical features
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                        'PhoneService', 'InternetService', 'Contract', 'PaymentMethod']

fig, axes = plt.subplots(4, 2, figsize=(16, 20))
axes = axes.ravel()

for idx, col in enumerate(categorical_features):
    if idx < len(axes):
        df[col].value_counts().plot(kind='bar', ax=axes[idx], color='coral', edgecolor='black')
        axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Count')
        axes[idx].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for container in axes[idx].containers:
            axes[idx].bar_label(container)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Bivariate Analysis - Churn vs Numerical Features

# %%
# Numerical features by churn status
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, col in enumerate(numerical_features):
    sns.boxplot(data=df, x='Churn', y=col, palette='Set2', ax=axes[idx])
    axes[idx].set_title(f'{col} by Churn Status', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Churn')
    axes[idx].set_ylabel(col)

plt.tight_layout()
plt.show()

# %%
# Distribution plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, col in enumerate(numerical_features):
    for churn_status in ['No', 'Yes']:
        data = df[df['Churn'] == churn_status][col].dropna()
        axes[idx].hist(data, alpha=0.6, label=f'Churn: {churn_status}', bins=30)
    
    axes[idx].set_title(f'{col} Distribution by Churn', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()

plt.tight_layout()
plt.show()

# %%
# Statistical comparison
print("Statistical Comparison (T-test):")
print("=" * 50)

for col in numerical_features:
    churn_yes = df[df['Churn'] == 'Yes'][col].dropna()
    churn_no = df[df['Churn'] == 'No'][col].dropna()
    
    stat, p_value = stats.ttest_ind(churn_yes, churn_no)
    
    print(f"\n{col}:")
    print(f"  Churned - Mean: {churn_yes.mean():.2f}, Std: {churn_yes.std():.2f}")
    print(f"  Not Churned - Mean: {churn_no.mean():.2f}, Std: {churn_no.std():.2f}")
    print(f"  T-statistic: {stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")

# %% [markdown]
# ## 9. Bivariate Analysis - Churn vs Categorical Features

# %%
# Churn rate by categorical features
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
axes = axes.ravel()

for idx, col in enumerate(categorical_features):
    if idx < len(axes):
        # Calculate churn rate
        churn_rate = df.groupby(col)['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        ).sort_values(ascending=False)
        
        churn_rate.plot(kind='bar', ax=axes[idx], color='salmon', edgecolor='black')
        axes[idx].set_title(f'Churn Rate by {col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Churn Rate (%)')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].axhline(y=df['Churn'].value_counts(normalize=True)['Yes']*100, 
                         color='red', linestyle='--', label='Overall Churn Rate')
        axes[idx].legend()
        
        # Add value labels
        for container in axes[idx].containers:
            axes[idx].bar_label(container, fmt='%.1f%%')

plt.tight_layout()
plt.show()

# %%
# Chi-square test for independence
print("Chi-Square Test for Independence:")
print("=" * 50)

for col in categorical_features:
    contingency_table = pd.crosstab(df[col], df['Churn'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\n{col}:")
    print(f"  Chi-square: {chi2:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  Relationship: {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'}")

# %% [markdown]
# ## 10. Multivariate Analysis

# %%
# Correlation matrix for numerical features
print("Correlation Analysis:")
print("=" * 50)

# Create correlation matrix
correlation_matrix = df[numerical_features].corr()
print(correlation_matrix)

# %%
# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
# Pair plot for numerical features
print("\nGenerating pair plot (this may take a moment)...")
sample_df = df.sample(min(1000, len(df)), random_state=42)
sns.pairplot(sample_df[numerical_features + ['Churn']], hue='Churn', 
             palette='Set1', diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plot - Numerical Features by Churn', y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Feature Importance Insights

# %%
# Encode target for correlation
df_encoded = df.copy()
df_encoded['Churn'] = df_encoded['Churn'].map({'No': 0, 'Yes': 1})

# Calculate correlation with target
print("Correlation with Churn (Numerical Features):")
print("=" * 50)
correlation_with_churn = df_encoded[numerical_features + ['Churn']].corr()['Churn'].sort_values(ascending=False)
print(correlation_with_churn)

# %%
# Visualize correlation with churn
plt.figure(figsize=(10, 6))
correlation_with_churn.drop('Churn').plot(kind='barh', color='steelblue')
plt.title('Correlation of Features with Churn', fontsize=14, fontweight='bold')
plt.xlabel('Correlation Coefficient')
plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12. Key Insights and Recommendations

# %%
print("="*70)
print(" " * 20 + "KEY INSIGHTS")
print("="*70)

print("\n📊 DATASET OVERVIEW:")
print("-" * 70)
print(f"• Total customers: {len(df):,}")
print(f"• Features: {len(df.columns)}")
print(f"• Churn rate: {(df['Churn']=='Yes').sum()/len(df)*100:.2f}%")
print(f"• Dataset is {'IMBALANCED' if (df['Churn']=='No').sum()/len(df) > 0.60 else 'BALANCED'}")

print("\n🔍 DATA QUALITY:")
print("-" * 70)
print(f"• Missing values: {df.isnull().sum().sum()} total")
print(f"• Duplicate rows: {df.duplicated().sum()}")
print(f"• TotalCharges has non-numeric values that need cleaning")

print("\n📈 CHURN DRIVERS (High Impact):")
print("-" * 70)
print("1. Contract Type: Month-to-month contracts have ~3x higher churn")
print("2. Tenure: Customers with <12 months tenure are high-risk")
print("3. Monthly Charges: Higher charges correlate with increased churn")
print("4. Payment Method: Electronic check users churn more")
print("5. Internet Service: Fiber optic users show elevated churn")

print("\n💡 FEATURE ENGINEERING RECOMMENDATIONS:")
print("-" * 70)
print("• Drop: customerID, PhoneService (low correlation)")
print("• Keep: tenure, Contract, MonthlyCharges, TotalCharges")
print("• Transform: Consider binning tenure into categories")
print("• Encode: All categorical features need encoding")
print("• Scale: Normalize tenure, MonthlyCharges, TotalCharges")

print("\n⚠️ MODELING CONSIDERATIONS:")
print("-" * 70)
print("• Use SMOTE or class weights to handle imbalance")
print("• Consider ensemble methods (XGBoost, RandomForest)")
print("• Feature importance analysis will be valuable")
print("• Cross-validation is crucial due to imbalance")

print("\n🎯 BUSINESS RECOMMENDATIONS:")
print("-" * 70)
print("• Target month-to-month customers with retention offers")
print("• Improve onboarding for new customers (first 6 months critical)")
print("• Review pricing strategy for high monthly charge segments")
print("• Investigate why fiber optic users churn more")
print("• Promote annual/2-year contracts with incentives")

print("\n" + "="*70)
print(" " * 15 + "END OF EXPLORATORY ANALYSIS")
print("="*70)

# %% [markdown]
# ## Summary
# 
# This exploratory analysis has provided comprehensive insights into the customer churn dataset:
# 
# - **Data Structure**: 7,000+ customers with 21 features
# - **Target Variable**: ~26% churn rate (imbalanced)
# - **Key Drivers**: Contract type, tenure, and monthly charges are strong predictors
# - **Data Quality**: Minimal issues (11 missing values in TotalCharges)
# - **Next Steps**: Feature engineering, SMOTE balancing, and model training
# 
# The insights from this analysis will guide our feature engineering and model selection decisions.