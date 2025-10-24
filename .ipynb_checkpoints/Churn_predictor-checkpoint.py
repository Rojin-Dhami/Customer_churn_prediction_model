#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Loading the dataset 

# In[2]:


df=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()


# Understanding the data 

# In[3]:


df.shape


# In[4]:


df.info()


# Note: Most of the features are of onject type , even feature like TotalCharges are object which is supposed to be float . This is most likely due to whitespaces and there is no need of customerID for the further analysis and model building 

# In[5]:


# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows where TotalCharges couldn't be converted
print("Missing values in TotalCharges:", df['TotalCharges'].isna().sum())
df = df.dropna(subset=['TotalCharges'])

# Reset index after drop
df.reset_index(drop=True, inplace=True)


# In[6]:


df.drop('customerID', axis=1, inplace=True)


# In[7]:


#checking missing values
df.isna().sum()


# In[8]:


df.duplicated().sum()


# In[9]:


df.drop_duplicates(inplace=True)



# In[10]:


df.describe()


# Summary:
# - TotalCharges=tenure*MonthlyCharges
# - tenure ranges from 1 to 72 months
# - MonthlyCharges range from $18.25 to $118.75
# - On avg. typical customers has been with the company ~32 months, pays around $65/month, and has spent about $2,200 in total.

# Checking the Target Variable:
# 
#  This section includes the checking of target variale which is Churn to see how the churn value distribution looks like.
# 

# In[11]:


df['Churn'].value_counts()


# In[12]:


df['Churn'].value_counts()/len(df)*100


# Univariate Analysis

# Churn by gender 

# In[13]:


df[['Churn', 'gender']].value_counts().unstack()


# In[14]:


plt.figure(figsize=(12, 6))
sns.countplot(x='Churn', hue='gender', data=df)
plt.title('Customer Churn by Gender', fontsize=16, fontweight='bold')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')
plt.legend(title='Gender')
plt.tight_layout()
plt.show()


# Observation:
# - Churn rate seems similar among the both gender
# - So, gender alone isn't that strong feature for prediction 

# In[15]:


# Distribution of numerical features by churn
numerical_features = ['MonthlyCharges', 'TotalCharges','tenure']
for col in numerical_features:
  plt.figure(figsize=(8, 4))
  sns.histplot(data=df, x=col, hue='Churn', kde=True)
  plt.title(f'Distribution of {col} by Churn')
  plt.xlabel(col)
  plt.ylabel('Frequency')
  plt.show()


# In[16]:


# Distribution of categorical features by churn
categorical_features = df.select_dtypes(include='object').columns.tolist()
categorical_features.remove('Churn')
categorical_features.remove('gender')
categorical_features.append('SeniorCitizen')

for col in categorical_features:
  plt.figure(figsize=(8, 4))
  sns.countplot(data=df, x=col, hue='Churn', palette='viridis')
  plt.title(f'Distribution of {col} by Churn')
  plt.xlabel(col)
  plt.ylabel('Count')
  plt.xticks(rotation=45, ha='right')
  plt.tight_layout()
  plt.show()


# Summary of Key Insights:
# 
# - Churn rate seems to depends on Monthlycharges as churned people tends to pay high
# - Gender have little infulence in churn
# - Contract type has a strong relationship with churn as shorter contracts result in more churn.
# - Shorter tenures are associated with high churn
# - From payment method with electronic checks have hihg churn

# Feature Engineering
# 
# Now we have explored and understand the dataset and the key drivers of the of target.
# 
# Here, we will perform data preprocessing which includes :
# - Dropping the irrelevant cols
# - encoding the categorical variable
# - Scaling the numeric features 
# 

# In[17]:


df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Now get correlation
df.corr(numeric_only=True)['Churn'].sort_values(ascending=False)


# In[18]:


from sklearn.preprocessing import MinMaxScaler,StandardScaler
mms = MinMaxScaler()
df['tenure'] = mms.fit_transform(df[['tenure']])
df['MonthlyCharges'] = mms.fit_transform(df[['MonthlyCharges']])
df['TotalCharges'] = mms.fit_transform(df[['TotalCharges']])
df.head()


# Normalization is done for features whose data does not display normal distribution and standardization is carried out for features that are normally distributed where their values are huge or very small as compared to other features.
# 
# So, tenure, MonthlyCharges and TotalCharges features are normalized as they displayed a right skewed and bimodal data distribution.

# In[19]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])


# Correlation Matrix:

# In[20]:


plt.figure(figsize=(20,5))
sns.heatmap(df.corr(), cmap='coolwarm', annot=True)


# In[21]:


corr = df.corrwith(df['Churn']).sort_values(ascending = False).to_frame()
corr.columns = ['Correlations']
plt.subplots(figsize = (5,5))
sns.heatmap(corr,annot = True,cmap = 'coolwarm',linewidths = 0.4,linecolor = 'black');
plt.title('Correlation w.r.t Outcome');


# MulipleLines, PhoneService, gender, StreamingTV, StreamingMovies and InternetService does not display any kind of correlation. We drop the features with correlation coefficient between (-0.1,0.1).
# 
# Remaining features either display a significant positive or negative correlation.

# In[22]:


df.drop(columns = ['PhoneService', 'gender','StreamingTV','StreamingMovies','MultipleLines','InternetService'],inplace = True)
df.head()


# Data Balancing :
# 
# As the given dataset is imbalanced ,so we need to solve this issue as models can be biased due to this. Thus , this is solved using SMOTE analysis 

# In[23]:


import imblearn
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# In[24]:


over = SMOTE(sampling_strategy = 1)

f1 = df.iloc[:,:13].values
t1 = df.iloc[:,13].values

f1, t1 = over.fit_resample(f1, t1)
Counter(t1)


# Modeling 
# 

# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve


# In[26]:


x_train, x_test, y_train, y_test = train_test_split(f1, t1, test_size = 0.20, random_state = 2)


# In[27]:


def model(classifier,x_train,y_train,x_test,y_test):
    
    classifier.fit(x_train,y_train)
    prediction = classifier.predict(x_test)
    cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3,random_state = 1)
    print("Cross Validation Score : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'roc_auc').mean()))
    print("ROC_AUC Score : ",'{0:.2%}'.format(roc_auc_score(y_test,prediction)))
    RocCurveDisplay.from_estimator(classifier, x_test, y_test)
    plt.title('ROC_AUC_Plot')
    plt.show()

def model_evaluation(classifier,x_test,y_test):
    
    # Confusion Matrix
    cm = confusion_matrix(y_test,classifier.predict(x_test))
    names = ['True Neg','False Pos','False Neg','True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm,annot = labels,cmap = 'Blues',fmt ='')
    
    # Classification Report
    print(classification_report(y_test,classifier.predict(x_test)))


# XGBoost Classifier

# In[28]:


from xgboost import XGBClassifier

classifier_xgb = XGBClassifier(learning_rate= 0.01,max_depth = 3,n_estimators = 1000)


# In[29]:


model(classifier_xgb,x_train,y_train,x_test,y_test)


# In[30]:


model_evaluation(classifier_xgb,x_test,y_test)


# LightGBM Classifier

# In[46]:


from lightgbm import LGBMClassifier

classifier_lgbm = LGBMClassifier(
    learning_rate=0.01,
    max_depth=3,
    n_estimators=1000,
    verbose=-1,            # disables LightGBM logs and warnings
    min_gain_to_split=0.0  # optional: prevents "no further splits" warning
)


# In[47]:


model(classifier_lgbm,x_train,y_train,x_test,y_test)


# In[48]:


model_evaluation(classifier_lgbm,x_test,y_test)


# RandomForest Classifier

# In[49]:


from sklearn.ensemble import RandomForestClassifier


# In[50]:


classifier_rf = RandomForestClassifier(max_depth = 4,random_state = 0)


# In[51]:


model(classifier_rf,x_train,y_train,x_test,y_test)


# In[52]:


model_evaluation(classifier_rf,x_test,y_test)


# DecisionTree Classifier

# In[53]:


from sklearn.tree import DecisionTreeClassifier


# In[54]:


classifier_dt = DecisionTreeClassifier(random_state = 1000,max_depth = 4,min_samples_leaf = 1)


# In[55]:


model(classifier_dt,x_train,y_train,x_test,y_test)


# In[56]:


model_evaluation(classifier_dt,x_test,y_test)


# Stack of XGBClassifier, LightGBMClassifier, Random Forest Classifer & Decision Tree Classifier

# In[57]:


from sklearn.ensemble import StackingClassifier

stack = StackingClassifier(estimators = [('classifier_xgb',classifier_xgb),
                                         ('classifier_lgbm',classifier_lgbm),
                                         ('classifier_rf',classifier_rf),
                                         ('classifier_dt',classifier_dt)],
                           final_estimator = classifier_lgbm)


# In[58]:


model(stack,x_train,y_train,x_test,y_test)


# In[59]:


model_evaluation(stack,x_test,y_test)


# In[ ]:




