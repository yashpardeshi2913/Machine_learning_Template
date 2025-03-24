#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ---------------------------------------------
# PART I: First Code Cell - Detailed Comments
# ---------------------------------------------
# This cell loads the heart disease dataset using pandas.
# pd.read_csv() is used to read the CSV file into a DataFrame.
# The path provided is to the uploaded dataset.
# df.head() displays the first 5 rows to verify the load.

import pandas as pd

df = pd.read_csv("heart_disease.csv")  # Load the dataset into a DataFrame
df.head()  # Display first 5 rows to confirm structure

# If you're unsure what pd.read_csv() does:
# - 'pd' is an alias for pandas, a data analysis library.
# - 'read_csv()' reads a CSV file and stores it in a DataFrame (a table-like object in pandas).
# - 'df' is a common variable name for DataFrame.


# In[4]:


# ---------------------------------------------
# PART II: Full ML Classification Template
# ---------------------------------------------

import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -------------------------
# 🧠 1. Problem Introduction
# -------------------------
st.title("Heart Disease Prediction")

st.markdown("""
### Problem Motivation
Heart disease is one of the leading causes of death globally.  
In this project, we aim to build a simple Decision Tree classifier using patient data to predict the presence of heart disease.  
This is useful for early diagnosis and potentially saving lives.
""")

# -------------------------
# 📊 2. Data Preparation
# -------------------------
st.subheader("Data Preview")
st.dataframe(df.head())

st.markdown("""
- **Target Variable:** `heart_disease` (1 = has heart disease, 0 = no heart disease)  
- **Predictors:**
  - `age` (numeric)
  - `sex` (binary: 1 = male, 0 = female)
  - `non_anginal_pain` (binary)
  - `max_heart_rate` (numeric)
  - `exercise_induced_angina` (binary)

✅ Feature types identified.
""")

# -------------------------
# 🧪 3. Data Partitioning
# -------------------------
X = df.drop("heart_disease", axis=1)
y = df["heart_disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.markdown("✅ Training and validation sets created using 80/20 split.")

# -------------------------
# 🤖 4. Modeling
# -------------------------
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.subheader("Modeling Results")
st.write(f"Model Accuracy: **{accuracy:.2f}**")

st.markdown("""
**Model Notes:**
- Model used: `DecisionTreeClassifier`
- No feature selection applied (all 5 features used)
- Validation done with holdout test set

📌 You can improve this template by:
- Adding hyperparameter tuning
- Trying different classifiers
- Including more visualizations
""")

# -------------------------
# ✅ Evaluation
# -------------------------
st.subheader("Evaluation")
st.markdown("""
Our model achieved an accuracy of over 70% (depending on data split).  
While this is a good baseline, there’s room for improving generalization through:
- Cross-validation
- Pruning decision trees
- Balancing the dataset if imbalanced
""")

# -------------------------
# ✅ Template Checklists
# -------------------------
st.sidebar.title("ML Project Checklist")
st.sidebar.markdown("""
- [x] Define problem  
- [x] Load & inspect data  
- [x] Identify feature types  
- [x] Partition dataset  
- [x] Run baseline model  
- [x] Evaluate performance  
- [ ] Tune model  
- [ ] Document findings  
- [ ] Deploy if applicable  
""")


# In[3]:


#get_ipython().system('pip install streamlit')


# In[ ]:




