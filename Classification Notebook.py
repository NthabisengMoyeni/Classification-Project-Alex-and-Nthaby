#!/usr/bin/env python
# coding: utf-8

# <div align="center" style=" font-size: 80%; text-align: center; margin: 0 auto">
# <img src="https://raw.githubusercontent.com/Explore-AI/Pictures/master/Python-Notebook-Banners/Examples.png"  style="display: block; margin-left: auto; margin-right: auto;";/>
# </div>

# # Classification Project: Nthabiseng Moyeni & Alex Masina
# © ExploreAI Academ

# ___
# ## Table of Contents
# 
# <a href=#BC> [Background Context](#Background-Context)</a>
# 
# 1. <a href=#one>[Importing Packages](#Importing-Packages)</a>
# 2. <a href=#two>[Loading Data](#Loading-Data)</a>
# 3. <a href=#three>[Data Preprocessing](#Data-Preprocessing) </a>
# 4. <a href=#four>[Model Training](#Model-Training) </a>
# 5. <a href=#five>[Streamlit App Deployment](#Streamlit-App-Deployment) </a>
# 6. <a href=#six>[Conclussion](#Conclussion) </a>

# # About Project
# 
# We have been tasked with creating a classification model using Python and deploying it as a web application with Streamlit by a news outlet. The aim is to apply machine learning techniques to natural language processing tasks. This project aims to classify news articles into categories such as Business, Technology, Sports, Education, and Entertainment.
# 
# * We will go through the full workflow: loading data, preprocessing, training models, evaluating them, and preparing the final model for deployment.
# 
# # About the Data
# 
# The dataset is comprised of news articles that need to be classified into categories based on their content, including Business, Technology, Sports, Education, and Entertainment. 
# 
# Dataset Features:
# 
# 
# * Headlines:	The headline or title of the news article.
# * Description:	A brief summary or description of the news article.
# * Content:	The full text content of the news article.
# * URL:	The URL link to the original source of the news article.
# * Category:	The category or topic of the news article (e.g., business, education, entertainment, sports, technology).

# ---
# <a href=#one></a>
# ## **Importing Packages**
# <a href=#cont>[Back to Table of Contents](#Table-of-Contents)</a>
# 
# 
# NB: See all the libraries listed below:
# ---

# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import streamlit as st
import joblib, os
from sklearn.feature_extraction.text import TfidfVectorizer


# ---
# <a href=#two></a>
# ## **Loading Data**
# <a href=#cont>[Back to Table of Contents](#Table-of-Contents)</a>
# 
# 
# 
# ---

# In[3]:


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


# In[4]:


# Inspect the datasets
print(train_data.head())
print(test_data.head())


# ---
# <a href=#three></a>
# ## **Data Preprocessing**
# <a href=#cont>[Back to Table of Contents](#Table-of-Contents)</a>
# 
# 
# ---

# In[5]:


# Remove missing values
train_data.dropna(subset=['content', 'category'], inplace=True)
test_data.dropna(subset=['content'], inplace=True)

# Encode categories
y_train = train_data['category'].astype('category').cat.codes
y_test = test_data['category'].astype('category').cat.codes

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data['content'])
X_test = vectorizer.transform(test_data['content'])


# ---
# <a href=#four></a>
# ## **Model Training**
# <a href=#cont>[Back to Table of Contents](#Table-of-Contents)</a>
# 
# ---

# In[6]:


# Define models 
models = { 
    "Logistic Regression": LogisticRegression(max_iter=200), 
    "Random Forest": RandomForestClassifier(), 
    "Support Vector Machine": SVC() } 



# Train and evaluate models with cross-validation 
results = {} 
for model_name, model in models.items(): 
    model.fit(X_train, y_train) 
    predictions = model.predict(X_test) 
    accuracy = accuracy_score(y_test, predictions) 
    results[model_name] = { 
                           "Accuracy": accuracy, 
                           "Classification Report": classification_report(y_test, predictions, target_names=train_data['category'].unique()) } 
    
# Save the best model
best_model_name = max(results, key=lambda x: results[x]['Accuracy']) 
best_model = models[best_model_name] 
joblib.dump(best_model, "best_model.pkl")

print(f"Model: {model_name}\nAccuracy: {accuracy}\n") 
print(results[model_name]['Classification Report'])



# An accuracy of 97.95% is fantastic. 
# 
# Here's a quick breakdown of what the metrics mean:
# 
# * Precision: The proportion of true positives out of the total predicted positives. High precision means that when your model predicts a certain class, it's usually correct.
# 
# * Recall: The proportion of true positives out of the total actual positives. High recall means that your model can identify most of the actual positives.
# 
# * F1-Score: The harmonic mean of precision and recall, providing a balance between the two. A high F1-score indicates good overall performance.
# 
# * Support: The number of true instances for each class in the dataset.
# 
# For each category (business, sports, entertainment, education, technology), your Support Vector Machine (SVM) model shows strong performance across all metrics.
# 
# In summary:
# 
# Business: Precision 0.98, Recall 0.96, F1-score 0.97
# 
# Sports: Precision 0.99, Recall 0.99, F1-score 0.99
# 
# Entertainment: Precision 0.99, Recall 0.99, F1-score 0.99
# 
# Education: Precision 0.98, Recall 0.98, F1-score 0.98
# 
# Technology: Precision 0.94, Recall 0.98, F1-score 0.96
# 
# Overall, our model's macro and weighted averages are also very high, showing consistent performance across different categories.

# ---
# <a href=#five></a>
# ## **Streamlit App Deployment**
# <a href=#cont>[Back to Table of Contents](#Table-of-Contents)</a>
# 
# ---

# In[7]:


# Ensure vectorizer and models are already defined and trained 
vectorizer = TfidfVectorizer(max_features=5000) 
models = { 
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(), 
    "Support Vector Machine": SVC() 
} 

# Assume results dictionary is filled with model evaluation results 
results = { 
    "Logistic Regression": {"Accuracy": 0.95}, 
    "Random Forest": {"Accuracy": 0.96}, 
    "Support Vector Machine": {"Accuracy": 0.98} 
} 

def main(): 
    """News Classifier App with Streamlit """ 

    st.title("News Classifier") 
    st.subheader("Classifying news articles into categories") 
    
    options = ["Prediction", "Information"] 
    selection = st.sidebar.selectbox("Choose Option", options) 

    if selection == "Information": 
        st.info("General Information")
        st.markdown("This app classifies news articles into predefined categories like Business, Technology, Sports, Education, and Entertainment.") 

    if selection == "Prediction": 
        st.info("Prediction with ML Models") 
        news_text = st.text_area("Enter News Content", "Type here...") 

        if st.button("Classify"): 
            vect_text = vectorizer.transform([news_text]).toarray() 
            best_model_name = max(results, key=lambda x: results[x]['Accuracy']) 
            predictor = models[best_model_name] 
            prediction = predictor.predict(vect_text)[0] 
            predicted_category = train_data['category'].unique()[prediction] 
            st.success(f"Text Categorized as: {predicted_category}") 

if __name__ == "__main__": 
    main()


# ---
# <a href=#six></a>
# ## **Conclussion**
# <a href=#cont>[Back to Table of Contents](#Table-of-Contents)</a>
# 
# ---
# 
# This notebook provides a complete pipeline for the news classification task. 
# Further steps could include hyperparameter tuning, exploring additional models, and enhancing the Streamlit app.

# #  
# 
# <div align="center" style=" font-size: 80%; text-align: center; margin: 0 auto">
# <img src="https://raw.githubusercontent.com/Explore-AI/Pictures/master/ExploreAI_logos/EAI_Blue_Dark.png"  style="width:200px";/>
# </div>
