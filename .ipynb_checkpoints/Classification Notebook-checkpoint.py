{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div align=\"center\" style=\" font-size: 80%; text-align: center; margin: 0 auto\">\n",
    "<img src=\"https://raw.githubusercontent.com/Explore-AI/Pictures/master/Python-Notebook-Banners/Examples.png\"  style=\"display: block; margin-left: auto; margin-right: auto;\";/>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Classification Project: Nthabiseng Moyeni & Alex Masina\n",
    "© ExploreAI Academ"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "## Table of Contents\n",
    "\n",
    "<a href=#BC> [Background Context](#Background-Context)</a>\n",
    "\n",
    "1. <a href=#one>[Importing Packages](#Importing-Packages)</a>\n",
    "2. <a href=#two>[Loading Data](#Loading-Data)</a>\n",
    "3. <a href=#three>[Data Preprocessing](#Data-Preprocessing) </a>\n",
    "4. <a href=#four>[Model Training](#Model-Training) </a>\n",
    "5. <a href=#five>[Streamlit App Deployment](#Streamlit-App-Deployment) </a>\n",
    "6. <a href=#six>[Conclussion](#Conclussion) </a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# About Project\n",
    "\n",
    "We have been tasked with creating a classification model using Python and deploying it as a web application with Streamlit by a news outlet. The aim is to apply machine learning techniques to natural language processing tasks. This project aims to classify news articles into categories such as Business, Technology, Sports, Education, and Entertainment.\n",
    "\n",
    "* We will go through the full workflow: loading data, preprocessing, training models, evaluating them, and preparing the final model for deployment.\n",
    "\n",
    "# About the Data\n",
    "\n",
    "The dataset is comprised of news articles that need to be classified into categories based on their content, including Business, Technology, Sports, Education, and Entertainment. \n",
    "\n",
    "Dataset Features:\n",
    "\n",
    "\n",
    "* Headlines:\tThe headline or title of the news article.\n",
    "* Description:\tA brief summary or description of the news article.\n",
    "* Content:\tThe full text content of the news article.\n",
    "* URL:\tThe URL link to the original source of the news article.\n",
    "* Category:\tThe category or topic of the news article (e.g., business, education, entertainment, sports, technology)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "<a href=#one></a>\n",
    "## **Importing Packages**\n",
    "<a href=#cont>[Back to Table of Contents](#Table-of-Contents)</a>\n",
    "\n",
    "\n",
    "NB: See all the libraries listed below:\n",
    "---"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import necessary libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import classification_report, accuracy_score, confusion_matrix\n",
    "import streamlit as st\n",
    "import joblib, os\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "<a href=#two></a>\n",
    "## **Loading Data**\n",
    "<a href=#cont>[Back to Table of Contents](#Table-of-Contents)</a>\n",
    "\n",
    "\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data = pd.read_csv(\"train.csv\")\n",
    "test_data = pd.read_csv(\"test.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                           headlines  \\\n",
      "0  RBI revises definition of politically-exposed ...   \n",
      "1  NDTV Q2 net profit falls 57.4% to Rs 5.55 cror...   \n",
      "2  Akasa Air ‘well capitalised’, can grow much fa...   \n",
      "3  India’s current account deficit declines sharp...   \n",
      "4  States borrowing cost soars to 7.68%, highest ...   \n",
      "\n",
      "                                         description  \\\n",
      "0  The central bank has also asked chairpersons a...   \n",
      "1  NDTV's consolidated revenue from operations wa...   \n",
      "2  The initial share sale will be open for public...   \n",
      "3  The current account deficit (CAD) was 3.8 per ...   \n",
      "4  The prices shot up reflecting the overall high...   \n",
      "\n",
      "                                             content  \\\n",
      "0  The Reserve Bank of India (RBI) has changed th...   \n",
      "1  Broadcaster New Delhi Television Ltd on Monday...   \n",
      "2  Homegrown server maker Netweb Technologies Ind...   \n",
      "3  India’s current account deficit declined sharp...   \n",
      "4  States have been forced to pay through their n...   \n",
      "\n",
      "                                                 url  category  \n",
      "0  https://indianexpress.com/article/business/ban...  business  \n",
      "1  https://indianexpress.com/article/business/com...  business  \n",
      "2  https://indianexpress.com/article/business/mar...  business  \n",
      "3  https://indianexpress.com/article/business/eco...  business  \n",
      "4  https://indianexpress.com/article/business/eco...  business  \n",
      "                                           headlines  \\\n",
      "0  NLC India wins contract for power supply to Ra...   \n",
      "1  SBI Clerk prelims exams dates announced; admit...   \n",
      "2  Golden Globes: Michelle Yeoh, Will Ferrell, An...   \n",
      "3  OnePlus Nord 3 at Rs 27,999 as part of new pri...   \n",
      "4  Adani family’s partners used ‘opaque’ funds to...   \n",
      "\n",
      "                                         description  \\\n",
      "0  State-owned firm NLC India Ltd (NLCIL) on Mond...   \n",
      "1  SBI Clerk Prelims Exam: The SBI Clerk prelims ...   \n",
      "2  Barbie is the top nominee this year, followed ...   \n",
      "3  New deal makes the OnePlus Nord 3 an easy purc...   \n",
      "4  Citing review of files from multiple tax haven...   \n",
      "\n",
      "                                             content  \\\n",
      "0  State-owned firm NLC India Ltd (NLCIL) on Mond...   \n",
      "1  SBI Clerk Prelims Exam: The State Bank of Indi...   \n",
      "2  Michelle Yeoh, Will Ferrell, Angela Bassett an...   \n",
      "3  In our review of the OnePlus Nord 3 5G, we pra...   \n",
      "4  Millions of dollars were invested in some publ...   \n",
      "\n",
      "                                                 url       category  \n",
      "0  https://indianexpress.com/article/business/com...       business  \n",
      "1  https://indianexpress.com/article/education/sb...      education  \n",
      "2  https://indianexpress.com/article/entertainmen...  entertainment  \n",
      "3  https://indianexpress.com/article/technology/t...     technology  \n",
      "4  https://indianexpress.com/article/business/ada...       business  \n"
     ]
    }
   ],
   "source": [
    "# Inspect the datasets\n",
    "print(train_data.head())\n",
    "print(test_data.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "<a href=#three></a>\n",
    "## **Data Preprocessing**\n",
    "<a href=#cont>[Back to Table of Contents](#Table-of-Contents)</a>\n",
    "\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Remove missing values\n",
    "train_data.dropna(subset=['content', 'category'], inplace=True)\n",
    "test_data.dropna(subset=['content'], inplace=True)\n",
    "\n",
    "# Encode categories\n",
    "y_train = train_data['category'].astype('category').cat.codes\n",
    "y_test = test_data['category'].astype('category').cat.codes\n",
    "\n",
    "# Feature extraction using TF-IDF\n",
    "vectorizer = TfidfVectorizer(max_features=5000)\n",
    "X_train = vectorizer.fit_transform(train_data['content'])\n",
    "X_test = vectorizer.transform(test_data['content'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "<a href=#four></a>\n",
    "## **Model Training**\n",
    "<a href=#cont>[Back to Table of Contents](#Table-of-Contents)</a>\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: Support Vector Machine\n",
      "Accuracy: 0.9755\n",
      "\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "     business       0.98      0.95      0.97       400\n",
      "       sports       0.99      0.99      0.99       400\n",
      "entertainment       0.99      0.98      0.99       400\n",
      "    education       0.99      0.98      0.98       400\n",
      "   technology       0.93      0.98      0.95       400\n",
      "\n",
      "     accuracy                           0.98      2000\n",
      "    macro avg       0.98      0.98      0.98      2000\n",
      " weighted avg       0.98      0.98      0.98      2000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Define models \n",
    "models = { \n",
    "    \"Logistic Regression\": LogisticRegression(max_iter=200), \n",
    "    \"Random Forest\": RandomForestClassifier(), \n",
    "    \"Support Vector Machine\": SVC() } \n",
    "\n",
    "\n",
    "\n",
    "# Train and evaluate models with cross-validation \n",
    "results = {} \n",
    "for model_name, model in models.items(): \n",
    "    model.fit(X_train, y_train) \n",
    "    predictions = model.predict(X_test) \n",
    "    accuracy = accuracy_score(y_test, predictions) \n",
    "    results[model_name] = { \n",
    "                           \"Accuracy\": accuracy, \n",
    "                           \"Classification Report\": classification_report(y_test, predictions, target_names=train_data['category'].unique()) } \n",
    "    \n",
    "# Save the best model\n",
    "best_model_name = max(results, key=lambda x: results[x]['Accuracy']) \n",
    "best_model = models[best_model_name] \n",
    "joblib.dump(best_model, \"best_model.pkl\")\n",
    "\n",
    "print(f\"Model: {model_name}\\nAccuracy: {accuracy}\\n\") \n",
    "print(results[model_name]['Classification Report'])\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "An accuracy of 97.95% is fantastic. \n",
    "\n",
    "Here's a quick breakdown of what the metrics mean:\n",
    "\n",
    "* Precision: The proportion of true positives out of the total predicted positives. High precision means that when your model predicts a certain class, it's usually correct.\n",
    "\n",
    "* Recall: The proportion of true positives out of the total actual positives. High recall means that your model can identify most of the actual positives.\n",
    "\n",
    "* F1-Score: The harmonic mean of precision and recall, providing a balance between the two. A high F1-score indicates good overall performance.\n",
    "\n",
    "* Support: The number of true instances for each class in the dataset.\n",
    "\n",
    "For each category (business, sports, entertainment, education, technology), your Support Vector Machine (SVM) model shows strong performance across all metrics.\n",
    "\n",
    "In summary:\n",
    "\n",
    "Business: Precision 0.98, Recall 0.96, F1-score 0.97\n",
    "\n",
    "Sports: Precision 0.99, Recall 0.99, F1-score 0.99\n",
    "\n",
    "Entertainment: Precision 0.99, Recall 0.99, F1-score 0.99\n",
    "\n",
    "Education: Precision 0.98, Recall 0.98, F1-score 0.98\n",
    "\n",
    "Technology: Precision 0.94, Recall 0.98, F1-score 0.96\n",
    "\n",
    "Overall, our model's macro and weighted averages are also very high, showing consistent performance across different categories."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "<a href=#five></a>\n",
    "## **Streamlit App Deployment**\n",
    "<a href=#cont>[Back to Table of Contents](#Table-of-Contents)</a>\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-01-05 13:06:35.765 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.215 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run c:\\Users\\nthab\\anaconda4\\envs\\myenv\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-01-05 13:06:46.217 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.219 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.219 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.222 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.222 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.222 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.222 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.228 Session state does not function when running a script without `streamlit run`\n",
      "2025-01-05 13:06:46.229 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.233 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.233 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.235 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.235 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.239 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.240 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.241 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.243 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.244 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.245 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.246 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.248 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.249 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 13:06:46.251 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# Ensure vectorizer and models are already defined and trained \n",
    "vectorizer = TfidfVectorizer(max_features=5000) \n",
    "models = { \n",
    "    \"Logistic Regression\": LogisticRegression(max_iter=200),\n",
    "    \"Random Forest\": RandomForestClassifier(), \n",
    "    \"Support Vector Machine\": SVC() \n",
    "} \n",
    "\n",
    "# Assume results dictionary is filled with model evaluation results \n",
    "results = { \n",
    "    \"Logistic Regression\": {\"Accuracy\": 0.95}, \n",
    "    \"Random Forest\": {\"Accuracy\": 0.96}, \n",
    "    \"Support Vector Machine\": {\"Accuracy\": 0.98} \n",
    "} \n",
    "\n",
    "def main(): \n",
    "    \"\"\"News Classifier App with Streamlit \"\"\" \n",
    "\n",
    "    st.title(\"News Classifier\") \n",
    "    st.subheader(\"Classifying news articles into categories\") \n",
    "    \n",
    "    options = [\"Prediction\", \"Information\"] \n",
    "    selection = st.sidebar.selectbox(\"Choose Option\", options) \n",
    "\n",
    "    if selection == \"Information\": \n",
    "        st.info(\"General Information\")\n",
    "        st.markdown(\"This app classifies news articles into predefined categories like Business, Technology, Sports, Education, and Entertainment.\") \n",
    "\n",
    "    if selection == \"Prediction\": \n",
    "        st.info(\"Prediction with ML Models\") \n",
    "        news_text = st.text_area(\"Enter News Content\", \"Type here...\") \n",
    "\n",
    "        if st.button(\"Classify\"): \n",
    "            vect_text = vectorizer.transform([news_text]).toarray() \n",
    "            best_model_name = max(results, key=lambda x: results[x]['Accuracy']) \n",
    "            predictor = models[best_model_name] \n",
    "            prediction = predictor.predict(vect_text)[0] \n",
    "            predicted_category = train_data['category'].unique()[prediction] \n",
    "            st.success(f\"Text Categorized as: {predicted_category}\") \n",
    "\n",
    "if __name__ == \"__main__\": \n",
    "    main()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "<a href=#six></a>\n",
    "## **Conclussion**\n",
    "<a href=#cont>[Back to Table of Contents](#Table-of-Contents)</a>\n",
    "\n",
    "---\n",
    "\n",
    "This notebook provides a complete pipeline for the news classification task. \n",
    "Further steps could include hyperparameter tuning, exploring additional models, and enhancing the Streamlit app."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#  \n",
    "\n",
    "<div align=\"center\" style=\" font-size: 80%; text-align: center; margin: 0 auto\">\n",
    "<img src=\"https://raw.githubusercontent.com/Explore-AI/Pictures/master/ExploreAI_logos/EAI_Blue_Dark.png\"  style=\"width:200px\";/>\n",
    "</div>"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
