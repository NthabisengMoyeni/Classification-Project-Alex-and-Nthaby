"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: ExploreAI Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# Vectorizer
#news_vectorizer = open("streamlit/tfidfvect.pkl","rb")
#test_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
#raw = pd.read_csv("streamlit/train.csv")

# The main function where we will build the actual app
def main():
    """News Classifier App with Streamlit """

    # Load model and vectorizer
    try:
        predictor = joblib.load("best_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
    except FileNotFoundError:
        st.error("Model files not found. Please train the model and save it as 'best_model.pkl' and 'vectorizer.pkl'.")
        return

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("News Classifier")
    st.subheader("Classifying news articles into categories")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        st.markdown("This app classifies news articles into predefined categories like Business, Technology, Sports, Education, and Entertainment.")

    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        news_text = st.text_area("Enter News Content", "Type here...")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = vectorizer.transform([news_text])
            prediction = predictor.predict(vect_text)[0]
            predicted_category = train_data['Category'].unique()[prediction]
            st.success(f"Text Categorized as: {predicted_category}")

if __name__ == "__main__":
    main()
