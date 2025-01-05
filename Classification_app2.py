# Create a Streamlit app for deploying the best-performing model

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