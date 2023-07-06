import streamlit as st
import pickle
import nltk
from nltk.stem import PorterStemmer
import string
from nltk.corpus import stopwords

vectorizer = pickle.load(open('cvVectorizer (2).pkl', 'rb'))
model = pickle.load(open('mnbModel (1).pkl', 'rb'))


def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)

    filtered_words = [word for word in words if word.isalnum(
    ) and word not in stopwords.words('english') and word not in string.punctuation]

    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    return " ".join(stemmed_words)


st.title('Email classification')

email = st.text_area('Enter email message')

if st.button('Detect'):

    # 1.preprocess
    transformed_email = transform_text(email)
    # 2 vectorize
    vectorized_email = vectorizer.transform([transformed_email])
    # 3. predict
    prediction = model.predict_proba(vectorized_email)[0][1] * 100
    # result = model.predict(vectorized_email)

    # 4. Display
    if prediction <= 50:
        st.markdown(
            f"#### This email is a Spam email.")
    elif prediction > 50:
        st.markdown(f"#### This email is {prediction: .2f}% Not Spam.")

    st.markdown("------------------------------------------")
    st.markdown(
        f"""
        ##### CONFIDENCE SCORE: {prediction: .2f}%
        """
    )

    st.markdown("------------------------------------------")

    st.markdown("##### YOUR INPUT:")
    st.markdown(f"```{email}````")
