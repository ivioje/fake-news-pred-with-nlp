from flask import Flask, render_template, request
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
ps = PorterStemmer()
# Load model and vectorizer
model = pickle.load(open('model/naiveBayes.pkl', 'rb'))
tfidfvect = pickle.load(open('model/tfidfvect.pkl', 'rb'))

# Build functionalities
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


def predict(text):
    new_text = re.sub('[^a-zA-Z]', ' ', text)
    new_text = new_text.lower()
    new_text = new_text.split()
    new_text = [ps.stem(word)
              for word in new_text if not word in stopwords.words('english')]
    new_text = ' '.join(new_text)
    new_text_vect = tfidfvect.transform([new_text]).toarray()
    prediction = 'FAKE' if model.predict(new_text_vect) == 0 else 'REAL'
    return prediction


@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    prediction = predict(text)
    return render_template('index.html', text=text, result=prediction)


if __name__ == "__main__":
    app.run()
