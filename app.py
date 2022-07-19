from flask import Flask, render_template, request
import re
import pickle
import nltk
from sklearn.metrics import classification_report
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
ps = PorterStemmer()
# Load model and vectorizer
model = pickle.load(open('/home/caleb/mlProject/fake-news-pred-with-nlp/model/naiveBayes.pkl', 'rb'))
tfidfvect = pickle.load(open('/home/caleb/mlProject/fake-news-pred-with-nlp/model/tfidfvect.pkl', 'rb'))

# Build functionalities
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


def detect(text):
    new_text = re.sub('[^a-zA-Z]', ' ', text)
    new_text = new_text.lower()
    new_text = new_text.split()
    new_text = [ps.stem(word)
              for word in new_text if not word in stopwords.words('english')]
    new_text = ' '.join(new_text)
    new_text_vect = tfidfvect.transform([new_text]).toarray()
    detection = 'FAKE' if model.predict(new_text_vect) == 0 else 'REAL'
    return detection

def detect(title):
    new_title = re.sub('[^a-zA-Z]', ' ', title)
    new_title = new_title.lower()
    new_title = new_title.split()
    new_title = [ps.stem(word)
              for word in new_title if not word in stopwords.words('english')]
    new_title = ' '.join(new_title)
    new_title_vect = tfidfvect.transform([new_title]).toarray()
    detection = 'FAKE' if model.predict(new_title_vect) == 0 else 'REAL'
    return detection

def detect(subject):
    new_subject = re.sub('[^a-zA-Z]', ' ', subject)
    new_subject = new_subject.lower()
    new_subject = new_subject.split()
    new_subject = [ps.stem(word)
              for word in new_subject if not word in stopwords.words('english')]
    new_subject = ' '.join(new_subject)
    new_subject_vect = tfidfvect.transform([new_subject]).toarray()
    detection = 'FAKE' if model.predict(new_subject_vect) == 0 else 'REAL'
    
    return detection

@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    title = request.form['title']
    subject = request.form['subject']
    result = detect(text)
    result = detect(title)
    result = detect(subject)
    #score = model.predict(text) * 100
    return render_template('index.html', text=text, title=title, subject=subject, result=result)


if __name__ == "__main__":
    app.run()
