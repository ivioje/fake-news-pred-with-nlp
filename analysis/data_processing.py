import pandas as pd
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


"""Steps in the pipeline for natural language processing 
  1. Acquiring and loading the data
  2. Cleaning the dataset
  3. Removing extra symbols 
  4. Removing punctuations
  5. Removing the stopwords
  6. Tokenization
  7. Pickle file
 """

# 1. Acquring and loading the data
# load datasets
FAKE_NEWS = pd.read_csv('/home/caleb/mlProject/fake-news-pred-with-nlp/data/Fake.csv')
REAL_NEWS = pd.read_csv('/home/caleb/mlProject/fake-news-pred-with-nlp/data/True.csv')
SAVE_PATH = '/home/caleb/mlProject/fake-news-pred-with-nlp/analysis/dataFor_modelling.pkl'

# 2. Cleaning the dataset
# Since there's no label feature in the two sets of data, we can create labels to distinguish if the news
# is fake or not.
REAL_NEWS['label'] = 1
FAKE_NEWS['label'] = 0

#drop unnecessary columns
real = REAL_NEWS.drop(['date'], axis=1)
fake = FAKE_NEWS.drop(['date'], axis=1)

# concatenate the datasets
data = pd.concat([real, fake], axis=0)

# concat title, subject and text into a column called content
data['content'] = data['title'] + ' ' + data['subject'] + ' ' + data['text']

# Check if the labels are balanced
data.label.value_counts().plot(kind="bar", color=["salmon", "lightblue"])
# Remove urls
text = re.sub(r'^https?:\/\/.*[\r\n]*', '', str(data['content']), flags=re.MULTILINE)

# Remove user @ references and ‘#’ from text
text = re.sub(r'\@\w+|\#',"", text)

# remove punctuations
text.translate(str.maketrans("","", string.punctuation))

stop_words = set(stopwords.words('english'))

# word tokenization
tokens = word_tokenize(text)
words = [w for w in tokens if not w in stop_words]

# convert the tokens into meaningful numbers with TF-IDF.
# Use the TF-IDF method to extract and build the features for 
# our machine learning pipeline.
tf_vector = TfidfVectorizer(sublinear_tf=True)
tf_vector.fit(data['content'])

# 7. pickle file
data.to_pickle(SAVE_PATH)