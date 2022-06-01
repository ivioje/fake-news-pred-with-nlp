import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
FAKE_NEWS = pd.read_csv('data/Fake.csv')
REAL_NEWS = pd.read_csv('data/True.csv')
SAVE_PATH = 'analysis/dataFor_modelling.pkl'

# 2. Cleaning the dataset
# Since there's no label feature in the two sets of data, we can create labels to distinguish if the news
# is fake or not.
REAL_NEWS['label'] = 1
FAKE_NEWS['label'] = 0

#drop unnecessary columns
real = REAL_NEWS.drop(['title', 'subject', 'date'], axis=1)
fake = FAKE_NEWS.drop(['title', 'subject', 'date'], axis=1)

# combine the datasets
data = pd.concat([real, fake], axis=0)

# Check if the labels are balanced
data.label.value_counts().plot(kind="bar", color=["salmon", "lightblue"])

# 3. Removing extra symbols
# Remove urls
text = re.sub(r'^https?:\/\/.*[\r\n]*', '', str(data.text), flags=re.MULTILINE)
# Remove user @ references and '#' from text
text = re.sub(r'\@\w+|\#',"", text)

# 4. Removing punctuations
text = text.translate(str.maketrans("","", string.punctuation))

# 5-6. Removing stop words & tokenization
stop_words = set(stopwords.words('english'))
# word tokenization
tokens = word_tokenize(text)
words = [w for w in tokens if not w in stop_words]

# 7. pickle file
data.to_pickle(SAVE_PATH)