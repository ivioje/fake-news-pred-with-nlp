import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB


# load datasets
FAKE_NEWS = pd.read_csv('/home/caleb/mlProject/fake-news-pred-with-nlp/data/Fake.csv')
REAL_NEWS = pd.read_csv('/home/caleb/mlProject/fake-news-pred-with-nlp/data/True.csv')

# Since there are no labels in the two sets of data, we can create labels to distinguish if the news
# is fake or not.
real = REAL_NEWS['label'] = 1
fake = FAKE_NEWS['label'] = 0

#drop unnecessary columns
real = REAL_NEWS.drop(['date'], axis=1)
fake = FAKE_NEWS.drop(['date'], axis=1)

# concatenate the datasets
data = pd.concat([real, fake], axis=0)

# Remove urls
text = re.sub(r'^https?:\/\/.*[\r\n]*', '', str(data['text']), flags=re.MULTILINE)

# Remove user @ references and ‘#’ from text
text = re.sub(r'\@\w+|\#',"", text)

# Remove urls
title = re.sub(r'^https?:\/\/.*[\r\n]*', '', str(data['title']), flags=re.MULTILINE)

# Remove user @ references and ‘#’ from text
title = re.sub(r'\@\w+|\#',"", title)

# Remove urls
subject = re.sub(r'^https?:\/\/.*[\r\n]*', '', str(data['subject']), flags=re.MULTILINE)

# Remove user @ references and ‘#’ from text
subject = re.sub(r'\@\w+|\#',"", subject)

dataset = text + title + subject

# remove punctuations
dataset = dataset.translate(str.maketrans("","", string.punctuation))

stop_words = set(stopwords.words('english'))

# word tokenization
tokens = word_tokenize(dataset)
words = [w for w in tokens if not w in stop_words]

# convert the tokens into meaningful numbers with TF-IDF.
# Use the TF-IDF method to extract and build the features for 
# our machine learning pipeline.
tf_vector = TfidfVectorizer(sublinear_tf=True)
t1 = tf_vector.fit(data['text'])
t2 = tf_vector.fit(data['title'])
t3 = tf_vector.fit(data['subject'])

#tf = t1 + t2 + t3

X_text = tf_vector.transform((t1, t2, t3).ravel())
# t2 = tf_vector.transform(data['title'].ravel())
# t3 = tf_vector.transform(data['subject'].ravel())

# separate the label from the other features and transform it

#X_text = t1 + t2 + t3
y_values = np.array(data['label'].ravel())

# encode ...
le = preprocessing.LabelEncoder()
le.fit(y_values)
le.transform(y_values)

# split data with sklearn train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_text, y_values, 
test_size=0.2, random_state=120)

# use MNB to train the model
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# model evaluation (MNB)
print('Accuracy score: ', round(accuracy_score(y_test, y_pred) *100), '%')
print('--------------------------------------- \n')
print(classification_report(y_test, y_pred))
print('--------------------------------------- \n')
print('Confusion matrix: \n',confusion_matrix(y_test, y_pred))

# Feature extration & TF-IDF
# convert the tokens into meaningful numbers with TF-IDF.
# Use the TF-IDF method to extract and build the features for
# the machine learning pipeline.
tf_vector = TfidfVectorizer(sublinear_tf=True)

y_data = data['label']
x_data = data.drop(['label'], axis=1)

t1 = tf_vector.transform(data['text'].ravel())
t2 = tf_vector.transform(data['title'].ravel())
t3 = tf_vector.transform(data['subject'].ravel())

tf_vector.fit(x_data)

# separate the text and label features
X = tf_vector.transform(x_data.ravel())
y = np.array(y_data.ravel())

# encode the labels
le = preprocessing.LabelEncoder()
le.fit(y)
le.transform(y)

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=120)

# model

model = MultinomialNB()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)


# model evaluation
print('Accuracy score: ', round(accuracy_score(y_test, y_predict) * 100), '%')
print('--------------------------------------- \n')
print(classification_report(y_test, y_predict))
print('--------------------------------------- \n')
print('Confusion matrix: \n', confusion_matrix(y_test, y_predict))

# saving the final model and vectoriser
pickle.dump(model, open('model/naiveBayes.pkl', 'wb'))

pickle.dump(tf_vector, open('model/tfidfvect.pkl', 'wb'))