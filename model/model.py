# import necessary modules
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# Read file
READ_PATH = 'analysis/dataFor_modelling.pkl'

# save the pickled file to a variable called heartData
data = pd.read_pickle(READ_PATH)

# Feature extration & TF-IDF
# convert the tokens into meaningful numbers with TF-IDF.
# Use the TF-IDF method to extract and build the features for
# the machine learning pipeline.
tf_vector = TfidfVectorizer(sublinear_tf=True)
tf_vector.fit(data['text'])

# separate the text and label features
X = tf_vector.transform(data['text'].ravel())
y = np.array(data['label'].ravel())

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