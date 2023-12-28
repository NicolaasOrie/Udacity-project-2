import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, auc

## HIER WAS JE GEBLEVEN ! CODE IN DEF LOAD_DATA IS GEKOPIEERD VAN DE OEFEN SPACE EN MOET NU KLOPPEND GEMAAKT WORDEN

def load_data(database_filepath):
    """ input : filepath to database
        output : dataframe containing all data"""
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM category_messages' , engine)

    df = df[df['related'] != '2']
    
    X = df['message'] 
    y = df.drop(['message', 'original','id', 'genre' ], axis=1)

    if len(X)!=len(y):
        raise ValueError('input and output dataframes have')
    
    return X, y

def tokenize(text):
    """ input : sentence (string)
        output : tokenized sentence (list)"""
    stemmer = PorterStemmer()
    
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Stem the tokens
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return stemmed_tokens



def build_model(X_train, y_train):
    """ input : X_train, y_train
        output : ML model (fitted SK learn pipeline)"""
    # Create a text vectorizer for 'message' column
    text_vectorizer = TfidfVectorizer(max_features=1000)
    # Create a standard scaler for 'genre' column
    genre_scaler = StandardScaler(with_mean=False)
    # Create a multi-output classifier (Random Forest, for example)
    classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, max_depth=50, random_state=42, n_jobs=5))
    #create MeanAmbeddingVectorizer
    from gensim.models.word2vec import Word2Vec
    model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)
    model.wv.index2word
    w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
    class MeanEmbeddingVectorizer(object):
        def __init__(self, word2vec):
            self.word2vec = word2vec
            self.dim = len(next(iter(word2vec.values())))
    def fit(self, X, y):
        return self
    def transform(self, X):
        return np.array([np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                         or [np.zeros(self.dim)], axis=0)
                         for words in X
                         ])
 
    # Create a pipeline that processes 'message'column and predicts the output columns
    pipeline = Pipeline([
    (("word2vec vectorizer", MeanEmbeddingVectorizer(w2v))),
    ('genre_scaler', genre_scaler),
    ('classifier', classifier)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, Y_test):
    """ input : mode, X-test, Y_test
        output : ML model (fitted SK learn pipeline)"""
    y_pred = model.predict(X_test)
    roc_auc_scores = []
    for column in Y_test.columns:
        print(column)
    # Evaluate the model's performance
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(Y_test[column].astype(int), y_pred[:, Y_test.columns.get_loc(column)].astype(int))
    print(f'Accuracy: {accuracy}')
    try:
        roc_score = roc_auc_score(Y_test[column].astype(int), y_pred[:, Y_test.columns.get_loc(column)].astype(int) )
        print(f'roc_auc_score : {roc_score}')
        roc_auc_scores.append(roc_score)
    except ValueError:
        pass    
    print(f' mean roc score {np.mean(roc_auc_scores)}')
    pass


def save_model(model, model_filepath):
    from sklearn.externals import joblib
    joblib.dump(model, model_filepath)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        print(database_filepath)
        X, Y = load_data(database_filepath)
        X = X.apply(tokenize)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train = [' '.join(inner_list) for inner_list in X_train]
        X_test = [' '.join(inner_list) for inner_list in X_test]
        print('Building model...')
        model = build_model(X_train, Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()