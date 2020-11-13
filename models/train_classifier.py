import sys
import pandas as pd
import sqlite3
import re
import nltk
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    
    '''
    Function that take a database and transform it to dataframe
    
    Input: database_filepath: The path of sql database
    Output: X: Messages, Y: Categories, category_names: Labels for categories
    '''

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterMessages', engine)
    X = df['message']
    Y = df.drop(['id','message', 'original', 'genre'], axis=1)
    category_names = list(df.columns[4:])
    
    return X, Y, category_names


def tokenize(text):
    '''
    Function that tokenize and lemmatize text
    
    Input: text: original message text
    
    Output: clean_tokens: Text that has been tokenized
    '''
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    '''
    untion that build a machine learning pipeline
    
    Output: Results of GridSearchCV
    
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    
    parameters = {
        
        # 'vect__max_df': (0.5, 0.75, 1.0),
        # 'clf__estimator__n_neighbors': (5, 7, 10),
        # 'clf__estimator__leaf_size': (20, 30, 40),
        'tfidf__use_idf': (True, False),
        # 'vect__lowercase': (True, False),
        
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function that evaluate the performance of the machine learning model
    
    Input: model: ML Model, X_test: Messages test data, Y_test: Categories (labels) test data, category_names: Labels for categories
    
    Output: Print the accuracy, precision, and recall of the model
    
    '''
   

    y_pred = model.predict(X_test)

    print(classification_report(Y_test, y_pred, target_names=category_names))

    

def save_model(model, model_filepath):
    
    '''
    Function that save the machine learning model as a pickle file 
    
    Input: model: ML Model to be saved, model_filepath: The path of the output (pickle file)
    
    Output: A pickle file of saved ML model
    '''

    pickle.dump(model, open(model_filepath, "wb"))


                

def main():
                
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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