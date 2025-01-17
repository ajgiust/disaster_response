import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
import pickle

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''Loads database table which contains messages along with message category  
    
    Arguments:
        database_filepath {str} -- path to db file
    
    Returns:
        X -- messages array
        Y -- matrix of message category labels
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_filepath, engine)
    X = df['message'].values
    category_names = df.drop(['id', 'message', 'original', 'genre'], axis=1).columns
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values

    return X, Y, category_names


def tokenize(text):
    '''Tokenizes text from each message
    
    Arguments:
        text {str} -- message text
    
    Returns:
        clean_tokens {str} -- list of str's
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    '''Builds a multi-output classifier with Random Forest from a ML pipeline.
    
    Returns:
        model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1, 
                                                            random_state=42)))
    ])

    parameters = {
        "clf__estimator__n_estimators": [200, 300, 400],
        "clf__estimator__min_samples_split": [10, 20, 30],
    }

    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluates performance of model for each category using precision, 
    recall, and f1 score
    
    Arguments:
        model {var} -- model variable
        X_test {array} -- X_test variable
        Y_test {array} -- Y_test variable
        category_names {array} -- array of category names
    '''
    best_clf = model.best_estimator_
    best_predictions = best_clf.predict(X_test)

    for i in range(len(best_predictions[0,:])):
        print('-' * 10)
        print(category_names[i])
        print('-' * 10)
        print(classification_report(Y_test[:,i], best_predictions[:,i]))
        print('\n')


def save_model(model, model_filepath):
    '''Export model to pickle fil
    
    Arguments:
        model {var} -- model variable
        model_filepath {str} -- file path to model
    '''
    pickle_out = open(model_filepath,"wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()


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