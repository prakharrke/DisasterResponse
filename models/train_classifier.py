import sys, pickle
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from nltk.corpus import stopwords
import time
def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('MessageCategorization', engine)
    X = df.message
    df_categories = df.copy()
    df_categories = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    y = df_categories
    category_names = df_categories.columns
    return X, y, category_names

def tokenize(text):
    # Convert the text to lower case
    text = text.lower()
    #Remove punctuations
    text_normalized = re.sub(r"[^a-zA-Z0-9]", " ", text)
    #Tokenizing the normalized text
    tokens = word_tokenize(text_normalized)
    #removal of stop words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words('english')]
    return clean_tokens   


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters =  {
              'clf__estimator__n_estimators': [50, 100]
             
              } 
    
    #Grid search was taking very long to train, due to limitations on the resources. Hence it has been commented out.
    #cv = GridSearchCV(pipeline, param_grid=parameters)
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    #print(classification_report(y_pred, Y_test.values, target_names=category_names))
    # print raw accuracy score 
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))    


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        start_time = time.time()
        model.fit(X_train, Y_train)
        print('time taken to train model : {}'.format(time.time() - start_time))
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