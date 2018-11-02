#!/usr/bin/env python

# author: Georg Knabl

from helpers import tokenizer

import pandas as pd
import argparse
from pprint import pprint
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def main(args):
    # hyperparameters
    test_size = 0.2
    random_state = args.random_state

    # load data
    dataset = pd.read_csv(args.source_file, delimiter = '\t', quoting = 3, header = None)

    # load static tests data
    if args.static_tests_file != None:
        dataset_static_tests = pd.read_csv(args.static_tests_file, delimiter = '\t', quoting = 3, header = None)
        static_tests_passwords = dataset_static_tests.iloc[:, 0].values
        static_tests_labels = dataset_static_tests.iloc[:, 1].values


    # prepare data
    X = dataset.iloc[:, 0].values
    y = dataset.iloc[:, 1].values

    # setting test configuration pipelines
    count_vectorizer = CountVectorizer(tokenizer = tokenizer, lowercase = False)
    tfidf_vectorizer = TfidfVectorizer(tokenizer = tokenizer, lowercase = False)
    pipelines = [
        Pipeline([('vect', count_vectorizer),('clf', LogisticRegression())]),
        Pipeline([('vect', tfidf_vectorizer),('clf', LogisticRegression())]),
        Pipeline([('vect', count_vectorizer),('clf', MultinomialNB())]),
        Pipeline([('vect', tfidf_vectorizer),('clf', MultinomialNB())]),
        Pipeline([('vect', count_vectorizer),('clf', LinearSVC())]),
        Pipeline([('vect', tfidf_vectorizer),('clf', LinearSVC())]),
        Pipeline([('vect', count_vectorizer),('clf', RandomForestClassifier())]),
        Pipeline([('vect', tfidf_vectorizer),('clf', RandomForestClassifier())]),
        Pipeline([('vect', count_vectorizer),('tfidf', TfidfTransformer()),('clf', LogisticRegression())])
    ]

    # find best model
    if args.test_pipelines:
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

        # on error: comment out:
        X_train = X_train.astype('U')
        y_train = y_train.astype('U')
        X_test = X_test.astype('U')
        y_test = y_test.astype('U')

        # parameters to test for pipeline, source: http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html
        parameters = {
            'vect__max_df': (0.5, 0.75, 1.0),
            'vect__max_features': (None, 5000, 10000, 50000),
            'vect__ngram_range': ((1, 1), (1, 2)), # unigrams or bigrams
            #'tfidf__use_idf': (True, False),
            #'tfidf__norm': ('l1', 'l2'),
            #'clf__alpha': (0.00001, 0.000001),
            #'clf__penalty': ('l2', 'elasticnet'),
            #'clf__n_iter': (10, 50, 80),
        }


        # evaluate pipelines and parameters

        # detailed single pipeline analysis
        if args.detailed_test_for_pipeline != None and args.detailed_test_for_pipeline >= 0 and args.detailed_test_for_pipeline < len(pipelines):
            pipeline = pipelines[args.detailed_test_for_pipeline]


            # find the best parameters for both the feature extraction and the classifier
            grid_search = GridSearchCV(pipeline, parameters, n_jobs = -1, verbose = 1)

            print("Performing grid search...")
            print("parameters:")
            pprint(parameters)
            grid_search.fit(X_train, y_train)
            print()
            print("Best score: %0.3f" % grid_search.best_score_)
            print("Best parameters set:")
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(parameters.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))
        # general pipeline analysis
        else:
            for i, pipeline in enumerate(pipelines):
                print('Testing pipeline {0}:'.format(i))
                print('Score without grid search:')
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                print(classification_report(y_test, y_pred))

    # train on selected model
    else:
#        # prepare vectorizer
#        vectorizer = TfidfVectorizer(tokenizer = tokenizer, lowercase = False)
#        X = vectorizer.fit_transform(X)
#
#        # split in training/test set
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
#
#        # train
#        classifier = LogisticRegression()
#        classifier.fit(X_train, y_train)
#
#        # predicting the test set results
#        y_pred = classifier.predict(X_test)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
        pipeline = pipelines[args.training_pipeline]

        # on error: comment out:
        X_train = X_train.astype('U')
        y_train = y_train.astype('U')
        X_test = X_test.astype('U')
        y_test = y_test.astype('U')

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # generating the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        # stat outputs
        print('report:')
        print(classification_report(y_test, y_pred, target_names = ['h', 'm']))
        print("\n" + 'confusion matrix:')
        pprint(cm)
        if args.static_tests_file != None:
            print("\n" + 'static tests:')
            #X_pred_static = vectorizer.transform(static_tests_passwords)
            #y_pred_static = classifier.predict(X_pred_static)
            y_pred_static = pipeline.predict(static_tests_passwords)
            for i, password in enumerate(static_tests_passwords):
                correct_label = 'correct' if y_pred_static[i] == static_tests_labels[i] else 'incorrect'
                print("password: {0}\tlabel: {1}\t{2}".format(static_tests_passwords[i], y_pred_static[i], correct_label))

        # save model
        if args.output_prefix != '':
            #export_obj = {'classifier': classifier, 'vectorizer': vectorizer}
            export_obj = pipeline
            joblib.dump(export_obj, args.output_prefix + '.pkl')

            # save mlmodel
            if args.export_coreml:
                # import coremltools because of upward compatibility issues (requires Python 2.7)
                import coremltools

                # extract components from pipeline
                classifier = pipeline.get_params()['clf']
                vectorizer = pipeline.get_params()['vect']

                # export mlmodel
                coreml_model = coremltools.converters.sklearn.convert(classifier, 'password', 'label')
                coreml_model.save(args.output_prefix + '.mlmodel')

                # export feature columns
                features_export = open(args.output_prefix + '-features.txt', 'w')
                for feature in vectorizer.get_feature_names():
                    features_export.write(feature.encode('utf-8') + '\n')
                features_export.close()

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description = 'Trains model on password list.')
    parser.add_argument('source_file', type = argparse.FileType('r'), help = 'Source file containing passwords and labels. TSV format: col0: password, col1: "h" or "m".')
    parser.add_argument('--output_prefix', default = 'model', help = 'file prefix to save model files. e.g. "model"')
    parser.add_argument('--static_tests_file', type = argparse.FileType('r'), help = 'file of TSV-passwords/labels to test at the end')
    parser.add_argument('--test_pipelines', action = 'store_true', help = 'Flag. Test pre-specified models on this dataset')
    parser.add_argument('--export_coreml', action = 'store_true', help = 'Flag. Export mlmodel for iOS11+. Requires Python 2.7.')
    parser.add_argument('--detailed_test_for_pipeline', type = int, help = 'Test single pipeline using grid_search. pipeline ID (zero-based)')
    parser.add_argument('--training_pipeline', type = int, default = 0, help = 'Pipeline ID used for training (zero-based)')
    parser.add_argument('--random_state', type = int, default = 0, help = 'PRNG seed as integer. Set for reproducable outcomes on same datasets.')
    args = parser.parse_args()

    try:
        main(args)
    except (KeyboardInterrupt, SystemExit):
        print("\r" + 'Cancelled by user')
