#!/usr/bin/env python

# author: Georg Knabl

import argparse
import sys
import fileinput
from helpers import tokenizer
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def main(args):
    # load and prepare model
    pipeline = joblib.load(args.model_file.name)

    # load data
    if args.password_list_file.name == '<stdin>':
        content = args.password_list_file.readlines()
    else:
        with open(args.password_list_file.name) as f:
            content = f.readlines()

    # extract data
    rows = [x.strip().split('\t') for x in content]
    X_test = [x[0] for x in rows] # passwords
    y_test = [x[1] for x in rows] # labels

    # predict
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description = 'Uses labeled password list to evaluate an existing model.')
    parser.add_argument('password_list_file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help = 'Password list with labels')
    parser.add_argument('model_file', type = argparse.FileType('r'), help = 'Model file. e.g. "model.pkl"')
    args = parser.parse_args()

    try:
        main(args)
    except (KeyboardInterrupt, SystemExit):
        print('\rCancelled by user')
