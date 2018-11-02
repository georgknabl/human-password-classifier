#!/usr/bin/env python

# author: Georg Knabl

import argparse
import sys
import fileinput
from helpers import tokenizer
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

def main(args):
    # load and prepare model
    pipeline = joblib.load(args.model_file.name)
    
    # load data
    if args.password_list_file.name == '<stdin>':
        content = args.password_list_file.readlines()
    else:
        with open(args.password_list_file.name) as f:
            content = f.readlines()
    # strip newline
    passwords = [x.strip() for x in content]

    # predict
    human_label_index = pipeline.classes_.tolist().index('h')
    human_likelinesses = pipeline.predict_proba(passwords)[:,human_label_index]
    for i, password in enumerate(passwords):
        print(password + '\t' + '{0:.10f}'.format(human_likelinesses[i]))

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description = 'Adds human-likeliness to password list.')
    parser.add_argument('password_list_file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help = 'Password list to filter')
    parser.add_argument('model_file', type = argparse.FileType('r'), help = 'Model file. e.g. "model.pkl"')
    args = parser.parse_args()
    
    try:
        main(args)
    except (KeyboardInterrupt, SystemExit):
        print('\rCancelled by user')
