#!/usr/bin/env python

# author: Georg Knabl

from helpers import tokenizer

import argparse
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

def main(args):
    # load and prepare model
    # models = joblib.load(args.model_file.name)
    # classifier = models['classifier']
    # vectorizer = models['vectorizer']
    pipeline = joblib.load(args.model_file.name)

    # prepare data
    X_predict = [args.password]

    # predict
    # X_predict = vectorizer.transform(X_predict)
    # y_predict = classifier.predict(X_predict)
    y_predict = pipeline.predict(X_predict)
    print(y_predict[0])

    # if args.output_mode == 'l' or args.output_mode == 'v':
    #     print(y_predict[0])
    # if args.output_mode == 'p' or args.output_mode == 'v':
    #     human_label_index = pipeline.classes_.tolist().index('h')
    #     human_likelinesses = pipeline.predict_proba([args.password])[:,human_label_index]
    #     print('{0:.10f}'.format(human_likelinesses[0]))

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description = 'Classifies a password based on a model to "h" (human-generated) or "m" (machine-generated)')
    parser.add_argument('model_file', type = argparse.FileType('r'), help = 'Model file. e.g. "model.pkl"')
    parser.add_argument('password', default = 'model.pkl', help = 'Password to test')
    # parser.add_argument('--output_mode', default = 'l', help = 'Output modes: l (label only), p (human likeliness only), v (both)')
    args = parser.parse_args()

    main(args)
