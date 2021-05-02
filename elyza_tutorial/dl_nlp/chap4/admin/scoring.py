import sys
import os
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

ANSWER_PATH = 'test.ja'
SUBMISSION_PATH = os.environ['USERSUBMISSION']


def compute_bleu(refs, preds):
    return corpus_bleu(refs, preds)


def score_submission():
    DIR = os.path.dirname(__file__)
    answer_path = os.path.join(DIR, ANSWER_PATH)

    with open(answer_path, 'r') as f:
        answers = [[line.rstrip().split()] for line in f]

    with open(SUBMISSION_PATH, 'r') as f:
        submissions = [line.rstrip().split() for line in f]

    score = compute_bleu(answers, submissions)

    return score

if __name__ == '__main__':
    print(score_submission())
