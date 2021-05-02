import os
from sklearn.metrics import f1_score


ANSWER_PATH = 'answer.csv'
SUBMISSION_PATH = os.environ['USERSUBMISSION']


def score_submission():
    DIR = os.path.dirname(__file__)
    answer_path = os.path.join(DIR, ANSWER_PATH)

    with open(answer_path, 'r') as f:
        answers = [int(v) for v in f.read().split(',')]

    with open(SUBMISSION_PATH, 'r') as f:
        submissions = [int(v) for v in f.read().split(',')]

    score = f1_score(answers, submissions, average='macro')

    return score

if __name__ == '__main__':
    print(score_submission())