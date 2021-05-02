from scipy.stats import rankdata
import numpy as np
import os

ANSWER_PATH = 'wordsim.csv'
SUBMISSION_PATH = os.environ['USERSUBMISSION']


def evaluate():
    DIR = os.path.dirname(__file__)
    answer_path = os.path.join(DIR, ANSWER_PATH)

    word_pairs = []
    human_scores = []
    with open (answer_path, "r") as f:
        for line in f:
            line = line.strip().split(",")
            human_scores.append(float(line[2]))
    human_rank = rankdata(human_scores)
    
    N = len(word_pairs)
    pred_scores = []
    with open (SUBMISSION_PATH, "r") as f:
        for line in f:
            line = line.strip().split(",")
            pred_scores.append(float(line[2]))
    pred_rank = rankdata(pred_scores)
    rho = 1 - 6 * np.sum((human_rank - pred_rank)**2) / (N**3 - N)
    return rho


if __name__ == '__main__':
    print(evaluate())
