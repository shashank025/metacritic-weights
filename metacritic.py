"""
What are metacritic's relative weights for different critics/publications?
"""

import os
import sys
import optparse
import pickle
from operator import itemgetter
from random import random
from collections import defaultdict

import numpy as np
import scipy.optimize as sci

DEBUG = True
TECHNIQUES = frozenset(['nnls', 'lstsq'])

def debug(msg):
    if debug:
        print >> sys.stderr, msg
        sys.stderr.flush()

def warn(msg):
    print >> sys.stderr, "***warning*** " + msg
    sys.stderr.flush()

def error(msg):
    print >> sys.stderr, "***error*** " + msg
    sys.stderr.flush()

def get_critics(many_ratings):
    """returns a dictionary: critic name -> number of movies rated"""

    critics = defaultdict(int)
    for _, ratings in many_ratings:
        for c in ratings:
            critics[c] += 1
    return critics

def calc_overall_rating(weights, critic_ratings):
    unnormalized = sum(weights[critic] * score for critic, score in critic_ratings.items())
    total_weight = sum(weights[critic] for critic in critic_ratings)
    return unnormalized/total_weight

def show_accuracy(url, actual, predicted, ratings):
    error_pct = (predicted - actual) * 100./actual
    print "[p: %3.2f, a: %3.2f, e: %3.2f] %s" % (predicted, actual, error_pct, url)

def extract_training_set(urls, training_pct):
    frac = training_pct / 100.
    return set([url for url in urls if random() < frac])

def train_and_test(movie_ratings, training_pct, tech):
    """
    """
    training_set = extract_training_set(movie_ratings, training_pct)
    all_critics = get_critics(movie_ratings.values())
    predicted_weights = infer_weights([movie_ratings[u]
                                       for u in training_set
                                       if u in movie_ratings],
                                      all_critics,
                                      tech)
    pretty_print_weights(all_critics, predicted_weights)
    test_set = set(movie_ratings).difference(training_set)
    for u in test_set:
        if u not in movie_ratings:
            continue
        actual_overall, ratings = movie_ratings[u]
        if not (actual_overall and ratings):
            continue
        predicted_overall = calc_overall_rating(predicted_weights, ratings)
        show_accuracy(u, actual_overall, predicted_overall, ratings)

def build_A(multiple_ratings, critic_index):
    num_critics = len(critic_index)
    num_movies = len(multiple_ratings)
    A = np.zeros((num_movies + 1, num_critics))
    total_ratings = 0
    for i, (score, ratings) in enumerate(multiple_ratings):
        for name, rating in ratings.items():
            # normalization
            j, val = critic_index[name], rating - score
            A[i, j] = val
            total_ratings += 1
    # critic weights should add up to 1
    for j in range(num_critics):
        A[num_movies, j] = 1
    return A, total_ratings

def build_B(num_movies):
    B = np.zeros(num_movies + 1)
    B[num_movies] = 1
    return B

def infer_weights(multiple_ratings, all_critics, tech='lstsq'):
    """Return a dictionary mapping critics to relative weights.

    all_critics: a set of critic names;

    multiple_ratings: is a list of data, one per movie, of the form:

        (overall_score, critic_scores)

    where critic_scores is itself a dictionary of the form:

        critic_name -> critic rating for the movie.

    Uses linear regression to find the best fitting weights;
    notice that the linear system might be overdetermined if:

        num_movies > num_critics.
    """

    assert tech in TECHNIQUES

    num_critics = len(all_critics)
    num_movies = len(multiple_ratings)
    # critic name -> critic number
    critic_index = dict((c, i) for i, c in enumerate(all_critics))
    A, total_ratings = build_A(multiple_ratings, critic_index)
    print ">>>> %s with %d ratings of %d movies by %d critics" % (tech,
                                                                  total_ratings,
                                                                  num_movies,
                                                                  num_critics)
    B = build_B(num_movies)
    # min ||AX - B|| (optionally, with X >= 0)
    print A
    if tech == 'nnls':
        X, _ = sci.nnls(A, B)
    else:
        X = np.linalg.lstsq(A, B)[0]
    return dict((name, X[i]) for name, i in critic_index.items())

def pretty_print_weights(all_critics, weights):
    for critic_name, weight in sorted(weights.items(), key=itemgetter(1), reverse=True):
        print "%2.2f %s (%d ratings)" % (weight * 100., critic_name, all_critics[critic_name])

def testitout():
    weights = {
        'w1': 0.4,
        'w2': 0.15,
        'w3': 0.27,
        'w4': 0.18 }

    mcr = [{'w1': 45, 'w3': 79},
           {'w1': 72, 'w3': 64, 'w4': 59},
           {'w2': 89, 'w4': 81},
           {'w2': 95, 'w3': 81, 'w4': 77},
           {'w1': 59, 'w3': 79, 'w4': 91}]

    input = [(calc_overall_rating(weights, unr), unr) for unr in mcr]
    predicted_weights = infer_weights(input, set(weights))
    print ">>>> actual weights:", weights
    print ">>>> predicted_weights:", predicted_weights

if __name__ == '__main__':

    args_parser = optparse.OptionParser(usage="%prog --tech (nnls|lstsq) { --test | --training-pct <int>} < ratings.pkl")
    args_parser.add_option('-s', '--tech',
                           dest="tech",
                           help="what solution technique to use? "
                           "currently supported values are: "
                           "'lstsq': standard least squared error; "
                           "'nnls': like lstsq, but restrict solutions to non-negative space")
    args_parser.add_option('-r', '--test',
                           action='store_true',
                           dest="test_only",
                           help="just test, dont look at std input")
    args_parser.add_option('-t', '--training-pct',
                           dest="train_with",
                           type='int',
                           help="what percentage of data should be used to train?")
    options, args = args_parser.parse_args()
    if options.test_only:
        testitout()
        exit(0)
    if not options.train_with:
        args_parser.error("how much data to train with?")
    if options.tech not in TECHNIQUES:
        args_parser.error("tech should be one of: %s" % (', '.join(TECHNIQUES),))
    movie_ratings = pickle.loads(sys.stdin.read())
    train_and_test(movie_ratings, options.train_with, options.tech)
