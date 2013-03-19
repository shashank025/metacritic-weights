"""
What are metacritic's relative weights for different critics/publications?

XXX Things to try:

1. normalize weights
2. regularization
3. logistic regression
"""

import os
import sys
import optparse
import pickle
from operator import itemgetter
from random import random
from collections import defaultdict
from math import sqrt

import numpy as np
from numpy import linalg as LA
import scipy.optimize as sci

DEBUG = True
TECHNIQUES = frozenset(['SLSQP'])

def debug(msg):
    if DEBUG:
        print >> sys.stderr, msg
        sys.stderr.flush()

def warn(msg):
    print >> sys.stderr, "***warning*** " + str(msg)
    sys.stderr.flush()

def error(msg):
    print >> sys.stderr, "***error*** " + str(msg)
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

def err(actual, predicted):
    return (predicted - actual) * 1.0 /actual

def show_accuracy(url, actual, predicted, error_pct, ratings):
    print "[p: %3.2f, a: %3.2f, e: %3.2f] %s" % (predicted, actual, error_pct, url)

def extract_training_set(urls, training_pct):
    frac = training_pct / 100.
    return set([url for url in urls if random() < frac])

def rmse(errors):
    n = len(errors)
    return sqrt(sum(e * e for e in errors))/n if n else 0.

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
    errors = []
    for u in test_set:
        if u not in movie_ratings:
            continue
        actual_overall, ratings = movie_ratings[u]
        if not (actual_overall and ratings):
            continue
        predicted_overall = calc_overall_rating(predicted_weights, ratings)
        error = err(actual_overall, predicted_overall)
        show_accuracy(u, actual_overall, predicted_overall, error * 100, ratings)
        errors.append(actual_overall - predicted_overall)
    m = rmse(errors)
    print
    print "**********************"
    print "RMSE: %.6f" % (m,)
    print "**********************"

def build_A(multiple_ratings, critic_index):
    num_critics = len(critic_index)
    num_movies = len(multiple_ratings)
    A = np.zeros((num_movies, num_critics))
    total_ratings = 0
    for i, (overall, ratings) in enumerate(multiple_ratings):
        for name, rating_ij in ratings.items():
            j = critic_index[name]
            # normalized rating
            A[i, j] =  rating_ij - overall
            total_ratings += 1
    return A, total_ratings

def infer_weights(multiple_ratings, all_critics, tech):
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
    print A

    # min ||AX|| (optionally with X >= 0 and sum X = 1)

    # 1. objective function, that needs to be minimized
    def obj_f(theta):
        return LA.norm(A.dot(theta))

    # 2. initial, random solution: XXX what if this is not feasible?
    theta0 = np.array([random() for i in range(num_critics)])

    # 3. bounds
    bounds = [(0, 1) for i in range(num_critics)]

    # 4. constraints
    constraints = {'type': 'eq', 'fun': lambda theta: sum(theta) - 1}

    res = sci.minimize(obj_f, theta0, method=tech, bounds=bounds, constraints=constraints)
    result = dict((name, res.x[i]) for name, i in critic_index.items())
    if not res.success:
        msg = res.message
        error( ">>>> Current result:" )
        error( result )
        raise Exception("unable to converge to a solution using %(tech)s: %(msg)s" % locals())
    return result

def pretty_print_weights(all_critics, weights):
    for critic_name, weight in sorted(weights.items(), key=itemgetter(1), reverse=True):
        print "%2.2f %s (%d ratings)" % (weight * 100., critic_name, all_critics[critic_name])

def testitout(tech):
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
    predicted_weights = infer_weights(input, set(weights), tech)
    print ">>>> actual weights:", weights
    print ">>>> predicted_weights:", predicted_weights

if __name__ == '__main__':

    args_parser = optparse.OptionParser(usage="%prog --tech (COBYLA|SLSQP) { --test | --training-pct <int>} < ratings.pkl")
    args_parser.add_option('-s', '--tech',
                           dest="tech",
                           help="what solution technique to use? "
                           "currently supported values are: COBYLA and SLSQP; "
                           "see http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize")
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
        testitout('COBYLA')
        exit(0)
    if not options.train_with:
        args_parser.error("how much data to train with?")
    if options.tech not in TECHNIQUES:
        args_parser.error("tech should be one of: %s" % (', '.join(TECHNIQUES),))
    movie_ratings = pickle.loads(sys.stdin.read())
    train_and_test(movie_ratings, options.train_with, options.tech)
