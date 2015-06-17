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
TECHNIQUES = frozenset(['SLSQP',])
SIGNIFICANCE_RATING_COUNT = 5         # critic must rate at least these many movies to be considered.
OOB_PENALTY = 10                      # how much to penalize the objective fn when a theta value is out of bounds.
NIH_PENALTY = 10                      # how much to penalize the objective fn when theta values dont add up to 1.

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

def calc_overall_rating(weights, critic_ratings):
    unnormalized = sum(weights[critic] * score for critic, score in critic_ratings.items())
    total_weight = sum(weights[critic] for critic in critic_ratings)
    return unnormalized/total_weight

def get_critics(many_ratings):
    """returns a dictionary: critic name -> number of movies rated"""

    critics = defaultdict(int)
    for _, ratings in many_ratings:
        for c in ratings:
            critics[c] += 1
    return critics

def err(actual, predicted):
    return (predicted - actual) * 1.0 /actual

def show_accuracy(url, actual, predicted, error_pct, ratings):
    print "[p: %3.2f, a: %3.2f, e: %3.2f] %s" % (predicted, actual, error_pct, url)

def extract_training_keys(urls, training_pct):
    frac = training_pct / 100.
    return set([url for url in urls if random() < frac])

def rmse(errors):
    n = len(errors)
    return sqrt(sum(e * e for e in errors))/n if n else 0.

def extract_computed_weights(result, all_critics):
    """Return the computed weight for each critic.

    result:
        a scipy.optimize.OptimizeResult object.
    all_critics:
        ordered list of critics.
    """
    return {name : result.x[i] for i, name in enumerate(all_critics)}

def train_and_test(ratings_data, training_pct, tech):
    """
    """
    # --- 1. list of all critics
    # convert to list for consistent enumeration
    all_critics = get_critics(ratings_data.values())
    significant_critics = {critic:movies_rated
                           for critic, movies_rated in all_critics.items()
                           if movies_rated > SIGNIFICANCE_RATING_COUNT}
    # --- 2. extract training set
    training_keys = extract_training_keys(ratings_data, training_pct)
    training_data = [ratings for url, ratings in ratings_data.items()
                     if url in training_keys]
    result = infer_weights(training_data,
                           list(significant_critics),
                           tech)
    if not result.success:
        error("optimization failed [%s]: %s" % (result.status, result.message))
    predicted_weights = extract_computed_weights(result, significant_critics)
    pretty_print_weights(significant_critics, predicted_weights)
    test_keys = set(ratings_data).difference(training_keys)
    errors = []
    for u in test_keys:
        actual_overall, ratings = ratings_data[u]
        if not (actual_overall and ratings):
            continue
        # while computing this, throw away insignificant critics
        predicted_overall = calc_overall_rating(predicted_weights,
                                                {c:r for c, r in ratings.items()
                                                 if c in predicted_weights})
        e = err(actual_overall, predicted_overall)
        show_accuracy(u, actual_overall, predicted_overall, e * 100, ratings)
        errors.append(actual_overall - predicted_overall)
    m = rmse(errors)
    print
    print "**********************"
    print "RMSE: %.6f" % (m,)
    print "**********************"

def infer_weights(training_data, all_critics, tech):
    """Return a dictionary mapping critics to relative weights.

    training_data: is a list of data, one per movie, of the form:

        (overall_score, critic_scores)

    where critic_scores is itself a dictionary of the form:

        critic_name -> critic rating for the movie.

    all_critics: an ordered list of critic names;
    """

    assert tech in TECHNIQUES

    # --- 1. calculate the r_dash matrix
    r_dash = []
    for _, ratings in training_data:
        r_dash.append([ratings.get(c, 0) for c in all_critics])
    r_dash = np.matrix(r_dash)
    # --- 2. calculate the e matrix
    e_matrix = []
    for _, ratings in training_data:
        e_matrix.append([1 if c in ratings else 0 for c in all_critics])
    e_matrix = np.matrix(e_matrix)
    # --- 3. construct the p vector
    p_vector = np.array([overall for overall, _ in training_data])
    # --- 4. the actual model
    m = len(training_data)
    n = len(all_critics)
    # all theta values are positive
    bounds = [(0, 1)] * n         # min, max
    theta0 = np.array([1.0/n] * n)   # initial values
    debug("m: %(m)s, n: %(n)s" % locals())
    debug("r_dash: %(r_dash)s" % locals())
    debug("e_matrix: %(e_matrix)s" % locals())
    debug("p_vector: %(p_vector)s" % locals())
    def y(theta):
        theta = np.transpose(np.asmatrix(theta))
        numerator = r_dash * theta   # each of these is an m-vector:
        denom = e_matrix * theta     # (m x n) times (n x 1).
        # element-wise division:
        # denom elements are _guaranteed_ to be positive, because:
        #   - theta values are always positive, and
        #   - e values are non-negative, and never all zeros.
        # so we should never see a divide by zero.
        return numerator / denom
    def d(theta):
        return p_vector - y(theta)
    def tub(val, lo, hi):
        return 1 if lo <= val <= hi else 0
    def obj_f(theta):
        """This is the function whose value will be minimized.

        In addition to the L-2 norm of the d() function,
        we will also add the following penalty terms:
        1) Out of Bound Penalty:
            - penalize when a theta value is outside the range [0, 1].
        2) Not in Hyperplane Penalty:
            - penalize when theta values don't add up exactly to one.
        """
        standard_error = LA.norm(d(theta))
        oob_penalty = sum(OOB_PENALTY * tub(x, 0, 1) for x in theta)
        nih_penalty = NIH_PENALTY * ( (sum(theta) - 1) ** 2 )
        return standard_error + oob_penalty + nih_penalty

    constraints = {'type': 'eq', 'fun': lambda theta: sum(theta) - 1}
    # --- 5. actual call to optimize
    return sci.minimize(obj_f, theta0, bounds=bounds, constraints=constraints,
                        method=tech, options={'disp':True})

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
    result = infer_weights(input, set(weights), tech)
    if not result.success:
        error("optimization failed [%s]: %s" % (result.status, result.message))
    predicted_weights = extract_computed_weights(infer_weights(input, set(weights), tech), weights)
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
        testitout(options.tech)
        exit(0)
    if not options.train_with:
        args_parser.error("how much data to train with?")
    if options.tech not in TECHNIQUES:
        args_parser.error("tech should be one of: %s" % (', '.join(TECHNIQUES),))
    movie_ratings = pickle.loads(sys.stdin.read())
    train_and_test(movie_ratings, options.train_with, options.tech)
