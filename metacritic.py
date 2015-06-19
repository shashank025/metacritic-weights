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
TECHNIQUES = frozenset(['SLSQP','COBYLA'])
RATING_COUNT_THRESHOLD = 5            # critic must rate at least these many movies to be considered.
OOB_PENALTY = 100                     # how much to penalize the objective fn when a theta value is out of bounds.
NIH_PENALTY = 100                     # how much to penalize the objective fn when theta values dont add up to 1.

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

def prune_ratings(movie_ratings, significant_critics):
    return {c: r for c, r in movie_ratings.items()
            if c in significant_critics}

def construct_opt_params(ratings_data, critics):
    r_dash = []
    for _, individual_ratings in ratings_data:
        r_dash.append([individual_ratings.get(c, 0) for c in critics])
    r_dash = np.matrix(r_dash)
    # --- 3. calculate the r_dash and e matrices
    e_matrix = []
    for _, individual_ratings in ratings_data:
        e_matrix.append([1 if c in individual_ratings else 0 for c in critics])
    e_matrix = np.matrix(e_matrix)
    p_vector = np.array([metascore for metascore, _ in ratings_data])
    debug("r_dash: %(r_dash)s" % locals())
    debug("e_matrix: %(e_matrix)s" % locals())
    debug("p_vector: %(p_vector)s" % locals())
    return r_dash, e_matrix, p_vector

def scale_critic_weights(predicted_weights):
    """Scale weights to be in interval [0, 1].

    Returns a triple of the form:
        (lo, hi, critic_weights)
    where critic_weights is a dictionary of the form:
        critic_name -> scaled_weight.
    """
    all_weights = predicted_weights.values()
    hi = max(all_weights)
    lo = min(all_weights)
    assert hi > lo
    m = 1.0 / (hi - lo)  # slope
    c = lo * m           # intercept
    return (lo, hi, {critic_name : (m * weight - c)
                     for critic_name, weight in predicted_weights.items()})

def train_and_test(ratings_data, training_pct, tech, significant_critics):
    """
    ratings_data:
        a dictionary of the form:
            movie_url -> (metascore, movie_ratings)
        where movie_ratings is itself a dictionary of the form:
            critic_name -> critic_rating.
    training_pct:
        what fraction of input data do we want to use for training?
    tech:
        what optimization technology do we want to use?
    """

    # --- 1. extract training set
    training_keys = extract_training_keys(ratings_data, training_pct)
    training_data = [ratings for url, ratings in ratings_data.items()
                     if url in training_keys]
    # --- 2. construct optimization parameters
    r_dash, e_matrix, p_vector = construct_opt_params(training_data, significant_critics)

    # --- 3. actual constrained optimization
    result = infer_weights(r_dash, e_matrix, p_vector, tech)
    if not result.success:
        error("optimization failed [%s]: %s" % (result.status, result.message))

    # --- 4. how well did it do?
    predicted_weights = extract_computed_weights(result, significant_critics)
    lo, hi, scaled_weights = scale_critic_weights(predicted_weights)
    pretty_print_weights(significant_critics, scaled_weights)
    test_keys = set(ratings_data).difference(training_keys)
    errors = []
    for u in test_keys:
        actual_overall, ratings = ratings_data[u]
        if not (actual_overall and ratings):
            continue
        # while computing this, throw away insignificant critics
        predicted_overall = calc_overall_rating(predicted_weights, ratings)
        e = err(actual_overall, predicted_overall)
        show_accuracy(u, actual_overall, predicted_overall, e * 100, ratings)
        errors.append(actual_overall - predicted_overall)
    m = rmse(errors)
    print
    print "**********************"
    print "RMSE: %.6f" % (m,)
    print "**********************"

def infer_weights(r_dash, e_matrix, p_vector, tech):
    """Return a dictionary mapping critics to relative weights."""

    assert tech in TECHNIQUES

    # basic statistics
    m, n = np.shape(r_dash)
    total_ratings = np.count_nonzero(e_matrix)
    debug("m: %(m)s, n: %(n)s, total_ratings: %(total_ratings)s" % locals())

    # all theta values are positive
    bounds = [(0, 1)] * n            # min, max
    theta0 = np.array([1.0/n] * n)   # initial values
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
        return 0 if lo <= val <= hi else 1
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
        oob_penalty = OOB_PENALTY * sum(tub(x, 0, 1) for x in theta)
        nih_penalty = NIH_PENALTY * ( (sum(theta) - 1) ** 2 )
        return standard_error + oob_penalty + nih_penalty

    constraints = {'type': 'eq', 'fun': lambda theta: sum(theta) - 1} if tech == 'SLSQP' else []
    # --- 5. actual call to optimize
    return sci.minimize(obj_f, theta0, bounds=bounds, constraints=constraints,
                        method=tech, options={'disp':True})

def pretty_print_weights(all_critics, weights):
    for critic_name, weight in sorted(weights.items(), key=itemgetter(1), reverse=True):
        print "%2.6f %s (%d ratings)" % (weight * 100., critic_name, all_critics[critic_name])

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
    r_dash, e_matrix, p_vector = construct_opt_params(input, weights)
    result = infer_weights(r_dash, e_matrix, p_vector, tech)
    if not result.success:
        error("optimization failed [%s]: %s" % (result.status, result.message))
    predicted_weights = extract_computed_weights(result, weights)
    print ">>>> actual weights:", weights
    print ">>>> predicted_weights:", predicted_weights

def preprocess(ratings_data):
    all_critics = get_critics(ratings_data.values())
    significant_critics = {critic:movies_rated
                           for critic, movies_rated in all_critics.items()
                           if movies_rated > RATING_COUNT_THRESHOLD}
    # pruning actually happens here
    ratings_data = {url : (metascore, prune_ratings(ratings, significant_critics))
                    for url, (metascore, ratings) in ratings_data.items()}
    return significant_critics, ratings_data

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

    # --- 1. load previously extracted movie rating data
    # ratings_data is a dictionary of the form:
    #     movie_url -> (metascore, individual_ratings)
    # where individual_ratings is itself a dictionary of the form:
    #     critic_name -> rating
    ratings_data = pickle.loads(sys.stdin.read())
    # --- 2. preprocessing: remove data corresponding to insignificant critics
    # dictionary of the form: critic_name -> movies_rated
    significant_critics, ratings_data = preprocess(ratings_data)
    train_and_test(ratings_data, options.train_with, options.tech, significant_critics)
