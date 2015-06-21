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
DEFAULT_RATING_COUNT_THRESHOLD = 5    # critic must rate at least these many movies to be considered.
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

def pretty_print_weights(all_critics, weights):
    for critic_name, weight in sorted(weights.items(), key=itemgetter(1), reverse=True):
        print "%.6f %s" % (weight, critic_name)

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

def report(theta):
    lo, hi, scaled_weights = scale_critic_weights(theta)
    pretty_print_weights(theta, scaled_weights)

def calc_overall_rating(weights, critic_ratings):
    unnormalized = sum(weights[critic] * score for critic, score in critic_ratings.items())
    total_weight = sum(weights[critic] for critic in critic_ratings)
    return unnormalized/total_weight

def get_critics(ratings_data):
    """returns a dictionary:

        critic name -> number of movies rated
    """

    critics = defaultdict(int)
    for _, (_, ratings) in ratings_data.items():
        for c in ratings:
            critics[c] += 1
    return critics

def get_significant_critics(counts, threshold):
    return {critic:movies_rated
            for critic, movies_rated in counts.items()
            if movies_rated > threshold}

def prune(ratings_data, significant_critics):
    def _prune(movie_ratings):
        return {c: r for c, r in movie_ratings.items()
                if c in significant_critics}
    return {url : (metascore, _prune(individual_ratings))
            for url, (metascore, individual_ratings) in ratings_data.items()}

def extract_training_keys(urls, frac):
    return set([url for url in urls if random() < frac])

def partition(ratings_data, frac):
    training_keys = extract_training_keys(ratings_data, frac)
    training_data = {url: ratings for url, ratings in ratings_data.items()
                     if url in training_keys}
    test_data = {url: ratings for url, ratings in ratings_data.items()
                 if url not in training_keys}
    return training_data, test_data

def extract_computed_weights(result, all_critics):
    """Return the computed weight for each critic.

    result:
        a scipy.optimize.OptimizeResult object.
    all_critics:
        ordered list of critics.
    """
    return {name : float(result.x[i]) for i, name in enumerate(all_critics)}

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
                        method=tech)

def construct_opt_params(ratings_data, critics):
    """
    ratings_data:
        the usual metacritic data structure.
    critics:
        a *list* of critics (not a set).
    """

    # make an ordered list first
    url_list = list(ratings_data)
    r_dash = []
    for url in url_list:
        _, individual_ratings = ratings_data[url]
        r_dash.append([individual_ratings.get(c, 0)
                       for c in critics])
    r_dash = np.matrix(r_dash)
    e_matrix = []
    for url in url_list:
        _, individual_ratings = ratings_data[url]
        e_matrix.append([1 if c in individual_ratings else 0
                         for c in critics])
    e_matrix = np.matrix(e_matrix)
    p_vector = np.array([ratings_data[url][0]
                         for url in url_list])
    debug("r_dash: %(r_dash)s" % locals())
    debug("e_matrix: %(e_matrix)s" % locals())
    debug("p_vector: %(p_vector)s" % locals())
    return r_dash, e_matrix, p_vector

def train(ratings_data, critic_list, tech):
    """
    ratings_data:
        a dictionary of the form:
            movie_url -> (metascore, movie_ratings)
        where movie_ratings is itself a dictionary of the form:
            critic_name -> critic_rating.

    critic_list:
        an ordered list of critic names.

    tech:
        what optimization technology do we want to use?
    """
    r_dash, e_matrix, p_vector = construct_opt_params(ratings_data, critic_list)
    result = infer_weights(r_dash, e_matrix, p_vector, tech)
    if not result.success:
        error("optimization failed [%s]: %s" % (result.status, result.message))
    return extract_computed_weights(result, critic_list)

def predict(test_data, theta):
    result = {}
    for movie, (_, individual_ratings) in test_data.items():
        predicted_metascore = calc_overall_rating(theta, individual_ratings)
        result[movie] = predicted_metascore
    return result

def err(actual, predicted):
    return (predicted - actual) * 1.0 /actual

def get_accuracy(url, actual, predicted, error_pct):
    return "[p: %3.2f, a: %3.2f, e: %3.2f] %s" % (predicted, actual, error_pct, url)

def rmse(errors):
    n = len(errors)
    return sqrt(sum(e * e for e in errors))/n if n else 0.

def performance_report(ratings_data, predictions):
    errors = []
    report = []
    for movie, predicted in predictions.items():
        actual, _ = ratings_data[movie]
        e = err(actual, predicted)
        errors.append(e)
        report.append(get_accuracy(movie, actual, predicted, e * 100))
    m = rmse(errors)
    print
    print "**********************"
    print "RMSE: %.6f" % (m,)
    print "**********************"
    print
    for row in report:
        print row
