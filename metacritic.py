"""
What are metacritic's relative weights for different critics/publications?
"""

import os
import sys
import subprocess
import urllib2
import optparse
import tempfile
import pickle
from operator import itemgetter
from random import random
from collections import defaultdict

import numpy as np
import scipy.optimize as sci
# import xml.etree.ElementTree as ET
from lxml import etree as ET

DEBUG = True
XPATHTOOLS = '/Users/sramapra/Downloads/software/xpathtool-20071102/xpathtool/xpathtool.sh'
DEFAULT_CACHE_FILE = '.mccache'
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

def extract_movies(rss_filename):
    """returns a list of (movie name, metacritic_link) tuples."""

    tree = ET.parse(rss_filename)
    root = tree.getroot()
    return [(item.find('title').text, item.find('link').text)
            for item in root.findall('//item')]

def xpathtools(args, file_handle):
    cmd = [XPATHTOOLS] + args
    debug("Running cmd: %s" % (cmd,))
    sys.stderr.flush()
    return subprocess.check_output(cmd, stdin=file_handle, shell=False)

def get_overall_score(file_handle):
    args = ['--otext',
            '--ihtml',
            '//div[@class="metascore_wrap feature_metascore"]/div[contains(concat(" ", @class, " "), " data metascore ")]/span[@class="score_value"]']
    score_text = xpathtools(args, file_handle)
    try:
        return int(score_text)
    except:
        return None

def get_review_stats_xml(file_handle):
    args = ['--oxml', '--ihtml',  '//div[@class="review_stats"]']
    return xpathtools(args, file_handle)

def extract_ratings(reviews_xml):
    ratings = {}
    root = ET.fromstring(reviews_xml)
    for div in root.xpath('.//div[@class="review_stats"]'):
        match = div.find('.//div[@class="source"]/a')
        pub = match.text.strip() if match is not None else None
        match = div.find('.//div[@class="author"]/a')
        auth = match.text.strip() if match is not None else None
        match = div.xpath('.//div[starts-with(@class, "review_grade")]')
        if not match:
            warn("no score found for review; moving on ...")
        score = int(match[0].text)
        if not auth and not pub:
            warn("no auth/pub for review; moving on ...")
            continue
        if auth and pub:
            auth = '_'.join([auth, pub])
        if not auth and pub:
            auth = pub
        ratings[auth] = score
    return ratings

def get_content(url):
    url = url + '/critic-reviews' if not url.endswith('/critic-reviews') else url
    debug("trying to download from %s" % (url,))
    try:
        content = urllib2.urlopen(url).read()
    except urllib2.HTTPError, e:
        if 'HTTP Error 404: Not Found' in str(e):
            error("url %s is invalid" % (url,))
            return None
        elif 'HTTP Error 503: Service Unavailable' in str(e):
            error("service unavailable for url %s" % (url,))
            return None
        raise e
    else:
        return content

def get_movie_stats(contents):
    with tempfile.NamedTemporaryFile() as f:
        f.write(contents)
        f.seek(0)
        overall_score = get_overall_score(f)
        debug(">>>> overall score: %s" % (overall_score,))
        f.seek(0)
        review_stats_xml = get_review_stats_xml(f)
    all_ratings = extract_ratings(review_stats_xml)
    return (overall_score, all_ratings)

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
    training_set = set()
    frac = training_pct / 100.
    for url in set(urls):
        if random() < frac:
            training_set.add(url)
    return training_set

def train_and_test(ratings_cache, all_urls, training_set, tech):
    """
    """
    all_critics = get_critics(ratings_cache.values())
    predicted_weights = infer_weights([ratings_cache[u]
                                       for u in training_set
                                       if u in ratings_cache],
                                      all_critics,
                                      tech)
    pretty_print_weights(all_critics, predicted_weights)
    test_set = set(all_urls).difference(training_set)
    for u in test_set:
        if u not in ratings_cache:
            continue
        actual_overall, ratings = ratings_cache[u]
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

def infer_weights_from_rss(rss_url):
    urls = [url for _, url in extract_movies(rss_url)]
    return infer_weights_from_urls(urls)

def load(cache_file):
    if not os.path.isfile(cache_file):
        return {}
    with open(cache_file, 'rb') as f:
        return pickle.load(f)

def save(new_values, cache_file):
    with open(cache_file, 'wb') as f:
        pickle.dump(new_values, f)

def get_ratings(url):
    content = get_content(url)
    return get_movie_stats(content) if content else (None, None)

def collect_ratings(movie_urls, cache_file=DEFAULT_CACHE_FILE):
    cache = load(cache_file)
    for u in movie_urls:
        if u not in cache:
            cache[u] = get_ratings(u)
    save(cache, cache_file)
    return dict([(u, (overall, r))
                for u, (overall, r) in cache.items()
                if (overall and r)])

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

    args_parser = optparse.OptionParser(usage="%prog --tech (nnls|lstsq) { --test | --training-pct <int>}")
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
    # read urls from the input
    urls = [line.strip() for line in sys.stdin.readlines()]
    training_set = extract_training_set(urls, options.train_with)
    ratings_cache = collect_ratings(urls)
    train_and_test(ratings_cache, urls, training_set, options.tech)
