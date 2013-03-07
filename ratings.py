"""
Extract ratings for specified movies
"""

import os
import sys
import subprocess
import optparse
import pickle

from lxml import etree as ET

DEBUG = True

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

def xpathtools(args, file_handle):
    XPATHTOOLS = os.environ['XPATHTOOLS']
    cmd = [XPATHTOOLS] + args
    debug("Running cmd: %s" % (cmd,))
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

def get_movie_stats(f):
    f.seek(0)
    overall_score = get_overall_score(f)
    debug(">>>> overall score: %s" % (overall_score,))
    f.seek(0)
    review_stats_xml = get_review_stats_xml(f)
    all_ratings = extract_ratings(review_stats_xml)
    return (overall_score, all_ratings)

def get_ratings(suffix):
    if os.path.exists(suffix):
        with open(suffix, 'rb') as f:
            return get_movie_stats(f)
    return (None, None)

def collect_ratings(url_suffixes):
    return dict([(u, (overall, r))
                 for u, (overall, r) in [(u, get_ratings(u))
                                         for u in url_suffixes]
                 if (overall and r)])

if __name__ == '__main__':

    args_parser = optparse.OptionParser(usage="""
    %prog < url_suffixes > pickledump

    Outputs a pickle dump of a Python dictionary that holds critic ratings for specified movies.
    """)
    options, args = args_parser.parse_args()
    url_suffixes = [line.strip() for line in sys.stdin.readlines()]
    movie_ratings = collect_ratings(url_suffixes)
    print pickle.dumps(movie_ratings)
