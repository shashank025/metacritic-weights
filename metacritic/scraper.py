"""Library for scraping metacritic movie rating info"""

import asyncio
import os
import re
import requests
import sys
import traceback

from aiohttp import ClientSession
from lxml import html
from pathlib import Path

from metacritic.common import debug
from metacritic.common import err
from metacritic.common import warning
from metacritic.common import info

USER_AGENT = 'Mozilla/5.0'
PREFIX_PATTERN = re.compile(r'''^/movie/''')
SUFFIX_PATTERN = re.compile('/(critic|user)-reviews$')
DEFAULT_SENTINEL = '__CONTENT_UPDATED'

MOVIE_HREF_PATTERN = '//a[starts-with(@href, "/movie/")]'
METASCORE_SPAN_PATTERN = '//table[@class="simple_summary marg_btm1"]//span[starts-with(@class, "metascore_w")]'
REVIEW_PATTERN = '//div[starts-with(@class, "review")]'
INDIVIDUAL_RATING_PATTERN = './/div[starts-with(@class, "metascore_w")]'
AUTHOR_PATTERN = './/span[contains(@class, "author")]'
PUBLICATION_PATTERN = './/span[contains(@class, "source")]//img[@title]'


def scrape_movie_urls(html_content_filename):
    """Extract metacritic movie url suffixes from specified metacritic html content"""

    tree = html.parse(html_content_filename)
    urls = set()
    for a_node in tree.xpath(MOVIE_HREF_PATTERN):
        href = a_node.get('href')
        href = PREFIX_PATTERN.sub('', href)
        href = SUFFIX_PATTERN.sub('', href)
        urls.add(href)
    return urls


def get_suffixes_to_download(dir, sentinel_filename, refresh=False):
    suffixes = (s for s in os.listdir(dir) if s != sentinel_filename)
    if not refresh:
        def is_empty(filename):
            return os.stat(filename).st_size == 0
        # only fetch content if file is empty
        suffixes = (s for s in suffixes if is_empty(os.path.join(dir, s)))
    return suffixes


async def download_and_write_urls(dir, suffixes, concurrency):
    """Asynchronously download urls and write to disk."""
    async with ClientSession() as session:
        tasks = [download_task(s, dir, session) for s in suffixes]
        # list, where each element is of the form (url, content)
        return await gather_with_concurrency(concurrency, *tasks)


async def download_task(suffix, dir, session):
    url = f"https://www.metacritic.com/movie/{suffix}/critic-reviews"
    response = await session.get(url, headers={'User-Agent': USER_AGENT})
    if response.status != 200:
        err(f"{response.status} while downloading {url}")
        return False
    html = await response.text()
    dest = os.path.join(dir, suffix)
    with open(dest, 'w') as f:
        debug(f"writing downloaded content for {suffix} to {dest} ...")
        f.write(html)
        return True


async def gather_with_concurrency(n, *tasks):
    semaphore = asyncio.Semaphore(n)
    async def sem_task(task):
        async with semaphore:
            return await task
    return await asyncio.gather(*(sem_task(task) for task in tasks))


def find(node, xpath_pattern):
    """replacement for lxml find() that supports full xpath power"""
    matches = node.xpath(xpath_pattern)
    return matches[0] if matches else None


def get_overall_score(content_tree):
    span = find(content_tree, METASCORE_SPAN_PATTERN)
    try:
        return int(span.text_content())
    except:
        err("get_overall_score: " + traceback.format_exc())
        return None


def extract_one_rating(review_node):
    rating_node = find(review_node, INDIVIDUAL_RATING_PATTERN)
    try:
        return int(rating_node.text_content())
    except:
        err("extract_one_rating: " + traceback.format_exc())
        return None


def extract_author(review_node):
    author_node = find(review_node, AUTHOR_PATTERN)
    return author_node.text_content() if author_node is not None else None


def extract_publication(review_node):
    img_node = find(review_node, PUBLICATION_PATTERN)
    return img_node.get('title') if img_node is not None else None


def extract_ratings_for_movie(filename, group_pub=False):
    tree = html.parse(filename)
    if not tree.getroot():
        # if the file is empty, there is no tree!
        return None
    metascore = get_overall_score(tree)
    if not metascore:
        err(f"could not extract metascore for: {filename}")
        return None
    review_nodes = tree.xpath(REVIEW_PATTERN)
    if not review_nodes:
        err(f"no critic reviews found for: {filename}")
        return None
    ratings = {}
    total = len(review_nodes)
    succeeded = 0
    for node in review_nodes:
        critic_score = extract_one_rating(node)
        critic_name = extract_author(node)
        publication = extract_publication(node)
        if not (critic_score and critic_name and publication):
            continue
        if not group_pub:
            critic_name = critic_name + ' (' + publication + ')'
        else:
            critic_name = publication
        ratings[critic_name] = critic_score
        succeeded += 1
    if succeeded < total:
        warning(f"only parsed {succeeded} out of {total} reviews for {filename}")
    return (metascore, ratings)


def extract_all_ratings(content_dir, sentinel, group_pub=False):
    ratings = {}
    for suffix in os.listdir(content_dir):
        if suffix == sentinel:
            continue
        debug(f"extracting ratings for: {suffix} ...")
        rating_for_movie = extract_ratings_for_movie(
            os.path.join(content_dir, suffix), group_pub=group_pub)
        if not rating_for_movie:
            err(f"could not extract ratings for: {suffix}")
            continue
        ratings[suffix] = rating_for_movie
    return ratings
