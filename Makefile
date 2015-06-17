# if you see a new version of xpathtool, you only need to change this
XPT_VERSION=xpathtool-20071102

XPT_URL=http://www.semicomplete.com/files/xpathtool/${XPT_VERSION}.tar.gz
XPT=${XPT_VERSION}/xpathtool/xpathtool.sh
SCRAPE_FROM=http://www.metacritic.com/browse/movies/release-date/theaters/metascore?view=condensed
BASE_URL=http://www.metacritic.com
PYTHON=/usr/bin/python

# --- 1. get necessary tools
${XPT}:
	wget ${XPT_URL} -O xpt.tar.gz
	gzip -d xpt.tar.gz
	tar xvf xpt.tar
	rm xpt.tar

# --- 2. this html contains a bunch of metacritic urls
urls_html:
	wget ${SCRAPE_FROM} -O urls_html

# --- 3. get movie suffixes
url_suffixes: urls_html
	cat urls_html | ${XPT} --ihtml '//@href' | grep '^/movie' | sed -e 's/^\///g' | sort | uniq > /tmp/url_suffixes
	if [ ! -e url_suffixes ]; then \
		mv /tmp/url_suffixes url_suffixes; \
	else \
		if cmp url_suffixes /tmp/url_suffixes; then \
			sort /tmp/url_suffixes url_suffixes | uniq > /tmp/url_suffixes.1; \
			mv /tmp/url_suffixes.1 url_suffixes; \
		fi \
	fi
	touch url_suffixes # otherwise, urls_xml will be newer than url_suffixes

# --- 4. download and cache movie critic reviews html
# url suffix -> actual critic ratings:
#   movie/a-haunted-house -> http://www.metacritic.com/movie/a-haunted-house/critic-reviews
movie: url_suffixes
	mkdir -p movie
	while read suffix; do \
		if [ ! -e $$suffix ]; then \
			wget ${BASE_URL}/$$suffix/critic-reviews -O $$suffix; \
		fi \
	done < url_suffixes

# --- 6. pickle dump of Python dictionary that contains movie ratings
# requires: Python module lxml
ratings.pkl: url_suffixes ratings.py movie
	XPATHTOOLS=${XPT} $(PYTHON) ratings.py < url_suffixes > /tmp/ratings.pkl && mv /tmp/ratings.pkl ratings.pkl

# --- 7. Sequential Least SQuares Programming
out.slsqp: ratings.pkl metacritic.py
	$(PYTHON) metacritic.py -s SLSQP -t 80 < ratings.pkl > out.slsqp 2> err.slsqp

all: out.slsqp

clean_movie:
	rm -rf movie

clean: clean_movie
