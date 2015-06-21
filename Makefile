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

install:
	$(PYTHON) setup.py install

# --- 6. pickle dump of Python dictionary that contains movie ratings
# requires: Python module lxml
ratings.pkl: url_suffixes movie
	XPATHTOOLS=${XPT} mc_extract_raw_ratings < url_suffixes > /tmp/ratings.pkl && mv /tmp/ratings.pkl ratings.pkl

# --- 7. extract critics who've rated at least a few movies
sig.pkl: ratings.pkl
	mc_extract_significant_critics < ratings.pkl > /tmp/sig.pkl && mv /tmp/sig.pkl sig.pkl

# --- 8. eliminate ratings from insignificant critics
pruned.pkl: sig.pkl
	mc_prune -s sig.pkl < ratings.pkl > /tmp/pruned.pkl && mv /tmp/pruned.pkl pruned.pkl

# --- 9. partition data into train and test set
train.pkl: pruned.pkl
	mc_partition -f 80 --test test.pkl --train train.pkl < pruned.pkl

# --- 10. train the models
theta_slsqp.pkl: train.pkl
	mc_train -s SLSQP --significant-critics sig.pkl < train.pkl > /tmp/theta_slsqp.pkl && mv /tmp/theta_slsqp.pkl theta_slsqp.pkl

theta_cobyla.pkl: train.pkl
	mc_train -s COBYLA --significant-critics sig.pkl < train.pkl >  /tmp/theta_cobyla.pkl && mv /tmp/theta_cobyla.pkl theta_cobyla.pkl

# --- 11. report: theta values
theta_slsqp.report: theta_slsqp.pkl
	mc_report_weights < theta_slsqp.pkl > theta_slsqp.report

theta_cobyla.report: theta_cobyla.pkl
	mc_report_weights < theta_cobyla.pkl > theta_cobyla.report

# --- 12. predict metascores
predict_slsqp.pkl: theta_slsqp.pkl
	mc_predict --theta theta_slsqp.pkl < test.pkl > predict_slsqp.pkl

predict_cobyla.pkl: theta_cobyla.pkl
	mc_predict --theta theta_cobyla.pkl < test.pkl > predict_cobyla.pkl

# --- 12. how did they do?
perf_slsqp.report:
	mc_perf_report -p predict_slsqp.pkl -i pruned.pkl > perf_slsqp.report

perf_cobyla.report:
	mc_perf_report -p predict_cobyla.pkl -i pruned.pkl > perf_cobyla.report

clean:
	rm *.pkl *.report
