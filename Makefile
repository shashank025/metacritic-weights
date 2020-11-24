PYTHON=/usr/bin/env python3

NEW_RELEASES_PAGE='https://www.metacritic.com/browse/movies/release-date/theaters/metascore?view=condensed'
TOP_MOVIES_PAGE='https://www.metacritic.com/browse/movies/score/metascore/all/filtered?sort=desc&page=[0-99]'

MOVIE_DIR=movies
REVIEW_DIR=${MOVIE_DIR}/reviews
HTML_DIR=${MOVIE_DIR}/html

NEW_RELEASES_HTML_FILE=${HTML_DIR}/new.html
SCRAPED_MOVIE_URLS_FILE=${MOVIE_DIR}/movie_urls

CURL_USER_AGENT_ARG=User-Agent: Mozilla/5.0
CURL_PARALLEL_MAX=4

SENTINEL=__CONTENT_UPDATED

TRAIN_FRACTION=.3

# --- 0. download metacritic new release html locally

# download html of pages with movie urls
${NEW_RELEASES_HTML_FILE}:
	mkdir -p ${HTML_DIR}
	curl -o ${NEW_RELEASES_HTML_FILE} ${NEW_RELEASES_PAGE}
	curl -Z -H "${CURL_USER_AGENT_ARG}" \
		--parallel-max ${CURL_PARALLEL_MAX} \
		-o "${HTML_DIR}/top_#1.html" \
		${TOP_MOVIES_PAGE}


# --- 1. extract movie urls from html
${SCRAPED_MOVIE_URLS_FILE}: ${NEW_RELEASES_HTML_FILE}
	touch ${SCRAPED_MOVIE_URLS_FILE}
	ls ${HTML_DIR} | while read f; do \
		mc_scrape_movie_urls ${HTML_DIR}/$$f 2> /tmp/scrape.err >> /tmp/movie_urls ; \
	done
	sort ${SCRAPED_MOVIE_URLS_FILE} /tmp/movie_urls | uniq >> ${SCRAPED_MOVIE_URLS_FILE}
	rm /tmp/movie_urls


# --- 2. create new file for each movie url in subdirectory
${REVIEW_DIR}: ${SCRAPED_MOVIE_URLS_FILE}
	mkdir -p ${REVIEW_DIR}
	cat ${SCRAPED_MOVIE_URLS_FILE} | while read url; do \
		if [ ! -e ${REVIEW_DIR}/$${url} ]; then \
			touch ${REVIEW_DIR}/$${url}; \
		fi; \
	done


# --- 3. download raw HTML content of critic reviews for each movie, in parallel
${REVIEW_DIR}/${SENTINEL}: ${REVIEW_DIR}
	mc_download_content --sentinel ${SENTINEL} ${REVIEW_DIR} 2> /tmp/download.err

# --- 4. extract raw ratings
data/ratings.pkl: ${REVIEW_DIR}/${SENTINEL}
	mkdir -p data
	mc_extract_raw_ratings \
			--current-data data/ratings.pkl \
			${REVIEW_DIR} > /tmp/ratings.pkl 2> /tmp/ratings.err && \
		mv /tmp/ratings.pkl data/ratings.pkl

# --- 5. extract critics who've rated at least a few movies
data/sig.pkl: data/ratings.pkl
	mc_extract_significant_critics < data/ratings.pkl \
		2> /tmp/sig.err > /tmp/sig.pkl && \
		mv /tmp/sig.pkl data/sig.pkl

# --- 6. eliminate ratings from insignificant critics
data/pruned.pkl: data/sig.pkl
	mc_prune -s data/sig.pkl < data/ratings.pkl > /tmp/pruned.pkl && mv /tmp/pruned.pkl data/pruned.pkl

# --- 7. partition data into train and test set
data/train.pkl: data/pruned.pkl
	mc_partition -f ${TRAIN_FRACTION} --test data/test.pkl --train data/train.pkl < data/pruned.pkl

# --- 8. train the models
data/theta_slsqp.pkl: data/train.pkl
	mc_train \
		-s SLSQP \
		--significant-critics data/sig.pkl < data/train.pkl > /tmp/theta_slsqp.pkl && \
		mv /tmp/theta_slsqp.pkl data/theta_slsqp.pkl

data/theta_cobyla.pkl: data/train.pkl
	mc_train \
		-s COBYLA \
		--significant-critics data/sig.pkl < data/train.pkl >  /tmp/theta_cobyla.pkl && \
	mv /tmp/theta_cobyla.pkl data/theta_cobyla.pkl

# --- 9. report: theta values
data/theta_slsqp.report: data/theta_slsqp.pkl
	mc_report_weights < data/theta_slsqp.pkl > data/theta_slsqp.report

data/theta_cobyla.report: data/theta_cobyla.pkl
	mc_report_weights < data/theta_cobyla.pkl > data/theta_cobyla.report

# --- 10. predict metascores
data/predict_slsqp.pkl: data/theta_slsqp.pkl
	mc_predict --theta data/theta_slsqp.pkl < data/test.pkl > data/predict_slsqp.pkl

data/predict_cobyla.pkl: data/theta_cobyla.pkl
	mc_predict --theta data/theta_cobyla.pkl < data/test.pkl > data/predict_cobyla.pkl

# --- 11. how did they do?
data/perf_slsqp.report: data/predict_slsqp.pkl data/pruned.pkl
	mc_perf_report -p data/predict_slsqp.pkl -i data/pruned.pkl > data/perf_slsqp.report

data/perf_cobyla.report: data/predict_cobyla.pkl data/pruned.pkl
	mc_perf_report -p data/predict_cobyla.pkl -i data/pruned.pkl > data/perf_cobyla.report

all: data/perf_slsqp.report data/perf_cobyla.report data/theta_slsqp.report data/theta_cobyla.report


# TODO: add addiitonal make targets to re-download content, etc.
clean:
	rm data/*.pkl data/*.report

# run as sudo
clean_dist:
	$(PYTHON) setup.py clean --all

# run as sudo
install:
	$(PYTHON) setup.py install
