PYTHON=/usr/bin/env python3

METACRITIC_NEW_RELEASES_PAGE='https://www.metacritic.com/browse/movies/release-date/theaters/metascore?view=condensed'
METACRITIC_TOP_MOVIE_PAGES='https://www.metacritic.com/browse/movies/score/metascore/all/filtered?sort=desc&page=[0-99]'

HTML_DOWNLOAD_DIR=/tmp/metacritic
NEW_RELEASES_HTML_FILE=/tmp/new_releases.html
SCRAPED_NEW_MOVIE_URLS_FILE=/tmp/movie_urls_new
SCRAPED_TOP_MOVIE_URLS_FILE=/tmp/movie_urls_top

SENTINEL=__CONTENT_UPDATED
CURL_USER_AGENT_ARG=User-Agent: Mozilla/5.0

# --- 0. download metacritic new release html locally

# download html of page with latest releases
${NEW_RELEASES_HTML_FILE}:
	curl -o ${NEW_RELEASES_HTML_FILE} ${METACRITIC_NEW_RELEASES_PAGE}

# download html of top rated movies
${HTML_DOWNLOAD_DIR}:
	mkdir -p ${HTML_DOWNLOAD_DIR}
	curl -Z -H "${CURL_USER_AGENT_ARG}" \
		-o "${HTML_DOWNLOAD_DIR}/top_#1.html" \
		${METACRITIC_TOP_MOVIE_PAGES}


# --- 1. extract movie urls from html
${SCRAPED_NEW_MOVIE_URLS_FILE}: ${NEW_RELEASES_HTML_FILE}
	mc_scrape_movie_urls ${NEW_RELEASES_HTML_FILE} 2> /tmp/scrape.err > ${SCRAPED_NEW_MOVIE_URLS_FILE}

${SCRAPED_TOP_MOVIE_URLS_FILE}: ${HTML_DOWNLOAD_DIR}
	ls ${HTML_DOWNLOAD_DIR} | while read f; do \
		mc_scrape_movie_urls ${HTML_DOWNLOAD_DIR}/$$f 2> /tmp/scrape.err >> ${SCRAPED_TOP_MOVIE_URLS_FILE}; \
	done


# --- 2. create new file for each movie url in subdirectory
movies: ${SCRAPED_NEW_MOVIE_URLS_FILE} ${SCRAPED_TOP_MOVIE_URLS_FILE}
	mkdir -p movies
	cat ${SCRAPED_NEW_MOVIE_URLS_FILE} ${SCRAPED_TOP_MOVIE_URLS_FILE} | while read url; do \
		if [ ! -e movies/$${url} ]; then \
			touch movies/$${url}; \
		fi; \
	done


# --- 3. download raw HTML content of critic reviews for each movie, in parallel
movies/${SENTINEL}: movies
	mc_download_content --sentinel ${SENTINEL} movies 2> /tmp/download.err

# --- 4. extract raw ratings
data/ratings.pkl: movies/${SENTINEL}
	mkdir -p data
	mc_extract_raw_ratings \
			--current-data data/ratings.pkl \
			movies > /tmp/ratings.pkl 2> /tmp/ratings.err && \
		mv /tmp/ratings.pkl data/ratings.pkl

# --- 5. extract critics who've rated at least a few movies
data/sig.pkl: data/ratings.pkl
	mc_extract_significant_critics < data/ratings.pkl > /tmp/sig.pkl && mv /tmp/sig.pkl data/sig.pkl

# --- 6. eliminate ratings from insignificant critics
data/pruned.pkl: data/sig.pkl
	mc_prune -s data/sig.pkl < data/ratings.pkl > /tmp/pruned.pkl && mv /tmp/pruned.pkl data/pruned.pkl

# --- 7. partition data into train and test set
data/train.pkl: data/pruned.pkl
	mc_partition -f .8 --test data/test.pkl --train data/train.pkl < data/pruned.pkl

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
