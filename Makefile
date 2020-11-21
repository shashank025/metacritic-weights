PYTHON=/usr/bin/env python3
METACRITIC_NEW_RELEASES_PAGE=https://www.metacritic.com/browse/movies/release-date/theaters/metascore?view=condensed
HTML_CONTENT_FILE=/tmp/metacritic.html
SENTINEL=__CONTENT_UPDATED

# --- 0. download metacritic new release html locally
${HTML_CONTENT_FILE}:
	curl -o ${HTML_CONTENT_FILE} ${METACRITIC_NEW_RELEASES_PAGE}

# --- 1. extract movie urls, and create new file for each url in subdirectory
movies: ${HTML_CONTENT_FILE}
	mkdir -p movies
	mc_scrape_movie_urls ${HTML_CONTENT_FILE} 2> /tmp/scrape.err > /tmp/movie_urls
	while read url; do \
		if [ ! -e movies/$${url} ]; then \
			touch movies/$${url}; \
		fi; \
	done < /tmp/movie_urls
	rm /tmp/movie_urls

# --- 2. download raw HTML content of critic reviews for each movie, in parallel
movies/__CONTENT_UPDATED: movies
	mkdir -p data
	mc_download_content --sentinel ${SENTINEL} movies

# --- 3. extract raw ratings
data/ratings.pkl: movies/${SENTINEL}
	mkdir -p data
	mc_extract_raw_ratings \
			--current-data data/ratings.pkl \
			movies > /tmp/ratings.pkl 2> /tmp/ratings.err && \
		mv /tmp/ratings.pkl data/ratings.pkl

# --- 4. extract critics who've rated at least a few movies
data/sig.pkl: data/ratings.pkl
	mc_extract_significant_critics < data/ratings.pkl > /tmp/sig.pkl && mv /tmp/sig.pkl data/sig.pkl

# --- 5. eliminate ratings from insignificant critics
data/pruned.pkl: data/sig.pkl
	mc_prune -s data/sig.pkl < data/ratings.pkl > /tmp/pruned.pkl && mv /tmp/pruned.pkl data/pruned.pkl

# --- 6. partition data into train and test set
data/train.pkl: data/pruned.pkl
	mc_partition -f .8 --test data/test.pkl --train data/train.pkl < data/pruned.pkl

# --- 7. train the models
data/theta_slsqp.pkl: data/train.pkl
	mc_train \
		-s SLSQP \
		--significant-critics data/sig.pkl < data/train.pkl > /tmp/theta_slsqp.pkl && \
		mv /tmp/theta_slsqp.pkl data/theta_slsqp.pkl

data/theta_cobyla.pkl: data/train.pkl
	mc_train \
		-s COBYLA \
		--significant-critics sig.pkl < data/train.pkl >  /tmp/theta_cobyla.pkl && \
	mv /tmp/theta_cobyla.pkl data/theta_cobyla.pkl

# --- 8. report: theta values
data/theta_slsqp.report: data/theta_slsqp.pkl
	mc_report_weights < data/theta_slsqp.pkl > data/theta_slsqp.report

data/theta_cobyla.report: data/theta_cobyla.pkl
	mc_report_weights < data/theta_cobyla.pkl > data/theta_cobyla.report

# --- 9. predict metascores
data/predict_slsqp.pkl: data/theta_slsqp.pkl
	mc_predict --theta data/theta_slsqp.pkl < data/test.pkl > data/predict_slsqp.pkl

data/predict_cobyla.pkl: data/theta_cobyla.pkl
	mc_predict --theta data/theta_cobyla.pkl < data/test.pkl > data/predict_cobyla.pkl

# --- 10. how did they do?
data/perf_slsqp.report: data/predict_slsqp.pkl data/pruned.pkl
	mc_perf_report -p data/predict_slsqp.pkl -i data/pruned.pkl > data/perf_slsqp.report

data/perf_cobyla.report: data/predict_cobyla.pk data/pruned.pkl
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
