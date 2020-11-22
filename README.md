Reverse Engineering Metacritic
==============================

_For complete detail, please see
[this blog post](https://shashank.ramaprasad.com/2015/06/14/reverse-engineering-the-metacritic-movie-ratings)_.

Install
-------

Run:

    sudo make install


Getting Results
---------------

Run the following commands _after_ you make install.

    make all

You should see the following output files (among others):

* `theta_*.report`: pretty printing of the learned theta values for each learning algorithm
* `perf_*.report`: pretty printing of the performance analysis of each learning algorithm


Scraping Critic Reviews for a Movie on metacritic
---

For a given movie, Metacritic lists all critic reviews it considered
when creating a metascore on a page with the following url format:

    https://www.metacritic.com/movie/<movie-url-suffix>/critic-reviews

For example, individual critic reviews for the movies
[Oliver Sacks: His Own Life](https://www.imdb.com/title/tt10887164/)
(2020) and [Tenet](https://www.imdb.com/title/tt6723592/) can be accessed at
the following two Metacritic pages, respectively:

    https://www.metacritic.com/movie/oliver-sacks-his-own-life/critic-reviews
    https://www.metacritic.com/movie/tenet/critic-reviews

The overall numeric Metascore for a movie can be scraped by processing the
following HTML tree structure on the above page:

    <table class="simple_summary marg_btm1">
      ...
      <table class="score_wrapper">
        ...
        <span class="metascore_w larger movie positive">82</span>

Individual numeric critic ratings can be scraped from following HTML structure:

    <div class="review pad_top1 pad_btm1">
      <div class="left fl">
        <div class="metascore_w large movie positive indiv">85</div>
      </div>
      <div class="right fl">
         <div class="title pad_btm_half">
           <span class="source">
             <a href="/publication/film-threat?filter=movies">
               <img
                 src="....90_291422295.png"
                 alt="Film Threat"
                 title="Film Threat"
                 class="pub-img" />
             </a>
           </span>
           <span class="author">Kyle Bain</span>
           ...

The
[New Movie Releases](https://www.metacritic.com/browse/movies/release-date/theaters/metascore?view=condensed)
page on Metacritic is a good place to scrape Critic Review links for new movies.

Strategy
---

0. Open the previously gathered ratings data (if any), stored as a cPickle file,
from the following subdirectory:

    data/
      ratings.pkl

Remember, each rating is a mapping of the form:

    movie_url => (metascore, individual_ratings)

where `individual_ratings` is also a mapping of the form:

    critic_name => numeric_rating

1. Scrape movie urls from the New Releases page.

2. If the movie url is already in ratings.pkl, then throw it away (because
we have already extracted ratings for this movie).

3. For each remaining movie url, check whether its HTML contents need to be
downloaded by looking for an appropriately named file in the movies/
subdirectory, and download if necessary:

    movies/
      tenet-critic-reviews
      oliver-sack-his-own-life-critic-reviews
      ...

Note: this step involves a network call for downloading each movie's content,
and can be parallelized, which should result in a speed up.

4. For each movie, parse the HTML to extract the overall metascore, and
individual critic ratings and append them to the ratings.pkl data structure.

5. Extract critics who've rated at least a few movies and dump those into
their own file:

    data/
      sig.pkl

6. Create a new ratings mapping that only contains ratings from
significant critics, and dump those into another file:

    data/
      pruned.pkl

7. Partition this into a test/train set:

    data/
      test.pkl train.pkl

8. Train models, and so on ...
