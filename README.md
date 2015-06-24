Reverse Engineering Metacritic
==============================

_For complete detail, please see
[this blog post](http://shashank.ramaprasad.com/2015/06/14/reverse-engineering-the-metacritic-movie-ratings/)_.

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
