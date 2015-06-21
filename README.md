Reverse Engineering Metacritic
==============================

_For complete detail,
please see [this blog post](http://shashank025.github.io/2015/06/14/reverse-engineering-the-metacritic-movie-ratings/)_.

Run ``make all'' in this directory.

After a rather long process that involves downloading metacritic review html,
parsing and extracting ratings, etc., and finally running the learning algorithms,
you should see two output files:

     out.lstsq and out.nnls,

that represent the output of the regression analysis.

Prerequisites:
 * Python 2.7
 * lxml Python module
 * numpy, scipy

If someone else can package all this up neatly in a setup.py script,
I will be much obliged.
