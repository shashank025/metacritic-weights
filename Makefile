XPT=/Users/sramapra/Downloads/software/xpathtool-20071102/xpathtool/xpathtool.sh

urls_html:
	wget http://www.metacritic.com/browse/movies/release-date/theaters/metascore?view=condensed -O urls_html

urls_xml: urls_html
	${XPT} --oxml --ihtml '//a[starts-with(@href, "/movie")]' < urls_html > urls_xml

new_suffixes:
	${XPT} --ihtml '//a/@href' < urls_xml > new_suffixes

urls: new_suffixes
	awk '{ printf "http://www.metacritic.com%s\n", $$1 }' new_suffixes >> urls
	sort urls | uniq > urls.1
	mv urls.1 urls

out.nnls: urls
	/opt/local/bin/python2.7 metacritic.py -s nnls -t 95 < urls > out.nnls

out.lstsq: urls
	/opt/local/bin/python2.7 metacritic.py -s lstsq -t 95 < urls > out.lstsq
