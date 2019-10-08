# Add options to handle UTF-8 charset and to ignore CSS code
# htlatex main.tex "html5mathjax,charset=utf-8" " -chunihft -utf8"
htlatex main.tex "custom"
bibtex main
htlatex main.tex "custom"
cp main.html standalone
cp main.css standalone
zip proposal.zip -r standalone