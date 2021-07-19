#!/usr/bin/gnuplot
#References:
# http://www.gnuplotting.org/
# http://people.duke.edu/~hpgavin/save.plt
# http://gnuplot.sourceforge.net/docs_4.2/node410.html

set title "Tempo médio de execução entre diferentes configurações" font "Helvetica,12"
set xlabel "Quantidade (blocos ou threads)" font "Helvetica,12"
set ylabel "Tempo(s)" font "Helvetica,12"

plot "./best-gpu-config-test-data.txt" u 5:8 smooth unique w linespoints title 'Graph (Nr. blocks)', "./best-gpu-config-test-data.txt" u 6:8 smooth unique w linespoints title 'Graph (Nr. threads)', "./best-gpu-config-test-data.txt" u ($5*$6):8 w linespoints title 'Graph (N. threads x N. Blocks)'

pause -1

