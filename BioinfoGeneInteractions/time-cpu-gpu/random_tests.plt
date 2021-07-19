#!/usr/bin/gnuplot

# Para executar, no terminal digite:
# gnuplot random_tests_table_in_6.plt

#References:
# http://www.gnuplotting.org/
# http://people.duke.edu/~hpgavin/save.plt
# http://gnuplot.sourceforge.net/docs_4.2/node410.html

# Desvio padrão 
# http://xanadu-1999.blogspot.com.br/2010/01/desvio-padrao-gnuplot.html
# http://pt.numberempire.com/statisticscalculator.php

# Impressao em arquivo
#filename='nome_arquivo.png'
#print 'Generating Image output file: ' .filename
#set terminal png small font arial 
#set output "".filename
# ADICIONE AQUI O PLOT
#set output
#set term pop

input2 = "./random_tests_table_in_2.txt"
input6 = "./random_tests_table_in_6.txt"
input8 = "./random_tests_table_in_8.txt"

# TEMPO DE EXECUÇÃO PARA DIFERENTES METODOS

set title "Tempo de execução para diferentes métodos"
set xlabel "Nº de Genes"
set ylabel "Tempo de Execução (s)"
set key top left
set grid

# Serial
plot "".input2 u 1:11 smooth unique w linespoints title 'Serial k = 2'
rep "".input2 u 1:11:10 smooth unique w errorbars title 'Desv. Padrão'
pause -1
plot "".input6 u 1:11 smooth unique w linespoints title 'Serial k = 6'
rep "".input6 u 1:11:10 smooth unique w errorbars title 'Desv. Padrão'
pause -1
plot "".input8 u 1:11 smooth unique w linespoints title 'Serial k = 8'
rep "".input8 u 1:11:10 smooth unique w errorbars title 'Desv. Padrão'
pause -1
plot "".input2 u 1:11 smooth unique w linespoints title 'Serial k = 2'
rep "".input6 u 1:11 smooth unique w linespoints title 'Serial k = 6'
rep "".input8 u 1:11 smooth unique w linespoints title 'Serial k = 8'
pause -1

# Graph
plot "".input2 u 1:3 smooth unique w linespoints title 'Graph GPU k = 2'
rep "".input2 u 1:3:4 smooth unique w errorbars title 'Desv. Padrão'
pause -1
plot "".input6 u 1:3 smooth unique w linespoints title 'Graph GPU k = 6'
rep "".input6 u 1:3:4 smooth unique w errorbars title  'Desv. Padrão'
pause -1
plot "".input8 u 1:3 smooth unique w linespoints title 'Graph GPU k = 8'
rep "".input8 u 1:3:4 smooth unique w errorbars title  'Desv. Padrão'
pause -1
plot "".input2 u 1:3 smooth unique w linespoints title 'Graph GPU k = 2'
rep "".input6 u 1:3 smooth unique w linespoints title 'Graph GPU k = 6'
rep "".input8 u 1:3 smooth unique w linespoints title 'Graph GPU k = 8'
pause -1

# Parallel
plot "".input2 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 2'
rep "".input2 u 1:7:8 smooth unique w errorbars title 'Desv. Padrão'
pause -1
plot "".input6 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 6'
rep "".input6 u 1:7:8 smooth unique w errorbars title 'Desv. Padrão'
pause -1
plot "".input8 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 8'
rep "".input8 u 1:7:8 smooth unique w errorbars title 'Desv. Padrão'
pause -1
plot "".input2 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 2'
rep "".input6 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 6'
rep "".input8 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 8'
pause -1

# SAT
set logscale y
set xrange [19:29.5]
set yrange [1e-1:1e4]
set ytics (0.3,0.5, 1, 2, 4, 8 10, 20, 40, 80, 160, 320, 640, 1280)
plot "".input2 u 1:13 smooth unique w linespoints title 'SAT k = 2'
rep "".input2 u 1:13:14 smooth unique w errorbars title 'Desv. Padrão'
pause -1
plot "".input6 u 1:13 smooth unique w linespoints title 'SAT k = 6'
rep "".input6 u 1:13:14 smooth unique w errorbars title 'Desv. Padrão'
pause -1
plot "".input8 u 1:13 smooth unique w linespoints title 'SAT k = 8'
rep "".input8 u 1:13:14 smooth unique w errorbars title 'Desv. Padrão'
pause -1
plot "".input2 u 1:13 smooth unique w linespoints title 'SAT k = 2'
rep "".input6 u 1:13 smooth unique w linespoints title 'SAT k = 6'
rep "".input8 u 1:13 smooth unique w linespoints title 'SAT k = 8'
pause -1

# Todos em um unico grafico
set grid
set logscale y
set xrange [19:29.5]
set yrange [1e-1:1e4]
set ytics (0.3,0.5, 1, 2, 4, 8 10, 20, 40, 80, 160, 320, 640, 1280)

set title "Tempo de execução para diferentes métodos k = 2"
plot "".input2 u 1:3 smooth unique w linespoints title 'Graph GPU',  "".input2 u 1:7 smooth unique w linespoints title 'Parallel GPU',  "".input2 u 1:11 smooth unique w linespoints title 'Serial', "".input2 u 1:13 smooth unique w linespoints title 'SAT'
pause -1

set title "Tempo de execução para diferentes métodos k = 6"
plot "".input6 u 1:3 smooth unique w linespoints title 'Graph GPU',  "".input6 u 1:7 smooth unique w linespoints title 'Parallel GPU',  "".input6 u 1:11 smooth unique w linespoints title 'Serial', "".input6 u 1:13 smooth unique w linespoints title 'SAT'
pause -1

set title "Tempo de execução para diferentes métodos k = 8"
plot "".input8 u 1:3 smooth unique w linespoints title 'Graph GPU',  "".input8 u 1:7 smooth unique w linespoints title 'Parallel GPU',  "".input8 u 1:11 smooth unique w linespoints title 'Serial', "".input8 u 1:13 smooth unique w linespoints title 'SAT'
pause -1

# SPEEDUP
reset
clear

set title "Desempenho"
set xlabel "Nº de Genes"
set ylabel "Tempo de execução (s)"
set key top left

set grid
set logscale y
set xrange [19.5:29.5]
set yrange [1e-1:1e4]
set ytics (0.3,0.5, 1, 2, 4, 8 10, 20, 40, 80, 160, 320, 640, 1280)

# Speedup Serial x Parallel GPU (serial/parallel)
plot "".input2 u 1:11 smooth unique w linespoints title 'Serial k = 2'
rep "".input2 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 2' 
rep "".input2 u 1:($11/$7) smooth unique w linespoints title 'Speedup'
pause -1
plot "".input6 u 1:11 smooth unique w linespoints title 'Serial k = 6'
rep "".input6 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 6'
rep "".input6 u 1:($11/$7) smooth unique w linespoints title 'Speedup'
pause -1
plot "".input8 u 1:11 smooth unique w linespoints title 'Serial k = 8'
rep "".input8 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 8' 
rep "".input8 u 1:($11/$7) smooth unique w linespoints title 'Speedup'
pause -1

# Speedup Parallel x Graph (graph/parallel)
plot "".input2 u 1:3 smooth unique w linespoints title 'Graph GPU k = 2'
rep "".input2 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 2'
rep "".input2 u 1:($3/$7) smooth unique w linespoints title 'Speedup'
pause -1

plot "".input6 u 1:3 smooth unique w linespoints title 'Graph GPU k = 6'
rep "".input6 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 6'
rep "".input6 u 1:($3/$7) smooth unique w linespoints title 'Speedup'
pause -1

plot "".input8 u 1:3 smooth unique w linespoints title 'Graph GPU k = 8'
rep "".input8 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 8'
rep "".input8 u 1:($3/$7) smooth unique w linespoints title 'Speedup'
pause -1

# Speedup SAT x Parallel (sat/parallel)
plot "".input2 u 1:13  smooth unique w linespoints title 'SAT k = 2'
rep "".input2 u 1:7  smooth unique w linespoints title 'Parallel GPU k = 2'
rep "".input2 u 1:($13/$7) smooth unique w linespoints title 'Speedup'
pause -1

plot "".input6 u 1:13  smooth unique w linespoints title 'SAT k = 6'
rep "".input6 u 1:7  smooth unique w linespoints title 'Parallel GPU k = 6'
rep "".input6 u 1:($13/$7) smooth unique w linespoints title 'Speedup'
pause -1

plot "".input8 u 1:13 smooth unique w linespoints title 'SAT k = 8'
rep "".input8 u 1:7  smooth unique w linespoints title 'Parallel GPU k = 8'
rep "".input8 u 1:($13/$7) smooth unique w linespoints title 'Speedup'
pause -1


#### ESTATISTICAS SOBRE ATRATORES
# N. de atratores x N. de genes
reset
clear
set title "Nº Médio de Atratores por Rede"
set xlabel "Nº de Genes"
set ylabel ""
set xrange [19.5:29.5]
set grid

set logscale y
set xrange [19.5:29.5]
set yrange [8:1e4+3010]
set ytics (10, 20, 40, 80, 160, 320, 640, 1280,  2560, 5000)

plot "".input2 u 1:15 smooth unique w linespoints title 'Nº. Médio de Atratores k = 2'  
rep "".input6 u 1:15 smooth unique w linespoints title 'Nº. Médio de Atratores k = 6'
rep "".input8 u 1:15 smooth unique w linespoints title 'Nº. Médio de Atratores k = 8'
pause -1

reset
clear
set title "Ocupação das Maiores Bacias de Atração"
set xlabel "Nº de Genes"
set ylabel "Espaço de Estados"
set xrange [18:32]
set format y "%g %%"
set grid

plot "".input2 u 1:((100 * $16)/2**$1) smooth unique w boxes title 'Maior Bacia de Atração k = 2'
rep "".input6 u 1:((100 * $16)/2**$1) smooth unique w boxes title 'Maior Bacia de Atração k = 6'
rep "".input8 u 1:((100 * $16)/2**$1) smooth unique w boxes title 'Maior Bacia de Atração k = 8'
pause -1

