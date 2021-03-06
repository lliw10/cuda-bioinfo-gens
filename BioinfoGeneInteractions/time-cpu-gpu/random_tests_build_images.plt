set terminal png small font arial 

input2 = "./random_tests_table_in_2.txt"
input6 = "./random_tests_table_in_6.txt"
input8 = "./random_tests_table_in_8.txt"

# TEMPO DE EXECUÇÃO PARA DIFERENTES MÉTODOS
set title "Tempo de execução para diferentes métodos "
set xlabel "Nº de Genes"
set ylabel "Tempo de Execução (s)"
set key top left
set grid

# Serial
filename='tempo_serial_in_2.png'
print 'Generating Image output file: ' .filename
set output "".filename
plot "".input2 u 1:11 smooth unique w linespoints title 'Serial k = 2', "".input2 u 1:11:10 smooth unique w errorbars title 'Desv. Padrão'

filename='tempo_serial_in_6.png'
print 'Generating Image output file: ' .filename
set output "".filename
plot "".input6 u 1:11 smooth unique w linespoints title 'Serial k = 6', "".input6 u 1:11:10 smooth unique w errorbars title 'Desv. Padrão'

filename='tempo_serial_in_8.png'
print 'Generating Image output file: ' .filename
set output "".filename
plot "".input8 u 1:11 smooth unique w linespoints title 'Serial k = 8', "".input8 u 1:11:10 smooth unique w errorbars title 'Desv. Padrão'

filename='tempo_serial_in_2_6_8.png'
print 'Generating Image output file: ' .filename
set output "".filename
plot "".input2 u 1:11 smooth unique w linespoints title 'Serial k = 2', "".input6 u 1:11 smooth unique w linespoints title 'Serial k = 6', "".input8 u 1:11 smooth unique w linespoints title 'Serial k = 8'

# Graph
filename='tempo_graph_gpu_in_2.png'
print 'Generating Image output file: ' .filename
set output "".filename
plot "".input2 u 1:3 smooth unique w linespoints title 'Graph GPU k = 2', "".input2 u 1:3:4 smooth unique w errorbars title 'Desv. Padrão'

filename='tempo_graph_gpu_in_6.png'
print 'Generating Image output file: ' .filename
set output "".filename
plot "".input6 u 1:3 smooth unique w linespoints title 'Graph GPU k = 6', "".input6 u 1:3:4 smooth unique w errorbars title  'Desv. Padrão'

filename='tempo_graph_gpu_in_8.png'
print 'Generating Image output file: ' .filename
set output "".filename
plot "".input8 u 1:3 smooth unique w linespoints title 'Graph GPU k = 8', "".input8 u 1:3:4 smooth unique w errorbars title  'Desv. Padrão'

filename='tempo_graph_gpu_in_2_6_8.png'
print 'Generating Image output file: ' .filename
set output "".filename
plot "".input2 u 1:3 smooth unique w linespoints title 'Graph GPU k = 2', "".input6 u 1:3 smooth unique w linespoints title 'Graph GPU k = 6', "".input8 u 1:3 smooth unique w linespoints title 'Graph GPU k = 8'

# Parallel
filename='tempo_parallel_gpu_in_2.png'
print 'Generating Image output file: ' .filename
set output "".filename
plot "".input2 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 2', "".input2 u 1:7:8 smooth unique w errorbars title 'Desv. Padrão'

filename='tempo_parallel_gpu_in_6.png'
print 'Generating Image output file: ' .filename
set output "".filename
plot "".input6 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 6', "".input6 u 1:7:8 smooth unique w errorbars title 'Desv. Padrão'

filename='tempo_parallel_gpu_in_8.png'
print 'Generating Image output file: ' .filename
set output "".filename
plot "".input8 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 8', "".input8 u 1:7:8 smooth unique w errorbars title 'Desv. Padrão'

filename='tempo_parallel_gpu_in_2_6_8.png'
print 'Generating Image output file: ' .filename
set output "".filename
plot "".input2 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 2', "".input6 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 6', "".input8 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 8'


# SAT
#set logscale y
#set xrange [19:29.5]
#set yrange [1e-1:1e4]
#set ytics (0.3,0.5, 1, 2, 4, 8 10, 20, 40, 80, 160, 320, 640, 1280)

filename='tempo_sat_in_2.png'
print 'Generating Image output file: ' .filename
set output "".filename
plot "".input2 u 1:13 smooth unique w linespoints title 'SAT k = 2', "".input2 u 1:13:14 smooth unique w errorbars title 'Desv. Padrão'

filename='tempo_sat_in_6.png'
print 'Generating Image output file: ' .filename
set output "".filename
plot "".input6 u 1:13 smooth unique w linespoints title 'SAT k = 6', "".input6 u 1:13:14 smooth unique w errorbars title 'Desv. Padrão'

filename='tempo_sat_in_8.png'
print 'Generating Image output file: ' .filename
set output "".filename
plot "".input8 u 1:13 smooth unique w linespoints title 'SAT k = 8', "".input8 u 1:13:14 smooth unique w errorbars title 'Desv. Padrão'

filename='tempo_sat_in_2_6_8.png'
print 'Generating Image output file: ' .filename
set output "".filename
plot "".input2 u 1:13 smooth unique w linespoints title 'SAT k = 2', "".input6 u 1:13 smooth unique w linespoints title 'SAT k = 6', "".input8 u 1:13 smooth unique w linespoints title 'SAT k = 8'

# Todos em um unico grafico
set grid
set logscale y
set xrange [19:29.5]
set yrange [1e-1:1e4]
set ytics (0.3,0.5, 1, 2, 4, 8 10, 20, 40, 80, 160, 320, 640, 1280)

filename='tempo_comparacao_todos_in_2.png'
print 'Generating Image output file: ' .filename
set output "".filename
set title "Tempo de execução para diferentes métodos (k = 2)"
plot "".input2 u 1:3 smooth unique w linespoints title 'Graph GPU',  "".input2 u 1:7 smooth unique w linespoints title 'Parallel GPU',  "".input2 u 1:11 smooth unique w linespoints title 'Serial', "".input2 u 1:13 smooth unique w linespoints title 'SAT'

filename='tempo_comparacao_todos_in_6.png'
print 'Generating Image output file: ' .filename
set output "".filename
set title "Tempo de execução para diferentes métodos (k = 6)"
plot "".input6 u 1:3 smooth unique w linespoints title 'Graph GPU',  "".input6 u 1:7 smooth unique w linespoints title 'Parallel GPU',  "".input6 u 1:11 smooth unique w linespoints title 'Serial', "".input6 u 1:13 smooth unique w linespoints title 'SAT'

filename='tempo_comparacao_todos_in_8.png'
print 'Generating Image output file: ' .filename
set output "".filename
set title "Tempo de execução para diferentes métodos (k = 8)"
plot "".input8 u 1:3 smooth unique w linespoints title 'Graph GPU',  "".input8 u 1:7 smooth unique w linespoints title 'Parallel GPU',  "".input8 u 1:11 smooth unique w linespoints title 'Serial', "".input8 u 1:13 smooth unique w linespoints title 'SAT'

# SPEEDUP
reset
clear

set xlabel "Nº de Genes"
set ylabel "Tempo de execução (s)"
set key top left

set grid
set logscale y
set xrange [19.5:29.5]
set yrange [1e-1:1e4]
set ytics (0.3,0.5, 1, 2, 4, 8 10, 20, 40, 80, 160, 320, 640, 1280)

# Speedup Serial x Parallel GPU (serial/parallel)
filename='desemp_serial_parallel_gpu_in_2.png'
print 'Generating Image output file: ' .filename
set title "Desempenho Serial x Parallel GPU (k = 2)"
set output "".filename
plot "".input2 u 1:11 smooth unique w linespoints title 'Serial k = 2', "".input2 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 2'
set output "speedup_".filename
print 'Generating Image output file: ' ."speedup_".filename
set ylabel "Speedup"
plot "".input2 u 1:($11/$7) smooth unique w linespoints title 'Speedup'
set ylabel "Tempo de execução (s)"

filename='desemp_serial_parallel_gpu_in_6.png'
print 'Generating Image output file: ' .filename
set title "Desempenho Serial x Parallel GPU (k = 6)"
set output "".filename
plot "".input6 u 1:11 smooth unique w linespoints title 'Serial k = 6', "".input6 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 6'
set output "speedup_".filename
print 'Generating Image output file: ' ."speedup_".filename
set ylabel "Speedup"
plot "".input6 u 1:($11/$7) smooth unique w linespoints title 'Speedup'
set ylabel "Tempo de execução (s)"

filename='desemp_serial_parallel_gpu_in_8.png'
print 'Generating Image output file: ' .filename
set title "Desempenho Serial x Parallel GPU (k = 8)"
set output "".filename
plot "".input8 u 1:11 smooth unique w linespoints title 'Serial k = 8', "".input8 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 8' 
set output "speedup_".filename
print 'Generating Image output file: ' ."speedup_".filename
set ylabel "Speedup"
plot "".input8 u 1:($11/$7) smooth unique w linespoints title 'Speedup'
set ylabel "Tempo de execução (s)"

# Speedup Parallel x Graph (graph/parallel)

filename='desemp_graph_parallel_gpu_in_2.png'
print 'Generating Image output file: ' .filename
set title "Desempenho Graph GPU x Parallel GPU (k = 2)"
set output "".filename
plot "".input2 u 1:3 smooth unique w linespoints title 'Graph GPU k = 2', "".input2 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 2'
set output "speedup_".filename
print 'Generating Image output file: ' ."speedup_".filename
set ylabel "Speedup"
plot "".input2 u 1:($3/$7) smooth unique w linespoints title 'Speedup'
set ylabel "Tempo de execução (s)"

filename='desemp_graph_parallel_gpu_in_6.png'
print 'Generating Image output file: ' .filename
set title "Desempenho Graph GPU x Parallel GPU (k = 6)"
set output "".filename
plot "".input6 u 1:3 smooth unique w linespoints title 'Graph GPU k = 6', "".input6 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 6'
set output "speedup_".filename
print 'Generating Image output file: ' ."speedup_".filename
set ylabel "Speedup"
plot "".input6 u 1:($3/$7) smooth unique w linespoints title 'Speedup'
set ylabel "Tempo de execução (s)"

filename='desemp_graph_parallel_gpu_in_8.png'
print 'Generating Image output file: ' .filename
set title "Desempenho Graph GPU x Parallel GPU (k = 8)"
set output "".filename
plot "".input8 u 1:3 smooth unique w linespoints title 'Graph GPU k = 8', "".input8 u 1:7 smooth unique w linespoints title 'Parallel GPU k = 8'
set output "speedup_".filename
print 'Generating Image output file: ' ."speedup_".filename
set ylabel "Speedup"
plot "".input8 u 1:($3/$7) smooth unique w linespoints title 'Speedup'
set ylabel "Tempo de execução (s)"

# Speedup SAT x Parallel (sat/parallel)

filename='desemp_sat_parallel_gpu_in_2.png'
print 'Generating Image output file: ' .filename
set title "Desempenho SAT x Parallel GPU (k = 2)"
set output "".filename
plot "".input2 u 1:13  smooth unique w linespoints title 'SAT k = 2', "".input2 u 1:7  smooth unique w linespoints title 'Parallel GPU k = 2'
set output "speedup_".filename
print 'Generating Image output file: ' ."speedup_".filename
set ylabel "Speedup"
plot "".input2 u 1:($13/$7) smooth unique w linespoints title 'Speedup'
set ylabel "Tempo de execução (s)"

filename='desemp_sat_parallel_gpu_in_6.png'
print 'Generating Image output file: ' .filename
set title "Desempenho SAT x Parallel GPU (k = 6)"
set output "".filename
plot "".input6 u 1:13  smooth unique w linespoints title 'SAT k = 6', "".input6 u 1:7  smooth unique w linespoints title 'Parallel GPU k = 6'
set output "speedup_".filename
print 'Generating Image output file: ' ."speedup_".filename
set ylabel "Speedup"
plot "".input6 u 1:($13/$7) smooth unique w linespoints title 'Speedup'
set ylabel "Tempo de execução (s)"

filename='desemp_sat_parallel_gpu_in_8.png'
print 'Generating Image output file: ' .filename
set title "Desempenho SAT x Parallel GPU (k = 8)"
set output "".filename
plot "".input8 u 1:13 smooth unique w linespoints title 'SAT k = 8', "".input8 u 1:7  smooth unique w linespoints title 'Parallel GPU k = 8'
set output "speedup_".filename
print 'Generating Image output file: ' ."speedup_".filename
set ylabel "Speedup"
plot "".input8 u 1:($13/$7) smooth unique w linespoints title 'Speedup'
set ylabel "Tempo de execução (s)"

#### ESTATISTICAS SOBRE ATRATORES
# N. de atratores x N. de genes
reset
clear
set title "Nº Médio de Atratores por Rede "
set xlabel "Nº de Genes"
set ylabel ""
set xrange [19.5:29.5]
set grid

set logscale y
set xrange [19.5:29.5]
set yrange [8:1e4+3010]
set ytics (10, 20, 40, 80, 160, 320, 640, 1280,  2560, 5000)

filename='estatistica_n_medio_atratores.png'
print 'Generating Image output file: ' .filename
set output "".filename
plot "".input2 u 1:15 smooth unique w linespoints title 'Nº. Médio de Atratores k = 2', "".input6 u 1:15 smooth unique w linespoints title 'Nº. Médio de Atratores k = 6', "".input8 u 1:15 smooth unique w linespoints title 'Nº. Médio de Atratores k = 8'

reset
clear
set title "Ocupação das Maiores Bacias de Atração"
set xlabel "Nº de Genes"
set ylabel "Espaço de Estados"
set xrange [18:32]
set format y "%g %%"
set grid
filename='estatistica_tamanho_maior_bacia.png'
print 'Generating Image output file: ' .filename
set output "".filename
plot "".input2 u 1:((100 * $16)/2**$1) smooth unique w boxes title 'Maior Bacia de Atração k = 2', "".input6 u 1:((100 * $16)/2**$1) smooth unique w boxes title 'Maior Bacia de Atração k = 6', "".input8 u 1:((100 * $16)/2**$1) smooth unique w boxes title 'Maior Bacia de Atração k = 8'

