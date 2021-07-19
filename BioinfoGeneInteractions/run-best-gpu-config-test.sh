#!/bin/bash

# This automated test run with different setup GPU configurations of number
# of blocks and threads.

checkEmpty() {
	value=$1
    if [ -z "$value" ] 
	then 
		value="-"		
	fi 	
	
	# Return value		
	echo "$value"		
}

testsDir=test-best-gpu-config
mat=mat

MIN_GENES=15
MAX_GENES=20
N_REPETITIONS=1
MAX_INPUT_DEGREE=2

# Time out 10 min
TIME_OUT_SEC=120s

# Params to parallel graph based
ONE_GPU=1
DOUBLE_GPU=2

out=out
parallel=parallel
parallelDouble=parallelDouble
graph=graph
time=time

totalSimulations=`echo "$MAX_GENES - $MIN_GENES + 1" |bc -l`

echo "# Graph and parallel execution kernel parameters simulation"
echo "# Running $totalSimulations simulations of $MIN_GENES until $MAX_GENES genes"
echo "# - Networks have max input degree $MAX_INPUT_DEGREE"
echo "# - Simulations have $N_REPETITIONS repetitions to each configuration of (4096 2048 1024 512 256 128 64 32 16 8 4 2 1)divBlocks x (2 4 8 16)divThreads"

echo "#"
echo "#Simulation\tRepetition\tDivBlocks\tDivThreads\tNumBlocks\tNumThreads\tNumGenes\tKernelGraphTime\tHostGraphTime\tHostGraphProgramTime\tKernelParallelTime\tHostParallelTime\tHostParallelProgramTime\tKernelParallelDoubleTime\tHostParallelDoubleTime\tHostParallelDoubleProgramTime\tNumAttractors\tMaxBasinSize"

for n in `seq $MIN_GENES $MAX_GENES`;
do		
	simId=0
	for gridDivSize in 4096 2048 1024 512 256 128 64 32 16 8 4 2 1
	do
		for blockDivSize in 2 4 8 16
		do	
			simId=`expr $simId + 1`
			matrixFileName=mat_"$n"_conf_"$gridDivSize"_"$blockDivSize".txt
			matrixFile=$testsDir/$matrixFileName

			# Random matrix generation - one matrix of size n
			./build/BioinfoGeneInteractions -genmat 1 $n $MAX_INPUT_DEGREE > $matrixFile

			for it in `seq 1 $N_REPETITIONS`;
			do
				outGraph=$testsDir/$out"_"$graph"_"$matrixFileName
				outParallel=$testsDir/$out"_"$parallel"_"$matrixFileName
				outParallelDouble=$testsDir/$out"_"$parallelDouble"_"$matrixFileName

				timeGraphFile=$testsDir/$time"_"$graph"_"$matrixFileName
				timeParallelFile=$testsDir/$time"_"$parallel"_"$matrixFileName
				timeParallelDoubleFile=$testsDir/$time"_"$parallelDouble"_"$matrixFileName

				# Execution: Graph Based
				/usr/bin/timeout $TIME_OUT_SEC /usr/bin/time -v -o $timeGraphFile ./build/BioinfoGeneInteractions -f $matrixFile -g $gridDivSize $blockDivSize > $outGraph
		
				# Execution: Parallel Find Based With one GPU
				/usr/bin/timeout $TIME_OUT_SEC /usr/bin/time -v -o $timeParallelFile ./build/BioinfoGeneInteractions -f $matrixFile -p $gridDivSize $blockDivSize $ONE_GPU > $outParallel

				# Execution: Parallel Find Based With two GPUs
				/usr/bin/timeout $TIME_OUT_SEC /usr/bin/time -v -o $timeParallelDoubleFile ./build/BioinfoGeneInteractions -f $matrixFile -p $gridDivSize $blockDivSize $DOUBLE_GPU > $outParallelDouble

				# Extract execution data
				# General Variables
				blocks=`cat $outParallel | grep -m 1 "Device\[0\] - Number of blocks used: "`
				blocks=$(echo $blocks|sed 's/Device\[0\] - Number of blocks used: / /g')

				threads=`cat $outParallel | grep -m 1 "Device\[0\] - Number of threads used: "`
				threads=$(echo $threads|sed 's/Device\[0\] - Number of threads used: / /g')

				numberOfAttractors=`cat $outParallel | grep -m 1 "executeFindAttractorsKernel Number of Attractors: "`
				numberOfAttractors=$(echo $numberOfAttractors|sed 's/executeFindAttractorsKernel Number of Attractors: / /g')

				maxBasinSize=`cat $outParallel | grep -m 1 "executeFindAttractorsKernel Max Basin Size: "`
				maxBasinSize=$(echo $maxBasinSize|sed 's/executeFindAttractorsKernel Max Basin Size: / /g')

				#########################################
				# Graph Time Variables
				#########################################

				initKernelTime=`cat $outGraph | grep -m 1 "kernelInitializeConectedComponents time (s): "`		
				initKernelTime=$(echo $initKernelTime|sed 's/kernelInitializeConectedComponents time (s): / /g')

				buildGraphKernelTime=`cat $outGraph | grep -m 1 "kernelBuildGraphStates time (s): "`
				buildGraphKernelTime=$(echo $buildGraphKernelTime|sed 's/kernelBuildGraphStates time (s): / /g')

				kernelLabelCompTime=`cat $outGraph | grep -m 1 "kernelLabelComponents time (s): "`
				kernelLabelCompTime=$(echo $kernelLabelCompTime|sed 's/kernelLabelComponents time (s): / /g')
				graphKernelTime=`echo "$initKernelTime + $buildGraphKernelTime + $kernelLabelCompTime" |bc -l`
				graphKernelTime=`printf "%2.3f" "$graphKernelTime"`

				graphHostTime=`cat $outGraph | grep -m 1 "Total matrix execution time (s): "`
				graphHostTime=$(echo $graphHostTime|sed 's/Total matrix execution time (s): / /g')

				graphHostTimeProgram=`cat $timeGraphFile | grep -m 1 "User time (seconds): "`
				graphHostTimeProgram=$(echo $graphHostTimeProgram|sed 's/User time (seconds): / /g')

				#########################################
				# Parallel Kernel Time with one GPU
				#########################################

				parallelKernelTime=`cat $outParallel | grep -m 1 "executeFindAttractorsKernel Device total time (s): "`
				parallelKernelTime=$(echo $parallelKernelTime|sed 's/executeFindAttractorsKernel Device total time (s): / /g')
		
				parallelHostTime=`cat $outParallel | grep -m 1 "executeFindAttractorsKernel Host total time (s): "`
				parallelHostTime=$(echo $parallelHostTime|sed 's/executeFindAttractorsKernel Host total time (s): / /g')
				
				parallelHostTimeProgram=`cat $timeParallelFile | grep -m 1 "User time (seconds): "`
				parallelHostTimeProgram=$(echo $parallelHostTime|sed 's/User time (seconds): / /g')

				#########################################
				# Parallel Kernel Time with two GPUs
				#########################################

				parallelDoubleKernelTime=`cat $outParallelDouble | grep -m 1 "executeFindAttractorsKernel Device total time (s): "`
				parallelDoubleKernelTime=$(echo $parallelDoubleKernelTime|sed 's/executeFindAttractorsKernel Device total time (s): / /g')
				parallelDoubleKernelTime=`echo "$parallelDoubleKernelTime / $DOUBLE_GPU" |bc -l`
				parallelDoubleKernelTime=`printf "%2.3f" "$parallelDoubleKernelTime"`

				parallelDoubleHostTime=`cat $outParallelDouble | grep -m 1 "executeFindAttractorsKernel Host total time (s): "`
				parallelDoubleHostTime=$(echo $parallelDoubleHostTime|sed 's/executeFindAttractorsKernel Host total time (s): / /g')

				parallelDoubleHostTimeProgram=`cat $timeParallelDoubleFile | grep -m 1 "User time (seconds): "`
				parallelDoubleHostTimeProgram=$(echo $parallelDoubleHostTimeProgram|sed 's/User time (seconds): / /g')

				#########################################
				# Print
				#########################################

				blockDivSize=$(checkEmpty $blockDivSize)
				gridDivSize=$(checkEmpty $gridDivSize)
				blocks=$(checkEmpty $blocks)
				threads=$(checkEmpty $threads)
				numberOfAttractors=$(checkEmpty $numberOfAttractors)
				maxBasinSize=$(checkEmpty $maxBasinSize)

				graphKernelTime=$(checkEmpty $graphKernelTime)
				graphHostTime=$(checkEmpty $graphHostTime)
				graphHostTimeProgram=$(checkEmpty $graphHostTimeProgram)

				parallelKernelTime=$(checkEmpty $parallelKernelTime)
				parallelHostTime=$(checkEmpty $parallelHostTime)
				parallelHostTimeProgram=$(checkEmpty $parallelHostTimeProgram)
				
				parallelDoubleKernelTime=$(checkEmpty $parallelDoubleKernelTime)
				parallelDoubleHostTime=$(checkEmpty $parallelDoubleHostTime)
				parallelDoubleHostTimeProgram=$(checkEmpty $parallelDoubleHostTimeProgram)

				echo "$simId\t$it\t$gridDivSize\t$blockDivSize\t$blocks\t$threads\t$n\t$graphKernelTime\t$graphHostTime\t$graphHostTimeProgram\t$parallelKernelTime\t$parallelHostTime\t$parallelHostTimeProgram\t$parallelDoubleKernelTime\t$parallelDoubleHostTime\t$parallelDoubleHostTimeProgram\t$numberOfAttractors\t$maxBasinSize"
			done
		done
	done
done
exit 0


