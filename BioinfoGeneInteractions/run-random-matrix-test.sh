#!/bin/bash

checkEmpty() {
	value=$1
    if [ -z "$value" ] 
	then 
		value="-"		
	fi 	
	
	# Return value		
	echo "$value"		
}

testOutputFile=out
testsDir=test-random-matrix
mat=mat
nGPU=1

MIN_GENES=20
MAX_GENES=23
N_SIMULATIONS=20
MAX_INPUT_DEGREE=6

# Time out 10 min
TIME_OUT_SEC=5000s

# Params to parallel graph based
GRID_DIV_SIZE=16
BLOCK_DIV_SIZE=4
NUM_GPUS=2

out=out
parallel=parallel
graph=graph
serial=serial
bns=bns
time=time
cnet=cnet

echo "#Running $N_SIMULATIONS Simulations of $MIN_GENES until $MAX_GENES genes of networks with max input degree $MAX_INPUT_DEGREE"
echo "#NumGenes\tSimulation\tKernelGraphTime\tHostGraphTime\tKernelParallelTime\tHostParallelTime\tSerialTime\tBnsTime\tNumAttractors\tMaxBasinSize"
for n in `seq $MIN_GENES $MAX_GENES`;
do	
	for simId in `seq 1 $N_SIMULATIONS`;
	do
		matrixFileName=mat_"$n"_sim_"$simId".txt 
		matrixFile=$testsDir/$matrixFileName
		
		outParallelFile=$testsDir/$out"_"$parallel"_"$matrixFileName
		outGraphFile=$testsDir/$out"_"$graph"_"$matrixFileName
		outSerialFile=$testsDir/$out"_"$serial"_"$matrixFileName		
		outBnsFile=$testsDir/$out"_"$bns"_"$matrixFileName

		timeParallelFile=$testsDir/$time"_"$parallel"_"$matrixFileName
		timeGraphFile=$testsDir/$time"_"$graph"_"$matrixFileName
		timeSerialFile=$testsDir/$time"_"$serial"_"$matrixFileName
		timeBnsFile=$testsDir/$time"_"$bns"_"$matrixFileName

		# Random matrix generation - 1 matrix of size n
		#./build/BioinfoGeneInteractions -genmat 1 $n $MAX_INPUT_DEGREE > $matrixFile

		# Graph Based
		#/usr/bin/timeout $TIME_OUT_SEC /usr/bin/time -v -o $timeGraphFile ./build/BioinfoGeneInteractions -f $matrixFile -g $GRID_DIV_SIZE $BLOCK_DIV_SIZE > $outGraphFile
		
		# Parallel Find Based
		#/usr/bin/timeout $TIME_OUT_SEC /usr/bin/time -v -o $timeParallelFile ./build/BioinfoGeneInteractions -f $matrixFile -p $GRID_DIV_SIZE $BLOCK_DIV_SIZE $NUM_GPUS > $outParallelFile

		# Simple Serial CPU Based
		#/usr/bin/timeout $TIME_OUT_SEC /usr/bin/timeout $TIME_OUT_SEC /usr/bin/time -v -o $timeSerialFile ./serial/diagrama_estados $matrixFile > $outSerialFile
		
		# BNS SAT Based
		cnetFile=$testsDir/$cnet"_"$matrixFileName
		#./build/BioinfoGeneInteractions -f $matrixFile -cnet > $cnetFile
		#/usr/bin/timeout $TIME_OUT_SEC /usr/bin/time -v -o $timeBnsFile ./bns/bns-sat $cnetFile > $outBnsFile

		# Variables Graph based Kernel
		initKernelTime=`cat $outGraphFile | grep -m 1 "kernelInitializeConectedComponents time (s): "`		
		buildGraphKernelTime=`cat $outGraphFile | grep -m 1 "kernelBuildGraphStates time (s): "`
		kernelLabelCompTime=`cat $outGraphFile | grep -m 1 "kernelLabelComponents time (s): "`
		graphHostTime=`cat $outGraphFile | grep -m 1 "Total matrix execution time (s): "`

		buildGraphKernelTime=$(echo $buildGraphKernelTime|sed 's/kernelBuildGraphStates time (s): / /g')
		initKernelTime=$(echo $initKernelTime|sed 's/kernelInitializeConectedComponents time (s): / /g')
		kernelLabelCompTime=$(echo $kernelLabelCompTime|sed 's/kernelLabelComponents time (s): / /g')	
		graphKernelTime=`echo "$initKernelTime + $buildGraphKernelTime + $kernelLabelCompTime" |bc -l`
		graphKernelTime=`printf "%2.3f" "$graphKernelTime"`

		#graphHostTime=$(echo $graphHostTime|sed 's/Total matrix execution time (s): / /g')
		graphHostTime=`cat $timeGraphFile | grep -m 1 "User time (seconds): "`
		graphHostTime=$(echo $graphHostTime|sed 's/User time (seconds): / /g')

		# Variables parallel based Kernel
		parallelKernelTime=`cat $outParallelFile | grep -m 1 "executeFindAttractorsKernel Device total time (s): "`
		parallelKernelTime=$(echo $parallelKernelTime|sed 's/executeFindAttractorsKernel Device total time (s): / /g')
		parallelKernelTime=`echo "$parallelKernelTime / $NUM_GPUS" |bc -l`
		parallelKernelTime=`printf "%2.3f" "$parallelKernelTime"`
		
		#parallelHostTime=`cat $outParallelFile | grep -m 1 "executeFindAttractorsKernel Host total time (s): "`
		#parallelHostTime=$(echo $parallelHostTime|sed 's/executeFindAttractorsKernel Host total time (s): / /g')
		parallelHostTime=`cat $timeParallelFile | grep -m 1 "User time (seconds): "`
		parallelHostTime=$(echo $parallelHostTime|sed 's/User time (seconds): / /g')

		numberOfAttractors=`cat $outParallelFile | grep -m 1 "executeFindAttractorsKernel Number of Attractors: "`
		numberOfAttractors=$(echo $numberOfAttractors|sed 's/executeFindAttractorsKernel Number of Attractors: / /g')

		maxBasinSize=`cat $outParallelFile | grep -m 1 "executeFindAttractorsKernel Max Basin Size: "`
		maxBasinSize=$(echo $maxBasinSize|sed 's/executeFindAttractorsKernel Max Basin Size: / /g')

		# Variables Simple Serial based Kernel
		serialTime=`cat $timeSerialFile | grep -m 1 "User time (seconds): "`
		serialTime=$(echo $serialTime|sed 's/User time (seconds): / /g')
	
		# Variables BNS SAT based Kernel
		bnsTime=`cat $timeBnsFile | grep -m 1 "User time (seconds): "`
		bnsTime=$(echo $bnsTime|sed 's/User time (seconds): / /g')

		# Check Empty Values
		
		graphKernelTime=$(checkEmpty $graphKernelTime)
		graphHostTime=$(checkEmpty $graphHostTime)
		parallelKernelTime=$(checkEmpty $parallelKernelTime)
		parallelHostTime=$(checkEmpty $parallelHostTime)
		serialTime=$(checkEmpty $serialTime)
		bnsTime=$(checkEmpty $bnsTime)
		numberOfAttractors=$(checkEmpty $numberOfAttractors)
		maxBasinSize=$(checkEmpty $maxBasinSize)

		echo "$n\t$simId\t$graphKernelTime\t$graphHostTime\t$parallelKernelTime\t$parallelHostTime\t$serialTime\t$bnsTime\t$numberOfAttractors\t$maxBasinSize"

	done
done

exit 0
