#!/bin/bash

testsDir=bat-tests
testInputFile=out
mat=mat
testOutputFile=execution-time

for file in $testsDir/mat*txt
do
	matrixSize=`echo "$file" | cut -d'_' -f2`
	matrixSize=`echo "$matrixSize" | cut -d'.' -f1`

	output=$testsDir/$testOutputFile-$mat-$matrixSize.txt

	# Clean file and write header
	# > write
	# >> append
	#
	echo "#Time execution data " > $output
	echo "#N_BLOCKS	N_THREADS	N_GENES	T_PARALLEL_FIND T_BUILD_GRAPH	T_INIT_KERNEL	T_LABEL_COMP	T_TOTAL" >> $output
	echo "#------------		------------		------------	------------	------------	------------" >> $output

	for nblockDiv in 1 2 4 8 16 32 64 128
	do
		for nthreadDiv in 1 2 4 8 16
		do 
			file=$testsDir/$testInputFile-$mat-$matrixSize-$nblockDiv-$nthreadDiv.txt
	
			echo "Generated data to $file in $output"

			blocks=`cat $file | grep -m 1 "Device\[0\] - Number of blocks used: "`
			threads=`cat $file | grep -m 1 "Device\[0\] - Number of threads used: "`
			genes=`cat $file | grep -m 1 "Number of genes: "`

			# Variables Graph based Kernel			
			buildGraphKernelTime=`cat $file | grep -m 1 "kernelBuildGraphStates time (s): "`
			initKernelTime=`cat $file | grep -m 1 "kernelInitializeConectedComponents time (s): "`
			kernelLabelCompTime=`cat $file | grep -m 1 "kernelLabelComponents time (s): "`
			totalTime=`cat $file | grep -m 1 "Total matrix execution time (s): "`

			# Variables Paralell based Kernel
			kernelFindAttTime=`cat $file | grep -m 1 "Device\[0\] - kernelFindAttractorsKernel total time (s): "`

			blocks=$(echo $blocks|sed 's/Device\[0\] - Number of blocks used: / /g')
			threads=$(echo $threads|sed 's/Device\[0\] - Number of threads used: / /g')
			genes=$(echo $genes|sed 's/Number of genes: / /g')
			buildGraphKernelTime=$(echo $buildGraphKernelTime|sed 's/kernelBuildGraphStates time (s): / /g')
			initKernelTime=$(echo $initKernelTime|sed 's/kernelInitializeConectedComponents time (s): / /g')
			kernelLabelCompTime=$(echo $kernelLabelCompTime|sed 's/kernelLabelComponents time (s): / /g')	
			totalTime=$(echo $totalTime|sed 's/Total matrix execution time (s): / /g')
			kernelFindAttTime=$(echo $kernelFindAttTime|sed 's/Device\[0\] - kernelFindAttractorsKernel total time (s): / /g')
		
			echo "$blocks	$threads	$genes	$kernelFindAttTime $buildGraphKernelTime	$initKernelTime	$kernelLabelCompTime	$totalTime" >> $output
			# echo "$blocks	$threads	$totalTime"
			# echo "$blocks	$threads	$totalTime" >> $output
		done
			# Break line
			echo >> $output
	done
done
