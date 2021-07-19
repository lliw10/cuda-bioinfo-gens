#! /bin/bash

# Input
scriptName=$0
n=$1
numberOfInputs=$2

if [ -z "$n" -o -z "$numberOfInputs" ];
then
	echo "Necessário informar o [tamanho da rede] e [numero de inputs]."
	echo "Execução: sh" $scriptName  "[tamanho da rede] [numero de inputs]"
	exit 0
fi

# Configuration
nBlocks=16
nThreads=2
nGpus=1
matrixFile="mat_"$n"_"$numberOfInputs"_rand.txt"

# Execution
echo "1. Generate matrix"
echo "Running ./build/BioinfoGeneInteractions -genmat 1 $n $numberOfInputs > $matrixFile"
./build/BioinfoGeneInteractions -genmat 1 $n $numberOfInputs > $matrixFile

echo "2. Processing matrix"
echo "Running: ./build/BioinfoGeneInteractions -f $matrixFile -p $nBlocks $nThreads $nGpus"
./build/BioinfoGeneInteractions -f $matrixFile -p $nBlocks $nThreads $nGpus
