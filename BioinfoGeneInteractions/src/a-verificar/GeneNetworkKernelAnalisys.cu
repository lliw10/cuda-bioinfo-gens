// includes, system
#include <stdio.h>
#include <math.h>
#include "KernelUtils.h"
#include "HostUtils.h"
#include "GeneNetworkKernelAnalisys.h"
#include <list>
#include <vector>
#include <queue>

using namespace std;


texture<int, 1, cudaReadModeElementType> regulationMatrixTextRef;

/**
 *  Verify if the state v2 arrives on  v1 (bitsPathernV1)
 */
__device__ void verifyBeforeStateMult( int regMatrixSizeX,
		int regMatrixSizeY,
		int v1,
		int v2,
		bool *outCreateEdge ) {

	bool stopComputation = false;
	*(outCreateEdge) = true;

	for (int row = 0; (row < regMatrixSizeY) && !stopComputation; row++) {
		int number = v2;
		int bitPathernV2 = 0;
		int bitQPermanent = 0;

		for (int col = regMatrixSizeX - 1; (col >= 0) && (number > 0); col--) {

			int idxMatrix = row * regMatrixSizeY + col;

			// Generate digits in reverse order
			int bitQ = number & 1; // number % 2

			if( row == col ) {
				bitQPermanent = bitQ;
			}

			bitPathernV2 += tex1Dfetch( regulationMatrixTextRef, idxMatrix )
					* bitQ;

			number /= 2;
		}

		// Normalization
		bitPathernV2 = bitPathernV2 == 0 ? bitQPermanent : bitPathernV2 < 0 ? 0
				: 1;

		//		int bitPathernV1 = (v1 / (int) powf( 2, regMatrixSizeY - row - 1 )) % 2;

		long one = 1;

		int bitPathernV1 = (v1 / (one << (regMatrixSizeX - row - 1))) & 1;

		// Fast binary operations:
		// (i & (n-1))  = i % n
		// v1 /	(1 << (regMatrixSizeX - row - 1))

		// Verify result:
		// All bits in v1 must be equals bitPathernV2
		if( bitPathernV1 != bitPathernV2 ) {
			*(outCreateEdge) = false;
			stopComputation = true;
		}
	}

}

/**
 *  Compute all states that arrive on v1 (bitsPathernV1),
 *  that is, all states immediately previous to v1
 */
__global__ void kernelCalculateBeforeStates( int regMatrixSizeX,
		int regMatrixSizeY,
		long v1,
		bool *visitedStates,
		int *outBeforeStates,
		int *outSizeBeforeStates,
		int maxNumberOfStates,
		int offset ) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	int v2 = tid + offset;

	// Process state v2 only if isn't visited yet
	if( v2 < maxNumberOfStates && visitedStates[v2] == false ) {

		bool createEdge = true;

		// Verify if v2 state arrives in v1
		verifyBeforeStateMult( regMatrixSizeX, regMatrixSizeY, v1, v2,
				&createEdge );

		if( createEdge ) {
			// Exists a vertex from v2 to v1
			int nextIndex = atomicAdd( &outSizeBeforeStates[0], 1 );
			outBeforeStates[nextIndex] = v2;
			visitedStates[v2] = true;
		}
	}
}

__global__ void kernelCalculateNextState( int v1,
		int regMatrixSizeX,
		int regMatrixSizeY,
		int *outV2 ) {

	// One row per thread
	int row = blockIdx.x + threadIdx.x;

	int number = v1;
	int bitPathernV2 = 0;
	int bitQPermanent = 0;

	for (int col = regMatrixSizeX - 1; (col >= 0) && (number > 0); col--) {

		int idxMatrix = row * regMatrixSizeY + col;

		// Generate digits in reverse order
		int bitQ = number & 1; // number % 2

		if( row == col ) {
			bitQPermanent = bitQ;
		}

		bitPathernV2 += tex1Dfetch( regulationMatrixTextRef, idxMatrix ) * bitQ;

		number /= 2;
	}

	// Normalization
	bitPathernV2 = bitPathernV2 == 0 ? bitQPermanent : bitPathernV2 < 0 ? 0 : 1;

	// Exists an arc v1 -> outV2
	//	int value = bitPathernV2 * powf( 2, (regMatrixSizeY - row - 1) );
	long one = 1;
	int value = bitPathernV2 * (one << (regMatrixSizeX - row - 1));

	atomicAdd( &outV2[0], value );

}

int executeKernelCalculateNextState( long long int v1,
		int *regulationMatrix,
		int regMatrixSizeX,
		int regMatrixSizeY ) {

	printf( "....Start: executeKernelCalculateNextStage \n\n" );

	// Part 1 of 6: define kernel configuration
	int numBlocksOnGridX = 1;
	int numBlocksOnGridY = 1;

	int numThreadsPerBlockX = regMatrixSizeX;
	int numThreadsPerBlockY = 1;

	// Number of blocks on grid
	dim3 dimGrid( numBlocksOnGridX, numBlocksOnGridY );

	// Number of threads per block
	dim3 dimBlock( numThreadsPerBlockX, numThreadsPerBlockY );

	int sizeOutV2 = 1;
	int sizeRegulationMatrix = (regMatrixSizeX * regMatrixSizeY);

	size_t memSizeOutV2 = sizeOutV2 * sizeof(int);
	size_t memSizeRegulationMatrix = sizeRegulationMatrix * sizeof(int);

	printf( "Number of blocks used: %d x %d = %d\n", numBlocksOnGridX,
			numBlocksOnGridY, (numBlocksOnGridX * numBlocksOnGridY) );

	printf( "Number of threads used: %d x %d = %d\n", numThreadsPerBlockX,
			numThreadsPerBlockY, (numThreadsPerBlockX * numThreadsPerBlockY) );

	// Part 2 of 6: allocate host memory
	int *outV2 = getPointerToMatrix( memSizeOutV2 );
	outV2[0] = 0;

	int *outV2Dev = NULL;
	int *regulationMatrixDev = NULL;

	// Regulation matrix allocation memory
	cudaMalloc( (void **) &regulationMatrixDev, memSizeRegulationMatrix );
	cudaBindTexture( 0, regulationMatrixTextRef, regulationMatrixDev,
			memSizeRegulationMatrix );
	cudaMemcpy( regulationMatrixDev, regulationMatrix, memSizeRegulationMatrix,
			cudaMemcpyHostToDevice );

	checkCUDAError( "regulationMatrixDev Memory Allocation" );

	cudaMalloc( (void **) &outV2Dev, memSizeOutV2 );
	cudaMemcpy( outV2Dev, outV2, memSizeOutV2, cudaMemcpyHostToDevice );

	checkCUDAError( "outV2Dev Memory Allocation" );

	// Part 5 of 6: launch kernel
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );

	kernelCalculateNextState<<< dimGrid , dimBlock>>>(v1,regMatrixSizeX,
			regMatrixSizeY, outV2Dev );

	// block until the device has completed
	cudaThreadSynchronize();

	// Compute time of kernel execution in milliseconds
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf( "time %f s \n", (time / 1000) );

	checkCUDAError( "Kernel execution" );

	cudaMemcpy( outV2, outV2Dev, memSizeOutV2, cudaMemcpyDeviceToHost );

	// Check for any CUDA errors
	checkCUDAError( "Memory copy" );

	// free device memory
	cudaFree( outV2Dev );
	cudaFree( regulationMatrixDev );
	cudaUnbindTexture( regulationMatrixTextRef );

	return outV2[0];
}

int MAX_POSSIBLE_BEFORE_NODES = 10000;

bool * allocVisitedStatesOnCuda( size_t memVisitedStates ) {
	bool *visitedStates = NULL;
	cudaHostAlloc( (void **) &visitedStates, memVisitedStates,
			cudaHostAllocMapped );
	return visitedStates;
}

long executeKernelAccountBasinOfAtraction( list<long> atractorsList,
		bool *visitedStates,
		int sizeStatesVisited,
		int *regulationMatrix,
		int regMatrixSizeX,
		int regMatrixSizeY ) {

	printf( "....Start: executeKernelAccountBasinOfAtraction \n\n" );

	// Part 1: define kernel configuration
	long long int maxNumberOfStates = (int) pow( 2.0, regMatrixSizeX );

	long long int numBlocksOnGrid = 0;
	long long int numThreadsPerBlock = 0;
	long long int numIterations = 0;
	long long int restThreadsToExecute = 0;

	calculateKernelLaunchConfiguration( maxNumberOfStates, &numThreadsPerBlock,
			&numBlocksOnGrid, &numIterations, &restThreadsToExecute );

	long long int totalNumIterationsKernel = numIterations
			+ (restThreadsToExecute <= 0 ? 0 : 1);

	int sizeMatrix = (regMatrixSizeX * regMatrixSizeY);
	int sizeBeforeStatesNum = 1;

	size_t memMatrixSize = sizeMatrix * sizeof(int);
	size_t memSizeNumBeforeStates = sizeBeforeStatesNum * sizeof(int);
	size_t memBeforeStates = MAX_POSSIBLE_BEFORE_NODES * sizeof(int);
	size_t memVisitedStates = maxNumberOfStates * sizeof(bool);

	dim3 dimBlock( numThreadsPerBlock );
	dim3 dimGrid( numBlocksOnGrid );

	printf( "Number of genes: %d \n", regMatrixSizeX );
	printf( "Size solution (2^%d) = %lld \n", regMatrixSizeX, maxNumberOfStates );
	// 1MB = 1024^2
	printf( "Number of blocks used: %lld \n", numBlocksOnGrid );
	printf( "Number of threads used: %lld \n", numThreadsPerBlock );
	printf( "Iterations: %lld + (rest: %lld) = %lld \n", numIterations,
			restThreadsToExecute, totalNumIterationsKernel );

	// Host memory allocation
	int *beforeStates = getPointerToMatrix( memBeforeStates );
	int *numBeforeStates = getPointerToMatrix( memSizeNumBeforeStates );

	if( beforeStates == NULL || beforeStates <= 0 ) {
		printf( "Host error: beforeStages Memory Allocation \n" );
		exit( 0 );
	}
	numBeforeStates[0] = 0;
	// TODO: Verificar criação de um kernel para inicializar beforeStates
	for (int i = 0; i < MAX_POSSIBLE_BEFORE_NODES; i++) {
		beforeStates[i] = -1;
	}

	// Device out allocation memory
	int *regulationMatrixDev = NULL;
	bool *visitedStatesDev = NULL;
	int *outBeforeStatesDev = NULL;
	int *outNumBeforeStatesDev = NULL;

	cudaMalloc( (void **) &regulationMatrixDev, memMatrixSize );
	cudaBindTexture( 0, regulationMatrixTextRef, regulationMatrixDev,
			memMatrixSize );
	cudaMemcpy( regulationMatrixDev, regulationMatrix, memMatrixSize,
			cudaMemcpyHostToDevice );

	checkCUDAError( "Memory Allocation" );

	//------- Allocate Zero Copy memory -------
	//	cudaHostAlloc( (void **) &visitedStates, memVisitedStates,
	//			cudaHostAllocMapped );
	cudaHostGetDevicePointer( (void **) &visitedStatesDev,
			(void *) visitedStates, 0 );

	checkCUDAError( "visitedStatesDev Memory Allocation" );
	//	-------

	cudaMalloc( (void **) &outBeforeStatesDev, memBeforeStates );
	checkCUDAError( "outBeforeStatesDev Memory Allocation" );

	cudaMalloc( (void **) &outNumBeforeStatesDev, memSizeNumBeforeStates );
	checkCUDAError( "outBeforeStatesDev Memory Allocation" );

	list<long>::iterator it;
	for (it = atractorsList.begin(); it != atractorsList.end(); ++it) {
		visitedStates[*it] = true;
	}

	long countConectedComponents = 0;

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );

	while( atractorsList.size() > 0 ) {

		int v1 = atractorsList.front();
		atractorsList.pop_front();
		int offset = 0;

		for (int i = 0; i < totalNumIterationsKernel; i++) {

			// Part 5 of 6: launch kernel
			//			cudaEvent_t start, stop;
			//			float time;
			//			cudaEventCreate( &start );
			//			cudaEventCreate( &stop );
			//			cudaEventRecord( start, 0 );

			numBeforeStates[0] = 0;

			cudaMemcpy( outBeforeStatesDev, beforeStates, memBeforeStates,
					cudaMemcpyHostToDevice );
			checkCUDAError( "outBeforeStatesDev Memory Allocation" );

			cudaMemcpy( outNumBeforeStatesDev, numBeforeStates,
					memSizeNumBeforeStates, cudaMemcpyHostToDevice );
			checkCUDAError( "outNumBeforeStatesDev Memory Allocation" );

			kernelCalculateBeforeStates <<< dimGrid , dimBlock>>>( regMatrixSizeX, regMatrixSizeY, v1,
					visitedStatesDev, outBeforeStatesDev, outNumBeforeStatesDev,
					maxNumberOfStates, offset );

			cudaThreadSynchronize();

			offset += dimBlock.x * dimGrid.x;

			checkCUDAError( "Kernel execution" );

			cudaMemcpy( beforeStates, outBeforeStatesDev, memBeforeStates,
					cudaMemcpyDeviceToHost );
			cudaMemcpy( numBeforeStates, outNumBeforeStatesDev,
					memSizeNumBeforeStates, cudaMemcpyDeviceToHost );
			cudaMemcpy( visitedStates, visitedStatesDev, memVisitedStates,
					cudaMemcpyDeviceToHost );

			//			// Compute time of kernel execution in milliseconds
			//			cudaEventRecord( stop, 0 );
			//			cudaEventSynchronize( stop );
			//			cudaEventElapsedTime( &time, start, stop );
			//			cudaEventDestroy( start );
			//			cudaEventDestroy( stop );
			//
			//						printf( "time %f s \n", (time / 1000) );

			//			printf( "Number of before states of %d: %d\n", v1,
			//					numBeforeStates[0] );

			countConectedComponents += numBeforeStates[0];
			for (int i = 0; i < numBeforeStates[0]; i++) {
				atractorsList.push_back( beforeStates[i] );
				beforeStates[i] = -1;
				//				 printf( "before[%d]: %d \t ", i, beforeStates[i] );
				//				 printf( "visitedState[%d]: %d\n", beforeStates[i], visitedStates[beforeStates[i]] );
			}
		}
	}

	// free device memory
	cudaFree( outBeforeStatesDev );
	cudaFree( outNumBeforeStatesDev );
	cudaFree( regulationMatrixDev );
	cudaUnbindTexture( regulationMatrixTextRef );

	// Compute time of kernel execution in milliseconds
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	printf( "Total Before Nodes reached: %ld \n", countConectedComponents );
	printf( "time %f s \n\n", (time / 1000) );

	checkCUDAError( "executeKernelAccountBasinOfAtraction Cuda Free" );

	return countConectedComponents;
}
