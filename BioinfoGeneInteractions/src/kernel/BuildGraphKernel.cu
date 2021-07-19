// includes, system
#include <stdio.h>
#include <math.h>

#include "KernelUtils.h"
#include "HostUtils.h"
#include "BuildGraphKernel.h"

__global__ void kernelBuildGraphStates( int regMatrixSizeX,
		int regMatrixSizeY,
		int *regulationMatrix,
		long long int *countSolutions,
		long long int sizeCountSolutions,
		long long int offset ) {

	long long int tid = blockDim.x * blockIdx.x + threadIdx.x;
	long long int v1 = tid + offset;

	if( v1 < sizeCountSolutions ) {
		long long int v2 = 0;

		for (int row = 0; row < regMatrixSizeY; row++) {
			long long int number = v1;
			int bitPathernV2 = 0;
			int bitQPermanent = 0;

			for (int col = regMatrixSizeX - 1; (col >= 0) && (number > 0); col--) {

				int idxMatrix = row * regMatrixSizeY + col;

				// Generate digits in reverse order
				//				long long int bitQ = number % 2;
				int bitQ = number & 1;

				if( row == col ) {
					bitQPermanent = bitQ;
				}

				//				bitPathernV2 += tex1Dfetch( regulationMatrixTextRef, idxMatrix )
				//						* bitQ;
				bitPathernV2 += regulationMatrix[idxMatrix] * bitQ;

				// [Desempenho] Operacao realizada: number /= 2;
				number = number >> 1;
			}

			// Normalization
			bitPathernV2 = bitPathernV2 == 0 ? bitQPermanent
					: bitPathernV2 < 0 ? 0 : 1;

			// Exists an arc v1 -> outV2
			// [Desempenho] Operacao realizada:
			// v2 += bitPathernV2 * (long long int) powf( 2, (regMatrixSizeX - row	- 1) );
			long long int one = 1;
			v2 += bitPathernV2 * (one << (regMatrixSizeX - row - 1));
		}
		countSolutions[tid] = v2;
	}
}

long long int * executeBuildGraphKernel( int *regulationMatrix,
		int m,
		int n,
		unsigned long long int *outSizeCountSolutions ) {

	printf( "\n----------- Start: executeBuildGraphKernel --------- \n\n" );

	// Part 1: define kernel configuration
	int sizeMatrix = (m * n);

	unsigned long long int sizeCountSolutions = (unsigned long long int) pow( 2.0, m );
	*(outSizeCountSolutions) = sizeCountSolutions;

	long long int numBlocksOnGrid = 0;
	long long int numThreadsPerBlock = 0;
	long long int numIterations = 0;
	long long int restThreadsToExecute = 0;

	calculateKernelLaunchConfiguration( sizeCountSolutions,
			&numThreadsPerBlock, &numBlocksOnGrid, &numIterations,
			&restThreadsToExecute );

	long long int totalNumIterationsKernel = numIterations
			+ (restThreadsToExecute <= 0 ? 0 : 1);

	long long int memMatrixSize = sizeMatrix * (long long int) sizeof(int);
	long long int memCountSolutions = sizeCountSolutions
			* (long long int) sizeof(long long int);

	dim3 dimBlock( numThreadsPerBlock );
	dim3 dimGrid( numBlocksOnGrid );

	printf( "Number of genes: %d \n", m );
	printf( "Size solution (2^%d) = %lld \n", m, sizeCountSolutions );
	// 1MB = 1024^2
	printf( "Device memory allocated (%lld  + %lld) = %f MB \n", memMatrixSize,
			memCountSolutions, ((double) (memMatrixSize + memCountSolutions)
					/ (1024.0 * 1024.0)) );

	printBlockAndThreadConfig();
	printf( "Number of blocks used: %lld \n", numBlocksOnGrid );
	printf( "Number of threads used: %lld \n", numThreadsPerBlock );
	printf( "Iterations: %lld + (rest: %lld) = %lld \n", numIterations,
			restThreadsToExecute, totalNumIterationsKernel );

	// Part 2: allocate device memory
	long long int *graph;
	graph = NULL;
	graph = (long long int*) malloc( memCountSolutions );
	if( graph == NULL || graph <= 0 ) {
		printf( "Host error: graph Memory Allocation \n" );
		exit( 0 );
	}

	int *regulationMatrixDev = NULL;

	cudaMalloc( (void **) &regulationMatrixDev, memMatrixSize );

	// Part 4 of 6: host to device copy
	cudaMemcpy( regulationMatrixDev, regulationMatrix, memMatrixSize,
			cudaMemcpyHostToDevice );

	checkCUDAError( "Matrix Memory Allocation" );

	long long int offset = 0;

	long long int *countSolutionsPartialDev = NULL;
	long long int previousMemCountSolutionsToProcess = -1;

	cudaEvent_t startTotalTime, stopTotalTime;
	float totalTime;
	cudaEventCreate( &startTotalTime );
	cudaEventCreate( &stopTotalTime );
	cudaEventRecord( startTotalTime, 0 );

	for (long long int i = 0; i < totalNumIterationsKernel; i++) {

		// Part 5 of 6: launch kernel
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate( &start );
		cudaEventCreate( &stop );
		cudaEventRecord( start, 0 );

		int sizeCountSolutionsToProcess;
		if( i == (totalNumIterationsKernel - 1) && restThreadsToExecute > 0 ) {
			// Last iteration
			sizeCountSolutionsToProcess = restThreadsToExecute;

			calculateKernelLaunchConfiguration( sizeCountSolutionsToProcess,
					&numThreadsPerBlock, &numBlocksOnGrid, &numIterations,
					&restThreadsToExecute );

			dim3 dimBlock2( numThreadsPerBlock );
			dim3 dimGrid2( numBlocksOnGrid );

			dimBlock = dimBlock2;
			dimGrid = dimGrid2;

		} else {
			sizeCountSolutionsToProcess = (dimGrid.x * dimBlock.x);
		}

		long long int memCountSolutionsToProcess = sizeCountSolutionsToProcess
				* (long long int) sizeof(long long int);

		long long int *countSolutionsPartial = (graph + offset);

		// Only allocate memory when need
		if( previousMemCountSolutionsToProcess != memCountSolutionsToProcess ) {
			if( countSolutionsPartialDev != NULL ) {
				cudaFree( countSolutionsPartialDev );
				checkCUDAError( "countSolutionsPartialDev [partial]  Free" );
			}

			printf( "CountSolutionsPartial [MEM: %f MB] \n",
					(((double) memCountSolutionsToProcess) / (1024.0 * 1024.0)) );

			cudaMalloc( (void **) &countSolutionsPartialDev,
					memCountSolutionsToProcess );
			checkCUDAError( "countSolutionsPartialDev [malloc]  Free" );

			previousMemCountSolutionsToProcess = memCountSolutionsToProcess;
		}

		cudaMemcpy( countSolutionsPartialDev, countSolutionsPartial,
				memCountSolutionsToProcess, cudaMemcpyHostToDevice );

		checkCUDAError( "countSolutionsPartialDev Memory Allocation" );

		kernelBuildGraphStates<<< dimGrid, dimBlock>>>(m, n, regulationMatrixDev, countSolutionsPartialDev, sizeCountSolutions, offset);

		cudaError_t errAsync = cudaThreadSynchronize();
		if( cudaSuccess != errAsync ) {
			fprintf( stderr, "Cuda Async Error kernelBuildGraphStates %s.\n",
					cudaGetErrorString( errAsync ) );
		}
		checkCUDAError( "Kernel execution" );

		// copiar para o host usando + offset para os indices
		cudaMemcpy( countSolutionsPartial, countSolutionsPartialDev,
				memCountSolutionsToProcess, cudaMemcpyDeviceToHost );

		checkCUDAError( "countSolutionsPartial MemCpy" );
		offset += dimBlock.x * dimGrid.x;

		// Compute time of kernel execution in milliseconds
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &time, start, stop );
		cudaEventDestroy( start );
		cudaEventDestroy( stop );

		if( i % 999 == 0 ) {
			printf( "Iteration %lld/%lld time (s): %f \n", (i + 1),
					totalNumIterationsKernel, (time / 1000) );
		}

	}

	if( countSolutionsPartialDev != NULL ) {
		cudaFree( countSolutionsPartialDev );
		checkCUDAError( "countSolutionsPartialDev [end] Free" );
	}

	// free device memory
	cudaFree( regulationMatrixDev );
	checkCUDAError( "regulationMatrixDev Free" );

	// Calculate total time for algorithm execution
	cudaEventRecord( stopTotalTime, 0 );
	cudaEventSynchronize( stopTotalTime );
	cudaEventElapsedTime( &totalTime, startTotalTime, stopTotalTime );
	cudaEventDestroy( startTotalTime );
	cudaEventDestroy( stopTotalTime );

	printf( "kernelBuildGraphStates time (s): %f \n", (totalTime / 1000) );

	printf( "............End: executeGeneNetworkKernel \n\n" );

	return graph;

}
