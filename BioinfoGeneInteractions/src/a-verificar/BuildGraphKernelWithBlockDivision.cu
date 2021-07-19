// includes, system
#include <stdio.h>
#include <math.h>
#include "KernelUtils.h"
#include "HostUtils.h"
#include "BuildGraphKernelWithBlockDivision.h"
#include "Lock.h"

const int DIV_IN_BLOCK_FACTOR_STATIC = 64;

__global__ void kernelBuildGraphStatesWithBlockDivision( int m,
		int n,
		int *regulationMatrix,
		long long int *countSolutions,
		long long int sizeCountSolutions,
		long long int offset ) {

	long long int tid = (blockDim.x * blockIdx.x + threadIdx.x) / m;

	long long int v1 = tid + offset;

	__shared__ unsigned int v2[DIV_IN_BLOCK_FACTOR_STATIC];
	int sharedIdx = threadIdx.x / m;
	v2[sharedIdx] = 0;

	if( v1 < sizeCountSolutions ) {
		// Inicio do laço externo
		{
			int row = threadIdx.x % m;
			long long int number = v1;
			int bitPathernV2 = 0;
			int bitQPermanent = 0;

			for (int col = m - 1; (col >= 0) && (number > 0); col--) {

				int idxMatrix = row * n + col;

				// Generate digits in reverse order
				// long long int bitQ = number % 2;
				int bitQ = number & 1;

				if( row == col ) {
					bitQPermanent = bitQ;
				}

				bitPathernV2 += regulationMatrix[idxMatrix] * bitQ;

				// [Desempenho] Operacao realizada: number /= 2;
				number = number >> 1;
			}

			// Normalization
			bitPathernV2 = bitPathernV2 == 0 ? bitQPermanent
					: bitPathernV2 < 0 ? 0 : 1;

			// Necessário compartilhamento de dados entre as threads.
			// Para consolidar o próximo estado

			// Exists an arc v1 -> outV2
			// [Desempenho] Operacao realizada:
			// v2 += bitPathernV2 * (long long int) powf( 2, (regMatrixSizeX - row	- 1) );
			long long int one = 1;
			//			v2 += bitPathernV2 * (one << (n - row - 1));

			atomicAdd( &(v2[sharedIdx]),
					(bitPathernV2 * (one << (m - row - 1))) );
		}
		// Exists a vertex from v1 to v2
		//countSolutions[tid] = v2;
		countSolutions[tid] = v2[sharedIdx];
	}
}

void calculateKernelLaunchConfiguration2( long long int m,
		long long int sizeOfData,
		long long int *outThreadsPerBlock,
		long long int *outBlocksOnGrid,
		long long int *outNumIterations,
		long long int *outRestThreadsToExecute ) {

	cudaDeviceProp deviceProp;
	int device;

	int deviceCount;
	cudaGetDeviceCount( &deviceCount );

	long long int maxGridSize;
	long long int maxThreadsPerBlock;

	for (device = 0; device < deviceCount; ++device) {
		cudaGetDeviceProperties( &deviceProp, device );
		if( device <= deviceCount ) {
			if( deviceProp.major == 9999 && deviceProp.minor == 9999 ) {
				printf( "There is no device supporting CUDA.\n" );
			}
		}

		cudaGetDevice( &device );
		cudaGetDeviceProperties( &deviceProp, device );

		int blockDivSize = (blockDiv <= 1) ? 2 : blockDiv;
		int threadDivSize = (threadDiv <= 1) ? 1 : threadDiv;

		maxThreadsPerBlock = deviceProp.maxThreadsPerBlock / threadDivSize;
		maxGridSize = deviceProp.maxGridSize[0] / blockDivSize;

		break;
	}

	int statesPerBlock = maxThreadsPerBlock / m;
	maxThreadsPerBlock = statesPerBlock * m;

	*outThreadsPerBlock = maxThreadsPerBlock;

	long long int numBlocksOnGrid = sizeOfData / statesPerBlock;
	*outBlocksOnGrid = numBlocksOnGrid > maxGridSize ? maxGridSize
			: numBlocksOnGrid;

	*outNumIterations = numBlocksOnGrid / maxGridSize;
	*outRestThreadsToExecute = (numBlocksOnGrid % maxGridSize) * m;
}

long long int * executeBuildGraphKernelWithBlockDivision( int *regulationMatrix,
		int m,
		int n,
		long long int *outSizeCountSolutions ) {

	printf(
			"\n----------- Start: executeBuildGraphKernelWithBlockDivision --------- \n\n" );

	// Part 1: define kernel configuration
	int sizeMatrix = (m * n);

	long long int sizeCountSolutions = (long long int) pow( 2.0, m );
	*(outSizeCountSolutions) = sizeCountSolutions;

	long long int numBlocksOnGrid = 0;
	long long int numThreadsPerBlock = 0;
	long long int numIterations = 0;
	long long int restThreadsToExecute = 0;

	calculateKernelLaunchConfiguration2( m, sizeCountSolutions,
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
		printf( "Host error: countSolutions Memory Allocation \n" );
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

	for (int i = 0; i < totalNumIterationsKernel; i++) {

		// Part 5 of 6: launch kernel
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate( &start );
		cudaEventCreate( &stop );
		cudaEventRecord( start, 0 );

		int sizeCountSolutionsToProcess;
		if( i == (totalNumIterationsKernel - 1) && restThreadsToExecute > 0 ) {
			// Last iteration
			sizeCountSolutionsToProcess = restThreadsToExecute / m;

			calculateKernelLaunchConfiguration2( m,
					sizeCountSolutionsToProcess, &numThreadsPerBlock,
					&numBlocksOnGrid, &numIterations, &restThreadsToExecute );

			dim3 dimBlock2( numThreadsPerBlock );
			dim3 dimGrid2( numBlocksOnGrid );

			dimBlock = dimBlock2;
			dimGrid = dimGrid2;
			//			sizeCountSolutionsToProcess = restThreadsToExecute(
			//					(dimBlock.x / m) * dimGrid.x );

		} else {
			sizeCountSolutionsToProcess = ((dimBlock.x / m) * dimGrid.x);
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

		kernelBuildGraphStatesWithBlockDivision<<< dimGrid, dimBlock>>>(m, n, regulationMatrixDev, countSolutionsPartialDev, sizeCountSolutions, offset);

		cudaThreadSynchronize();

		checkCUDAError( "Kernel execution" );

		// copiar para o host usando + offset para os indices
		cudaMemcpy( countSolutionsPartial, countSolutionsPartialDev,
				memCountSolutionsToProcess, cudaMemcpyDeviceToHost );

		checkCUDAError( "countSolutionsPartial MemCpy" );
		offset += sizeCountSolutionsToProcess;

		// Compute time of kernel execution in milliseconds
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &time, start, stop );
		cudaEventDestroy( start );
		cudaEventDestroy( stop );

		printf( "Iteration %d time (s): %f \n", i, (time / 1000) );
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

	printf( "............End: executeBuildGraphKernelWithBlockDivision \n\n" );

	return graph;

}
