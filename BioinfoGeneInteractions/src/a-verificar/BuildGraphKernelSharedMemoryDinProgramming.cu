// includes, system
#include <stdio.h>
#include <math.h>
#include "KernelUtils.h"
#include "HostUtils.h"
#include "BuildGraphKernelSharedMemoryDinProgramming.h"

const long long int ONE = 1;
const int MATRIX_SIZE = 30;

__device__ long long int grayCode( long long int i ) {
	return i ^ (i >> ONE);
}

__global__ void kernelBuildGraphStatesByDinamicProgramming( int *regulationMatrix,
		int regMatrixSizeX,
		int regMatrixSizeY,
		long long int *graph,
		long long int sizeCountSolutions,
		long long int offset ) {

	__shared__ int cacheSum[MATRIX_SIZE];
	__shared__ int v2;

	long long int v1;
	long long int bit;
	long long int bitBef;
	long int bitChanged;
	long int bitChangedIdx;
	int col;
	int row;
	long long int i;
	long long int graycodeBef;

	i = blockIdx.x * (sizeCountSolutions / gridDim.x) + offset;
	long long int maxBlockState = ((blockIdx.x + 1) * (sizeCountSolutions
			/ gridDim.x)) + offset;
	row = threadIdx.x;
	v2 = 0;

	if( i < maxBlockState ) {
		// Cache initialization by previous state
		// Each block has cache of previous result
		long long int previousState = grayCode( i - 1 );
		int bitPathernV2Init = 0;
		if( i > 0 ) {
			for (int col = regMatrixSizeX - 1; (col >= 0)
					&& (previousState > 0); col--) {

				int idxMatrix = row * regMatrixSizeY + col;

				// Generate digits in reverse order
				//				long long int bitQ = number % 2;
				long long int bitQ = previousState & 1;
				bitPathernV2Init += regulationMatrix[idxMatrix] * bitQ;
				previousState /= 2;
			}
		}
		cacheSum[row] = bitPathernV2Init;
	}

	__syncthreads();

	// Dimanic Programming algorithm: buildGraphStates
	while( i < maxBlockState ) {

		// ^ is XOR bit operation
		v1 = grayCode( i );
		v2 = 0;
		if( v1 > 0 ) {
			graycodeBef = grayCode( i - 1 );
			bitChanged = graycodeBef ^ v1;
			bitChangedIdx = bitChanged == 0 ? 0 : log2( (float) bitChanged );
			col = (regMatrixSizeX - bitChangedIdx - 1);
			bit = (v1 / (ONE << (regMatrixSizeX - col - ONE))) & ONE;
			bitBef = (graycodeBef / (ONE << (regMatrixSizeX - col - ONE)))
					& ONE;

			// For each thread, multiply the column element
			int idxMatrix = row * regMatrixSizeY + col;
			int mat = regulationMatrix[idxMatrix];
			cacheSum[row] += (mat * bit) - (mat * bitBef);

			int bitQPermanent = (v1 / (ONE << (regMatrixSizeX - row - 1))) & 1;
			{
				int bitPathernV2 = cacheSum[row] == 0 ? bitQPermanent
						: cacheSum[row] < 0 ? 0 : 1;
				//			v2 += bitPathernV2 * (long long int) powf( 2, (regMatrixSizeX - row
				//					- 1) );

				// Decimal value transformation, add factor on state v2
				int v2fac = bitPathernV2 * (ONE << (regMatrixSizeX - row - 1));
				atomicAdd( &v2, v2fac );
			}
		}

		__syncthreads();

		// Exists a vertex from v1 to v2
		//		graph[v1] = v2;
		graph[i - offset] = v2;
		i += 1;
	}

}

long long int * executeGeneNetworkKernelByDinamicalProgramming( int *regulationMatrix,
		int m,
		int n,
		long long int *outSizeCountSolutions ) {

	printf(
			"\n----------- Start: executeGeneNetworkKernelByDinamicalProgramming --------- \n\n" );

	// Part 1: define kernel configuration
	int sizeMatrix = (m * n);

	long long int sizeCountSolutions = ((long long int) pow( 2.0, m ));
	*(outSizeCountSolutions) = sizeCountSolutions;

	long long int numBlocksOnGrid = 0;
	long long int numThreadsPerBlock = 0;

	numBlocksOnGrid = 64;
	numThreadsPerBlock = m;

	long long int memMatrixSize = sizeMatrix * sizeof(int);
	long long int memCountSolutions = sizeCountSolutions
			* (long long int) sizeof(long long int);

	dim3 dimBlock( numThreadsPerBlock );
	dim3 dimGrid( numBlocksOnGrid );

	printf( "Number of genes: %d \n", m );
	printf( "Size solution (2^%d) = %lld \n", m, sizeCountSolutions );
	// 1MB = 1024^2
	printf( "Device memory allocated (%lld  + %lld) = %f MB \n", memMatrixSize,
			memCountSolutions, ((float) (memMatrixSize + memCountSolutions)
					/ (1024.0 * 1024.0)) );
	printf( "Number of blocks used: %lld \n", numBlocksOnGrid );
	printf( "Number of threads used: %lld \n", numThreadsPerBlock );

	// Part 2: allocate device memory
	long long int *graph;
	graph = NULL;
	graph = (long long int*) malloc( memCountSolutions );
	if( graph == NULL || graph <= 0 ) {
		printf( "Host error: countSolutions Memory Allocation \n" );
		exit( 0 );
	}

	int *regulationMatrixDev = NULL;

	// Part 4 of 6: host to device copy
	cudaMalloc( (void **) &regulationMatrixDev, memMatrixSize );
	cudaMemcpy( regulationMatrixDev, regulationMatrix, memMatrixSize,
			cudaMemcpyHostToDevice );

	checkCUDAError( "Memory Allocation" );

	long long int offset = 0;

	long long int *countSolutionsPartialDev = NULL;
	long long int *countSolutionsPartial = NULL;

	// Part 5 of 6: launch kernel
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );

	int maxMemSize = 16; // MB

	long long int maxSizeCountSolutions = (maxMemSize * 1024.0 * 1024.0)
			/ ((long long int) sizeof(long long int));

	int numInteractions = sizeCountSolutions / maxSizeCountSolutions;
	if( numInteractions == 0 ) {
		numInteractions = 1;
	}

	long long int previousMemCountSolutionsToProcess = -1;

	for (int i = 0; i < numInteractions; i++) {

		long long int sizeCountSolutionsPartial =
				(i + 1) == numInteractions ? sizeCountSolutions - offset
						: maxSizeCountSolutions;

		long long int memCountSolutionsToProcess = sizeCountSolutionsPartial
				* ((long long int) sizeof(long long int));

		countSolutionsPartial = (graph + offset);

		printf( "Iteration %d/%d ", (i + 1), numInteractions );
		printf( "CountSolutionsPartial [MEM: %f MB] \n",
				(((float) memCountSolutionsToProcess) / (1024.0 * 1024.0)) );

		if( previousMemCountSolutionsToProcess != memCountSolutionsToProcess ) {
			if( countSolutionsPartialDev != NULL ) {
				cudaFree( countSolutionsPartialDev );
				checkCUDAError( "countSolutionsPartialDev [partial] Free" );
			}
			cudaMalloc( (void **) &countSolutionsPartialDev,
					memCountSolutionsToProcess );
			checkCUDAError( "countSolutionsPartialDev [malloc] " );
			previousMemCountSolutionsToProcess = memCountSolutionsToProcess;
		}

		cudaMemcpy( countSolutionsPartialDev, countSolutionsPartial,
				memCountSolutionsToProcess, cudaMemcpyHostToDevice );

		checkCUDAError( "countSolutionsPartialDev Memory Allocation" );

		kernelBuildGraphStatesByDinamicProgramming<<< dimGrid, dimBlock>>>(regulationMatrixDev, m, n, countSolutionsPartialDev, sizeCountSolutionsPartial, offset );

		cudaThreadSynchronize();

		checkCUDAError( "Kernel execution" );

		// Copiar para o host usando + offset para os indices
		cudaMemcpy( countSolutionsPartial, countSolutionsPartialDev,
				memCountSolutionsToProcess, cudaMemcpyDeviceToHost );

		checkCUDAError( "countSolutionsPartial MemCpy" );

		offset += sizeCountSolutionsPartial;

	}

	// Compute time of kernel execution in milliseconds
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	// free device memory
	cudaFree( regulationMatrixDev );
	checkCUDAError( "regulationMatrixDev Free" );

	if( countSolutionsPartialDev != NULL ) {
		cudaFree( countSolutionsPartialDev );
		checkCUDAError( "countSolutionsPartialDev [final] Free" );
	}

	printf( "kernelBuildGraphStates time (s): %f \n", (time / 1000) );

	printf( "............End: executeGeneNetworkKernel \n\n" );

	return graph;
}

