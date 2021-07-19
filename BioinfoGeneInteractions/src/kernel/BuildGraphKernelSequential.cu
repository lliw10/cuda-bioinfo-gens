// includes, system
#include <stdio.h>
#include <math.h>
#include "KernelUtils.h"
#include "HostUtils.h"
#include "BuildGraphKernelSequential.h"

/*****************************************************
 *
 *
 * 				SEQUENTIAL ALGORITHMS
 *
 * - Execute sequential algorithms for gene network analisys
 * - Implementation base for time compare.
 *
 *****************************************************/

void kernelBuildGraphStatesSequentialByDinamicProgramming( int *regulationMatrix,
		int regMatrixSizeX,
		int regMatrixSizeY,
		long long int *countSolutions,
		long long int sizeCountSolutions ) {

	int *cache = (int*) malloc( sizeof(int) * regMatrixSizeX * regMatrixSizeY );
	int *cacheSum = (int*) malloc( sizeof(int) * regMatrixSizeY );

	for (int i = 0; i < (regMatrixSizeX * regMatrixSizeY); i++) {
		cache[i] = 0;
		if( i < regMatrixSizeY ) {
			cacheSum[i] = 0;
		}
	}

	// Dimanic Programming algorithm: buildGraphStates
	for (long long int i = 0; i < sizeCountSolutions; i++) {

		// ^ is XOR bit operation
		long long int v1 = i ^ (i >> 1);
		long long int graycodeBef = (i - 1) ^ ((i - 1) >> 1);

		long int bitChanged = (graycodeBef ^ v1);
		long int bitChangedIdx = (bitChanged == 0 ? 0 : log2(
				(float) bitChanged ));

		long long int v2 = 0;
		long long int ONE = 1;

		long long int col = (regMatrixSizeX - bitChangedIdx - 1);

		// bitQPermanent = (v1 / (long long int) powf( 2, row )) % 2;
		long long int bit = (v1 / (ONE << (regMatrixSizeX - col - 1))) & ONE;
		long long int bitBef = (graycodeBef / (ONE
				<< (regMatrixSizeX - col - 1))) & ONE;

		long long int number = v1;

		for (int row = 0; row < regMatrixSizeX && number > 0; row++) {
			int idxMatrix = row * regMatrixSizeY + col;

			//			cacheSum[row] -= cache[idxMatrix];
			//
			//			cache[idxMatrix] = regulationMatrix[idxMatrix] * bit;
			//
			//			cacheSum[row] += cache[idxMatrix];
			cacheSum[row] += (regulationMatrix[idxMatrix] * bit)
					- (regulationMatrix[idxMatrix] * bitBef);
		}

		// Normalization
		for (int row = 0; row < regMatrixSizeX && number > 0; row++) {
			long long int bitPathernV2 = 0;

			long long int bitQPermanent = (v1 / (ONE << (regMatrixSizeX - row
					- 1))) & 1;

			bitPathernV2 = cacheSum[row] == 0 ? bitQPermanent : cacheSum[row]
					< 0 ? 0 : 1;

			long long int one = 1;
			//			v2 += bitPathernV2 * (long long int) powf( 2, (regMatrixSizeX - row
			//					- 1) );
			v2 += bitPathernV2 * (one << (regMatrixSizeX - row - 1));
		}

		// Exists a vertex from v1 to v2
		countSolutions[v1] = v2;
	}

	free( cache );
	free( cacheSum );
}

void kernelBuildGraphStatesSequential( int *regulationMatriz,
		int regMatrixSizeX,
		int regMatrixSizeY,
		long long int *countSolutions,
		long long int sizeCountSolutions ) {

	for (int i = 0; i < sizeCountSolutions; i++) {

		long long int v1 = i;
		long long int v2 = 0;

		for (int row = 0; row < regMatrixSizeY; row++) {
			long long int number = v1;
			long long int bitPathernV2 = 0;
			long long int bitQPermanent = 0;

			for (int col = regMatrixSizeX - 1; (col >= 0) && (number > 0); col--) {

				int idxMatrix = row * regMatrixSizeY + col;

				// Generate digits in reverse order
				//				long long int bitQ = number % 2;
				long long int bitQ = number & 1;

				if( row == col ) {
					bitQPermanent = bitQ;
				}

				bitPathernV2 += regulationMatriz[idxMatrix] * bitQ;

				number /= 2;
			}

			// Normalization
			bitPathernV2 = bitPathernV2 == 0 ? bitQPermanent
					: bitPathernV2 < 0 ? 0 : 1;

			// Exists an arc v1 -> outV2
			long long int one = 1;
			//			v2 += bitPathernV2 * (long long int) powf( 2, (regMatrixSizeX - row
			//					- 1) );
			v2 += bitPathernV2 * (one << (regMatrixSizeX - row - 1));
		}

		// Exists a vertex from v1 to v2

		countSolutions[i] = v2;
	}
}

long long int * executeGeneNetworkKernelSequential( int *regulationMatrix,
		int m,
		int n,
		long long int *outSizeCountSolutions ) {

	printf(
			"\n----------- Start: executeGeneNetworkKernelSequential --------- \n\n" );

	long long int sizeCountSolutions = (long long int) pow( 2.0, m );
	*(outSizeCountSolutions) = sizeCountSolutions;

	long long int memCountSolutions = sizeCountSolutions
			* (long long int) sizeof(long long int);

	long long int *countSolutions;
	countSolutions = NULL;
	countSolutions = (long long int*) malloc( memCountSolutions );
	if( countSolutions == NULL || countSolutions <= 0 ) {
		printf( "Host error: countSolutions Memory Allocation \n" );
		exit( 0 );
	}

	// Calculate total time for algorithm execution
	cudaEvent_t startTotalTime, stopTotalTime;
	float totalTime;
	cudaEventCreate( &startTotalTime );
	cudaEventCreate( &stopTotalTime );
	cudaEventRecord( startTotalTime, 0 );

	kernelBuildGraphStatesSequentialByDinamicProgramming( regulationMatrix, m,
			n, countSolutions, sizeCountSolutions );

	cudaEventRecord( stopTotalTime, 0 );
	cudaEventSynchronize( stopTotalTime );
	cudaEventElapsedTime( &totalTime, startTotalTime, stopTotalTime );
	cudaEventDestroy( startTotalTime );
	cudaEventDestroy( stopTotalTime );

	printf( "kernelBuildGraphStatesSequential time %f s \n", (totalTime / 1000) );

	printf( "............End: executeGeneNetworkKernelSequential \n\n" );

	return countSolutions;
}

long long int * executeGeneNetworkKernelSequentialByDinamicalProgramming( int *regulationMatrix,
		int m,
		int n,
		long long int *outSizeCountSolutions ) {

	printf(
			"\n----------- Start: executeGeneNetworkKernelSequential --------- \n\n" );

	long long int sizeCountSolutions = (long long int) pow( 2.0, m );
	*(outSizeCountSolutions) = sizeCountSolutions;

	long long int memCountSolutions = sizeCountSolutions
			* (long long int) sizeof(long long int);

	long long int *countSolutions;
	countSolutions = NULL;
	countSolutions = (long long int*) malloc( memCountSolutions );
	if( countSolutions == NULL || countSolutions <= 0 ) {
		printf( "Host error: countSolutions Memory Allocation \n" );
		exit( 0 );
	}

	// Calculate total time for algorithm execution
	cudaEvent_t startTotalTime, stopTotalTime;
	float totalTime;
	cudaEventCreate( &startTotalTime );
	cudaEventCreate( &stopTotalTime );
	cudaEventRecord( startTotalTime, 0 );

	kernelBuildGraphStatesSequentialByDinamicProgramming( regulationMatrix, m,
			n, countSolutions, sizeCountSolutions );

	cudaEventRecord( stopTotalTime, 0 );
	cudaEventSynchronize( stopTotalTime );
	cudaEventElapsedTime( &totalTime, startTotalTime, stopTotalTime );
	cudaEventDestroy( startTotalTime );
	cudaEventDestroy( stopTotalTime );

	printf( "kernelBuildGraphStatesSequential time %f s \n", (totalTime / 1000) );

	printf( "............End: executeGeneNetworkKernelSequential \n\n" );

	return countSolutions;
}

