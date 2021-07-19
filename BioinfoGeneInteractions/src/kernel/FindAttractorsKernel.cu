// includes, system
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>
#include <set>
#include <list>
#include <map>

#include "KernelUtils.h"
#include "HostUtils.h"
#include "BuildGraphKernel.h"
#include "ProgressMonitor.h"
#include "FindAttractorsKernel.h"

//#ifdef DEBUG
//#define printf printf
//#define printtime printf
//#else
//#define printf
//#define printtime
//#endif
//
//#define DEBUG

using std::map;
using std::list;
using std::set;

// Max seconds to status processing result
#define MAX_SEC_TO_STATUS 15

// Max size of stack inside GPU
#define MAX_STACK_SIZE 128

//#define MAX_MATRIX_SIZE (50*50)

texture<int, 1, cudaReadModeElementType> regulationMatrixTexture;
//__constant__ int regulationMatrixConstant[MAX_MATRIX_SIZE];

texture<int2, 1, cudaReadModeElementType> matrixInfoTexture;
texture<int, 1, cudaReadModeElementType> matrixInfoSizeArrayDevTexture;

__device__ inline bool foundAttractor( long long int circularStack[],
		int top,
		int stackSize ) {

	//int i = ((top - 1) + stackSize) % stackSize;
	int i = ((top - 1) + stackSize) & (stackSize - 1);

	long long int topState = circularStack[top];
	while( i != top && (topState != circularStack[i]) ) {
		//i = ((i - 1) + stackSize) % stackSize;
		i = ((i - 1) + stackSize) & (stackSize - 1);
	}
	return (topState == circularStack[i] && i != top);
}

__global__ void kernelFindAttractorsReducedMatrix( int n,
//		int *matrixInfoSize,
		//		MatrixInfo *matrixInfoArray,
		long long int *attractorSummary,
		long long int sizeCountSolutions,
		long long int offset ) {

	long long int tid = blockDim.x * blockIdx.x + threadIdx.x;
	long long int v1 = tid + offset;

	if( v1 < sizeCountSolutions ) {

		// Initialize stack with v1 on top
		long long int circularStack[MAX_STACK_SIZE];
		int top = 0;
		int stackSize = 1;
		circularStack[top] = v1;

		long long int v2;
		do {
			v2 = 0;
			int matInfoIdx = 0;

			for (int row = 0; row < n; row++) {

				long long int bitPathernV2 = 0;
				int idx = tex1Dfetch( matrixInfoSizeArrayDevTexture, row );
				while( idx-- ) {
					int2 matInfo = tex1Dfetch( matrixInfoTexture, matInfoIdx++ );

					//
					// bit(k) = (val / (2^(k)) ) % 2
					// The value n - col - 1 is because bits values are generated in reverse order.
					// Example to 4 bit word:
					// idx(43210)
					// bit(00001)
					//
					int bitQ = (v1 / (1LL << (n - matInfo.x - 1))) & 1;
					bitPathernV2 += matInfo.y * bitQ;
				}

				// [Performance]
				// Solve warp divergence
				// We want calculate:
				// bitPathernV2 = bitPathernV2 == 0 ? bitQPermanent : bitPathernV2 < 0 ? 0 : 1;
				//
				// bit(k) = (n / (2^(k)) ) % 2
				// int bitQPermanent = (number / (1LL << (n - row - 1))) & 1;
				// bitPathernV2 = (bitPathernV2 == 0 && bitQPermanent) || (bitPathernV2 > 0);
				//
				bitPathernV2 = (bitPathernV2 > 0) || (bitPathernV2 == 0 && (v1
						/ (1LL << (n - row - 1))) & 1);

				// Exists an arc v1 -> v2
				// [Performance] Operation:
				// v2 += bitPathernV2 * (long long int) powf( 2, (regMatrixSizeX - row	- 1) );
				v2 += bitPathernV2 * (1LL << (n - row - 1));
			}

			// push
			stackSize = (stackSize + 1) < MAX_STACK_SIZE ? (stackSize + 1)
					: MAX_STACK_SIZE;
			//top = (top + 1) % stackSize;
			top = (top + 1) & (stackSize - 1);
			circularStack[top] = v2;

			sizeCountSolutions--;

			v1 = v2;

		} while( !foundAttractor( circularStack, top, stackSize )
				&& sizeCountSolutions > 0 );

		// At this point, v2 is an attractor state.
		// Thus, find minor attractor state identifier
		// Each thread writte on our on position index (no conflit and coalesed writte)
		attractorSummary[tid] = v2;
	}
}

__global__ void kernelFindAttractors( int n,
		int *regulationMatrix,
		long long int *attractorSummary,
		long long int sizeCountSolutions,
		long long int offset ) {

	long long int tid = blockDim.x * blockIdx.x + threadIdx.x;
	long long int v1 = tid + offset;

	if( v1 < sizeCountSolutions ) {

		// Initialize stack with v1 on top
		long long int circularStack[MAX_STACK_SIZE];
		int top = 0;
		int stackSize = 1;
		circularStack[top] = v1;

		long long int v2 = 0;
		do {
			v2 = 0;
			for (int row = 0; row < n; row++) {
				long long int number = v1;
				long long int bitPathernV2 = 0;
				long long int bitQPermanent = 0;

				for (int col = n - 1; (col >= 0) && (number > 0); col--) {

					// Generate digits in reverse order
					//				long long int bitQ = number % 2;
					int bitQ = number & 1;

					if( row == col ) {
						bitQPermanent = bitQ;
					}

					int idxMatrix = row * n + col;
					//bitPathernV2 += regulationMatrix[idxMatrix] * bitQ;
					bitPathernV2 += tex1Dfetch( regulationMatrixTexture,
							idxMatrix ) * bitQ;
					//					bitPathernV2 += regulationMatrixConstant[idxMatrix] * bitQ;

					// [Desempenho] Operacao realizada: number /= 2;
					number = number >> 1;
				}

				// Solve warp divergence
				//				bitPathernV2 = bitPathernV2 == 0 ? bitQPermanent : bitPathernV2
				//						< 0 ? 0 : 1;
				bitPathernV2 = (bitPathernV2 == 0 && bitQPermanent)
						|| (bitPathernV2 > 0);

				// Exists an arc v1 -> v2
				// [Desempenho] Operacao realizada:
				// v2 += bitPathernV2 * (long long int) powf( 2, (regMatrixSizeX - row	- 1) );

				v2 += bitPathernV2 * (1LL << (n - row - 1));
			}

			// push
			stackSize = (stackSize + 1) < MAX_STACK_SIZE ? (stackSize + 1)
					: MAX_STACK_SIZE;
			top = (top + 1) % stackSize;
			circularStack[top] = v2;

			sizeCountSolutions--;

			v1 = v2;

		} while( !foundAttractor( circularStack, top, stackSize )
				&& sizeCountSolutions > 0 );

		// At this point, v2 is an attractor state.
		// Thus, find minor attractor state identifier
		// Each thread writte on our on position index (no conflit and coalesed writte)
		attractorSummary[tid] = v2;
	}
}

long long int nextState( long long int v1,
		int regMatrixSizeX,
		int regMatrixSizeY,
		int *regulationMatrix ) {

	long long int v2 = 0;
	for (int row = 0; row < regMatrixSizeX; row++) {
		long long int number = v1;
		long long int bitPathernV2 = 0;
		long long int bitQPermanent = 0;

		for (int col = regMatrixSizeX - 1; (col >= 0) && (number > 0); col--) {

			int idxMatrix = row * regMatrixSizeY + col;

			// Generate digits in reverse order
			int bitQ = number & 1; // number % 2

			if( row == col ) {
				bitQPermanent = bitQ;
			}

			bitPathernV2 += regulationMatrix[idxMatrix] * bitQ;

			number /= 2;
		}

		// Normalization
		bitPathernV2 = bitPathernV2 == 0 ? bitQPermanent : bitPathernV2 < 0 ? 0
				: 1;

		// Exists an arc v1 -> v2
		//	int value = bitPathernV2 * powf( 2, (regMatrixSizeY - row - 1) );
		long long int one = 1;
		long long int value = bitPathernV2
				* (one << (regMatrixSizeX - row - 1));

		v2 += value;
	}
	return v2;
}

void findAllAttractorStates( long long int v,
		int *regulationMatrix,
		int m,
		int n,
		list<long long int> *atractorsList ) {

	list<long long int> stackFinder;
	set<long long int> visitedSet;

	stackFinder.push_back( v );
	visitedSet.insert( v );

	long long int v2 = -1;
	bool found = false;

	while( !found ) {
		long long int v1 = stackFinder.back();
		v2 = nextState( v1, m, n, regulationMatrix );

		if( visitedSet.count( v2 ) <= 0 ) {
			// Not founded
			stackFinder.push_back( v2 );
			visitedSet.insert( v2 );
		} else {
			// Founded - Cycle
			found = true;
		}
	}

	if( !stackFinder.empty() && v2 != -1 ) {
		long long int vAtractor = -1;
		while( stackFinder.size() > 0 && vAtractor != v2 ) {

			vAtractor = stackFinder.back();
			stackFinder.pop_back();

			atractorsList->push_back( vAtractor );
		}
	}
}

int2 * buildMatrixInfo( int *regulationMatrix,
		int n,
		int **outMatrixInfoSizeArray,
		int *outMatrixInfoTotalSize ) {
	(*outMatrixInfoSizeArray) = (int*) malloc( n * sizeof(int) );
	int matrixInfoTotalSize = 0;
	for (int row = 0; row < n; row++) {
		int size = 0;
		for (int col = 0; col < n; col++) {
			int idxMatrix = row * n + col;
			int value = regulationMatrix[idxMatrix];
			if( value != 0 ) {
				size++;
			}
		}
		matrixInfoTotalSize += size;
		(*outMatrixInfoSizeArray)[row] = size;
	}

	(*outMatrixInfoTotalSize) = matrixInfoTotalSize;
	int2 *matrixInfoArray = (int2*) malloc( matrixInfoTotalSize * sizeof(int2) );
	int matInfoIdx = 0;
	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			int idxMatrix = row * n + col;
			int value = regulationMatrix[idxMatrix];
			if( value != 0 ) {
				matrixInfoArray[matInfoIdx].x = col;
				matrixInfoArray[matInfoIdx].y = value;

				matInfoIdx++;
			}
		}
	}
	return matrixInfoArray;
}

map<long long int, long long int> * executeFindAttractorsKernel( int *regulationMatrix,
		int m,
		int n,
		unsigned long long int *outSizeCountSolutions,
		ProgressMonitor *progressMonitor ) {

	printf( "\n----------- Start: executeFindAttractorsKernel --------- \n\n" );

	unsigned long long int sizeCountSolutions = (long long int) pow( 2.0, m );
	*(outSizeCountSolutions) = sizeCountSolutions;
	long long int sizeMatrix = (m * n);
	long long int memMatrixSize = sizeMatrix * (long long int) sizeof(int);
	long long int memCountSolutions = sizeCountSolutions
			* (long long int) sizeof(long long int);

	// Reduced representation of matrix
	int2 *matrixInfoArray = NULL;
	int *matrixInfoSizeArray = NULL;
	int matrixInfoTotalSize = 0;
	matrixInfoArray = buildMatrixInfo( regulationMatrix, n,
			&matrixInfoSizeArray, &matrixInfoTotalSize );

	long long int memMatrixInfoSizeArray = n * (long long int) sizeof(int);
	long long int memMatrixInfoArray = matrixInfoTotalSize
			* (long long int) sizeof(int2);

	printf( "Number of genes: %d \n", m );
	printf( "State Space Size (2^%d) = %lld \n", m, sizeCountSolutions );
	// 1MB = 1024^2
	printf( "Total Memory required (%lld  + %lld) = %f MB \n", memMatrixSize,
			memCountSolutions, ((double) (memMatrixSize + memCountSolutions)
					/ (1024.0 * 1024.0)) );
	printBlockAndThreadConfig();

	map<long long int, long long int> *globalCountMap = new map<long long int,
			long long int> ();

	// Executa número de threads informado - uma para cada GPU - caso exista mais de uma.
	int configNumberOfGPUThreads = getNumberOfGPUThreads();
	int deviceCount = 0;
	cudaGetDeviceCount( &deviceCount );

	// Number of processing result threads (one by default)
	int ompProcessResultThreadCount = 1;
	int ompThreadCount = configNumberOfGPUThreads == 0 ? deviceCount
			: configNumberOfGPUThreads;
	int ompThreadResultId = (ompProcessResultThreadCount + ompThreadCount - 1);
	omp_set_num_threads( ompProcessResultThreadCount + ompThreadCount );

	printf( "\n\n" );
	printf(
			"--------------- [%d] Device%s - Running %d OMPDeviceThread%s and %d OMPResultThread ---------------",
			deviceCount, (deviceCount > 1 ? "s" : ""), ompThreadCount,
			(ompThreadCount > 1 ? "s" : ""), ompProcessResultThreadCount );
	printf( "\n\n" );

	clock_t startTotalExecTime = clock();

	list<long long int*> queueProcessing;
	list<long long int> queueProcessingSize;
	int finishCount = 0;
	float kernelTotalExecTime = 0;

	// Realiza a execução para cada GPU em uma thread separada
#pragma omp parallel shared(deviceCount, ompThreadCount, ompThreadResultId, finishCount, kernelTotalExecTime, progressMonitor, globalCountMap, queueProcessing, queueProcessingSize, sizeCountSolutions, memMatrixSize)
	{
		clock_t timeToStatus = clock();
		int deviceId = omp_get_thread_num();
		if( deviceId == ompThreadResultId ) {
			// Thread id=0 only process results
			printf( "OMPResultThread[%d] - Started...\n", deviceId );
			progressMonitor->progressStarted( ompThreadCount, deviceId, 0, 0 );
			do {
				if( queueProcessing.empty() ) {
					int seconds = 3;
					printf(
							"OMPResultThread[%d] - sleeping %d seconds. Queue size[%zd].\n",
							deviceId, seconds, queueProcessing.size() );

					sleep( seconds );

					printf(
							"OMPResultThread[%d] - wake up. Queue [%zd], FinishCount[%d/%d].\n",
							deviceId, queueProcessing.size(), finishCount,
							ompThreadCount );
				} else {
					long long int *attractorSummary = queueProcessing.front();
					long long int sizeArr = queueProcessingSize.front();
					queueProcessing.pop_front();
					queueProcessingSize.pop_front();

					if( attractorSummary == NULL || attractorSummary == 0 ) {
						printf( "OMPResultThread[%d] - array NULL.\n", deviceId );
					} else {
						for (long long int idxAttr = 0; idxAttr < sizeArr; idxAttr++) {
							if( idxAttr < sizeCountSolutions ) {
								long long int attractorId =
										attractorSummary[idxAttr];
								(*globalCountMap)[attractorId] += 1;
							} else {
								printf(
										"OMPResultThread[%d] [WARN] - consolidate attractor state idx more than number of states: idx = %lld\n",
										deviceId, idxAttr );
							}
						}

						free( attractorSummary );
					}
				}
			} while( (finishCount != ompThreadCount)
					|| !queueProcessing.empty() );

			printf( "OMPResultThread[%d] - Finish...\n", deviceId );
		} else {

			printf( "OMP[Device]Thread[%d] - Started...\n", deviceId );

			bool initError = false;

			long long int sizeCountSolutionsPerThread = sizeCountSolutions
					/ ompThreadCount;

			long long int offset = ((long long int) deviceId)
					* sizeCountSolutionsPerThread;

			// Last thread process rest of division states
			if( (deviceId + 1) == ompThreadCount ) {
				sizeCountSolutionsPerThread += (sizeCountSolutions
						% ompThreadCount);
			}
			long long int numBlocksOnGrid = 0;
			long long int numThreadsPerBlock = 0;
			long long int numIterations = 0;
			long long int restThreadsToExecute = 0;
			calculateKernelLaunchConfiguration( sizeCountSolutionsPerThread,
					&numThreadsPerBlock, &numBlocksOnGrid, &numIterations,
					&restThreadsToExecute );

			long long int totalNumIterationsKernel = numIterations
					+ (restThreadsToExecute <= 0 ? 0 : 1);
			long long int startIdx = 0;
			long long int endIdx = totalNumIterationsKernel;

			dim3 *dimBlockPt = new dim3( numThreadsPerBlock );
			dim3 *dimGridPt = new dim3( numBlocksOnGrid );

			progressMonitor->progressStarted( ompThreadCount, deviceId,
					numBlocksOnGrid, numThreadsPerBlock );

			printf( "Device[%d] - Number of states to analize: %lld \n",
					deviceId, sizeCountSolutionsPerThread );
			printf( "Device[%d] - Number of blocks used: %lld \n", deviceId,
					numBlocksOnGrid );
			printf( "Device[%d] - Number of threads used: %lld \n", deviceId,
					numThreadsPerBlock );
			printf( "Device[%d] - Iterations: %lld + (rest: %lld) = %lld \n",
					deviceId, numIterations, restThreadsToExecute,
					totalNumIterationsKernel );

			// Clears all the runtime state for the current thread
			cudaThreadExit();
			// Use device
			int selectedDeviceId = deviceId % deviceCount;
			cudaSetDevice( selectedDeviceId );
			if( deviceId != selectedDeviceId ) {
				printf(
						"Device[%d] - using device id[%d] unless %d. DeviceCount is [%d].\n",
						deviceId, selectedDeviceId, deviceId, deviceCount );
			}
			if( checkCUDAError( "Error on set device ID" ) ) {
				fprintf( stderr, "Error on set device ID [%d].\n", deviceId );
				initError = true;
			}

			long long int *attractorSummary = NULL;
			long long int *attractorSummaryDev = NULL;
			int *regulationMatrixDev = NULL;
			int *matrixInfoSizeArrayDev = NULL;
			int2 *matrixInfoArrayDev = NULL;

			bool isReducedMatrix = true; // (matrixInfoTotalSize <= 4 * n);
			if( !isReducedMatrix ) {
				// Matrix memory allocation

				cudaMalloc( (void **) &regulationMatrixDev, memMatrixSize );
				if( regulationMatrixDev == NULL || regulationMatrixDev == 0
						|| checkCUDAError( "Matrix Memory Allocation" ) ) {
					fprintf( stderr,
							"Matrix regulationMatrixDev Memory Allocation Device Error.\n" );
					initError = true;
				}
				cudaMemcpy( regulationMatrixDev, regulationMatrix,
						memMatrixSize, cudaMemcpyHostToDevice );
				if( checkCUDAError( "Matrix Memory Allocation" ) ) {
					initError = true;
				}
				cudaBindTexture( 0, regulationMatrixTexture,
						regulationMatrixDev, memMatrixSize );
				if( checkCUDAError( "Matrix Bind Texture Memory" ) ) {
					initError = true;
				}

			} else {
				// Reduced matrix memory allocation
				cudaMalloc( (void **) &matrixInfoSizeArrayDev,
						memMatrixInfoSizeArray );
				if( matrixInfoSizeArrayDev == NULL || matrixInfoSizeArrayDev
						== 0 || checkCUDAError(
						"Matrix matrixInfoSizeArrayDev Memory Allocation" ) ) {
					fprintf( stderr,
							"Matrix matrixInfoSizeArrayDev Memory Allocation Device Error.\n" );
					initError = true;
				}
				cudaMemcpy( matrixInfoSizeArrayDev, matrixInfoSizeArray,
						memMatrixInfoSizeArray, cudaMemcpyHostToDevice );
				if( checkCUDAError( "Matrix info size array Memory Allocation" ) ) {
					initError = true;
				}

				cudaMalloc( (void **) &matrixInfoArrayDev, memMatrixInfoArray );
				if( matrixInfoArrayDev == NULL || matrixInfoArrayDev == 0
						|| checkCUDAError(
								"Matrix matrixInfoArrayDev Memory Allocation" ) ) {
					fprintf( stderr,
							"Matrix matrixInfoArrayDev Memory Allocation Device Error.\n" );
					initError = true;
				}
				cudaMemcpy( matrixInfoArrayDev, matrixInfoArray,
						memMatrixInfoArray, cudaMemcpyHostToDevice );
				if( checkCUDAError( "Matrix info array Memory Allocation" ) ) {
					initError = true;
				}

				cudaBindTexture( 0, matrixInfoSizeArrayDevTexture,
						matrixInfoSizeArrayDev, memMatrixInfoSizeArray );
				if( checkCUDAError(
						"Matrix matrixInfoSizeArrayDevTexture Bind Texture Memory" ) ) {
					initError = true;
				}

				cudaBindTexture( 0, matrixInfoTexture, matrixInfoArrayDev,
						memMatrixInfoArray );
				if( checkCUDAError(
						"Matrix matrixInfoTexture Bind Texture Memory" ) ) {
					initError = true;
				}
			}
			map<long long int, long long int> countMap;
			long long int previousMemCountSolutionsToProcess = -1;

			cudaEvent_t startTotalTime, stopTotalTime;
			float totalTime;
			cudaEventCreate( &startTotalTime );
			cudaEventCreate( &stopTotalTime );
			cudaEventRecord( startTotalTime, 0 );

			printf(
					"Device[%d] - Workspace StartIdx[%lld], EndIdx[%lld], startOffset[%lld] , InitError: %d\n",
					deviceId, startIdx, endIdx, offset, initError );

			if( !initError ) {
				for (long long int itIdx = startIdx; itIdx < endIdx; itIdx++) {

					cudaEvent_t start, stop;
					float kernelTime;
					cudaEventCreate( &start );
					cudaEventCreate( &stop );
					cudaEventRecord( start, 0 );

					// Compute iteration and memory size
					long long int sizeCountSolutionsToProcess = 0;
					long long int memCountSolutionsToProcess = 0;
					if( itIdx == (endIdx - 1) && restThreadsToExecute > 0 ) {
						// The last iteration: recompute block and grid size
						sizeCountSolutionsToProcess = restThreadsToExecute;
						long long int numBlocksOnGridRest = 0;
						long long int numThreadsPerBlockRest = 0;
						long long int numIterationsRest = 0;
						long long int restThreadsToExecuteRest = 0;
						calculateKernelLaunchConfiguration(
								sizeCountSolutionsToProcess,
								&numThreadsPerBlockRest, &numBlocksOnGridRest,
								&numIterationsRest, &restThreadsToExecuteRest );

						if( dimBlockPt ) {
							delete dimBlockPt;
						}
						if( dimGridPt ) {
							delete dimGridPt;
						}
						dimBlockPt = new dim3( numThreadsPerBlockRest );
						dimGridPt = new dim3( numBlocksOnGridRest );
						printf( "nT=%lld, nb = %lld\n", numThreadsPerBlockRest,
								numBlocksOnGridRest );
					}

					dim3 dimBlock = (*dimBlockPt);
					dim3 dimGrid = (*dimGridPt);

					sizeCountSolutionsToProcess = dimBlock.x * dimGrid.x;

					memCountSolutionsToProcess = sizeCountSolutionsToProcess
							* (long long int) sizeof(long long int);

					// Only allocate memory when need
					if( previousMemCountSolutionsToProcess
							!= memCountSolutionsToProcess ) {

						if( attractorSummaryDev != NULL ) {
							cudaFree( attractorSummaryDev );
							if( checkCUDAError( "attractorSummaryDev [free]" ) ) {
								break;
							}
						}

						printf(
								"Device[%d] - CountSolutionsPartial [Size: %lld] [MEM: %2.2f MB] \t Kernel[Blocks: %d,  Threads: %d] \n",
								deviceId, sizeCountSolutionsToProcess,
								(((double) memCountSolutionsToProcess)
										/ (1024.0 * 1024.0)), dimGrid.x,
								dimBlock.x );

						cudaMalloc( (void **) &attractorSummaryDev,
								memCountSolutionsToProcess );
						if( checkCUDAError( "attractorSummaryDev [malloc]" ) ) {
							break;
						}

						previousMemCountSolutionsToProcess
								= memCountSolutionsToProcess;
					}

					// This allocated memory will be released by thread[0]
					attractorSummary = (long long int*) malloc(
							memCountSolutionsToProcess );
					if( attractorSummary == NULL || attractorSummary == 0 ) {
						fprintf( stderr,
								"Host attractorSummary malloc space error.\n" );
						break;
					}

					char *kernelType;
					if( isReducedMatrix ) {
						kernelFindAttractorsReducedMatrix <<< dimGrid, dimBlock>>>(n, attractorSummaryDev, sizeCountSolutions, offset );
						kernelType = "kernel reduced_mat";
					} else {
						kernelFindAttractors <<< dimGrid, dimBlock>>>( n, regulationMatrixDev, attractorSummaryDev, sizeCountSolutions, offset );
						kernelType = "kernel general";
					}

					cudaError_t errAsync = cudaThreadSynchronize();
					if( cudaSuccess != errAsync || checkCUDAError(
							"Kernel execution" ) ) {
						fprintf(
								stderr,
								"Cuda Async Error kernelFindAttractorsKernel %s.\n",
								cudaGetErrorString( errAsync ) );
						break;
					}

					// Copy device data to host
					cudaMemcpy( attractorSummary, attractorSummaryDev,
							memCountSolutionsToProcess, cudaMemcpyDeviceToHost );
					if( checkCUDAError(
							"attractorSummaryDev MemCpy DeviceToHost" ) ) {
						fprintf(
								stderr,
								"Device[%d] - attractorSummaryDev MemCpy DeviceToHost error [%s].\n",
								deviceId, cudaGetErrorString( errAsync ) );
						break;
					}

					offset += (sizeCountSolutionsToProcess / ompThreadCount);

					// Compute time of kernel execution in milliseconds
					cudaEventRecord( stop, 0 );
					cudaEventSynchronize( stop );
					cudaEventElapsedTime( &kernelTime, start, stop );
					cudaEventDestroy( start );
					cudaEventDestroy( stop );

					// Critical region where multiple threads publish results
#pragma omp critical
					{
						queueProcessing.push_back( attractorSummary );
						queueProcessingSize.push_back(
								sizeCountSolutionsToProcess );
					}

					double timeSpent = (double) (clock() - timeToStatus)
							/ CLOCKS_PER_SEC;

					if( timeSpent >= MAX_SEC_TO_STATUS || itIdx == 0 || ((itIdx
							+ 1) == endIdx) ) {
						timeToStatus = clock();

						printf(
								"Device[%d] - Iteration %lld/%lld [%2.2f%s] - %s\tIt. Kernel Time(s): %f\tSpent Time(s): %f\tN. Attr: %zd \tQueue: %zd\n",
								deviceId, (itIdx + 1), endIdx, ((itIdx + 1)
										* 100.0) / (double) endIdx, "%",
								kernelType, (kernelTime / 1000.0), timeSpent,
								globalCountMap->size(), queueProcessing.size() );

						progressMonitor->progressStatus( ompThreadCount,
								deviceId, (itIdx + 1), endIdx, ((itIdx + 1)
										* 100.0) / (double) endIdx, kernelTime,
								(long long int) globalCountMap->size(),
								globalCountMap );
					}

				} // end main loop
			} // end if main loop

			// Release all allocated memory
			if( dimBlockPt ) {
				delete dimBlockPt;
			}
			if( dimGridPt ) {
				delete dimGridPt;
			}

			if( attractorSummaryDev != NULL ) {
				cudaFree( attractorSummaryDev );
				checkCUDAError( "attractorSummaryDev [end] Free" );
			}

			if( regulationMatrixDev != NULL ) {
				cudaFree( regulationMatrixDev );
				checkCUDAError( "regulationMatrixDev Free" );

				// Only release
				if( deviceId == (numberOfGPUThreads - 1) ) {
					cudaUnbindTexture( regulationMatrixTexture );
					checkCUDAError( "regulationMatrixTexture Unbind Free" );
				}
			}

			if( matrixInfoSizeArrayDev != NULL ) {
				cudaFree( matrixInfoSizeArrayDev );
				checkCUDAError( "matrixInfoSizeArray Free" );

				if( deviceId == (numberOfGPUThreads - 1) ) {
					cudaUnbindTexture( matrixInfoSizeArrayDevTexture );
					checkCUDAError( "matrixInfoSizeArrayDevTexture Unbind Free" );
				}
			}
			if( matrixInfoArrayDev != NULL ) {
				cudaFree( matrixInfoArrayDev );
				checkCUDAError( "matrixInfoArrayDev Free" );

				if( deviceId == (numberOfGPUThreads - 1) ) {
					cudaUnbindTexture( matrixInfoTexture );
					checkCUDAError( "matrixInfoTexture Unbind Free" );
				}
			}

			// Calculate total time for algorithm execution
			cudaEventRecord( stopTotalTime, 0 );
			cudaEventSynchronize( stopTotalTime );
			cudaEventElapsedTime( &totalTime, startTotalTime, stopTotalTime );
			cudaEventDestroy( startTotalTime );
			cudaEventDestroy( stopTotalTime );

			// Critical region where multiple threads notify finish
#pragma omp critical
			{
				finishCount++;
				kernelTotalExecTime += totalTime;
			}

			printf( "Device[%d] - kernelFindAttractors total time (s): %f \n",
					deviceId, (totalTime / 1000) );

			printf( "OMP[Device]Thread[%d] - Finish\n", deviceId );

		} // else block: GPU Processing OMPThreads

		printf( "OMPThread[%d] - Notify Progress Finish\n", deviceId );
		progressMonitor->notifyThreadFinish( deviceId );
		printf( "OMPThread[%d] - End...\n\n", deviceId );

#pragma omp barrier

	} // End multiple threads section


	// Release host memory
	if( matrixInfoSizeArray != NULL && matrixInfoSizeArray ) {
		free( matrixInfoSizeArray );
	}
	if( matrixInfoArray != NULL && matrixInfoArray ) {
		free( matrixInfoArray );
	}

	bool PRINT_ATTRACTORS = false;
	long long int maxBasinSize = 0;
	int attCount = 1;

	printf( "---------------------------------------------------------\n" );
	for (map<long long int, long long int>::iterator itCountMap =
			globalCountMap->begin(); itCountMap != globalCountMap->end(); ++itCountMap, attCount++) {

		long long int attractorState = itCountMap->first;

		list<long long int> allAttractorStates;
		findAllAttractorStates( attractorState, regulationMatrix, m, n,
				&allAttractorStates );

		bool okAttractor = false;
		for (list<long long int>::iterator itList = allAttractorStates.begin(); itList
				!= allAttractorStates.end(); ++itList) {
			long long int att = (long long int) *itList;

			// Group different attractor states at same basin of attraction.
			if( att != attractorState ) {
				(*globalCountMap)[attractorState] += (*globalCountMap)[att];
				globalCountMap->erase( att );
			} else {
				// O estado procurado faz parte do atrator
				okAttractor = true;
			}
		}

		if( !okAttractor ) {
			printf(
					"Attractor [%lld] assertion ERROR. Attractor state %lld not found at attractor.\n",
					attractorState, attractorState );
		} else {

			// At last get updated basin size
			long long int basinSize =
					globalCountMap->find( attractorState )->second;

			if( maxBasinSize < basinSize ) {
				maxBasinSize = basinSize;
			}

			if( PRINT_ATTRACTORS ) {
				printf( "Attractor [%d] > ", attCount );
				printf( "Size: %zd, ", allAttractorStates.size() );
				printf( "Basin size: %lld,\t", basinSize );
				//printf( "Ref. State: %lld,\t", attractorState );
				printf( "Attractor States:" );
				for (list<long long int>::iterator it =
						allAttractorStates.begin(); it
						!= allAttractorStates.end(); it++) {
					printf( " %lld ", *it );
				}
				printf( "\n" );
			}
		}

	}

	double programTotalTime = (double) (clock() - startTotalExecTime)
			/ CLOCKS_PER_SEC;

	printf( "---------------------------------------------------------\n" );

	printf(
			"\nNumber of Genes: %d, Number of States: %lld, Number of Attractors: %zd \n",
			n, sizeCountSolutions, globalCountMap->size() );

	printf(
			"#ResultStatisticsTitle: NUM_GENES\tSPACE_SIZE\tNUM_ATTRACTORS\tMAX_BASIN_SIZE\tKERNEL_TIME(s)\tTOTAL_TIME(s)\n" );
	printf( "#ResultStatistics: %d\t%lld\t%zd\t%lld\t%0.4f\t%0.4f\n", n,
			sizeCountSolutions, globalCountMap->size(), maxBasinSize,
			(kernelTotalExecTime / 1000), programTotalTime );

	printf( "executeFindAttractorsKernel Device total time (s): %0.3f \n",
			(kernelTotalExecTime / 1000) );

	printf( "executeFindAttractorsKernel Host total time (s): %0.3f \n",
			programTotalTime );

	printf( "executeFindAttractorsKernel Number of Attractors: %zd \n",
			globalCountMap->size() );

	printf( "executeFindAttractorsKernel Max Basin Size: %lld \n", maxBasinSize );

	progressMonitor->progressFinished( ompThreadCount, sizeCountSolutions,
			programTotalTime, globalCountMap );

	return globalCountMap;
}
