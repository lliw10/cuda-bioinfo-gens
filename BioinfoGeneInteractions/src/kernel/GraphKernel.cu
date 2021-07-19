// includes, system
#include <stdio.h>
#include <math.h>
#include "KernelUtils.h"
#include "HostUtils.h"
#include "GraphKernel.h"

__global__ void kernelLabelComponents( long long int *graph,
		unsigned long long int *components,
		int *hasChange,
		long long int sizeComponents,
		long long offset ) {

	unsigned long long int tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long long int idx = tid;
	if( idx < sizeComponents ) {
		unsigned long long int ev1 = idx + offset;
		unsigned long long int ev2 = graph[idx];

		unsigned long long int cv1 = components[ev1];
		unsigned long long int cv2 = components[ev2];

		if( cv1 < cv2 ) {
			//			atomicMin( &components[e.v2], cv1 );
			components[ev2] = cv1;
			hasChange[0] = 1;
		} else if( cv1 > cv2 ) {
			//			atomicMin( &components[e.v1], cv2 );
			components[ev1] = cv2;
			hasChange[0] = 1;
		}
	}
}

__global__ void kernelInitializeConectedComponents( unsigned long long int *components,
		long long int sizeComponents,
		long long int offset ) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	long long int idx = tid + offset;

	if( idx < sizeComponents ) {
		// comentar (somente para kernelSharedMemory)
		//		idx = (idx ^ (idx >> 1));
		components[idx] = idx;
	}
}

unsigned long long int * executeKernelLabelComponents( long long int *graph,
		unsigned long long int nVertices ) {

	printf( "....Start: executeKernelLabelComponents \n\n" );

	unsigned long long int *components = NULL;
	int *hasChangeHost = NULL;

	long long int *graphDev = NULL;
	unsigned long long int *componentsDev;
	int *hasChangeDev = NULL;

	long long int numThreadsPerBlock;
	long long int numBlocksOnGrid;

	long long int restThreadsToExecute;
	long long int numIterations;

	// Part 1 of 6: define kernel configuration
	// Number of threads per block

	// Part 3 of 6: allocate device memory
	long long int memComponentsSize = nVertices
			* (long long int) sizeof(long long int);
	int memHasChangeSize = sizeof(int);

	hasChangeHost = getPointerToMatrix( 1 );
	cudaMalloc( (void **) &hasChangeDev, memHasChangeSize );
	if( checkCUDAError( "GraphKernel::cudaMalloc. Aborting..." ) ) {
		return NULL;
	}

	//------- Allocate Zero Copy memory -------
	cudaHostAlloc( (void **) &components, memComponentsSize,
			cudaHostAllocMapped );
	if( !components || checkCUDAError(
			"GraphKernel::cudaHostAlloc. Aborting..." ) ) {
		if( !components ) {
			printf(
					"GraphKernel::cudaHostAlloc. Cannot allocate memory of size: %lld.\n",
					memComponentsSize );
		}
		return NULL;
	}

	cudaHostGetDevicePointer( (void **) &componentsDev, (void *) components, 0 );
	if( !componentsDev || checkCUDAError(
			"GraphKernel::cudaHostGetDevicePointer. Aborting..." ) ) {
		printf(
				"GraphKernel::cudaHostGetDevicePointer. Cannot allocate memory of size: %lld.\n",
				memComponentsSize );
		return NULL;
	}

	//-------

	//------- Kernel Initialization Execution
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );

	calculateKernelLaunchConfiguration( (long long int) nVertices,
			(long long int *) &numThreadsPerBlock,
			(long long int *) &numBlocksOnGrid,
			(long long int *) &numIterations,
			(long long int *) &restThreadsToExecute );

	dim3 dimBlockInitVertices( numThreadsPerBlock );
	dim3 dimGridInitVertices( numBlocksOnGrid );

	long long int offset = 0;
	int numIterationsKernelInit = numIterations
			+ (restThreadsToExecute <= 0 ? 0 : 1);

	for (int i = 0; i < numIterationsKernelInit; i++) {
		kernelInitializeConectedComponents <<< dimGridInitVertices, dimBlockInitVertices>>> ( componentsDev, nVertices, offset );
		offset += dimBlockInitVertices.x * dimGridInitVertices.x;
	}

	cudaThreadSynchronize();
	checkCUDAError( "Error: kernelInitializeConectedComponents" );

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	float timeKernelInitialize = (time / 1000);

	int steps = 0;

	cudaEvent_t startTotalTime, stopTotalTime;
	cudaEventCreate( &startTotalTime );
	cudaEventCreate( &stopTotalTime );
	cudaEventRecord( startTotalTime, 0 );
	//-------

	//------- Kernel label components execution

	calculateKernelLaunchConfiguration( (long long int) nVertices,
			&numThreadsPerBlock, &numBlocksOnGrid, &numIterations,
			&restThreadsToExecute );

	dim3 dimBlockLabelComp( numThreadsPerBlock );
	dim3 dimGridLabelComp( numBlocksOnGrid );

	printf( "Number of blocks used: %lld \n", numBlocksOnGrid );
	printf( "Number of threads used: %lld \n", numThreadsPerBlock );
	printf( "Internal iterations: %lld \n", numIterations );

	long long int previousMemGraphSize = -1;

	do {

		if( steps % 1000 == 0 ) {
			cudaEventCreate( &start );
			cudaEventCreate( &stop );
			cudaEventRecord( start, 0 );
		}

		steps++;
		hasChangeHost[0] = 0;

		cudaMemcpy( hasChangeDev, hasChangeHost, memHasChangeSize,
				cudaMemcpyHostToDevice );

		offset = 0;
		for (int i = 0; i < numIterationsKernelInit; i++) {

			// Part 4 of 6: host to device copy
			int graphDevSize;
			if( i == (numIterationsKernelInit - 1) && restThreadsToExecute > 0 ) {
				graphDevSize = restThreadsToExecute;

				calculateKernelLaunchConfiguration( graphDevSize,
						&numThreadsPerBlock, &numBlocksOnGrid, &numIterations,
						&restThreadsToExecute );

				dim3 dimBlock2( numThreadsPerBlock );
				dim3 dimGrid2( numBlocksOnGrid );

				dimBlockLabelComp = dimBlock2;
				dimGridLabelComp = dimGrid2;

			} else {
				graphDevSize = (dimGridLabelComp.x * dimBlockLabelComp.x);
			}

			long long int memGraphSize = graphDevSize
					* (long long int) sizeof(long long int);

			long long int *partialGraph = (graph + offset);

			if( previousMemGraphSize != memGraphSize ) {
				if( graphDev != NULL ) {
					cudaFree( graphDev );
					checkCUDAError( "edgesDev [partial] Free" );
				}
				printf( "GraphPartial size [MEM: %f MB] \n",
						((double) (memGraphSize) / (1024.0 * 1024.0)) );

				cudaMalloc( (void **) &graphDev, memGraphSize );
				previousMemGraphSize = memGraphSize;
			}
			cudaMemcpy( graphDev, partialGraph, memGraphSize,
					cudaMemcpyHostToDevice );
			checkCUDAError( "edgesDev Memory Allocation" );

			kernelLabelComponents <<< dimGridLabelComp, dimBlockLabelComp>>>(graphDev, componentsDev, hasChangeDev, graphDevSize, offset);
			offset += dimGridLabelComp.x * dimBlockLabelComp.x;

			cudaThreadSynchronize();
			checkCUDAError( "Kernel execution" );

		}

		cudaMemcpy( hasChangeHost, hasChangeDev, memHasChangeSize,
				cudaMemcpyDeviceToHost );

		checkCUDAError( "Memory copy" );

		if( steps % 1000 == 0 ) {
			cudaEventRecord( stop, 0 );
			cudaEventSynchronize( stop );
			cudaEventElapsedTime( &time, start, stop );
			cudaEventDestroy( start );
			cudaEventDestroy( stop );
			//			printf( "step: %d - kernelLabelComponents time %f s \n", steps,
			//					(time / 1000) );
		}

	} while( hasChangeHost[0] == 1 );

	offset += dimGridLabelComp.x * dimBlockLabelComp.x;

	cudaThreadSynchronize();
	cudaMemcpy( components, componentsDev, memComponentsSize,
			cudaMemcpyDeviceToHost );

	checkCUDAError( "Memory copy verticesComponentOut " );

	// Calculate total time for algorithm execution
	cudaEventRecord( stopTotalTime, 0 );
	cudaEventSynchronize( stopTotalTime );
	cudaEventElapsedTime( &time, startTotalTime, stopTotalTime );
	cudaEventDestroy( startTotalTime );
	cudaEventDestroy( stopTotalTime );
	float timeKernelLabelComponents = (time / 1000);

	printf( "kernelInitializeConectedComponents time (s): %f \n",
			timeKernelInitialize );
	printf( "kernelLabelComponents time (s): %f \n", timeKernelLabelComponents );
	printf( "Number of steps: %d \n", steps );

	// free device memory
	if( graphDev != NULL ) {
		cudaFree( graphDev );
		checkCUDAError( "edgesDev [end] Free" );
	}
	cudaFree( hasChangeDev );
	checkCUDAError( "hasChangeDev Free" );

	printf( "\n\n....End: executeKernelLabelComponents\n\n" );

	return components;
}
