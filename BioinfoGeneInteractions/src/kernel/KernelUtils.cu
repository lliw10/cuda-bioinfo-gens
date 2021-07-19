#include <stdio.h>
#include "KernelUtils.h"

void printDeviceProps() {
	cudaDeviceProp deviceProp;
	int deviceCount = 0;
	cudaGetDeviceCount( &deviceCount );

	if( deviceCount == 0 ) {
		printf( "There is no device supporting CUDA\n\n" );
	} else if( deviceCount == 1 )
		printf( "There is 1 device supporting CUDA\n\n" );
	else {
		printf( "There are %d devices supporting CUDA\n\n", deviceCount );
	}

	for (int device = 0; device < deviceCount; device++) {

		cudaGetDeviceProperties( &deviceProp, device );

		printf( "Properties for device [%d] .........: %s\n\n", device,
				deviceProp.name );
		printf( "Clock rate: %d\n", deviceProp.clockRate );
		printf( "  CUDA Capability Major revision number: %d\n",
				deviceProp.major );
		printf( "  CUDA Capability Minor revision number: %d\n",
				deviceProp.minor );
		printf( "Max Grid size: [%d, %d, %d]\n", deviceProp.maxGridSize[0],
				deviceProp.maxGridSize[1], deviceProp.maxGridSize[2] );
		printf( "Max Thread dim: [%d, %d, %d]\n", deviceProp.maxThreadsDim[0],
				deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2] );
		printf( "Max Thread per block: %d\n", deviceProp.maxThreadsPerBlock );
		printf( "Total number of registers available per block: %d\n",
				deviceProp.regsPerBlock );
		printf( "Max multi processor count: %d\n",
				deviceProp.multiProcessorCount );
		printf( "Number of cores: %d\n", (8 * deviceProp.multiProcessorCount) );
		printf( "Warp Size: %d\n", deviceProp.warpSize );
		printf( "Max total const memory: %zd\n", deviceProp.totalConstMem );
		printf( "Max global memory: %zd\n", deviceProp.totalGlobalMem );
		printf( "Can map host memory: %d\n", deviceProp.canMapHostMemory );
		printf( "Device overlap: %d\n", deviceProp.deviceOverlap );
		printf( "Run time limit on kernels: %s\n",
				deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No" );
		printf( "  Integrated: %s\n", deviceProp.integrated ? "Yes" : "No" );
		printf( "  Support host page-locked memory mapping: %s\n",
				deviceProp.canMapHostMemory ? "Yes" : "No" );
		printf(
				"  Compute mode: %s\n",
				deviceProp.computeMode == cudaComputeModeDefault ? "Default (multiple host threads can use this device simultaneously)"
						: deviceProp.computeMode == cudaComputeModeExclusive ? "Exclusive (only one host thread at a time can use this device)"
								: deviceProp.computeMode
										== cudaComputeModeProhibited ? "Prohibited (no host thread can use this device)"
										: "Unknown" );
		printf(
				"..........................................................\n\n" );
	}
}

bool checkCUDAError( const char *msg ) {
	bool error = false;
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err ) {
		fprintf( stderr, "CudaError %s ErrMsg:[%s].\n", msg,
				cudaGetErrorString( err ) );
		error = true;
	}
	return error;
}

void setDeviceFlags() {
	// Used on zero copy
	cudaSetDeviceFlags( cudaDeviceMapHost);
	checkCUDAError( "setDeviceFlags setting cudaDeviceMapHost" );
}

void finalizeCudaResources() {
	printf( "Releasing cuda resources" );
	cudaThreadExit();
	checkCUDAError( "cudaThreadExit on finalize cuda resources" );
}

void freeOnCuda( void *pointerToZeroCopyMemory ) {
	cudaFreeHost( pointerToZeroCopyMemory );
	checkCUDAError( "freeOnCuda Free" );
}

void calculateNumberOfIteractions( long long int sizeOfData,
		long long int totalThreadsExecuted,
		long long int *outNumIterations,
		long long int *outRestThreadsToExecute ) {

	if( totalThreadsExecuted <= sizeOfData ) {
		*outNumIterations = sizeOfData / totalThreadsExecuted;
	} else {
		*outNumIterations = 0;
	}
	*outRestThreadsToExecute = sizeOfData % totalThreadsExecuted;
}

void printBlockAndThreadConfig() {
	printf( "Block division factor: %d \n", blockDiv );
	printf( "Threads division factor: %d \n", threadDiv );
	printf( "Number Of GPU Threads: %d \n", numberOfGPUThreads );
}

void setBlockAndThreadConfig( int numBlocksDiv, int numThreadsDiv ) {
	blockDiv = numBlocksDiv;
	threadDiv = numThreadsDiv;
}

void setNumberOfGPUThreads( int numGPUThreads ) {
	numberOfGPUThreads = numGPUThreads;
}

int getNumberOfGPUThreads() {
	return numberOfGPUThreads;
}

void calculateKernelLaunchConfiguration( long long int sizeOfData,
		long long int *outThreadsPerBlock,
		long long int *outBlocksOnGrid,
		long long int *outNumIterations,
		long long int *outRestThreadsToExecute ) {

	cudaDeviceProp deviceProp;
	int device;

	int deviceCount;
	cudaGetDeviceCount( &deviceCount );

	long long int maxGridSize = 0;
	long long int maxThreadsPerBlock = 0;

	for (device = 0; device < deviceCount; ++device) {
		cudaGetDeviceProperties( &deviceProp, device );
		if( device <= deviceCount ) {
			if( deviceProp.major == 9999 && deviceProp.minor == 9999 ) {
				printf( "There is no device supporting CUDA.\n" );
			}
		}

		cudaGetDevice( &device );
		cudaGetDeviceProperties( &deviceProp, device );

		int blockDivSize = (blockDiv <= 0) ? 1 : blockDiv;
		int threadDivSize = (threadDiv <= 0) ? 1 : threadDiv;

		maxThreadsPerBlock = deviceProp.maxThreadsPerBlock / threadDivSize;
		maxGridSize = (deviceProp.maxGridSize[0] + 1) / blockDivSize;
		if( maxGridSize > deviceProp.maxGridSize[0] ) {
			maxGridSize = deviceProp.maxGridSize[0];
		}

		break;
	}

	*outThreadsPerBlock = maxThreadsPerBlock > sizeOfData ? sizeOfData
			: maxThreadsPerBlock;

	long long int numBlocksOnGrid = (sizeOfData + ((*outThreadsPerBlock) - 1))
			/ (*outThreadsPerBlock);
	*outBlocksOnGrid = numBlocksOnGrid > maxGridSize ? maxGridSize
			: numBlocksOnGrid;

	calculateNumberOfIteractions( sizeOfData, (*outBlocksOnGrid)
			* (*outThreadsPerBlock), outNumIterations, outRestThreadsToExecute );
}
