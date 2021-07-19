/*
 * KernelUtils.h
 *
 *  Created on: 21/02/2011
 *      Author: william
 */
/**
 * Variaveis para parametrização de numero de threads e blocos
 */
static int blockDiv = 1;
static int threadDiv = 1;
/**
 * Armazena o numero de GPU threads - 0 utiliza o numero de GPUs como default. valor > 0,  utiliza o valor.
 * É usado para definir o numero de threads OMP.
 */
static int numberOfGPUThreads = 0;

extern void printBlockAndThreadConfig();

extern void setBlockAndThreadConfig( int numBlocksDiv, int numThreadsDiv );

extern void setNumberOfGPUThreads( int numGPUThreads );

extern int getNumberOfGPUThreads();

/**
 * Print device properties on console.
 * This includes, device name, clock rate,
 * max properties, and more.
 */
extern void printDeviceProps();

/**
 * Print last cuda error message for CUDA runtime errors
 * @param msg Aditional message information for user.
 */
extern bool checkCUDAError( const char *msg );

extern void calculateKernelLaunchConfiguration( long long int sizeOfData,
		long long int *outThreadsPerBlock,
		long long int *outBlocksOnGrid,
		long long int *outNumIterations,
		long long int *outRestThreadsToExecute );

extern void calculateKernelLaunchConfiguration2( long long int sizeOfData,
		long long int *outThreadsPerBlock,
		long long int *outBlocksOnGrid,
		long long int *outNumIterations,
		long long int *outRestThreadsToExecute );

extern void setDeviceFlags();

extern void freeOnCuda( void *pointerToZeroCopyMemory );

extern void finalizeCudaResources();
