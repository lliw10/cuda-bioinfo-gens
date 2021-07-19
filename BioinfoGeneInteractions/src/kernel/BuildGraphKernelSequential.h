/**
 * Execution of Sequential Kernel implementation.
 */
extern long long int
* executeGeneNetworkKernelSequential( int *regulationMatrix,
		int m,
		int n,
		long long int *outSizeCountSolutions );

extern long long int
		* executeGeneNetworkKernelSequentialByDinamicalProgramming( int *regulationMatrix,
				int m,
				int n,
				long long int *outSizeCountSolutions );
