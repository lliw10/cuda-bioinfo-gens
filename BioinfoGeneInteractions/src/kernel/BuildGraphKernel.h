
/**
 * Global function of the build graph Kernel implementation. This function build all graph states 2^N generated by regulation matrix of size N x N.
 */
extern long long int * executeBuildGraphKernel( int *regulationMatrix,
		int m,
		int n,
		unsigned long long int *outSizeCountSolutions );
