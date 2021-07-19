/**
 * Learn value structure. Store number of ones, zeros and pathern.
 */
typedef struct EDGE {
		long long int v1;
		long long int v2;
} Edge;

/**
 * Global function do the Kernel implementation.
 */
extern unsigned long long int
		* executeKernelLabelComponents( long long int *graph,
				unsigned long long int nVertices );
