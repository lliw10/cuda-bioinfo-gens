#include <map>

using std::map;

/**
 * Global function to find attractors kernel call.
 * @param regulationMatrix
 * @param m
 * @param n
 * @param outSizeCountSolutions
 * @return
 */
extern map<long long int, long long int>
* executeFindAttractorsKernel( int *regulationMatrix,
		int m,
		int n,
		unsigned long long int *outSizeCountSolutions,
		ProgressMonitor *progressMonitor );
