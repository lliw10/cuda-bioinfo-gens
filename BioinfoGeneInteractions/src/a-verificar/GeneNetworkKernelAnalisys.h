#include <list>
#include <vector>

/**
 * Calculate next stage of a v1 (bitPathernV1).
 * Verify (in parallel) all stages generated within regulationMatriz, until find one that v1 reaches.
 */
extern int executeKernelCalculateNextState( long long int v1,
		int *regulationMatriz,
		int regMatrixSizeX,
		int regMatrixSizeY );

/**
 * Computes number of components that has pathway to vertex v.
 */
extern long
executeKernelAccountBasinOfAtraction( std::list<long> atractorsList,
		bool *statesVisited,
		int sizeStatesVisited,
		int *regulationMatrix,
		int m,
		int n );

extern bool * allocVisitedStatesOnCuda( size_t memVisitedStates );

