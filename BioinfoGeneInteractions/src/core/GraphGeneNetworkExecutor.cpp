/*
 * GraphGeneNetworkExecutor.cpp
 *
 *  Created on: 13/02/2011
 *      Author: william
 */

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <list>
#include <map>
#include <stack>
#include <set>

#include "HostUtils.h"
#include "KernelUtils.h"
#include "GraphKernel.h"
#include "GeneExecutor.h"
#include "DataProvider.h"
#include "ProgressMonitor.h"
#include "BuildGraphKernel.h"
//#include "BuildGraphKernelSequential.h"
//#include "BuildGraphKernelSharedMemoryDinProgramming.h"
//#include "BuildGraphKernelWithBlockDivision.h"
#include "FindAttractorsKernel.h"
#include "GraphGeneNetworkExecutor.h"

using namespace std;

GraphGeneNetworkExecutor::GraphGeneNetworkExecutor( DataProvider *_dataProvider,
		ProgressMonitor *_progressMonitor ) :
	GeneExecutor( "GraphGeneNetworkExecutor" ) {
	dataProvider = _dataProvider;
	progressMonitor = _progressMonitor;
}

GraphGeneNetworkExecutor::~GraphGeneNetworkExecutor() {
}

void GraphGeneNetworkExecutor::doExecute() {
	bool printGraph = false;
	int m = -1;
	int n = -1;
	int matrixNumber = 0;

	int *regulationMatrix = dataProvider->getNextMatrix( &m, &n );
	if( regulationMatrix == NULL ) {
		return;
	}

	printDeviceProps();

	// Init device flags need to process (zero-copy)
	setDeviceFlags();

	while( regulationMatrix != NULL ) {

		clock_t start = clock();

		cout << "Matrix read " << matrixNumber << "\n";
		printMatrix( "Matrix read ", regulationMatrix, m, (m * n) );

		unsigned long long int sizeCountSolutions = 0;

		long long int *graph = NULL;

		//		graph = executeGeneNetworkKernelSequentialByDinamicalProgramming(
		//				regulationMatrix, m, n, &sizeCountSolutions );

		//		graph = executeGeneNetworkKernelByDinamicalProgramming(
		//				regulationMatrix, m, n, &sizeCountSolutions );

		//		graph = executeBuildGraphKernelWithBlockDivision( regulationMatrix, m,
		//				n, &sizeCountSolutions );

		graph = executeBuildGraphKernel( regulationMatrix, m, n,
				&sizeCountSolutions );
		// Validation of build graph result
		if( graph == NULL ) {
			printf( "NULL Graph Building. Returning...\n" );
			if( regulationMatrix != NULL ) {
				free( regulationMatrix );
			}
			return;
		}

		//		 DEBUGGING executeGeneNetworkKernelByDinamicalProgramming
		//		for (long long int i = 0; i < sizeCountSolutions; i++) {
		//			long long int grayI = (i ^ (i >> 1L));
		//			long long int s1 = graph[grayI];
		//			long long int s2 = graph2[i];
		//			if( s1 != s2 ) {
		//				cout << "Diff states: " << i << " -> " << s1;
		//				cout << " AND gray(" << i << ") = " << grayI << " -> " << s2
		//						<< "\n";
		//			}
		//		}
		//		long long int sizeCountValidator = 0;
		//		for (long long int i = 0; i < sizeCountSolutions; i++) {
		//			if( graph[i] == -1 ) {
		//				printf( "[Warning] Invalid Edge > %lld \n", i );
		//			}
		//			sizeCountValidator++;
		//		}
		//		if( sizeCountValidator != sizeCountSolutions ) {
		//			cout << "Invalid sizeCountSolutions [sizeCountSolutions = "
		//					<< sizeCountSolutions << "]\n";
		//		}

		// Execute label components

		unsigned long long int *components = executeKernelLabelComponents(
				graph, sizeCountSolutions );

		map<long long int, long long int> countMap;
		map<long long int, list<long long int> > attractorStatesMap;
		if( components ) {
			for (unsigned long long int i = 0; i < sizeCountSolutions; i++) {
				int attractorId = components[i];

				countMap[attractorId] = countMap[attractorId] + 1;

				// Find all attractor states to a connected component (only once)
				if( attractorStatesMap.count( attractorId ) == 0 ) {
					stack<long long int> stackFinder;
					set<long long int> visitedSet;

					stackFinder.push( attractorId );
					visitedSet.insert( attractorId );

					long long int nextState = -1;
					bool found = false;

					while( !found ) {
						long long int currentState = stackFinder.top();
						nextState = graph[currentState];

						if( visitedSet.count( nextState ) <= 0 ) {
							// Not visited
							stackFinder.push( nextState );
							visitedSet.insert( nextState );
						} else {
							// Cycle Founded
							found = true;
						}
					}

					if( !stackFinder.empty() && nextState != -1 ) {
						int vAtractor = -1;
						while( stackFinder.size() > 0 && vAtractor != nextState ) {

							vAtractor = stackFinder.top();

							stackFinder.pop();

							attractorStatesMap[attractorId].push_back(
									vAtractor );
						}
					}
				}
			}
		} // end components processing

		map<long long int, long long int>::iterator itCountMap;
		int i = 1;

		cout << "Number of Genes: " << n << ", Number of States: "
				<< sizeCountSolutions << ", Number of Attractors: "
				<< countMap.size() << endl;

		cout << "---------------------------------------------------------"
				<< endl;
		for (itCountMap = countMap.begin(); itCountMap != countMap.end(); ++itCountMap, i++) {
			long long int attractorId = itCountMap->first;

			list<long long int> attractorStates = attractorStatesMap.find(
					attractorId )->second;

			cout << "Attractor [" << i << "] > ";
			cout << "Size: " << attractorStates.size() << ", ";
			cout << "Basin size: " << itCountMap->second << ",\t";
			cout << "Attractor States:";
			for (list<long long int>::iterator it = attractorStates.begin(); it
					!= attractorStates.end(); it++) {
				cout << " " << *it;
			}

			cout << endl;
		}

		if( printGraph ) {
			cout << "Graph: " << endl;
			for (unsigned long long int i = 0; i < sizeCountSolutions; i++) {
				cout << i << "\t" << graph[i] << endl;
			}
		}

		double totalTime = ((double) clock() - start) / CLOCKS_PER_SEC;
		cout << "\nMatrix[" << matrixNumber << "] GeneNetworkKernelAnalisys";
		cout << "\nTotal matrix execution time (s): " << (totalTime) << "\n";

		cout << "---------------------------------------------------------"
				<< endl;

		// free host memory
		free( regulationMatrix );
		free( graph );
		freeOnCuda( components );

		matrixNumber++;
		m = -1;
		n = -1;
		regulationMatrix = dataProvider->getNextMatrix( &m, &n );

	}

	finalizeCudaResources();
}
