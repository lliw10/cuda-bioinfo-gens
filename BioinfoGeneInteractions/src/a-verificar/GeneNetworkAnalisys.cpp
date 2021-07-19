/*
 * MMExecutionOperator.cpp
 *
 *  Created on: 13/02/2011
 *      Author: william
 */

#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#include <iostream>
#include <list>
#include <stack>
#include <set>
#include <vector>
#include <map>

#include "HostUtils.h"
#include "KernelUtils.h"

#include "DataProvider.h"
#include "GeneExecutor.h"
#include "GeneNetworkAnalisys.h"
#include "GeneNetworkKernelAnalisys.h"

using namespace std;

GeneNetworkAnalisys::GeneNetworkAnalisys( DataProvider *_dataProvider ) :
	GeneExecutor( "GeneNetworkAnalisys" ) {
	dataProvider = _dataProvider;

}

GeneNetworkAnalisys::~GeneNetworkAnalisys() {
}

void findAtractors( long v,
		int *regulationMatrix,
		int m,
		int n,
		list<long> *atractorsList ) {

	stack<long> stackFinder;
	set<long> visitedSet;

	stackFinder.push( v );
	visitedSet.insert( v );

	long v2 = -1;
	bool found = false;

	while( !found ) {
		long long int v1 = stackFinder.top();
		v2 = executeKernelCalculateNextState( v1, regulationMatrix, m, n );

		if( visitedSet.count( v2 ) <= 0 ) {
			// Not founded
			stackFinder.push( v2 );
			visitedSet.insert( v2 );
		} else {
			// Founded - Cycle
			found = true;
		}
	}

	if( !stackFinder.empty() && v2 != -1 ) {
		int vAtractor = -1;
		while( stackFinder.size() > 0 && vAtractor != v2 ) {

			vAtractor = stackFinder.top();

			stackFinder.pop();

			atractorsList->push_back( vAtractor );
		}
	}
}

int nextStateNotVisited( long long int vState,
		bool *statesVisited,
		long long int numberOfStates ) {

	int vStateResult = vState;

	// while visited, find unvisited state
	while( statesVisited[vStateResult] == true ) {
		vStateResult++;
	}
	return vStateResult;
}

void GeneNetworkAnalisys::doExecute() {

	printDeviceProps();

	// Init device flags need to process (zero-copy)
	setDeviceFlags();

	int m = -1;
	int n = -1;
	int matrixNumber = 0;

	int *regulationMatrix = dataProvider->getNextMatrix( &m, &n );

	while( regulationMatrix != NULL ) {

		clock_t start = clock();

		const long long int numberOfStates = (int) pow( 2.0, m );

		cout << "Matrix read " << matrixNumber << "\n";
		printMatrix( "Matrix read ", regulationMatrix, m, (m * n) );

		map<list<long> , long> resultMap;
		bool *visitedStates = NULL;
		visitedStates
				= allocVisitedStatesOnCuda( numberOfStates * sizeof(bool) );

		int vState = 0;
		int totalComponents = 0;

		while( vState < numberOfStates ) {

			long quantityOfConectedComponents;
			list<long> atractorsList;

			vState
					= nextStateNotVisited( vState, visitedStates,
							numberOfStates );

			findAtractors( vState, regulationMatrix, m, n, &atractorsList );

			quantityOfConectedComponents
					= executeKernelAccountBasinOfAtraction( atractorsList,
							visitedStates, numberOfStates, regulationMatrix, m,
							n );

			resultMap.insert( pair<list<long> , long> ( atractorsList,
					quantityOfConectedComponents ) );

			totalComponents += quantityOfConectedComponents;
		}

		cout << "---------------------------------------------------------\n";
		cout << "Number of atractors: " << resultMap.size() << " \n";
		cout << "Total states reached by atractors: " << totalComponents
				<< " \n";
		cout << "Total states: " << numberOfStates << "\n\n";

		map<list<long> , long>::iterator itMap;
		int idxBasin = 0;
		for (itMap = resultMap.begin(); itMap != resultMap.end(); ++itMap) {

			list<long> atractorsList = itMap->first;
			list<long>::iterator it;

			int i = 0;
			int atractorListSize = atractorsList.size();
			cout << "Atractors: {";
			for (it = atractorsList.begin(); it != atractorsList.end(); ++it) {
				i++;
				if( i == atractorListSize ) {
					cout << *it << "}\n";
				} else {
					cout << *it << ", ";
				}
			}

			cout << "Basin[" << ++idxBasin << "] " << " size: "
					<< itMap->second << "\n";
		}

		double totalTime = ((double) clock() - start) / CLOCKS_PER_SEC;
		cout << "\nMatrix[" << matrixNumber << "] Total execution time: "
				<< (totalTime) << " (s) \n";

		cout << "---------------------------------------------------------\n";

		// zero copy free
		freeOnCuda( visitedStates );
		delete regulationMatrix;

		matrixNumber++;
		m = -1;
		n = -1;
		regulationMatrix = dataProvider->getNextMatrix( &m, &n );
	}
}
