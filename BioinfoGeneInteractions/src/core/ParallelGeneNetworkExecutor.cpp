/*
 * ParallelGeneNetworkExecutor.cpp
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
#include "FindAttractorsKernel.h"
#include "ParallelGeneNetworkExecutor.h"

using namespace std;

ParallelGeneNetworkExecutor::ParallelGeneNetworkExecutor( DataProvider *_dataProvider,
		ProgressMonitor *_progressMonitor ) :
	GeneExecutor( "ParallelGeneNetworkExecutor" ) {
	dataProvider = _dataProvider;
	progressMonitor = _progressMonitor;
}

ParallelGeneNetworkExecutor::~ParallelGeneNetworkExecutor() {
}

/**
 * Implementation of ParallelGeneNetworkExecutor::doExecute to find all attractors of
 * a boolean network.
 */
void ParallelGeneNetworkExecutor::doExecute() {

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

		cout << "Matrix read " << matrixNumber << "\n";
		printMatrix( "Matrix read ", regulationMatrix, m, (m * n) );

		unsigned long long int sizeCountSolutions = 0;

		clock_t start = clock();

		map<long long int, long long int> *globalMap =
				executeFindAttractorsKernel( regulationMatrix, m, n,
						&sizeCountSolutions, progressMonitor );

		double totalTime = ((double) (clock() - start)) / CLOCKS_PER_SEC;
		cout << "\nMatrix[" << matrixNumber << "] " << getName();
		cout << "\nTotal executeFindAttractorsKernel time (s): " << (totalTime)
				<< "\n";

		cout << "---------------------------------------------------------"
				<< endl;

		// free host memory
		free( regulationMatrix );
		delete globalMap;

		matrixNumber++;
		m = -1;
		n = -1;
		regulationMatrix = dataProvider->getNextMatrix( &m, &n );

	}

	finalizeCudaResources();
}

