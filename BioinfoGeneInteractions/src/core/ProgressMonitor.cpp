/*
 * ProgressMonitor.cpp
 *
 *  Created on: 17/01/2013
 *      Author: william
 */

#include <stdlib.h>
#include <stdio.h>
#include <map>

#include "ProgressMonitor.h"

using namespace std;

/**
 * A progress monitor share information over progress execution.
 * @return
 */
ProgressMonitor::ProgressMonitor() {
}

ProgressMonitor::~ProgressMonitor() {
}

void ProgressMonitor::progressStarted( int deviceCount,
		int deviceId,
		int blockDim,
		int threadDim ) {
	//	printf( "ProgressMonitor::progressStarted %d, %d, %d, %d\n", deviceCount, deviceId, blockDim,
	//			threadDim );

}

void ProgressMonitor::progressStatus( int deviceCount,
		int deviceId,
		long long int currentIteration,
		long long int endIteration,
		float processingStatus,
		float kernelTimeIteration,
		long long int numberOfAttractorFounded,
		map<long long int, long long int> *result ) {
	// Nada a ser feito. Uma implementação especifica para cada monitor deve ser realizada

	//	printf(
	//			"ProgressMonitor: Device[%lld] - Iteration %lld/%lld [%2.2f%s]\tKernel time(s): %f\tHost time(s): %f \tN. Att: %d\n",
	//			deviceId, currentIteration, endIteration,
	//			(currentIteration * 100.0) / (double) endIteration, "%",
	//			(kernelTimeIteration / 1000.0), (hostTimeIteration / 1000.0),
	//			result->size() );
}

void ProgressMonitor::notifyThreadFinish( int deviceId ) {
	// nada a ser feito
}

void ProgressMonitor::progressFinished( int deviceCount,
		long long int numberOfStates,
		double executionTime,
		map<long long int, long long int> *globalCountMap ) {

}
