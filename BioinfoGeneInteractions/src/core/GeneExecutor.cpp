/*
 * GeneExecutor.cpp
 *
 *  Created on: 09/02/2011
 *      Author: william
 */

#include <iostream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include <time.h>

#include "GeneExecutor.h"
using namespace std;

GeneExecutor::GeneExecutor( string _name ) {
	name = _name;
}

GeneExecutor::~GeneExecutor() {
}

string GeneExecutor::getName() {
	return name;
}

void GeneExecutor::execute() {

	printf( "#Executing: [%s]\n", getName().data() );

	clock_t start = clock();
	doExecute();
	double totalTime = ((double) clock() - start) / CLOCKS_PER_SEC;

	printf( "\n#Total execution time (s): %3f\n", totalTime );

}

void GeneExecutor::doExecute() {
	// do nothing on this implementation only on subclasses
}
