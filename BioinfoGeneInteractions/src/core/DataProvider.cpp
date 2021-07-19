/*
 * MatrixProvider.cpp
 *
 *  Created on: 16/08/2011
 *      Author: william
 */

#include <stdlib.h>
#include <stdio.h>

#include "DataProvider.h"

DataProvider::DataProvider() {

}

DataProvider::~DataProvider() {
}

int * DataProvider::getNextMatrix( int *m, int *n ) {
	// Empty implementation
	*(m) = 0;
	*(n) = 0;
	return NULL;
}
