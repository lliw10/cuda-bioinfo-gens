/*
 * MatrixFunctionConverterExecutor.cpp
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
#include "GeneExecutor.h"
#include "DataProvider.h"
#include "MatrixFunctionConverterExecutor.h"

using namespace std;

MatrixFunctionConverterExecutor::MatrixFunctionConverterExecutor( DataProvider *_dataProvider ) :
	GeneExecutor( "MatrixFunctionConverterExecutor" ) {
	dataProvider = _dataProvider;
}

MatrixFunctionConverterExecutor::~MatrixFunctionConverterExecutor() {
}

void convertToCnetFormat( int *regulationMatrix, int n );

void MatrixFunctionConverterExecutor::doExecute() {

	int m = -1;
	int n = -1;

	int *regulationMatrix = dataProvider->getNextMatrix( &m, &n );

	if( regulationMatrix != NULL ) {

		printMatrix( "#", "Matrix read ", regulationMatrix, m, (m * n) );
		printf( "\n" );

		convertToCnetFormat( regulationMatrix, n );

		// free host memory
		free( regulationMatrix );
	}
}

void convertToCnetFormat( int *regulationMatrix, int n ) {

	printf( "# Number of nodes (genes)\n" );
	printf( ".v %d\n", n );
	for (int node = 0; node < n; node++) {

		list<long long int> pres;
		set<long long int> presSet;
		int row = node;
		for (int col = n - 1; (col >= 0); col--) {
			int idxMatrix = row * n + col;
			long long int a = regulationMatrix[idxMatrix];
			if( a != 0 ) {
				pres.push_front( col );
				presSet.insert( col );
			}
		}

		map<long long int, long long int> outputByInput;

		bool dependsOnItself = false;
		long long int maxValue = (long long int) pow( 2.0, pres.size() );
		for (long long int value = 0; value < maxValue; value++) {

			long long int outputValue = 0;

			long long int idxVal = pres.size() - 1;
			for (list<long long int>::iterator itList = pres.begin(); itList
					!= pres.end(); ++itList, --idxVal) {
				long long int col = (long long int) *itList;

				int idxMatrix = row * n + col;

				long long int bitQ = (value / (1LL << idxVal)) & 1LL; // bit(k) = (n / 2^k) MOD 2

				outputValue += regulationMatrix[idxMatrix] * bitQ;
			}

			outputByInput[value] = outputValue;

			if( outputValue == 0 ) {
				dependsOnItself = true;
			}
		}

		list<long long int> orderedOuput;
		for (map<long long int, long long int>::iterator itMap =
				outputByInput.begin(); itMap != outputByInput.end(); ++itMap) {
			orderedOuput.push_back( itMap->first );
		}
		orderedOuput.sort();

		bool hasAditionalBit = dependsOnItself && presSet.count( row ) == 0;

		printf( "\n# Predecessor Nodes and Funcions\n" );
		printf( ".n %d %zd", node + 1, pres.size() + (hasAditionalBit ? 1 : 0) );

		int nBits = 1;
		if( hasAditionalBit ) {
			nBits = 2;
			printf( " %lld", (row + 1LL) );
		}
		for (list<long long int>::iterator itList = pres.begin(); itList
				!= pres.end(); ++itList) {
			printf( " %lld", (*itList) + 1 );
		}

		printf( "\n" );

		for (long long int aditionalBit = 0; aditionalBit < nBits; aditionalBit++) {

			for (list<long long int>::iterator itList = orderedOuput.begin(); itList
					!= orderedOuput.end(); ++itList) {

				long long int inputPred = *itList;
				long long int outputValue =
						outputByInput.find( inputPred )->second;

				// Normalization
				long long int bitPathern = outputValue == 0 ? aditionalBit
						: outputValue < 0 ? 0 : 1;

				// Nao tem bit adicional porem o ouput é zero, utiliza o valor do bit i
				// como saida da funcao
				if( !hasAditionalBit && outputValue == 0 ) {
					long long int idxFixedPre = pres.size() - 1;
					for (list<long long int>::iterator itList = pres.begin(); itList
							!= pres.end(); ++itList) {
						if( row == *itList ) {
							break;
						} else {
							idxFixedPre--;
						}
					}
					bitPathern = (inputPred / (1LL << idxFixedPre)) & 1;
				}
				if( bitPathern != 0 ) {
					break;
				}
			}

			for (list<long long int>::iterator itList = orderedOuput.begin(); itList
					!= orderedOuput.end(); ++itList) {

				long long int inputPred = *itList;
				long long int outputValue =
						outputByInput.find( inputPred )->second;

				// Normalization
				long long int bitPathern = outputValue == 0 ? aditionalBit
						: outputValue < 0 ? 0 : 1;

				// Nao tem bit adicional porem o ouput é zero, utiliza o valor do bit i
				// como saida da funcao
				if( !hasAditionalBit && outputValue == 0 ) {
					long long int idxFixedPre = pres.size() - 1;
					for (list<long long int>::iterator itList = pres.begin(); itList
							!= pres.end(); ++itList) {
						if( row == *itList ) {
							break;
						} else {
							idxFixedPre--;
						}
					}
					bitPathern = (inputPred / (1LL << idxFixedPre)) & 1;
				}

				// Print bits
				if( hasAditionalBit ) {
					printf( "%lld", aditionalBit );
				}

				for (int i = pres.size() - 1; i >= 0; i--) {
					printf( "%lld", (inputPred / (1LL << i)) & 1 );
				}

				// Print function value
				printf( " %lld\n", bitPathern );

			}
		}
	}
}
