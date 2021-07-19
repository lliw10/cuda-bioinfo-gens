/*
 * FileMatrixDataProvider.cpp
 *
 *  Created on: 16/08/2011
 *      Author: william
 */

#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <set>

#include "DataProvider.h"
#include "FileMatrixDataProvider.h"

using namespace std;
using std::set;

FileMatrixDataProvider::FileMatrixDataProvider() :
	DataProvider() {
	infile = NULL;
}

FileMatrixDataProvider::FileMatrixDataProvider( string _filename ) :
	DataProvider() {

	filename = _filename;
	infile = fopen( filename.data(), "r" );
}

FileMatrixDataProvider::~FileMatrixDataProvider() {
	if( infile ) {
		fclose( infile );
	}
}

int * FileMatrixDataProvider::getNextMatrix( int *m, int *n ) {

	// Verify if file was opened
	if( !infile || infile == NULL ) {
		cout << "While opening file " << filename << " an error is encountered"
				<< endl;
		fprintf( stderr, "File %s does not exist\n", filename.data() );

		return NULL;
	}

	// OK File was opened

	int numberOfGenes = 0;
	int ret = 0;
	ret = fscanf( infile, "%d", &numberOfGenes );
	if( ret != 1 ) {
		return NULL;
	}

	*(m) = numberOfGenes;
	*(n) = numberOfGenes;
	int size = numberOfGenes * numberOfGenes;
	int *regulationMatrix = new int[size];
	int idx = 0;

	int geneValue = 0;
	for (int i = 0; i < numberOfGenes; i++) {
		for (int j = 0; j < numberOfGenes; j++) {

			ret = fscanf( infile, "%d", &geneValue );
			if( ret != 1 ) {
				printf(
						"Bad file format reading matrix at [%d][%d]. End of file until matrix read complete.\n",
						i, j );
				return NULL;
			}

			regulationMatrix[idx++] = geneValue;

			// Erase
			geneValue = 0;
		}
	}

	return regulationMatrix;
}

/**
 * Generate random value: -1 or 1
 *
 * @return random values: -1 or 1.
 */
int getRandomRegulationValue() {

	// 0 - 1 = -1
	// 1 -1 = 0
	// 2 - 1 = 1

	//	return (rand() % 3) - 1;
	return (rand() % 2) == 0 ? -1 : 1;
}

void FileMatrixDataProvider::generateRandomMatrix( int quantityOfMatrix,
		int matrixSizeN,
		int numberOfInputs ) {

	sleep( 1 );
	// Generate new seed to peseudo random number generator
	srand( (unsigned) time( NULL ) );

	int n = matrixSizeN;

	for (int m = 0; m < quantityOfMatrix; m++) {

		printf( "%4d\n", n );

		for (int i = 0; i < n; i++) {
			set<int> inputIds;
			for (int in = 0; in < numberOfInputs; in++) {
				int inputPosition = rand() % n;
				inputIds.insert( inputPosition );
			}
			for (int input = 0; input < n; input++) {
				int val = 0;
				do {
					val = getRandomRegulationValue();
					// gene 1 -> 1 only negative or zero value
				} while( val == 1 && i == input );
				if( inputIds.count( input ) > 0 ) {
					printf( "%4d", val );
				} else {
					printf( "%4d", 0 );
				}
			}
			printf( "\n" );
		}
		printf( "\n" );
	}
}

