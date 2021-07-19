/*
 * FileMatrixDataProvider.h
 *
 *  Created on: 16/08/2011
 *      Author: william
 */
#include <string>
#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

using namespace std;

using std::string;
using std::ifstream;

class FileMatrixDataProvider: public DataProvider {

	public:

		FileMatrixDataProvider();

		FileMatrixDataProvider( string _filename );

		virtual ~FileMatrixDataProvider();

		virtual int * getNextMatrix( int *m, int *n );

		virtual void generateRandomMatrix( int quantityOfMatrix,
				int matrixSizeN,
				int numberOfInputs );

	protected:

		string filename;

		FILE * infile;
};

