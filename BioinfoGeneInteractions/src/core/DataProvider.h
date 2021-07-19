/*
 * MatrixProvider.h
 *
 *  Created on: 16/08/2011
 *      Author: william
 */
class DataProvider {

	public:
		DataProvider();

		virtual ~DataProvider();

		/**
		 * Return next matrix to process. If no matrix to read exists, result is NULL.
		 * @param m line size of matrix
		 * @param n column size of matrix
		 *
		 * @return next matrix to process.
		 */
		virtual int * getNextMatrix( int *m, int *n );

};

