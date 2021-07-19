/*
 * Testes.cpp
 *
 *  Created on: 11/04/2013
 *      Author: william
 */

string binary( unsigned long n ) {
	char result[(sizeof(unsigned long) * 8) + 1];
	unsigned index = sizeof(unsigned long) * 8;
	result[index] = '\0';

	do
		result[--index] = '0' + (n & 1);
	while( n >>= 1 );

	return string( result + index );
}

#include "math.h"
void testGrayCode() {
	long int n = 30;
	long int max = (1 << n);

	cout << "Value\t" << "GrayCode\t" << "BitChangedIdx\t" << "BinValue\t"
			<< "BinGrayCode\n";

	for (long int i = 0; i < max; i++) {

		long int graycodeBef = (i - 1) ^ ((i - 1) >> 1);
		long int graycode = i ^ (i >> 1);
		long int bitChanged = (graycodeBef ^ graycode);
		long int bitChangedIdx = (bitChanged == 0 ? 0 : log2(
				(float) bitChanged ));

		//		cout << i << "\t" << graycode << "\t" << "\t" << bitChangedIdx << "\t"
		//				<< binary( i ) << "\t" << binary( graycode ) << "\n";

	}
}
