/*
 * GrayCode.cpp
 *
 *  Created on: 27/09/2011
 *      Author: william
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <bitset>

using namespace std;

string binary( unsigned long n ) {
	bitset<sizeof(unsigned long) * 8> bits( n );
	string result =
			bits.to_string<char, char_traits<char> , allocator<char> > ();
	string::size_type x = result.find( '1' );
	if( x != string::npos )
		result = result.substr( x );
	return result;
}

void testGrayCode() {
	int n = 4;
	for (int i = 0; i < (n << 1); i++) {
		int grayCode = i ^ (i >> 1);

	}
}
